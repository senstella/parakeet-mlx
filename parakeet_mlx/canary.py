from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, cast

import mlx.core as mx
import mlx.nn as nn
from typing_extensions import Literal

from parakeet_mlx.alignment import (
    AlignedResult,
    AlignedToken,
    merge_longest_common_subsequence,
    merge_longest_contiguous,
    sentences_to_result,
    tokens_to_sentences,
)
from parakeet_mlx.audio import PreprocessArgs, get_logmel, load_audio
from parakeet_mlx.cache import TransformerDecoderCache
from parakeet_mlx.conformer import Conformer, ConformerArgs
from parakeet_mlx.tokenizer import CanaryTokenizer
from parakeet_mlx.transformer import (
    TransformerDecoder,
    TransformerDecoderArgs,
    TransformerHead,
    TransformerHeadArgs,
)


@dataclass
class CanaryArgs:
    preprocessor: PreprocessArgs
    encoder: ConformerArgs
    transf_decoder: TransformerDecoderArgs
    head: TransformerHeadArgs
    prompt_format: Literal["canary", "canary2"]
    tokenizer: dict


@dataclass
class CanaryDecodingConfig:
    decoding: Literal["greedy", "beam"] = "beam"
    beam_size: int = 5
    temperature: float = 0.0
    max_length: int = 512


class Canary(nn.Module):
    """Canary model"""

    def __init__(self, args: CanaryArgs):
        super().__init__()

        self.preprocessor_config = args.preprocessor
        self.encoder_config = args.encoder
        self.prompt_format: Literal["canary", "canary2"] = args.prompt_format

        self.tokenizer = CanaryTokenizer.from_data(args.tokenizer["data"])

        self.encoder = Conformer(args.encoder)
        self.transf_decoder = TransformerDecoder(args.transf_decoder)
        self.head = TransformerHead(args.head)

    def transcribe(
        self,
        path: Path | str,
        language: str,
        timestamps: bool = False,
        punctuation: bool = True,
        *,
        dtype: mx.Dtype = mx.bfloat16,
        chunk_duration: Optional[float] = None,
        overlap_duration: float = 15.0,
        chunk_callback: Optional[Callable] = None,
    ) -> AlignedResult | str:
        audio_path = Path(path)
        audio_data = load_audio(audio_path, self.preprocessor_config.sample_rate, dtype)

        if chunk_duration is None:
            prompt_tokens = prompt(
                self.tokenizer,
                self.prompt_format,
                language,
                language,
                punctuation,
                timestamp=timestamps,
            )
            mel = get_logmel(audio_data, self.preprocessor_config)
            return self.generate(mel, [prompt_tokens])[0]

        audio_length_seconds = len(audio_data) / self.preprocessor_config.sample_rate
        if audio_length_seconds <= chunk_duration:
            prompt_tokens = prompt(
                self.tokenizer,
                self.prompt_format,
                language,
                language,
                punctuation,
                timestamp=timestamps,
            )
            mel = get_logmel(audio_data, self.preprocessor_config)
            return self.generate(mel, [prompt_tokens])[0]

        chunk_samples = int(chunk_duration * self.preprocessor_config.sample_rate)
        overlap_samples = int(overlap_duration * self.preprocessor_config.sample_rate)
        all_tokens = []
        previous_text = ""

        for start in range(0, len(audio_data), chunk_samples - overlap_samples):
            end = min(start + chunk_samples, len(audio_data))

            if chunk_callback is not None:
                chunk_callback(end, len(audio_data))

            if end - start < self.preprocessor_config.hop_length:
                break

            if self.format == "canary2" and previous_text:
                prompt_tokens = prompt(
                    self.tokenizer,
                    self.prompt_format,
                    language,
                    language,
                    punctuation,
                    context=previous_text,
                    timestamp=timestamps,
                )
            else:
                prompt_tokens = prompt(
                    self.tokenizer,
                    self.prompt_format,
                    language,
                    language,
                    punctuation,
                    timestamp=timestamps,
                )

            chunk_audio = audio_data[start:end]
            chunk_mel = get_logmel(chunk_audio, self.preprocessor_config)
            chunk_result = self.generate(chunk_mel, [prompt_tokens])[0]

            if chunk_result.text:
                previous_text = chunk_result.text

            chunk_offset = start / self.preprocessor_config.sample_rate
            for sentence in chunk_result.sentences:
                for token in sentence.tokens:
                    token.start += chunk_offset
                    token.end = token.start + token.duration

            if all_tokens:
                try:
                    all_tokens = merge_longest_contiguous(
                        all_tokens,
                        chunk_result.tokens,
                        overlap_duration=overlap_duration
                        if timestamps
                        else chunk_duration,
                    )
                except RuntimeError:
                    all_tokens = merge_longest_common_subsequence(
                        all_tokens,
                        chunk_result.tokens,
                        overlap_duration=overlap_duration
                        if timestamps
                        else chunk_duration,
                    )
            else:
                all_tokens = chunk_result.tokens

        result = sentences_to_result(tokens_to_sentences(all_tokens))
        return result if timestamps else result.text

    def generate(
        self,
        mel: mx.array,
        prompts: list[list[int]],
        *,
        decoding_config: CanaryDecodingConfig = CanaryDecodingConfig(),
    ) -> list[AlignedResult]:
        if len(mel.shape) == 2:
            mel = mx.expand_dims(mel, 0)
        features, lengths = self.encoder(mel)
        mx.eval(features, lengths)
        decoded = self.decode(features, prompts, lengths, config=decoding_config)

        def parse_time(token):
            try:
                s = self.tokenizer.decode([token])
                if s.startswith("<|") and s.endswith("|>"):
                    time_str = s[2:-2]
                    if time_str.isdigit():
                        return (
                            int(time_str)
                            * self.encoder_config.subsampling_factor
                            / self.preprocessor_config.sample_rate
                            * self.preprocessor_config.hop_length
                        )
            except (ValueError, AttributeError, KeyError):
                pass
            return None

        result = []
        for batch_idx, batch in enumerate(decoded):
            aligned = []
            batch_length = float(
                lengths[batch_idx] if batch_idx < len(lengths) else lengths[-1]
            )
            max_time = (
                batch_length
                * self.encoder_config.subsampling_factor
                / self.preprocessor_config.sample_rate
                * self.preprocessor_config.hop_length
            )

            timestamp_indices = {}
            for i, t in enumerate(batch):
                if t in self.tokenizer.special_tokens:
                    time = parse_time(t)
                    if time is not None:
                        timestamp_indices[i] = time

            for i, t in enumerate(batch):
                if t not in self.tokenizer.special_tokens:
                    start = None
                    for j in range(i - 1, -1, -1):
                        if j in timestamp_indices:
                            start = timestamp_indices[j]
                            break
                    if start is None:
                        start = 0.0

                    end = None
                    for j in range(i + 1, len(batch)):
                        if j in timestamp_indices:
                            end = timestamp_indices[j]
                            break
                    if end is None:
                        end = (
                            max_time
                            if i == len(batch) - 1 or not timestamp_indices
                            else start
                        )

                    aligned.append(
                        AlignedToken(
                            t,
                            self.tokenizer.decode([t]),
                            start,
                            max(0, end - start),
                        )
                    )

            result.append(sentences_to_result(tokens_to_sentences(aligned)))

        return result

    def decode(
        self,
        features: mx.array,
        prompt: list[list[int]],
        lengths: Optional[mx.array] = None,
        *,
        config: CanaryDecodingConfig = CanaryDecodingConfig(),
    ) -> list[list[int]]:
        if config.decoding == "greedy":
            outputs = []
            for batch, p in enumerate(prompt):
                tokens = []
                inputs = p.copy()
                cache = [
                    TransformerDecoderCache()
                    for _ in range(len(self.transf_decoder.layers))
                ]

                feat = features[batch : batch + 1]
                if lengths is not None:
                    feat = feat[:, : int(lengths[batch])]

                while len(tokens) + len(p) < config.max_length:
                    logits = self.head(
                        self.transf_decoder(mx.array([inputs]), feat, cache=cache)
                    )
                    next_token = cast(int, mx.argmax(logits[:, -1], axis=-1).item())

                    if next_token == self.tokenizer.eos_id:
                        break

                    inputs = [next_token]
                    tokens.append(next_token)

                outputs.append(tokens)
            return outputs
        elif config.decoding == "beam":
            outputs = []

            for batch, p in enumerate(prompt):
                # (tokens, inputs, score)
                beams = [([], p.copy(), 0)]
                cache = [
                    TransformerDecoderCache()
                    for _ in range(len(self.transf_decoder.layers))
                ]

                feat = features[batch : batch + 1]
                if lengths is not None:
                    feat = feat[:, : int(lengths[batch])]

                for _ in range(config.max_length - len(p)):
                    logits = self.head(
                        self.transf_decoder(
                            mx.array([beam[1] for beam in beams]),
                            mx.repeat(feat, len(beams), 0),
                            cache=cache,
                        )
                    )
                    logprobs = nn.log_softmax(
                        logits[:, -1] / max(config.temperature, 1e-8)
                    )
                    accumulated_logprobs = logprobs.flatten() + mx.array(
                        [beam[2] for beam in beams for _ in range(logprobs.shape[1])]
                    )

                    indices = mx.argpartition(accumulated_logprobs, -config.beam_size)[
                        -config.beam_size :
                    ]
                    beam_indices = indices // logprobs.shape[1]
                    token_indices = indices % logprobs.shape[1]

                    # handle updates
                    for c in cache:
                        if c.keys is not None and c.values is not None:
                            c.keys = c.keys[beam_indices]
                            c.values = c.values[beam_indices]
                    beams = [
                        (
                            beams[int(beam_indices[i])][0] + [int(token_indices[i])],
                            [int(token_indices[i])],
                            float(accumulated_logprobs[indices[i]]),
                        )
                        if beams[int(beam_indices[i])][1][0] != self.tokenizer.eos_id
                        else (
                            beams[int(beam_indices[i])][0],
                            [self.tokenizer.eos_id],
                            beams[int(beam_indices[i])][2],
                        )
                        for i in range(config.beam_size)
                    ]

                    # exit condition
                    if all(beam[1][0] == self.tokenizer.eos_id for beam in beams):
                        beams = list(sorted(beams, key=lambda x: x[2], reverse=True))
                        outputs.append(beams[0][0][:-1])
                        break

                if len(outputs) < batch + 1:
                    # out of step
                    beams = list(sorted(beams, key=lambda x: x[2], reverse=True))
                    eos_beams = list(
                        filter(lambda x: x[1][0] == self.tokenizer.eos_id, beams)
                    )
                    if len(eos_beams) > 0:
                        outputs.append(eos_beams[0][0][:-1])
                    else:
                        outputs.append(beams[0][0])

            return outputs

        raise NotImplementedError


def prompt(
    tokenizer: CanaryTokenizer,
    prompt_format: Literal["canary", "canary2"],
    source_lang: str,
    target_lang: str,
    punctuation: bool,
    *,
    context: str = "",
    emotion: Literal["undefined", "neutral", "angry", "happy", "sad"] = "undefined",
    inverse_normalization: bool = False,
    timestamp: bool = False,
    diarize: bool = False,
):
    if prompt_format == "canary" and (
        len(context) > 0
        or emotion != "undefined"
        or inverse_normalization is True
        or timestamp is True
        or diarize is True
    ):
        raise ValueError(
            "`context`, `emotion`, `inverse_normalization`, `timestamp`, `diarize` are only supported in `canary2` prompt format."
        )

    src, tgt = f"<|{source_lang}|>", f"<|{target_lang}|>"
    pnc = "<|pnc|>" if punctuation else "<|nopnc|>"

    if prompt_format == "canary":
        task = "<|transcribe|>" if source_lang == target_lang else "<|translate|>"
        prompt_text = f"<|startoftranscript|>{src}{task}{tgt}{pnc}"
        return tokenizer.encode(prompt_text, lang_id="spl_tokens")

    emo = f"<|emo:{emotion}|>"
    itn = "<|itn|>" if inverse_normalization else "<|noitn|>"
    ts = "<|timestamp|>" if timestamp else "<|notimestamp|>"
    dia = "<|diarize|>" if diarize else "<|nodiarize|>"

    if context:
        ctx_tokens = tokenizer.encode(context, lang_id=target_lang)
        prompt_tokens = tokenizer.encode(
            f"<|startofcontext|><|startoftranscript|>{emo}{src}{tgt}{pnc}{itn}{ts}{dia}",
            lang_id="spl_tokens",
        )
        return prompt_tokens[:1] + ctx_tokens + prompt_tokens[1:]

    prompt_text = (
        f"<|startofcontext|><|startoftranscript|>{emo}{src}{tgt}{pnc}{itn}{ts}{dia}"
    )
    return tokenizer.encode(prompt_text, lang_id="spl_tokens")
