from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn

from parakeet_mlx import tokenizer
from parakeet_mlx.alignment import (
    AlignedResult,
    AlignedToken,
    sentences_to_result,
    tokens_to_sentences,
)
from parakeet_mlx.audio import PreprocessArgs, get_logmel, load_audio
from parakeet_mlx.cache import BaseCache, make_prompt_cache
from parakeet_mlx.conformer import Conformer, ConformerArgs
from parakeet_mlx.ctc import AuxCTCArgs, ConvASRDecoder, ConvASRDecoderArgs
from parakeet_mlx.rnnt import JointArgs, JointNetwork, PredictArgs, PredictNetwork


@dataclass
class TDTDecodingArgs:
    model_type: str
    durations: list[int]
    greedy: dict | None


@dataclass
class RNNTDecodingArgs:
    greedy: dict | None


@dataclass
class CTCDecodingArgs:
    greedy: dict | None


@dataclass
class ParakeetTDTArgs:
    preprocessor: PreprocessArgs
    encoder: ConformerArgs
    decoder: PredictArgs
    joint: JointArgs
    decoding: TDTDecodingArgs


@dataclass
class ParakeetRNNTArgs:
    preprocessor: PreprocessArgs
    encoder: ConformerArgs
    decoder: PredictArgs
    joint: JointArgs
    decoding: RNNTDecodingArgs


@dataclass
class RNNTState:
    hidden_state: list[tuple[mx.array, mx.array] | None]
    last_token: list[int | None]


@dataclass
class ParakeetCTCArgs:
    preprocessor: PreprocessArgs
    encoder: ConformerArgs
    decoder: ConvASRDecoderArgs
    decoding: CTCDecodingArgs


@dataclass
class ParakeetTDTCTCArgs(ParakeetTDTArgs):
    aux_ctc: AuxCTCArgs


class BaseParakeet(nn.Module):
    """Base parakeet model for interface purpose"""

    def __init__(self, preprocess_args: PreprocessArgs, encoder_args: ConformerArgs):
        super().__init__()

        self.preprocessor_config = preprocess_args
        self.encoder_config = encoder_args

    def generate(
        self,
        mel: mx.array,
        *,
        cache: Optional[list[BaseCache]] = None,
    ) -> list[AlignedResult]:
        """
        Generate with skip token logic for the Parakeet model, handling batches and single input. Uses greedy decoding.
        mel: [batch, sequence, mel_dim] or [sequence, mel_dim]
        """
        raise NotImplementedError

    def transcribe(
        self,
        path: Path | str,
        *,
        dtype: mx.Dtype = mx.bfloat16,
        chunk_duration: Optional[float] = None,
        overlap_duration: float = 15.0,
        chunk_callback: Optional[Callable] = None,
    ) -> AlignedResult:
        """
        Transcribe an audio file, with optional chunking for long files.

        Args:
            path: Path to the audio file
            dtype: Data type for processing
            chunk_duration: If provided, splits audio into chunks of this length for processing
            overlap_duration: Overlap between chunks (only used when chunking)
            chunk_callback: A function to call back when chunk is processed, called with (current_position, total_position)

        Returns:
            Transcription result with aligned tokens and sentences
        """
        audio_path = Path(path)
        audio_data = load_audio(audio_path, self.preprocessor_config.sample_rate, dtype)

        if chunk_duration is None:
            mel = get_logmel(audio_data, self.preprocessor_config)
            return self.generate(mel)[0]

        audio_length_seconds = len(audio_data) / self.preprocessor_config.sample_rate

        if audio_length_seconds <= chunk_duration:
            mel = get_logmel(audio_data, self.preprocessor_config)
            return self.generate(mel)[0]

        chunk_samples = int(chunk_duration * self.preprocessor_config.sample_rate)
        kv_length = int(
            (chunk_duration + overlap_duration)
            * self.preprocessor_config.sample_rate
            / self.encoder_config.subsampling_factor
        )

        all_tokens = []

        # stateful decoding
        cache = make_prompt_cache(self, kv_length)
        rnnt_state = None
        if isinstance(self, ParakeetTDT) or isinstance(self, ParakeetRNNT):
            rnnt_state = self.make_rnnt_state(1)

        for start in range(0, len(audio_data), chunk_samples):
            end = min(start + chunk_samples, len(audio_data))

            if chunk_callback is not None:
                chunk_callback(end, len(audio_data))

            chunk_audio = audio_data[start:end]
            chunk_mel = get_logmel(chunk_audio, self.preprocessor_config)

            if isinstance(self, ParakeetTDT) or isinstance(self, ParakeetRNNT):
                chunk_result = self.generate(
                    chunk_mel, cache=cache, rnnt_state=rnnt_state
                )[0]
            else:
                chunk_result = self.generate(chunk_mel, cache=cache)[0]

            chunk_offset = start / self.preprocessor_config.sample_rate
            for sentence in chunk_result.sentences:
                for token in sentence.tokens:
                    token.start += chunk_offset
                    token.end = token.start + token.duration

            chunk_tokens = []
            for sentence in chunk_result.sentences:
                chunk_tokens.extend(sentence.tokens)

            if all_tokens:
                all_tokens = all_tokens + chunk_tokens
            else:
                all_tokens = chunk_tokens

        result = sentences_to_result(tokens_to_sentences(all_tokens))
        return result


class ParakeetTDT(BaseParakeet):
    """MLX Implementation of Parakeet-TDT Model"""

    def __init__(self, args: ParakeetTDTArgs):
        super().__init__(args.preprocessor, args.encoder)

        assert args.decoding.model_type == "tdt", "Model must be a TDT model"

        self.vocabulary = args.joint.vocabulary
        self.durations = args.decoding.durations
        self.max_symbols: int | None = (
            args.decoding.greedy.get("max_symbols", None)
            if args.decoding.greedy
            else None
        )

        self.encoder = Conformer(args.encoder)
        self.decoder = PredictNetwork(args.decoder)
        self.joint = JointNetwork(args.joint)

    def make_rnnt_state(self, batch_size: int) -> RNNTState:
        return RNNTState([None] * batch_size, [len(self.vocabulary)] * batch_size)

    def generate(
        self,
        mel: mx.array,
        *,
        cache: Optional[list[BaseCache]] = None,
        rnnt_state: RNNTState | None = None,
    ) -> list[AlignedResult]:
        """
        Generate with skip token logic for the Parakeet model, handling batches and single input. Uses greedy decoding.
        mel: [batch, sequence, mel_dim] or [sequence, mel_dim]
        """
        batch_size: int = mel.shape[0]
        if len(mel.shape) == 2:
            batch_size = 1
            mel = mx.expand_dims(mel, 0)

        batch_features, lengths = self.encoder(mel, cache=cache)
        mx.eval(batch_features, lengths)

        if rnnt_state is None:
            rnnt_state = self.make_rnnt_state(batch_size)

        results = []
        for b in range(batch_size):
            features = batch_features[b : b + 1]
            max_length = int(lengths[b])

            hypothesis = []

            time = 0
            new_symbols = 0

            while time < max_length:
                feature = features[:, time : time + 1]

                current_token = (
                    mx.array([[rnnt_state.last_token[b]]], dtype=mx.int32)
                    if rnnt_state.last_token[b] != len(self.vocabulary)
                    else None
                )
                decoder_output, (hidden, cell) = self.decoder(
                    current_token, rnnt_state.hidden_state[b]
                )

                # cast
                decoder_output = decoder_output.astype(feature.dtype)
                proposed_decoder_hidden = (
                    hidden.astype(feature.dtype),
                    cell.astype(feature.dtype),
                )

                joint_output = self.joint(feature, decoder_output)

                pred_token = mx.argmax(
                    joint_output[0, 0, :, : len(self.vocabulary) + 1]
                )
                decision = mx.argmax(joint_output[0, 0, :, len(self.vocabulary) + 1 :])

                if pred_token != len(self.vocabulary):
                    hypothesis.append(
                        AlignedToken(
                            int(pred_token),
                            start=time
                            * self.encoder_config.subsampling_factor
                            / self.preprocessor_config.sample_rate
                            * self.preprocessor_config.hop_length,  # hop
                            duration=self.durations[int(decision)]
                            * self.encoder_config.subsampling_factor
                            / self.preprocessor_config.sample_rate
                            * self.preprocessor_config.hop_length,  # hop
                            text=tokenizer.decode([int(pred_token)], self.vocabulary),
                        )
                    )
                    rnnt_state.last_token[b] = int(pred_token)
                    rnnt_state.hidden_state[b] = proposed_decoder_hidden

                time += self.durations[int(decision)]
                new_symbols += 1

                if self.durations[int(decision)] != 0:
                    new_symbols = 0
                else:
                    if self.max_symbols is not None and self.max_symbols <= new_symbols:
                        time += 1
                        new_symbols = 0

            result = sentences_to_result(tokens_to_sentences(hypothesis))
            results.append(result)

        return results


class ParakeetRNNT(BaseParakeet):
    """MLX Implementation of Parakeet-RNNT Model"""

    def __init__(self, args: ParakeetRNNTArgs):
        super().__init__(args.preprocessor, args.encoder)

        self.vocabulary = args.joint.vocabulary
        self.max_symbols: int | None = (
            args.decoding.greedy.get("max_symbols", None)
            if args.decoding.greedy
            else None
        )

        self.encoder = Conformer(args.encoder)
        self.decoder = PredictNetwork(args.decoder)
        self.joint = JointNetwork(args.joint)

    def make_rnnt_state(self, batch_size: int) -> RNNTState:
        return RNNTState([None] * batch_size, [len(self.vocabulary)] * batch_size)

    def generate(
        self,
        mel: mx.array,
        *,
        cache: Optional[list[BaseCache]] = None,
        rnnt_state: RNNTState | None = None,
    ) -> list[AlignedResult]:
        """
        Generate with skip token logic for the Parakeet model, handling batches and single input. Uses greedy decoding.
        mel: [batch, sequence, mel_dim] or [sequence, mel_dim]
        """
        batch_size: int = mel.shape[0]
        if len(mel.shape) == 2:
            batch_size = 1
            mel = mx.expand_dims(mel, 0)

        batch_features, lengths = self.encoder(mel, cache=cache)
        mx.eval(batch_features, lengths)

        if rnnt_state is None:
            rnnt_state = self.make_rnnt_state(batch_size)

        results = []
        for b in range(batch_size):
            features = batch_features[b : b + 1]
            max_length = int(lengths[b])

            hypothesis = []

            time = 0
            new_symbols = 0

            while time < max_length:
                feature = features[:, time : time + 1]

                current_token = (
                    mx.array([[rnnt_state.last_token[b]]], dtype=mx.int32)
                    if rnnt_state.last_token[b] != len(self.vocabulary)
                    else None
                )
                decoder_output, (hidden, cell) = self.decoder(
                    current_token, rnnt_state.hidden_state[b]
                )

                # cast
                decoder_output = decoder_output.astype(feature.dtype)
                proposed_decoder_hidden = (
                    hidden.astype(feature.dtype),
                    cell.astype(feature.dtype),
                )

                joint_output = self.joint(feature, decoder_output)

                pred_token = mx.argmax(joint_output)

                if pred_token != len(self.vocabulary):
                    hypothesis.append(
                        AlignedToken(
                            int(pred_token),
                            start=time
                            * self.encoder_config.subsampling_factor
                            / self.preprocessor_config.sample_rate
                            * self.preprocessor_config.hop_length,  # hop
                            duration=1
                            * self.encoder_config.subsampling_factor
                            / self.preprocessor_config.sample_rate
                            * self.preprocessor_config.hop_length,  # hop
                            text=tokenizer.decode([int(pred_token)], self.vocabulary),
                        )
                    )
                    rnnt_state.last_token[b] = int(pred_token)
                    rnnt_state.hidden_state[b] = proposed_decoder_hidden

                    new_symbols += 1
                    if self.max_symbols is not None and self.max_symbols <= new_symbols:
                        time += 1
                        new_symbols = 0
                else:
                    time += 1
                    new_symbols = 0

            result = sentences_to_result(tokens_to_sentences(hypothesis))
            results.append(result)

        return results


class ParakeetCTC(BaseParakeet):
    """MLX Implementation of Parakeet-CTC Model"""

    def __init__(self, args: ParakeetCTCArgs):
        super().__init__(args.preprocessor, args.encoder)

        self.vocabulary = args.decoder.vocabulary

        self.encoder = Conformer(args.encoder)
        self.decoder = ConvASRDecoder(args.decoder)

    def generate(
        self,
        mel: mx.array,
        *,
        cache: Optional[list[BaseCache]] = None,
    ) -> list[AlignedResult]:
        """
        Generate with CTC decoding for the Parakeet model, handling batches and single input. Uses greedy decoding.
        mel: [batch, sequence, mel_dim] or [sequence, mel_dim]
        """
        batch_size: int = mel.shape[0]
        if len(mel.shape) == 2:
            batch_size = 1
            mel = mx.expand_dims(mel, 0)

        batch_features, lengths = self.encoder(mel, cache=cache)
        logits = self.decoder(batch_features)
        mx.eval(logits, lengths)

        results = []
        for b in range(batch_size):
            features_len = int(lengths[b])
            predictions = logits[b, :features_len]
            best_tokens = mx.argmax(predictions, axis=1)

            hypothesis = []
            token_boundaries = []
            prev_token = -1

            for t, token_id in enumerate(best_tokens):
                token_idx = int(token_id)

                if token_idx == len(self.vocabulary):
                    continue

                if token_idx == prev_token:
                    continue

                if prev_token != -1:
                    token_start_time = (
                        token_boundaries[-1][0]
                        * self.encoder_config.subsampling_factor
                        / self.preprocessor_config.sample_rate
                        * self.preprocessor_config.hop_length
                    )

                    token_end_time = (
                        t
                        * self.encoder_config.subsampling_factor
                        / self.preprocessor_config.sample_rate
                        * self.preprocessor_config.hop_length
                    )

                    token_duration = token_end_time - token_start_time

                    hypothesis.append(
                        AlignedToken(
                            prev_token,
                            start=token_start_time,
                            duration=token_duration,
                            text=tokenizer.decode([prev_token], self.vocabulary),
                        )
                    )

                token_boundaries.append((t, None))
                prev_token = token_idx

            if prev_token != -1:
                last_non_blank = features_len - 1
                for t in range(features_len - 1, token_boundaries[-1][0], -1):
                    if int(best_tokens[t]) != len(self.vocabulary):
                        last_non_blank = t
                        break

                token_start_time = (
                    token_boundaries[-1][0]
                    * self.encoder_config.subsampling_factor
                    / self.preprocessor_config.sample_rate
                    * self.preprocessor_config.hop_length
                )

                token_end_time = (
                    (last_non_blank + 1)
                    * self.encoder_config.subsampling_factor
                    / self.preprocessor_config.sample_rate
                    * self.preprocessor_config.hop_length
                )

                token_duration = token_end_time - token_start_time

                hypothesis.append(
                    AlignedToken(
                        prev_token,
                        start=token_start_time,
                        duration=token_duration,
                        text=tokenizer.decode([prev_token], self.vocabulary),
                    )
                )

            result = sentences_to_result(tokens_to_sentences(hypothesis))
            results.append(result)

        return results


class ParakeetTDTCTC(ParakeetTDT):
    """MLX Implementation of Parakeet-TDT-CTC Model

    Has ConvASRDecoder decoder in `.ctc_decoder` but `.generate` uses TDT decoder all the times (Please open an issue if you need CTC decoder use-case!)"""

    def __init__(self, args: ParakeetTDTCTCArgs):
        super().__init__(args)

        self.ctc_decoder = ConvASRDecoder(args.aux_ctc.decoder)
