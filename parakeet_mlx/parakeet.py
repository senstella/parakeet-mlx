import math
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from parakeet_mlx import tokenizer
from parakeet_mlx.alignment import (
    AlignedResult,
    AlignedToken,
    SentenceConfig,
    merge_longest_common_subsequence,
    merge_longest_contiguous,
    sentences_to_result,
    tokens_to_sentences,
)
from parakeet_mlx.audio import PreprocessArgs, get_logmel, load_audio
from parakeet_mlx.cache import ConformerCache, RotatingConformerCache
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
class ParakeetCTCArgs:
    preprocessor: PreprocessArgs
    encoder: ConformerArgs
    decoder: ConvASRDecoderArgs
    decoding: CTCDecodingArgs


@dataclass
class ParakeetTDTCTCArgs(ParakeetTDTArgs):
    aux_ctc: AuxCTCArgs


# API
@dataclass
class Greedy:
    pass


@dataclass
class Beam:
    beam_size: int = 5
    length_penalty: float = 1.0
    patience: float = 1.0
    duration_reward: float = 0.7  # TDT-only


@dataclass
class DecodingConfig:
    decoding: Union[Greedy, Beam] = field(default_factory=Greedy)
    sentence: SentenceConfig = field(default_factory=SentenceConfig)


# common methods
class BaseParakeet(nn.Module):
    """Base parakeet model for interface purpose"""

    def __init__(self, preprocess_args: PreprocessArgs, encoder_args: ConformerArgs):
        super().__init__()

        self.preprocessor_config = preprocess_args
        self.encoder_config = encoder_args

        self.encoder = Conformer(encoder_args)

    @property
    def time_ratio(self) -> float:
        return (
            self.encoder_config.subsampling_factor
            / self.preprocessor_config.sample_rate
            * self.preprocessor_config.hop_length
        )

    def generate(
        self, mel: mx.array, *, decoding_config: DecodingConfig = DecodingConfig()
    ) -> list[AlignedResult]:
        """
        Generate transcription results from the Parakeet model, handling batches and single input.
        Args:
            mel (mx.array):
                Mel-spectrogram input with shape [batch, sequence, mel_dim] for
                batch processing or [sequence, mel_dim] for single input.
            decoding_config (DecodingConfig, optional):
                Configuration object that controls decoding behavior and
                parameters for the generation process. Defaults to DecodingConfig().
        Returns:
            list[AlignedResult]: List of transcription results with aligned tokens
                and sentences, one for each input in the batch.
        """
        raise NotImplementedError

    def transcribe(
        self,
        path: Path | str,
        *,
        dtype: mx.Dtype = mx.bfloat16,
        decoding_config: DecodingConfig = DecodingConfig(),
        chunk_duration: Optional[float] = None,
        overlap_duration: float = 15.0,
        chunk_callback: Optional[Callable] = None,
    ) -> AlignedResult:
        """
        Transcribe an audio file, with optional chunking for long files.
        Args:
            path (Path | str):
                Path to the audio file to be transcribed.
            dtype (mx.Dtype, optional):
                Data type for processing the audio. Defaults to mx.bfloat16.
            chunk_duration (float, optional):
                If provided, splits audio into chunks of this length (in seconds)
                for processing. When None, processes the entire file at once.
                Defaults to None.
            overlap_duration (float, optional):
                Overlap between consecutive chunks in seconds. Only used when
                chunk_duration is specified. Defaults to 15.0.
            chunk_callback (Callable, optional):
                A function to call when each chunk is processed. The callback
                is called with (current_position, total_position) arguments
                to track progress. Defaults to None.
        Returns:
            AlignedResult: Transcription result with aligned tokens and sentences.
        """
        audio_path = Path(path)
        audio_data = load_audio(audio_path, self.preprocessor_config.sample_rate, dtype)

        if chunk_duration is None:
            mel = get_logmel(audio_data, self.preprocessor_config)
            return self.generate(mel, decoding_config=decoding_config)[0]

        audio_length_seconds = len(audio_data) / self.preprocessor_config.sample_rate

        if audio_length_seconds <= chunk_duration:
            mel = get_logmel(audio_data, self.preprocessor_config)
            return self.generate(mel, decoding_config=decoding_config)[0]

        chunk_samples = int(chunk_duration * self.preprocessor_config.sample_rate)
        overlap_samples = int(overlap_duration * self.preprocessor_config.sample_rate)

        all_tokens = []

        for start in range(0, len(audio_data), chunk_samples - overlap_samples):
            end = min(start + chunk_samples, len(audio_data))

            if chunk_callback is not None:
                chunk_callback(end, len(audio_data))

            if end - start < self.preprocessor_config.hop_length:
                break  # prevent zero-length log mel

            chunk_audio = audio_data[start:end]
            chunk_mel = get_logmel(chunk_audio, self.preprocessor_config)

            chunk_result = self.generate(chunk_mel, decoding_config=decoding_config)[0]

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
                        overlap_duration=overlap_duration,
                    )
                except RuntimeError:
                    all_tokens = merge_longest_common_subsequence(
                        all_tokens,
                        chunk_result.tokens,
                        overlap_duration=overlap_duration,
                    )
            else:
                all_tokens = chunk_result.tokens

        result = sentences_to_result(
            tokens_to_sentences(all_tokens, decoding_config.sentence)
        )
        return result

    def transcribe_stream(
        self,
        context_size: tuple[int, int] = (256, 256),
        depth=1,
        *,
        keep_original_attention: bool = False,
        decoding_config: DecodingConfig = DecodingConfig(),
    ) -> "StreamingParakeet":
        """
        Create a StreamingParakeet object for real-time (streaming) inference.
        Args:
            context_size (tuple[int, int], optional):
                A pair (left_context, right_context) for attention context windows.
            depth (int, optional):
                How many encoder layers will carry over their key/value
                cache (i.e. hidden state) exactly across chunks. Because
                we use local (non-causal) attention, the cache is only
                guaranteed to match a full forward pass up through each
                cached layer:
                    • depth=1 (default): only the first encoder layer's
                    cache matches exactly.
                    • depth=2: the first two layers match, and so on.
                    • depth=N (model's total layers): full equivalence to
                    a non-streaming forward pass.
                Setting `depth` larger than the model's total number
                of encoder layers won't have any impacts.
            keep_original_attention (bool, optional):
                Whether to preserve the original attention class
                during streaming inference. Defaults to False. (Will switch to local attention.)
            decoding_config (DecodingConfig, optional):
                Configuration object that controls decoding behavior
                Defaults to DecodingConfig().
        Returns:
            StreamingParakeet: A context manager for streaming inference.
        """
        return StreamingParakeet(
            self,
            context_size,
            depth,
            decoding_config=decoding_config,
            keep_original_attention=keep_original_attention,
        )


# models
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

        self.decoder = PredictNetwork(args.decoder)
        self.joint = JointNetwork(args.joint)

    def decode(
        self,
        features: mx.array,
        lengths: Optional[mx.array] = None,
        last_token: Optional[list[Optional[int]]] = None,
        hidden_state: Optional[list[Optional[tuple[mx.array, mx.array]]]] = None,
        *,
        config: DecodingConfig = DecodingConfig(),
    ) -> tuple[list[list[AlignedToken]], list[Optional[tuple[mx.array, mx.array]]]]:
        """Run TDT decoder with features, optional length and decoder state. Outputs list[list[AlignedToken]] and updated hidden state"""
        mx.eval(features)

        match config.decoding:
            case Greedy():
                return self.decode_greedy(
                    features, lengths, last_token, hidden_state, config=config
                )
            case Beam():
                return self.decode_beam(
                    features, lengths, last_token, hidden_state, config=config
                )
            case _:
                raise NotImplementedError(
                    f"{config.decoding} is not supported in TDT models."
                )

    def decode_beam(
        self,
        features: mx.array,
        lengths: Optional[mx.array] = None,
        last_token: Optional[list[Optional[int]]] = None,
        hidden_state: Optional[list[Optional[tuple[mx.array, mx.array]]]] = None,
        *,
        config: DecodingConfig = DecodingConfig(),
    ) -> tuple[list[list[AlignedToken]], list[Optional[tuple[mx.array, mx.array]]]]:
        assert isinstance(config.decoding, Beam)  # type guarntee

        beam_token = min(config.decoding.beam_size, len(self.vocabulary) + 1)
        beam_duration = min(config.decoding.beam_size, len(self.durations))
        max_candidates = round(config.decoding.beam_size * config.decoding.patience)

        @dataclass
        class Hypothesis:
            score: float
            step: int
            last_token: Optional[int]
            hidden_state: Optional[tuple[mx.array, mx.array]]
            stuck: int
            hypothesis: list[AlignedToken]

            def __hash__(self) -> int:
                return hash((self.step, tuple((x.id for x in self.hypothesis))))

        B, S, *_ = features.shape

        if hidden_state is None:
            hidden_state = list([None] * B)

        if lengths is None:
            lengths = mx.array([S] * B)

        if last_token is None:
            last_token = list([None] * B)

        results = []
        results_hidden = []
        for batch in range(B):
            feature = features[batch : batch + 1]
            length = int(lengths[batch])

            finished_hypothesis: list[Hypothesis] = []
            active_beam: list[Hypothesis] = [
                Hypothesis(
                    score=0.0,
                    step=0,
                    last_token=last_token[batch],
                    hidden_state=hidden_state[batch],
                    stuck=0,
                    hypothesis=[],
                )
            ]

            while len(finished_hypothesis) < max_candidates and active_beam:
                candidates: Dict[int, Hypothesis] = {}

                for hypothesis in active_beam:
                    decoder_out, (hidden, cell) = self.decoder(
                        mx.array([[hypothesis.last_token]])
                        if hypothesis.last_token is not None
                        else None,
                        hypothesis.hidden_state,
                    )
                    decoder_out = decoder_out.astype(feature.dtype)
                    decoder_hidden = (
                        hidden.astype(feature.dtype),
                        cell.astype(feature.dtype),
                    )

                    joint_out = self.joint(
                        feature[:, hypothesis.step : hypothesis.step + 1], decoder_out
                    )

                    token_logits, duration_logits = (
                        joint_out[0, 0, 0, : len(self.vocabulary) + 1],
                        joint_out[0, 0, 0, len(self.vocabulary) + 1 :],
                    )
                    token_logprobs, duration_logprobs = (
                        nn.log_softmax(token_logits, -1),
                        nn.log_softmax(duration_logits, -1),
                    )

                    # raw python ops might be faster from here
                    token_k, duration_k = (
                        typing.cast(
                            List[int],
                            mx.argpartition(token_logprobs, -beam_token)[
                                -beam_token:
                            ].tolist(),
                        ),
                        typing.cast(
                            List[int],
                            mx.argpartition(duration_logprobs, -beam_duration)[
                                -beam_duration:
                            ].tolist(),
                        ),
                    )

                    # for faster accessing
                    token_logprobs = typing.cast(List[float], token_logprobs.tolist())
                    duration_logprobs = typing.cast(
                        List[float], duration_logprobs.tolist()
                    )

                    for token in token_k:
                        is_blank = token == len(self.vocabulary)
                        for decision in duration_k:
                            duration = self.durations[decision]
                            stuck = 0 if duration != 0 else hypothesis.stuck + 1

                            if (
                                self.max_symbols is not None
                                and stuck >= self.max_symbols
                            ):
                                step = hypothesis.step + 1
                                stuck = 0
                            else:
                                step = hypothesis.step + duration

                            new_hypotheis = Hypothesis(
                                score=hypothesis.score
                                + token_logprobs[token]
                                * (1 - config.decoding.duration_reward)
                                + duration_logprobs[decision]
                                * (config.decoding.duration_reward),
                                step=step,
                                last_token=hypothesis.last_token if is_blank else token,
                                hidden_state=hypothesis.hidden_state
                                if is_blank
                                else decoder_hidden,
                                stuck=stuck,
                                hypothesis=hypothesis.hypothesis
                                if is_blank
                                else (
                                    list(hypothesis.hypothesis)
                                    + [
                                        AlignedToken(
                                            id=token,
                                            start=hypothesis.step
                                            * self.time_ratio,  # hop
                                            duration=duration * self.time_ratio,  # hop
                                            confidence=math.exp(
                                                token_logprobs[token]
                                                + duration_logprobs[decision]
                                            ),
                                            text=tokenizer.decode(
                                                [token], self.vocabulary
                                            ),
                                        )
                                    ]
                                ),
                            )

                            # merge if anyone took same path
                            key = hash(new_hypotheis)
                            if key in candidates:
                                other_hypothesis = candidates[key]

                                # log sum exp
                                maxima = max(
                                    other_hypothesis.score, new_hypotheis.score
                                )
                                score = maxima + math.log(
                                    math.exp(other_hypothesis.score - maxima)
                                    + math.exp(new_hypotheis.score - maxima)
                                )

                                if new_hypotheis.score > other_hypothesis.score:
                                    candidates[key] = new_hypotheis
                                candidates[key].score = score
                            else:
                                candidates[key] = new_hypotheis

                finished_hypothesis.extend(
                    [
                        hypothesis
                        for hypothesis in candidates.values()
                        if hypothesis.step >= length
                    ]
                )
                active_beam = sorted(
                    [
                        hypothesis
                        for hypothesis in candidates.values()
                        if hypothesis.step < length
                    ],
                    key=lambda x: x.score,
                    reverse=True,
                )[: config.decoding.beam_size]

            finished_hypothesis = finished_hypothesis + active_beam

            if not finished_hypothesis:
                results.append([])
                results_hidden.append(hidden_state[batch])
            else:
                length_penalty = (
                    config.decoding.length_penalty
                )  # mypy assumes weirdly so we go in safe way

                best = max(
                    finished_hypothesis,
                    key=lambda x: x.score
                    / (max(1, len(x.hypothesis)) ** length_penalty),
                )
                results.append(best.hypothesis)
                results_hidden.append(best.hidden_state)

        return results, results_hidden

    def decode_greedy(
        self,
        features: mx.array,
        lengths: Optional[mx.array] = None,
        last_token: Optional[list[Optional[int]]] = None,
        hidden_state: Optional[list[Optional[tuple[mx.array, mx.array]]]] = None,
        *,
        config: DecodingConfig = DecodingConfig(),
    ) -> tuple[list[list[AlignedToken]], list[Optional[tuple[mx.array, mx.array]]]]:
        assert isinstance(config.decoding, Greedy)  # type guarntee

        B, S, *_ = features.shape

        if hidden_state is None:
            hidden_state = list([None] * B)

        if lengths is None:
            lengths = mx.array([S] * B)

        if last_token is None:
            last_token = list([None] * B)

        results = []
        for batch in range(B):
            hypothesis = []

            feature = features[batch : batch + 1]
            length = int(lengths[batch])

            step = 0
            new_symbols = 0

            while step < length:
                # decoder pass
                decoder_out, (hidden, cell) = self.decoder(
                    mx.array([[last_token[batch]]])
                    if last_token[batch] is not None
                    else None,
                    hidden_state[batch],
                )
                decoder_out = decoder_out.astype(feature.dtype)
                decoder_hidden = (
                    hidden.astype(feature.dtype),
                    cell.astype(feature.dtype),
                )

                # joint pass
                joint_out = self.joint(feature[:, step : step + 1], decoder_out)

                # sampling
                token_logits = joint_out[0, 0, :, : len(self.vocabulary) + 1]
                pred_token = int(mx.argmax(token_logits))

                # compute confidence score using entropy-based method
                token_probs = mx.softmax(token_logits, axis=-1)
                vocab_size = len(self.vocabulary) + 1
                entropy = -mx.sum(token_probs * mx.log(token_probs + 1e-10), axis=-1)
                max_entropy = mx.log(mx.array(vocab_size, dtype=token_probs.dtype))
                confidence = float(1.0 - (entropy / max_entropy))

                decision = int(
                    mx.argmax(joint_out[0, 0, :, len(self.vocabulary) + 1 :])
                )

                # tdt decoding rule
                if pred_token != len(self.vocabulary):
                    hypothesis.append(
                        AlignedToken(
                            int(pred_token),
                            start=step * self.time_ratio,
                            duration=self.durations[decision] * self.time_ratio,
                            confidence=confidence,
                            text=tokenizer.decode([pred_token], self.vocabulary),
                        )
                    )
                    last_token[batch] = pred_token
                    hidden_state[batch] = decoder_hidden

                step += self.durations[int(decision)]

                # prevent stucking rule
                new_symbols += 1

                if self.durations[int(decision)] != 0:
                    new_symbols = 0
                else:
                    if self.max_symbols is not None and self.max_symbols <= new_symbols:
                        step += 1
                        new_symbols = 0

            results.append(hypothesis)

        return results, hidden_state

    def generate(
        self, mel: mx.array, *, decoding_config: DecodingConfig = DecodingConfig()
    ) -> list[AlignedResult]:
        if len(mel.shape) == 2:
            mel = mx.expand_dims(mel, 0)

        features, lengths = self.encoder(mel)
        mx.eval(features, lengths)

        result, _ = self.decode(features, lengths, config=decoding_config)

        return [
            sentences_to_result(
                tokens_to_sentences(hypothesis, decoding_config.sentence)
            )
            for hypothesis in result
        ]


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

        self.decoder = PredictNetwork(args.decoder)
        self.joint = JointNetwork(args.joint)

    def decode(
        self,
        features: mx.array,
        lengths: Optional[mx.array] = None,
        last_token: Optional[list[Optional[int]]] = None,
        hidden_state: Optional[list[Optional[tuple[mx.array, mx.array]]]] = None,
        *,
        config: DecodingConfig = DecodingConfig(),
    ) -> tuple[list[list[AlignedToken]], list[Optional[tuple[mx.array, mx.array]]]]:
        """Run TDT decoder with features, optional length and decoder state. Outputs list[list[AlignedToken]] and updated hidden state"""
        assert isinstance(config.decoding, Greedy), (
            "Only greedy decoding is supported for RNNT decoder now"
        )

        B, S, *_ = features.shape

        if hidden_state is None:
            hidden_state = list([None] * B)

        if lengths is None:
            lengths = mx.array([S] * B)

        if last_token is None:
            last_token = list([None] * B)

        results = []
        for batch in range(B):
            hypothesis = []

            feature = features[batch : batch + 1]
            length = int(lengths[batch])

            step = 0
            new_symbols = 0

            while step < length:
                # decoder pass
                decoder_out, (hidden, cell) = self.decoder(
                    mx.array([[last_token[batch]]])
                    if last_token[batch] is not None
                    else None,
                    hidden_state[batch],
                )
                decoder_out = decoder_out.astype(feature.dtype)
                decoder_hidden = (
                    hidden.astype(feature.dtype),
                    cell.astype(feature.dtype),
                )

                # joint pass
                joint_out = self.joint(feature[:, step : step + 1], decoder_out)

                # sampling
                token_logits = joint_out[0, 0]
                pred_token = int(mx.argmax(token_logits))

                # compute confidence score using entropy-based method
                token_probs = mx.softmax(token_logits, axis=-1)
                vocab_size = len(self.vocabulary) + 1
                entropy = -mx.sum(token_probs * mx.log(token_probs + 1e-10), axis=-1)
                max_entropy = mx.log(mx.array(vocab_size, dtype=token_probs.dtype))
                confidence = float(1.0 - (entropy / max_entropy))

                # rnnt decoding rule
                if pred_token != len(self.vocabulary):
                    hypothesis.append(
                        AlignedToken(
                            int(pred_token),
                            start=step * self.time_ratio,
                            duration=1 * self.time_ratio,
                            confidence=confidence,
                            text=tokenizer.decode([pred_token], self.vocabulary),
                        )
                    )
                    last_token[batch] = pred_token
                    hidden_state[batch] = decoder_hidden

                    # prevent stucking
                    new_symbols += 1
                    if self.max_symbols is not None and self.max_symbols <= new_symbols:
                        step += 1
                        new_symbols = 0
                else:
                    step += 1
                    new_symbols = 0

            results.append(hypothesis)

        return results, hidden_state

    def generate(
        self, mel: mx.array, *, decoding_config: DecodingConfig = DecodingConfig()
    ) -> list[AlignedResult]:
        if len(mel.shape) == 2:
            mel = mx.expand_dims(mel, 0)

        features, lengths = self.encoder(mel)
        mx.eval(features, lengths)

        result, _ = self.decode(features, lengths, config=decoding_config)

        return [
            sentences_to_result(
                tokens_to_sentences(hypothesis, decoding_config.sentence)
            )
            for hypothesis in result
        ]


class ParakeetCTC(BaseParakeet):
    """MLX Implementation of Parakeet-CTC Model"""

    def __init__(self, args: ParakeetCTCArgs):
        super().__init__(args.preprocessor, args.encoder)

        self.vocabulary = args.decoder.vocabulary

        self.decoder = ConvASRDecoder(args.decoder)

    def decode(
        self,
        features: mx.array,
        lengths: mx.array,
        *,
        config: DecodingConfig = DecodingConfig(),
    ) -> list[list[AlignedToken]]:
        """Run CTC decoder with features and lengths. Outputs list[list[AlignedToken]]."""
        B, S, *_ = features.shape

        logits = self.decoder(features)
        mx.eval(logits, lengths)

        results = []
        for batch in range(B):
            length = int(lengths[batch])
            predictions = logits[batch, :length]
            best_tokens = mx.argmax(predictions, axis=1)

            # Convert log probabilities to probabilities for confidence computation
            probs = mx.exp(predictions)

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
                    token_start_time = token_boundaries[-1][0] * self.time_ratio

                    token_end_time = t * self.time_ratio

                    token_duration = token_end_time - token_start_time

                    # Compute confidence using entropy-based method across token frames
                    token_start_frame = token_boundaries[-1][0]
                    token_end_frame = t
                    token_probs = probs[token_start_frame:token_end_frame]

                    # Compute average entropy across frames
                    vocab_size = len(self.vocabulary) + 1
                    entropies = -mx.sum(
                        token_probs * mx.log(token_probs + 1e-10), axis=-1
                    )
                    avg_entropy = mx.mean(entropies)
                    max_entropy = mx.log(mx.array(vocab_size, dtype=token_probs.dtype))
                    confidence = float(1.0 - (avg_entropy / max_entropy))

                    hypothesis.append(
                        AlignedToken(
                            prev_token,
                            start=token_start_time,
                            duration=token_duration,
                            confidence=confidence,
                            text=tokenizer.decode([prev_token], self.vocabulary),
                        )
                    )

                token_boundaries.append((t, None))
                prev_token = token_idx

            if prev_token != -1:
                last_non_blank = length - 1
                for t in range(length - 1, token_boundaries[-1][0], -1):
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

                # Compute confidence for last token
                token_start_frame = token_boundaries[-1][0]
                token_end_frame = last_non_blank + 1
                token_probs = probs[token_start_frame:token_end_frame]

                vocab_size = len(self.vocabulary) + 1
                entropies = -mx.sum(token_probs * mx.log(token_probs + 1e-10), axis=-1)
                avg_entropy = mx.mean(entropies)
                max_entropy = mx.log(mx.array(vocab_size, dtype=token_probs.dtype))
                confidence = float(1.0 - (avg_entropy / max_entropy))

                hypothesis.append(
                    AlignedToken(
                        prev_token,
                        start=token_start_time,
                        duration=token_duration,
                        confidence=confidence,
                        text=tokenizer.decode([prev_token], self.vocabulary),
                    )
                )

            results.append(hypothesis)

        return results

    def generate(
        self, mel: mx.array, *, decoding_config: DecodingConfig = DecodingConfig()
    ) -> list[AlignedResult]:
        if len(mel.shape) == 2:
            mel = mx.expand_dims(mel, 0)

        features, lengths = self.encoder(mel)

        result = self.decode(features, lengths, config=decoding_config)

        return [
            sentences_to_result(
                tokens_to_sentences(hypothesis, decoding_config.sentence)
            )
            for hypothesis in result
        ]


class ParakeetTDTCTC(ParakeetTDT):
    """MLX Implementation of Parakeet-TDT-CTC Model

    Has ConvASRDecoder decoder in `.ctc_decoder` but `.generate` uses TDT decoder all the times (Please open an issue if you need CTC decoder use-case!)"""

    def __init__(self, args: ParakeetTDTCTCArgs):
        super().__init__(args)

        self.ctc_decoder = ConvASRDecoder(args.aux_ctc.decoder)


# streaming
class StreamingParakeet:
    model: "BaseParakeet"
    cache: List[ConformerCache]

    audio_buffer: mx.array
    mel_buffer: Optional[mx.array]
    decoder_hidden: Optional[tuple[mx.array, mx.array]] = None
    last_token: Optional[int] = None

    finalized_tokens: list[AlignedToken]
    draft_tokens: list[AlignedToken]

    context_size: tuple[int, int]
    depth: int
    decoding_config: DecodingConfig
    keep_original_attention: bool = False

    def __init__(
        self,
        model: "BaseParakeet",
        context_size: tuple[int, int],
        depth: int = 1,
        *,
        keep_original_attention: bool = False,
        decoding_config: DecodingConfig = DecodingConfig(),
    ) -> None:
        self.context_size = context_size
        self.depth = depth
        self.decoding_config = decoding_config
        self.keep_original_attention = keep_original_attention

        self.model = model
        self.cache = [
            RotatingConformerCache(self.keep_size, cache_drop_size=self.drop_size)
            for _ in range(len(model.encoder.layers))
        ]

        self.audio_buffer = mx.array([])
        self.mel_buffer = None
        self.finalized_tokens = []
        self.draft_tokens = []

    def __enter__(self):
        if not self.keep_original_attention:
            self.model.encoder.set_attention_model(
                "rel_pos_local_attn", self.context_size
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.keep_original_attention:
            self.model.encoder.set_attention_model(
                "rel_pos"
            )  # hard-coded; might cache if there's actually new varient than rel_pos
        del self.audio_buffer
        del self.cache

        mx.clear_cache()

    @property
    def keep_size(self):
        """Indicates how many encoded feature frames to keep in KV cache"""
        return self.context_size[0]

    @property
    def drop_size(self):
        """Indicates how many encoded feature frames to drop"""
        return self.context_size[1] * self.depth

    @property
    def result(self) -> AlignedResult:
        """Transcription result"""
        return sentences_to_result(
            tokens_to_sentences(
                self.finalized_tokens + self.draft_tokens, self.decoding_config.sentence
            )
        )

    def add_audio(self, audio: mx.array) -> None:
        """Takes portion of audio and transcribe it.

        `audio` must be 1D array"""

        self.audio_buffer = mx.concat(
            [
                self.audio_buffer,
                audio,
            ],
            axis=0,
        )
        mel = get_logmel(
            self.audio_buffer[
                : (
                    len(self.audio_buffer)
                    // self.model.preprocessor_config.hop_length
                    * self.model.preprocessor_config.hop_length
                )
            ],
            self.model.preprocessor_config,
        )

        if self.mel_buffer is None:  # init
            self.mel_buffer = mel
        else:
            self.mel_buffer = mx.concat([self.mel_buffer, mel], axis=1)

        self.audio_buffer = self.audio_buffer[
            (mel.shape[1] * self.model.preprocessor_config.hop_length) :
        ]

        features, lengths = self.model.encoder(
            self.mel_buffer[
                :,
                : (
                    self.mel_buffer.shape[1]
                    // self.model.encoder_config.subsampling_factor
                    * self.model.encoder_config.subsampling_factor
                ),
            ],
            cache=self.cache,
        )
        mx.eval(features, lengths)
        length = int(lengths[0])

        # cache will automatically dropped in cache level
        leftover = self.mel_buffer.shape[1] - (
            length * self.model.encoder_config.subsampling_factor
        )
        self.mel_buffer = self.mel_buffer[
            :,
            -(
                self.drop_size * self.model.encoder_config.subsampling_factor + leftover
            ) :,
        ]

        # we decode in two phase
        # first phase: finalized region decode
        # second phase: draft region decode (will be dropped)
        finalized_length = max(0, length - self.drop_size)

        if isinstance(self.model, ParakeetTDT) or isinstance(self.model, ParakeetRNNT):
            finalized_tokens, finalized_state = self.model.decode(
                features,
                mx.array([finalized_length]),
                [self.last_token],
                [self.decoder_hidden],
                config=self.decoding_config,
            )

            self.decoder_hidden = finalized_state[0]
            self.last_token = (
                finalized_tokens[0][-1].id if len(finalized_tokens[0]) > 0 else None
            )

            draft_tokens, _ = self.model.decode(
                features[:, finalized_length:],
                mx.array(
                    [
                        features[:, finalized_length:].shape[1]
                    ]  # i believe in lazy evaluation
                ),
                [self.last_token],
                [self.decoder_hidden],
                config=self.decoding_config,
            )

            self.finalized_tokens.extend(finalized_tokens[0])
            self.draft_tokens = draft_tokens[0]
        elif isinstance(self.model, ParakeetCTC):
            finalized_tokens = self.model.decode(
                features, mx.array([finalized_length]), config=self.decoding_config
            )

            draft_tokens = self.model.decode(
                features[:, finalized_length:],
                mx.array(
                    [
                        features[:, finalized_length:].shape[1]
                    ]  # i believe in lazy evaluation
                ),
                config=self.decoding_config,
            )

            self.finalized_tokens.extend(finalized_tokens[0])
            self.draft_tokens = draft_tokens[0]
        else:
            raise NotImplementedError("This model does not support real-time decoding")
