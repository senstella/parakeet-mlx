from dataclasses import dataclass
from typing import Optional, cast

import mlx.core as mx
import mlx.nn as nn
from typing_extensions import Literal

from parakeet_mlx.audio import PreprocessArgs
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


@dataclass
class CanaryDecodingConfig:
    decoding: Literal["greedy", "beam"] = "beam"
    beam_size: int = 5
    temperature: float = 0.0
    max_length: int = 512


class Canary(nn.Module):
    """Canary model"""

    def __init__(self, args: CanaryArgs, tokenizer: CanaryTokenizer):
        super().__init__()

        self.preprocessor_config = args.preprocessor
        self.encoder_config = args.encoder
        self.tokenizer = tokenizer

        self.encoder = Conformer(args.encoder)
        self.transf_decoder = TransformerDecoder(args.transf_decoder)
        self.head = TransformerHead(args.head)

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
