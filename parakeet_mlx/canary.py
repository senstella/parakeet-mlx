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
                tokens = p.copy()
                inputs = tokens
                cache = [
                    TransformerDecoderCache()
                    for _ in range(len(self.transf_decoder.layers))
                ]

                feat = features[batch : batch + 1]
                if lengths is not None:
                    feat = feat[:, : lengths[batch]]

                while (
                    len(tokens) < config.max_length
                    and tokens[-1] != self.tokenizer.eos_id
                ):
                    logits = self.head(
                        self.transf_decoder(mx.array([inputs]), feat, cache=cache)
                    )
                    next_token = cast(int, mx.argmax(logits[:, -1], axis=-1).item())
                    inputs = [next_token]
                    tokens.append(next_token)

                outputs.append(tokens)
            return outputs

        raise NotImplementedError
