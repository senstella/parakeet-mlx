from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import mlx.nn as nn

from parakeet_mlx.attention import (
    FixedPositionalEncoding,
    MultiHeadAttention,
    create_causal_mask,
)
from parakeet_mlx.cache import TransformerDecoderCache
from parakeet_mlx.conformer import FeedForward


@dataclass
class TransformerDecoderArgs:
    vocab_size: int  # num_classes in head
    hidden_size: int
    inner_size: int
    num_layers: int
    num_attention_heads: int
    pre_ln: bool
    hidden_act: Literal["relu"]
    pre_ln_final_layer_norm: bool
    learn_positional_encodings: bool
    max_sequence_length: int


@dataclass
class TransformerHeadArgs:
    log_softmax: bool
    num_layers: int
    hidden_size: int
    num_classes: int  # this!


class TransformerDecoderBlock(nn.Module):
    def __init__(self, args: TransformerDecoderArgs):
        super().__init__()

        if args.pre_ln is False:
            raise NotImplementedError(
                "`pre_ln` = False for TransformerDecoder has not been implemented yet. Please open the issue in https://github.com/senstella/parakeet-mlx if you see this error."
            )

        self.layer_norm_1 = nn.LayerNorm(args.hidden_size, eps=1e-5)
        self.first_sub_layer = MultiHeadAttention(
            args.num_attention_heads,
            args.hidden_size,
        )
        self.layer_norm_2 = nn.LayerNorm(args.hidden_size, eps=1e-5)
        self.second_sub_layer = MultiHeadAttention(
            args.num_attention_heads,
            args.hidden_size,
        )
        self.layer_norm_3 = nn.LayerNorm(args.hidden_size, eps=1e-5)
        self.third_sub_layer = FeedForward(
            args.hidden_size, args.inner_size, activation=args.hidden_act
        )

    def __call__(
        self,
        x: mx.array,
        xa: mx.array,
        mask_x: mx.array | None = None,
        mask_xa: mx.array | None = None,
        cache: TransformerDecoderCache | None = None,
    ) -> mx.array:
        x_norm = self.layer_norm_1(x)
        x = x + self.first_sub_layer(x_norm, x_norm, x_norm, mask=mask_x, cache=cache)

        x_norm = self.layer_norm_2(x)
        x = x + self.second_sub_layer(x_norm, xa, xa, mask=mask_xa)

        x_norm = self.layer_norm_3(x)
        x = x + self.third_sub_layer(x_norm)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, args: TransformerDecoderArgs):
        super().__init__()

        self.token_embedding = nn.Embedding(
            args.vocab_size, args.hidden_size
        )  # vocab_size is num_classes in head, kind of confusing naming
        self.position_embedding = (
            nn.Embedding(args.max_sequence_length, args.hidden_size)
            if args.learn_positional_encodings
            else FixedPositionalEncoding(
                args.hidden_size, max_len=args.max_sequence_length
            )
        )
        self.embedding_layer_norm = nn.LayerNorm(args.hidden_size, eps=1e-5)

        self.layers = [TransformerDecoderBlock(args) for _ in range(args.num_layers)]
        self.final_layer_norm = (
            nn.LayerNorm(args.hidden_size, eps=1e-5)
            if args.pre_ln and args.pre_ln_final_layer_norm
            else None
        )

    def __call__(
        self,
        x: mx.array,
        xa: mx.array,
        mask_x: mx.array | None = None,
        mask_xa: mx.array | None = None,
        cache: list[TransformerDecoderCache] | None = None,
    ) -> mx.array:
        # embedding
        offset = 0 if cache is None else cache[0].offset

        x = self.token_embedding(x)
        x = x + (
            self.position_embedding(x, offset=offset)
            if isinstance(self.position_embedding, FixedPositionalEncoding)
            else self.position_embedding(mx.arange(offset, offset + x.shape[1]))
        )
        x = self.embedding_layer_norm(x)

        mask_x = (
            mask_x & create_causal_mask(x.shape[1], offset)
            if mask_x is not None
            else create_causal_mask(x.shape[1], offset)
        )
        for i, layer in enumerate(self.layers):
            x = layer(
                x, xa, mask_x, mask_xa, cache=cache[i] if cache is not None else None
            )

        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        return x


class TransformerHead(nn.Module):
    def __init__(self, args: TransformerHeadArgs):
        super().__init__()

        if args.num_layers != 1:
            raise NotImplementedError(
                "Classification head has non-supported layers. Please open an issue in https://github.com/senstella/parakeet-mlx"
            )

        self.log_softmax = args.log_softmax

        self.classifier = nn.Linear(args.hidden_size, args.num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.classifier(x)
        x = nn.log_softmax(x, -1) if self.log_softmax else x
        return x
