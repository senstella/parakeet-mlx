# adopted from mlx-lm
# https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py
from typing import List, Optional

import mlx.core as mx


class BaseCache:
    offset: int

    def is_trimmable(self) -> bool:
        raise NotImplementedError

    def trim(self, n: int) -> int:
        raise NotImplementedError

    def update_and_fetch_kv(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        raise NotImplementedError

    def update_and_fetch_conv(self, x: mx.array, padding: int = 0) -> mx.array:
        raise NotImplementedError


def make_prompt_cache(
    model,
    max_kv_size: Optional[int] = None,
) -> List[BaseCache]:
    num_layers = len(model.encoder.layers)
    if max_kv_size is not None:
        return [RotatingConformerCache(max_kv_size) for _ in range(num_layers)]
    else:
        return [ConformerCache() for _ in range(num_layers)]


class ConformerCache(BaseCache):
    keys: mx.array | None = None
    values: mx.array | None = None
    conv: mx.array | None = None
    offset: int = 0
    step: int = 256

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def update_and_fetch_kv(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        # k, v is [batch, head, seq, dim]
        prev = self.offset
        if (
            self.keys is None
            or self.values is None
            or (prev + keys.shape[2]) > self.keys.shape[2]
        ):
            B, H, S, D_KEYS = keys.shape
            _, _, _, D_VALUES = values.shape
            S_CACHE = ((self.step + S - 1) // self.step) * self.step

            new_k = mx.zeros((B, H, S_CACHE, D_KEYS), keys.dtype)
            new_v = mx.zeros((B, H, S_CACHE, D_VALUES), keys.dtype)

            if self.keys is None or self.values is None:  # type safety!
                self.keys, self.values = new_k, new_v
            else:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]


# skipping batch, it will mostly use maximum of it anyways
class RotatingConformerCache(BaseCache):
    def __init__(self, capacity: int):
        self.keys = None
        self.values = None
        self.capacity = capacity
        self.offset = 0

    def is_trimmable(self):
        return False

    def trim(self, n):
        return 0  # add if needed

    def update_and_fetch_kv(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        if self.keys is None or self.values is None:
            B, H, _, D_KEYS = keys.shape
            _, _, _, D_VALUES = values.shape
            self.keys = mx.zeros((B, H, self.capacity, D_KEYS), keys.dtype)
            self.values = mx.zeros((B, H, self.capacity, D_VALUES), values.dtype)

        S = keys.shape[2]

        pos = self.offset % self.capacity
        space_left = self.capacity - pos

        if S <= space_left:
            self.keys[..., pos : pos + S, :] = keys
            self.values[..., pos : pos + S, :] = values
        else:
            self.keys[..., pos:, :] = keys[..., :space_left, :]
            self.values[..., pos:, :] = values[..., :space_left, :]

            remaining_s = S - space_left
            self.keys[..., :remaining_s, :] = keys[..., space_left:, :]
            self.values[..., :remaining_s, :] = values[..., space_left:, :]

        self.offset += S

        if self.offset <= self.capacity:
            k_out = self.keys[..., : self.offset, :]
            v_out = self.values[..., : self.offset, :]
        else:
            shift_amount = -(self.offset % self.capacity)

            k_out = mx.roll(self.keys, shift_amount, 2)
            v_out = mx.roll(self.values, shift_amount, 2)

        return k_out, v_out
