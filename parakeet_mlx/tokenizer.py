import json
import re
from functools import cached_property

from tokenizers import Tokenizer


# For parakeet: decode some tokens for parakeet
def decode(tokens: list[int], vocabulary: list[str]):
    return "".join([vocabulary[token].replace("▁", " ") for token in tokens])


# intersting approach..
class CanaryTokenizer:
    CANARY_BOS = "<|startoftranscript|>"
    CANARY_EOS = "<|endoftext|>"
    CANARY_PAD = "<pad>"
    CANARY_NOSPEECH = "<|nospeech|>"
    CANARY_PNC = "<|pnc|>"
    CANARY_NOPNC = "<|nopnc|>"
    CANARY2_BOCTX = "<|startofcontext|>"

    def __init__(self, tokenizers: dict[str, Tokenizer]):
        self.tokenizers = tokenizers  # dict with py 3.7+ is basically ordered dict
        self.offsets = {
            lang: sum(len(t.get_vocab()) for t in list(tokenizers.values())[:i])
            for i, lang in enumerate(tokenizers.keys())
        }

        mappings = [
            (local_id + self.offsets[lang], tokenizer, local_id)
            for lang, tokenizer in tokenizers.items()
            for local_id in tokenizer.get_vocab().values()
        ]
        self.lookup_tokenizer = {gid: tok for gid, tok, _ in mappings}
        self.lookup_local = {gid: lid for gid, _, lid in mappings}

        self.special_tokens = {
            token: local_id + self.offsets["spl_tokens"]
            for token, local_id in tokenizers["spl_tokens"].get_vocab().items()
        }

    @staticmethod
    def from_data(tokenizer: dict[str, dict | str]):
        return CanaryTokenizer(
            {
                i: Tokenizer.from_str(json.dumps(v) if isinstance(v, dict) else v)
                for i, v in tokenizer.items()
            }
        )

    def encode(self, text: str, lang_id: str) -> list[int]:
        if lang_id == "spl_tokens":
            return [
                self.special_tokens[token] for token in re.findall(r"<\|[^|]+\|>", text)
            ]

        return [
            tid + self.offsets[lang_id]
            for tid in self.tokenizers[lang_id].encode(text).ids
        ]

    def decode(self, token_ids: list[int]) -> str:
        pieces = [
            self.lookup_tokenizer[tid].decode([self.lookup_local[tid]])
            for tid in token_ids
            if tid in self.lookup_tokenizer
        ]
        return "".join(pieces).replace("▁", " ")

    @cached_property
    def eos_id(self) -> int:
        return self.special_tokens[self.CANARY_EOS]

    @cached_property
    def bos_id(self) -> int:
        return self.special_tokens[self.CANARY_BOS]

    @cached_property
    def nospeech_id(self) -> int:
        return self.special_tokens[self.CANARY_NOSPEECH]

    @cached_property
    def pad_id(self) -> int:
        return self.special_tokens[self.CANARY_PAD]
