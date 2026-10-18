import re

# vocabulary entries that are markers, not text: <unk>, <pad> and the <|...|>
# family (language, task, ...) of the multilingual vocabulary
_CONTROL_TOKEN = re.compile(r"<unk>|<pad>|<\|[^|>]*\|>")


# decode some tokens (might edit it if to support other varients)
def decode(tokens: list[int], vocabulary: list[str]):
    return "".join(
        [
            ""
            if _CONTROL_TOKEN.fullmatch(vocabulary[token])
            else vocabulary[token].replace("▁", " ")
            for token in tokens
        ]
    )
