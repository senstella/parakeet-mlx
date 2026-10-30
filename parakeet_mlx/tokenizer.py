# decode some tokens (might edit it if to support other varients)
def decode(tokens: list[int], vocabulary: list[str]):
    return "".join([vocabulary[token].replace("▁", " ") for token in tokens])
