import re

space_before_punct = re.compile(r'(\s)(\.|!|\?)')

# decode some tokens (also strips space before punctuation as nemo does)
def decode(tokens: list[int], vocabulary: list[str]):
    pre = "".join([vocabulary[token].replace("▁", " ") for token in tokens])
    return space_before_punct.sub(r'\2', pre)
