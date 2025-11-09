from dataclasses import dataclass

import numpy as np


@dataclass
class AlignedToken:
    id: int
    text: str
    start: float
    duration: float
    confidence: float = 1.0  # confidence score (0.0 to 1.0)
    end: float = 0.0  # temporary

    def __post_init__(self) -> None:
        self.end = self.start + self.duration


@dataclass
class AlignedSentence:
    text: str
    tokens: list[AlignedToken]
    start: float = 0.0  # temporary
    end: float = 0.0  # temporary
    duration: float = 0.0  # temporary
    confidence: float = 1.0  # aggregate confidence score

    def __post_init__(self) -> None:
        self.tokens = list(sorted(self.tokens, key=lambda x: x.start))
        self.start = self.tokens[0].start
        self.end = self.tokens[-1].end
        self.duration = self.end - self.start
        # Compute geometric mean of token confidences
        confidences = np.array([t.confidence for t in self.tokens])
        self.confidence = float(np.exp(np.mean(np.log(confidences + 1e-10))))


@dataclass
class AlignedResult:
    text: str
    sentences: list[AlignedSentence]

    def __post_init__(self) -> None:
        self.text = self.text.strip()

    @property
    def tokens(self) -> list[AlignedToken]:
        return [token for sentence in self.sentences for token in sentence.tokens]


@dataclass
class SentenceConfig:
    max_words: int | None = None
    silence_gap: float | None = None
    max_duration: float | None = None


def tokens_to_sentences(
    tokens: list[AlignedToken], config: SentenceConfig = SentenceConfig()
) -> list[AlignedSentence]:
    sentences = []
    current_tokens: list[AlignedToken] = []

    for idx, token in enumerate(tokens):
        current_tokens.append(token)

        is_punctuation = (
            # hacky, will fix
            "!" in token.text
            or "?" in token.text
            or "。" in token.text
            or "？" in token.text
            or "！" in token.text
            or (
                "." in token.text
                and (idx == len(tokens) - 1 or " " in tokens[idx + 1].text)
            )
        )
        is_word_limit = (
            (config.max_words is not None)
            and (idx != len(tokens) - 1)
            and (
                len([x for x in current_tokens if " " in x.text])
                + (1 if " " in tokens[idx + 1].text else 0)
                > config.max_words
            )
        )
        is_long_silence = (
            (config.silence_gap is not None)
            and (idx != len(tokens) - 1)
            and (tokens[idx + 1].start - token.end >= config.silence_gap)
        )
        is_over_duration = (config.max_duration is not None) and (
            token.end - current_tokens[0].start >= config.max_duration
        )

        if is_punctuation or is_word_limit or is_long_silence or is_over_duration:
            sentence_text = "".join(t.text for t in current_tokens)
            sentence = AlignedSentence(text=sentence_text, tokens=current_tokens)
            sentences.append(sentence)

            current_tokens = []

    if current_tokens:
        sentence_text = "".join(t.text for t in current_tokens)
        sentence = AlignedSentence(text=sentence_text, tokens=current_tokens)
        sentences.append(sentence)

    return sentences


def sentences_to_result(sentences: list[AlignedSentence]) -> AlignedResult:
    return AlignedResult("".join(sentence.text for sentence in sentences), sentences)


def merge_longest_contiguous(
    a: list[AlignedToken],
    b: list[AlignedToken],
    *,
    overlap_duration: float,
):
    if not a or not b:
        return b if not a else a

    a_end_time = a[-1].end
    b_start_time = b[0].start

    if a_end_time <= b_start_time:
        return a + b

    overlap_a = [token for token in a if token.end > b_start_time - overlap_duration]
    overlap_b = [token for token in b if token.start < a_end_time + overlap_duration]

    enough_pairs = len(overlap_a) // 2

    if len(overlap_a) < 2 or len(overlap_b) < 2:
        cutoff_time = (a_end_time + b_start_time) / 2
        return [t for t in a if t.end <= cutoff_time] + [
            t for t in b if t.start >= cutoff_time
        ]

    best_contiguous = []
    for i in range(len(overlap_a)):
        for j in range(len(overlap_b)):
            if (
                overlap_a[i].id == overlap_b[j].id
                and abs(overlap_a[i].start - overlap_b[j].start) < overlap_duration / 2
            ):
                current = []
                k, l = i, j
                while (
                    k < len(overlap_a)
                    and l < len(overlap_b)
                    and overlap_a[k].id == overlap_b[l].id
                    and abs(overlap_a[k].start - overlap_b[l].start)
                    < overlap_duration / 2
                ):
                    current.append((k, l))
                    k += 1
                    l += 1

                if len(current) > len(best_contiguous):
                    best_contiguous = current

    if len(best_contiguous) >= enough_pairs:
        a_start_idx = len(a) - len(overlap_a)
        lcs_indices_a = [a_start_idx + pair[0] for pair in best_contiguous]
        lcs_indices_b = [pair[1] for pair in best_contiguous]

        result = []
        result.extend(a[: lcs_indices_a[0]])

        for i in range(len(best_contiguous)):
            idx_a = lcs_indices_a[i]
            idx_b = lcs_indices_b[i]

            result.append(a[idx_a])

            if i < len(best_contiguous) - 1:
                next_idx_a = lcs_indices_a[i + 1]
                next_idx_b = lcs_indices_b[i + 1]

                gap_tokens_a = a[idx_a + 1 : next_idx_a]
                gap_tokens_b = b[idx_b + 1 : next_idx_b]

                if len(gap_tokens_b) > len(gap_tokens_a):
                    result.extend(gap_tokens_b)
                else:
                    result.extend(gap_tokens_a)

        result.extend(b[lcs_indices_b[-1] + 1 :])
        return result
    else:
        raise RuntimeError(f"No pairs exceeding {enough_pairs}")


def merge_longest_common_subsequence(
    a: list[AlignedToken],
    b: list[AlignedToken],
    *,
    overlap_duration: float,
):
    if not a or not b:
        return b if not a else a

    a_end_time = a[-1].end
    b_start_time = b[0].start

    if a_end_time <= b_start_time:
        return a + b

    overlap_a = [token for token in a if token.end > b_start_time - overlap_duration]
    overlap_b = [token for token in b if token.start < a_end_time + overlap_duration]

    if len(overlap_a) < 2 or len(overlap_b) < 2:
        cutoff_time = (a_end_time + b_start_time) / 2
        return [t for t in a if t.end <= cutoff_time] + [
            t for t in b if t.start >= cutoff_time
        ]

    dp = [[0 for _ in range(len(overlap_b) + 1)] for _ in range(len(overlap_a) + 1)]

    for i in range(1, len(overlap_a) + 1):
        for j in range(1, len(overlap_b) + 1):
            if (
                overlap_a[i - 1].id == overlap_b[j - 1].id
                and abs(overlap_a[i - 1].start - overlap_b[j - 1].start)
                < overlap_duration / 2
            ):
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_pairs = []
    i, j = len(overlap_a), len(overlap_b)

    while i > 0 and j > 0:
        if (
            overlap_a[i - 1].id == overlap_b[j - 1].id
            and abs(overlap_a[i - 1].start - overlap_b[j - 1].start)
            < overlap_duration / 2
        ):
            lcs_pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    lcs_pairs.reverse()

    if not lcs_pairs:
        cutoff_time = (a_end_time + b_start_time) / 2
        return [t for t in a if t.end <= cutoff_time] + [
            t for t in b if t.start >= cutoff_time
        ]

    a_start_idx = len(a) - len(overlap_a)
    lcs_indices_a = [a_start_idx + pair[0] for pair in lcs_pairs]
    lcs_indices_b = [pair[1] for pair in lcs_pairs]

    result = []

    result.extend(a[: lcs_indices_a[0]])

    for i in range(len(lcs_pairs)):
        idx_a = lcs_indices_a[i]
        idx_b = lcs_indices_b[i]

        result.append(a[idx_a])

        if i < len(lcs_pairs) - 1:
            next_idx_a = lcs_indices_a[i + 1]
            next_idx_b = lcs_indices_b[i + 1]

            gap_tokens_a = a[idx_a + 1 : next_idx_a]
            gap_tokens_b = b[idx_b + 1 : next_idx_b]

            if len(gap_tokens_b) > len(gap_tokens_a):
                result.extend(gap_tokens_b)
            else:
                result.extend(gap_tokens_a)

    result.extend(b[lcs_indices_b[-1] + 1 :])

    return result
