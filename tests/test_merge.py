from itertools import pairwise

import pytest

from parakeet_mlx.alignment import (
    AlignedToken,
    merge_longest_common_subsequence,
    merge_longest_contiguous,
    tokens_to_sentences,
)

MERGES = [merge_longest_contiguous, merge_longest_common_subsequence]


def tok(id, text, start, duration=0.08):
    return AlignedToken(id=id, text=text, start=start, duration=duration)


def merge(fn, a, b, overlap_duration):
    try:
        return fn(a, b, overlap_duration=overlap_duration)
    except RuntimeError:  # contiguous gives up; transcribe() falls back to LCS
        pytest.skip("no contiguous match")


def check(merged, expected_text):
    # the merged order is the text; timestamps must agree with it, because
    # AlignedSentence sorts its tokens by start
    assert "".join(t.text for t in merged) == expected_text
    assert all(x.start <= y.start for x, y in pairwise(merged))
    for sentence in tokens_to_sentences(merged):
        assert sentence.text == "".join(t.text for t in sentence.tokens)


# The two counterexamples from the review of #54: the second pass catches a real
# token the first pass missed, with a start time earlier than its predecessor's.
# It must be kept, not dropped.


@pytest.mark.parametrize("fn", MERGES)
def test_missed_token_is_kept(fn):
    a = [tok(1, "t1", 5.00, 0.30), tok(2, "t2", 7.00, 0.30)]
    b = [tok(1, "t1", 4.50, 0.30), tok(3, "t3", 4.95, 0.30), tok(2, "t2", 7.00, 0.30)]
    check(merge(fn, a, b, overlap_duration=3.0), "t1t3t2")


@pytest.mark.parametrize("fn", MERGES)
def test_new_tokens_after_stretched_token_are_kept(fn):
    a = [tok(99, "t99", 4.50, 0.30), tok(1, "t1", 5.00, 0.30)]
    b = [
        tok(99, "t99", 4.50, 0.30),
        tok(1, "t1", 4.60, 0.30),
        tok(2, "t2", 4.95, 0.30),
        tok(3, "t3", 6.00, 0.30),
    ]
    check(merge(fn, a, b, overlap_duration=3.0), "t99t1t2t3")


# Shapes seen in real podcast audio (chunk_duration=30, overlap_duration=5).


@pytest.mark.parametrize("fn", MERGES)
def test_gap_inside_a_word(fn):
    # chunk a heard " Republicanvenge", chunk b " Republican Revenge": the b gap
    # " R" "e" goes between a's "an" and "ven" but carries b's later timestamps
    a = [
        tok(251, " R", 4374.64),
        tok(371, "ep", 4374.80),
        tok(502, "ub", 4374.96),
        tok(662, "lic", 4375.04),
        tok(25, "an", 4375.20),
        tok(264, "ven", 4375.52),
        tok(160, "ge", 4375.76),
        tok(1, " t", 4376.00),
        tok(249, "our", 4376.24),
        tok(841, ".", 4379.76),
    ]
    b = [
        tok(251, " R", 4375.08),
        tok(371, "ep", 4375.16),
        tok(502, "ub", 4375.24),
        tok(662, "lic", 4375.32),
        tok(25, "an", 4375.40),
        tok(251, " R", 4375.48),
        tok(820, "e", 4375.56),
        tok(264, "ven", 4375.64),
        tok(160, "ge", 4375.80),
        tok(1, " t", 4375.96),
        tok(249, "our", 4376.20),
        tok(59, " is", 4376.68),
    ]
    check(merge(fn, a, b, overlap_duration=5.0), " Republican Revenge tour is")


@pytest.mark.parametrize("fn", MERGES)
def test_last_token_of_a_pairs_with_a_later_one_in_b(fn):
    # a's final "." pairs with a "." in b 2.3 s later; the longer b gap before
    # it is kept, then a's "." arrives with its own, earlier, timestamp
    a = [
        tok(102, " with", 29.52),
        tok(152, "out", 29.60),
        tok(24, " p", 29.68),
        tok(41, "ar", 29.76),
        tok(22, "a", 29.84),
        tok(841, ".", 29.92),
    ]
    b = [
        tok(102, " with", 29.48),
        tok(152, "out", 29.56),
        tok(24, " p", 29.64),
        tok(41, "ar", 29.72),
        tok(57, "ot", 29.96),
        tok(841, ".", 30.20, 0.24),
        tok(212, " N", 30.44),
        tok(690, "ew", 30.52),
        tok(401, " year", 30.68),
        tok(839, ",", 30.92),
        tok(476, " new", 31.16),
        tok(16, " c", 31.48),
        tok(309, "are", 31.64),
        tok(12, "er", 31.80, 0.24),
        tok(841, ".", 32.20),
        tok(121, " B", 32.36),
        tok(831, "u", 32.44),
        tok(473, "ild", 32.52),
    ]
    check(
        merge(fn, a, b, overlap_duration=5.0),
        " without parot. New year, new career. Build",
    )


@pytest.mark.parametrize("fn", MERGES)
def test_ordered_input_is_unchanged(fn):
    a = [tok(i, f" w{i}", 20.0 + 0.5 * i) for i in range(20)]
    b = [tok(i, f" w{i}", 20.0 + 0.5 * i) for i in range(10, 30)]
    merged = merge(fn, a, b, overlap_duration=5.0)
    assert [(t.text, t.start, t.duration) for t in merged] == [
        (f" w{i}", 20.0 + 0.5 * i, 0.08) for i in range(30)
    ]
