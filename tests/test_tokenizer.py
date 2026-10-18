from types import SimpleNamespace

import mlx.core as mx

from parakeet_mlx.alignment import sentences_to_result, tokens_to_sentences
from parakeet_mlx.parakeet import ParakeetCTC
from parakeet_mlx.tokenizer import decode


def test_decode_joins_pieces():
    vocab = ["<unk>", "▁hello", "▁wor", "ld", "."]
    assert decode([1, 2, 3, 4], vocab) == " hello world."


def test_decode_skips_control_tokens():
    vocab = ["<unk>", "<|nospeech|>", "<pad>", "▁hello", "<|en|>", "▁world"]
    assert decode([0, 1, 2, 3, 4, 5], vocab) == " hello world"


def test_decode_skips_unk_inside_a_word():
    # an out-of-vocabulary character shows up as <unk> between two pieces
    assert decode([1, 0, 2], ["<unk>", "骨", "品"]) == "骨品"


def test_decode_keeps_pieces_that_only_look_like_markers():
    vocab = ["<", "<|", "|>", "<b>", "▁<unk"]
    assert decode([0, 1, 2, 3, 4], vocab) == "<<||><b> <unk"


def test_ctc_text_has_no_control_tokens_but_tokens_are_kept():
    vocab = ["<unk>", "▁a", "▁b"]  # the CTC blank is index len(vocab)
    frames = [1, 1, 0, 0, 2, 3]  # a a <unk> <unk> b blank
    logits = mx.log(mx.eye(len(vocab) + 1)[mx.array(frames)] + 1e-6)[None]
    model = SimpleNamespace(
        vocabulary=vocab,
        decoder=lambda features: logits,
        time_ratio=0.08,
        encoder_config=SimpleNamespace(subsampling_factor=8),
        preprocessor_config=SimpleNamespace(sample_rate=16000, hop_length=160),
    )

    (tokens,) = ParakeetCTC.decode(
        model, mx.zeros((1, len(frames), 1)), mx.array([len(frames)])
    )
    result = sentences_to_result(tokens_to_sentences(tokens))

    assert result.text == "a b"
    # the <unk> token stays, with its id and timing, so token-level consumers
    # (and streaming, which feeds the last token id back) are unaffected
    assert [t.id for t in result.tokens] == [1, 0, 2]
    assert [t.text for t in result.tokens] == [" a", "", " b"]
    assert result.tokens[1].start > result.tokens[0].start
