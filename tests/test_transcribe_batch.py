from types import SimpleNamespace

import mlx.core as mx
import pytest

from parakeet_mlx.alignment import AlignedResult
from parakeet_mlx.parakeet import BaseParakeet, DecodingConfig, Greedy


def make_model():
    model = BaseParakeet.__new__(BaseParakeet)
    object.__setattr__(
        model,
        "preprocessor_config",
        SimpleNamespace(sample_rate=16_000, hop_length=160),
    )
    return model


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_transcribe_batch_preprocesses_each_path_once_and_generates_one_ordered_batch(
    monkeypatch, batch_size
):
    """Catches batching that skips a source, reorders it, or invokes the model per file."""
    model = make_model()
    paths = [f"audio-{index}.wav" for index in range(batch_size)]
    loaded_paths = []
    preprocessed_audio = []
    generated = []
    dtype = mx.float32
    decoding_config = DecodingConfig(decoding=Greedy())

    def fake_load_audio(path, sample_rate, actual_dtype):
        loaded_paths.append((str(path), sample_rate, actual_dtype))
        return mx.full((320,), len(loaded_paths), dtype=actual_dtype)

    def fake_get_logmel(audio, preprocess_config):
        preprocessed_audio.append(int(audio[0]))
        return mx.full((1, 3, 2), int(audio[0]), dtype=audio.dtype)

    expected_results = [AlignedResult(path, []) for path in paths]

    def fake_generate(mel, *, decoding_config):
        generated.append((mel, decoding_config))
        return expected_results

    monkeypatch.setattr("parakeet_mlx.parakeet.load_audio", fake_load_audio)
    monkeypatch.setattr("parakeet_mlx.parakeet.get_logmel", fake_get_logmel)
    object.__setattr__(model, "generate", fake_generate)

    results = model.transcribe_batch(
        paths, dtype=dtype, decoding_config=decoding_config
    )

    assert results == expected_results
    assert loaded_paths == [(path, 16_000, dtype) for path in paths]
    assert preprocessed_audio == list(range(1, batch_size + 1))
    assert len(generated) == 1
    mel, actual_config = generated[0]
    assert mel.shape == (batch_size, 3, 2)
    assert mel.tolist() == [
        [[index, index], [index, index], [index, index]]
        for index in range(1, batch_size + 1)
    ]
    assert actual_config is decoding_config


def test_transcribe_batch_rejects_unequal_logmel_shapes_before_generate(monkeypatch):
    """Catches a variable-length batch reaching the encoder through padding or concatenation."""
    model = make_model()
    generate_calls = []

    monkeypatch.setattr(
        "parakeet_mlx.parakeet.load_audio",
        lambda path, sample_rate, dtype: mx.ones((320,), dtype=dtype),
    )
    mels = iter([mx.ones((1, 3, 2)), mx.ones((1, 4, 2))])
    monkeypatch.setattr("parakeet_mlx.parakeet.get_logmel", lambda *_: next(mels))
    object.__setattr__(
        model,
        "generate",
        lambda *args, **kwargs: generate_calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="same shape"):
        model.transcribe_batch(["first.wav", "second.wav"])

    assert generate_calls == []


@pytest.mark.parametrize("sample_count", [0, 159])
def test_transcribe_batch_rejects_audio_shorter_than_one_hop_before_logmel(
    monkeypatch, sample_count
):
    """Catches empty or undersized decoded audio reaching log-mel preprocessing."""
    model = make_model()
    preprocess_calls = []
    generate_calls = []

    monkeypatch.setattr(
        "parakeet_mlx.parakeet.load_audio",
        lambda path, sample_rate, dtype: mx.ones((sample_count,), dtype=dtype),
    )
    monkeypatch.setattr(
        "parakeet_mlx.parakeet.get_logmel",
        lambda *args: preprocess_calls.append(args),
    )
    object.__setattr__(
        model,
        "generate",
        lambda *args, **kwargs: generate_calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="at least one hop"):
        model.transcribe_batch(["short.wav"])

    assert preprocess_calls == []
    assert generate_calls == []


def test_transcribe_batch_rejects_an_empty_path_list():
    """Catches an empty request reaching concatenation or generation."""
    with pytest.raises(ValueError, match="at least one path"):
        make_model().transcribe_batch([])
