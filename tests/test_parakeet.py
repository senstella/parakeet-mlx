import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

from parakeet_mlx.audio import PreprocessArgs
from parakeet_mlx.parakeet import BaseParakeet


class TranscribeInputValidationTests(unittest.TestCase):
    def test_empty_decoded_audio_raises_clear_error_before_preprocessing(self):
        def evaluate_mel(mel, *, decoding_config):
            mx.eval(mel)
            return []

        model = SimpleNamespace(
            preprocessor_config=PreprocessArgs(
                sample_rate=16_000,
                normalize="per_feature",
                window_size=0.025,
                window_stride=0.01,
                window="hann",
                features=128,
                n_fft=512,
                dither=0.0,
            ),
            generate=evaluate_mel,
        )

        with patch(
            "parakeet_mlx.parakeet.load_audio",
            return_value=mx.array([], dtype=mx.float32),
        ):
            try:
                BaseParakeet.transcribe(model, "empty.wav", chunk_duration=120.0)
            except Exception as error:
                self.assertIsInstance(error, ValueError)
                self.assertIn("at least 160 decoded samples; got 0", str(error))
            else:
                self.fail("empty decoded audio was accepted")


if __name__ == "__main__":
    unittest.main()
