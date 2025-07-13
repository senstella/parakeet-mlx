# Parakeet MLX

An implementation of the Parakeet models - Nvidia's ASR(Automatic Speech Recognition) models - for Apple Silicon using MLX.

## Installation

> [!NOTE]
> Make sure you have `ffmpeg` installed on your system first, otherwise CLI won't work properly.

Using [uv](https://docs.astral.sh/uv/) - recommended way:

```bash
uv add parakeet-mlx -U
```

Or, for the CLI:

```bash
uv tool install parakeet-mlx -U
```

Using pip:

```bash
pip install parakeet-mlx -U
```

## CLI Quick Start

```bash
parakeet-mlx <audio_files> [OPTIONS]
```

## Arguments

- `audio_files`: One or more audio files to transcribe (WAV, MP3, etc.)

## Options

- `--model` (default: `mlx-community/parakeet-tdt-0.6b-v2`, env: `PARAKEET_MODEL`)
  - Hugging Face repository of the model to use

- `--output-dir` (default: current directory)
  - Directory to save transcription outputs

- `--output-format` (default: srt, env: `PARAKEET_OUTPUT_FORMAT`)
  - Output format (txt/srt/vtt/json/all)

- `--output-template` (default: `{filename}`, env: `PARAKEET_OUTPUT_TEMPLATE`)
  - Template for output filenames, `{parent}`, `{filename}`, `{index}`, `{date}` is supported.

- `--highlight-words` (default: False)
  - Enable word-level timestamps in SRT/VTT outputs

- `--verbose` / `-v` (default: False)
  - Print detailed progress information

- `--chunk-duration` (default: 120 seconds, env: `PARAKEET_CHUNK_DURATION`)
  - Chunking duration in seconds for long audio, `0` to disable chunking

- `--overlap-duration` (default: 15 seconds, env: `PARAKEET_OVERLAP_DURATION`)
  - Overlap duration in seconds if using chunking

- `--fp32` / `--bf16` (default: `bf16`, env: `PARAKEET_FP32` - boolean)
  - Determine the precision to use

- `--full-attention` / `--local-attention` (default: `full-attention`, env: `PARAKEET_LOCAL_ATTENTION` - boolean)
  - Use full attention or local attention (Local attention reduces intermediate memory usage)
  - Expected usage case is for long audio transcribing without chunking

- `--local-attention-context-size` (default: 256, env: `PARAKEET_LOCAL_ATTENTION_CTX`)
  - Local attention context size(window) in frames of Parakeet model

## Examples

```bash
# Basic transcription
parakeet-mlx audio.mp3

# Multiple files with word-level timestamps of VTT subtitle
parakeet-mlx *.mp3 --output-format vtt --highlight-words

# Generate all output formats
parakeet-mlx audio.mp3 --output-format all
```


## Python API Quick Start

Transcribe a file:

```py
from parakeet_mlx import from_pretrained

model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v2")

result = model.transcribe("audio_file.wav")

print(result.text)
```

Check timestamps:

```py
from parakeet_mlx import from_pretrained

model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v2")

result = model.transcribe("audio_file.wav")

print(result.sentences)
# [AlignedSentence(text="Hello World.", start=1.01, end=2.04, duration=1.03, tokens=[...])]
```

Do chunking:

```py
from parakeet_mlx import from_pretrained

model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v2")

result = model.transcribe("audio_file.wav", chunk_duration=60 * 2.0, overlap_duration=15.0)

print(result.sentences)
```

Use local attention:

```py
from parakeet_mlx import from_pretrained

model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v2")

model.encoder.set_attention_model(
    "rel_pos_local_attn", # Follows NeMo's naming convention
    (256, 256),
)

result = model.transcribe("audio_file.wav")

print(result.sentences)
```

## Timestamp Result

- `AlignedResult`: Top-level result containing the full text and sentences
  - `text`: Full transcribed text
  - `sentences`: List of `AlignedSentence`
- `AlignedSentence`: Sentence-level alignments with start/end times
  - `text`: Sentence text
  - `start`: Start time in seconds
  - `end`: End time in seconds
  - `duration`: Between `start` and `end`.
  - `tokens`: List of `AlignedToken`
- `AlignedToken`: Word/token-level alignments with precise timestamps
  - `text`: Token text
  - `start`: Start time in seconds
  - `end`: End time in seconds
  - `duration`: Between `start` and `end`.

## Streaming Transcription

For real-time transcription, use the `transcribe_stream` method which creates a streaming context:

```py
from parakeet_mlx import from_pretrained
from parakeet_mlx.audio import load_audio
import numpy as np

model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v2")

# Create a streaming context
with model.transcribe_stream(
    context_size=(256, 256),  # (left_context, right_context) frames
) as transcriber:
    # Simulate real-time audio chunks
    audio_data = load_audio("audio_file.wav", model.preprocessor_config.sample_rate)
    chunk_size = model.preprocessor_config.sample_rate  # 1 second chunks

    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i+chunk_size]
        transcriber.add_audio(chunk)

        # Access current transcription
        result = transcriber.result
        print(f"Current text: {result.text}")

        # Access finalized and draft tokens
        # transcriber.finalized_tokens
        # transcriber.draft_tokens
```

### Streaming Parameters

- `context_size`: Tuple of (left_context, right_context) for attention windows
  - Controls how many frames the model looks at before and after current position
  - Default: (256, 256)

- `depth`: Number of encoder layers that preserve exact computation across chunks
  - Controls how many layers maintain exact equivalence with non-streaming forward pass
  - depth=1: Only first encoder layer matches non-streaming computation exactly
  - depth=2: First two layers match exactly, and so on
  - depth=N (total layers): Full equivalence to non-streaming forward pass
  - Higher depth means more computational consistency with non-streaming mode
  - Default: 1

- `keep_original_attention`: Whether to keep original attention mechanism
  - False: Switches to local attention for streaming (recommended)
  - True: Keeps original attention (less suitable for streaming)
  - Default: False

## Low-Level API

To transcribe log-mel spectrum directly, you can do the following:

```python
import mlx.core as mx
from parakeet_mlx.audio import get_logmel, load_audio

# Load and preprocess audio manually
audio = load_audio("audio.wav", model.preprocessor_config.sample_rate)
mel = get_logmel(audio, model.preprocessor_config)

# Generate transcription with alignments
# Accepts both [batch, sequence, feat] and [sequence, feat]
# `alignments` is list of AlignedResult. (no matter if you fed batch dimension or not!)
alignments = model.generate(mel)
```

## Todo

- [X] Add CLI for better usability
- [X] Add support for other Parakeet variants
- [X] Streaming input (real-time transcription with `transcribe_stream`)
- [ ] Option to enhance chosen words' accuracy
- [ ] Chunking with continuous context (partially achieved with streaming)


## Acknowledgments

- Thanks to [Nvidia](https://www.nvidia.com/) for training these awesome models and writing cool papers and providing nice implementation.
- Thanks to [MLX](https://github.com/ml-explore/mlx) project for providing the framework that made this implementation possible.
- Thanks to [audiofile](https://github.com/audeering/audiofile) and [audresample](https://github.com/audeering/audresample), [numpy](https://numpy.org), [librosa](https://librosa.org) for audio processing.
- Thanks to [dacite](https://github.com/konradhalas/dacite) for config management.

## License

Apache 2.0
