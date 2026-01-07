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

- `--model` (default: `mlx-community/parakeet-tdt-0.6b-v3`, env: `PARAKEET_MODEL`)
  - Hugging Face repository of the model to use
  - https://huggingface.co/collections/mlx-community/parakeet

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

- `--decoding` (default: `greedy`, env: `PARAKEET_DECODING`)
  - Decoding method to use (`greedy` or `beam`)
  - `beam` is only available at TDT models for now

- `--chunk-duration` (default: 120 seconds, env: `PARAKEET_CHUNK_DURATION`)
  - Chunking duration in seconds for long audio, `0` to disable chunking

- `--overlap-duration` (default: 15 seconds, env: `PARAKEET_OVERLAP_DURATION`)
  - Overlap duration in seconds if using chunking

- `--beam-size` (default: 5, env: `PARAKEET_BEAM_SIZE`)
  - Beam size (only used while beam decoding)

- `--length-penalty` (default: 0.013, env: `PARAKEET_LENGTH_PENALTY`)
  - Length penalty in beam. 0.0 to disable (only used while beam decoding)

- `--patience` (default: 3.5, env: `PARAKEET_PATIENCE`)
  - Patience in beam. 1.0 to disable (only used while beam decoding)

- `--duration-reward` (default: 0.67, env: `PARAKEET_DURATION_REWARD`)
  - From 0.0 to 1.0, < 0.5 to favor token logprobs more, > 0.5 to favor duration logprobs more. (only used while beam decoding in TDT)

- `--max-words` (default: None, env: `PARAKEET_MAX_WORDS`)
  - Max words per sentence

- `--silence-gap` (default: None, env: `PARAKEET_SILENCE_GAP`)
  - Split sentence if it exceeds silence gap provided (seconds)

- `--max-duration` (default: None, env: `PARAKEET_MAX_DURATION`)
  - Max sentence duration (seconds)

- `--fp32` / `--bf16` (default: `bf16`, env: `PARAKEET_FP32` - boolean)
  - Determine the precision to use

- `--full-attention` / `--local-attention` (default: `full-attention`, env: `PARAKEET_LOCAL_ATTENTION` - boolean)
  - Use full attention or local attention (Local attention reduces intermediate memory usage)
  - Expected usage case is for long audio transcribing without chunking

- `--local-attention-context-size` (default: 256, env: `PARAKEET_LOCAL_ATTENTION_CTX`)
  - Local attention context size(window) in frames of Parakeet model

- `--cache-dir` (default: None, env: `PARAKEET_CACHE_DIR`)
  - Directory for HuggingFace model cache. If not specified, uses HF's default cache location [(~/.cache/huggingface or values you set in `HF_HOME` or `HF_HUB_CACHE` which is essentially `$HF_HOME/hub`)](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)

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

model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")

result = model.transcribe("audio_file.wav")

print(result.text)
```

Check timestamps:

```py
from parakeet_mlx import from_pretrained

model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")

result = model.transcribe("audio_file.wav")

print(result.sentences)
# [AlignedSentence(text="Hello World.", start=1.01, end=2.04, duration=1.03, tokens=[...])]
```

Do chunking:

```py
from parakeet_mlx import from_pretrained

model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")

result = model.transcribe("audio_file.wav", chunk_duration=60 * 2.0, overlap_duration=15.0)

print(result.sentences)
```

Do beam decoding:

```py
from parakeet_mlx import from_pretrained, DecodingConfig, Beam

model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")

config = DecodingConfig(
    decoding = decoding(
        beam_size=5, length_penalty=0.013, patience=3.5, duration_reward=0.67
        # Refer to CLI options for each parameters 
    )
)

result = model.transcribe("audio_file.wav", decoding_config=config)

print(result.sentences)
```

Use local attention:

```py
from parakeet_mlx import from_pretrained

model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")

model.encoder.set_attention_model(
    "rel_pos_local_attn", # Follows NeMo's naming convention
    (256, 256),
)

result = model.transcribe("audio_file.wav")

print(result.sentences)
```

Specifiy the sentence split options:

```py
from parakeet_mlx import from_pretrained, DecodingConfig, SentenceConfig

model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")

config = DecodingConfig(
    sentence = SentenceConfig(
        # Refer to CLI Options to see what those options does
        max_words=30, silence_gap=5.0, max_duration=40.0
    )
)

result = model.transcribe("audio_file.wav", decoding_config=config)

print(result.sentences)
```

## from_pretrained

Using `from_pretrained` downloads a model from Hugging Face and stores the downloaded model in HF's [cache folder](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache). You can specify the cache folder by passing it `cache_dir` args. It can return one of those Parakeet variants such as: `ParakeetTDT`, `ParakeetRNNT`, `ParakeetCTC`, or `ParakeetTDTCTC`. For general use cases, the `BaseParakeet` abstraction often suffices. However, if you want to call variant-specific functions like `.decode()` and want linters not to complain, `typing.cast` can be used.

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

model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")

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
from parakeet_mlx import DecodingConfig

# Load and preprocess audio manually
audio = load_audio("audio.wav", model.preprocessor_config.sample_rate)
mel = get_logmel(audio, model.preprocessor_config)

# Generate transcription with alignments
# Accepts both [batch, sequence, feat] and [sequence, feat]
# `alignments` is list of AlignedResult. (no matter if you fed batch dimension or not!)
alignments = model.generate(mel, decoding_config=DecodingConfig())
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
