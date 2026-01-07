import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import typer
from mlx.core import bfloat16, float32
from rich import print
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from typing_extensions import Annotated

from parakeet_mlx import AlignedResult, AlignedSentence, AlignedToken, from_pretrained
from parakeet_mlx.alignment import SentenceConfig
from parakeet_mlx.parakeet import Beam, DecodingConfig, Greedy

app = typer.Typer(no_args_is_help=True)


# helpers
def format_timestamp(
    seconds: float, always_include_hours: bool = True, decimal_marker: str = ","
) -> str:
    assert seconds >= 0
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000

    minutes = milliseconds // 60_000
    milliseconds %= 60_000

    seconds = milliseconds // 1_000
    milliseconds %= 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def to_txt(result: AlignedResult) -> str:
    """Format transcription result as plain text."""
    return result.text.strip()


def to_srt(result: AlignedResult, highlight_words: bool = False) -> str:
    """
    Format transcription result as an SRT file.
    """
    srt_content = []
    entry_index = 1
    if highlight_words:
        for sentence in result.sentences:
            for i, token in enumerate(sentence.tokens):
                start_time = format_timestamp(token.start, decimal_marker=",")
                end_time = format_timestamp(
                    token.end
                    if token == sentence.tokens[-1]
                    else sentence.tokens[i + 1].start,
                    decimal_marker=",",
                )

                text = ""
                for j, inner_token in enumerate(sentence.tokens):
                    if i == j:
                        text += inner_token.text.replace(
                            inner_token.text.strip(),
                            f"<u>{inner_token.text.strip()}</u>",
                        )
                    else:
                        text += inner_token.text
                text.strip()

                srt_content.append(f"{entry_index}")
                srt_content.append(f"{start_time} --> {end_time}")
                srt_content.append(text)
                srt_content.append("")
                entry_index += 1
    else:
        for sentence in result.sentences:
            start_time = format_timestamp(sentence.start, decimal_marker=",")
            end_time = format_timestamp(sentence.end, decimal_marker=",")
            text = sentence.text.strip()

            srt_content.append(f"{entry_index}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text)
            srt_content.append("")
            entry_index += 1

    return "\n".join(srt_content)


def to_vtt(result: AlignedResult, highlight_words: bool = False) -> str:
    """
    Format transcription result as a VTT file.
    """
    vtt_content = ["WEBVTT", ""]
    if highlight_words:
        for sentence in result.sentences:
            for i, token in enumerate(sentence.tokens):
                start_time = format_timestamp(token.start, decimal_marker=".")
                end_time = format_timestamp(
                    token.end
                    if token == sentence.tokens[-1]
                    else sentence.tokens[i + 1].start,
                    decimal_marker=".",
                )

                text_line = ""
                for j, inner_token in enumerate(sentence.tokens):
                    if i == j:
                        text_line += inner_token.text.replace(
                            inner_token.text.strip(),
                            f"<b>{inner_token.text.strip()}</b>",
                        )
                    else:
                        text_line += inner_token.text
                text_line = text_line.strip()

                vtt_content.append(f"{start_time} --> {end_time}")
                vtt_content.append(text_line)
                vtt_content.append("")
    else:
        for sentence in result.sentences:
            start_time = format_timestamp(sentence.start, decimal_marker=".")
            end_time = format_timestamp(sentence.end, decimal_marker=".")
            text_line = sentence.text.strip()

            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(text_line)
            vtt_content.append("")

    return "\n".join(vtt_content)


def _aligned_token_to_dict(token: AlignedToken) -> Dict[str, Any]:
    return {
        "text": token.text,
        "start": round(token.start, 3),
        "end": round(token.end, 3),
        "duration": round(token.duration, 3),
        "confidence": round(token.confidence, 3),
    }


def _aligned_sentence_to_dict(sentence: AlignedSentence) -> Dict[str, Any]:
    return {
        "text": sentence.text,
        "start": round(sentence.start, 3),
        "end": round(sentence.end, 3),
        "duration": round(sentence.duration, 3),
        "confidence": round(sentence.confidence, 3),
        "tokens": [_aligned_token_to_dict(token) for token in sentence.tokens],
    }


def to_json(result: AlignedResult) -> str:
    output_dict = {
        "text": result.text,
        "sentences": [
            _aligned_sentence_to_dict(sentence) for sentence in result.sentences
        ],
    }
    return json.dumps(output_dict, indent=2, ensure_ascii=False)


@app.command("transcribe")
def transcribe(
    audios: Annotated[
        List[Path],
        typer.Argument(
            help="Files to transcribe",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    model: Annotated[
        str,
        typer.Option(
            help="Hugging Face repository of model to use", envvar="PARAKEET_MODEL"
        ),
    ] = "mlx-community/parakeet-tdt-0.6b-v3",
    output_dir: Annotated[
        Path, typer.Option(help="Directory to save transcriptions")
    ] = Path("."),
    output_format: Annotated[
        str,
        typer.Option(
            help="Format for output files (txt, srt, vtt, json, all)",
            envvar="PARAKEET_OUTPUT_FORMAT",
        ),
    ] = "srt",
    output_template: Annotated[
        str,
        typer.Option(
            help="Template for output filenames, e.g. '{parent}/{filename}_{date}_{index}'",
            envvar="PARAKEET_OUTPUT_TEMPLATE",
        ),
    ] = "{filename}",
    highlight_words: Annotated[
        bool,
        typer.Option(help="Underline/timestamp each word as it is spoken in srt/vtt"),
    ] = False,
    decoding: Annotated[
        Literal["greedy", "beam"],
        typer.Option(
            help="Decoding method to use",
            envvar="PARAKEET_DECODING",
        ),
    ] = "greedy",
    chunk_duration: Annotated[
        float,
        typer.Option(
            help="Chunking duration in seconds for long audio, 0 to disable chunking.",
            envvar="PARAKEET_CHUNK_DURATION",
        ),
    ] = 60 * 2,
    overlap_duration: Annotated[
        float,
        typer.Option(
            help="Overlap duration in seconds if using chunking",
            envvar="PARAKEET_OVERLAP_DURATION",
        ),
    ] = 15,
    beam_size: Annotated[
        int,
        typer.Option(
            help="Beam size (only used while beam decoding)",
            envvar="PARAKEET_BEAM_SIZE",
        ),
    ] = 5,
    length_penalty: Annotated[
        float,
        typer.Option(
            help="Length penalty in beam. 0.0 to disable (only used while beam decoding)",
            envvar="PARAKEET_LENGTH_PENALTY",
        ),
    ] = 0.013,
    patience: Annotated[
        float,
        typer.Option(
            help="Patience in beam. 1.0 to disable (only used while beam decoding)",
            envvar="PARAKEET_PATIENCE",
        ),
    ] = 3.5,
    duration_reward: Annotated[
        float,
        typer.Option(
            help="From 0.0 to 1.0, < 0.5 to favor token logprobs more, > 0.5 to favor duration logprobs more. (only used while beam decoding in TDT)",
            envvar="PARAKEET_DURATION_REWARD",
        ),
    ] = 0.67,
    max_words: Annotated[
        Optional[int],
        typer.Option(
            "--max-words",
            help="Max words per sentence",
            envvar="PARAKEET_MAX_WORDS",
        ),
    ] = None,
    silence_gap: Annotated[
        Optional[float],
        typer.Option(
            "--silence-gap",
            help="Split at silence gaps (seconds)",
            envvar="PARAKEET_SILENCE_GAP",
        ),
    ] = None,
    max_duration: Annotated[
        Optional[float],
        typer.Option(
            "--max-duration",
            help="Max sentence duration (seconds)",
            envvar="PARAKEET_MAX_DURATION",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Print out process and debug messages"),
    ] = False,
    fp32: Annotated[
        bool,
        typer.Option(
            "--fp32/--bf16", help="Use FP32 precision", envvar="PARAKEET_FP32"
        ),
    ] = False,
    local_attention: Annotated[
        bool,
        typer.Option(
            "--local-attention/--full-attention",
            help="Use local attention (reduces intermediate memory usage for long audio)",
            envvar="PARAKEET_LOCAL_ATTENTION",
        ),
    ] = False,
    local_attention_context_size: Annotated[
        int,
        typer.Option(
            "--local-attention-context-size",
            help="Local attention context size (Only applies if using local attention)",
            envvar="PARAKEET_LOCAL_ATTENTION_CTX",
        ),
    ] = 256,
    cache_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Directory for HuggingFace model cache. If not specified, uses HF's default cache location",
            envvar="PARAKEET_CACHE_DIR",
        ),
    ] = None,
):
    """
    Transcribe audio files using Parakeet MLX models.
    """
    if verbose:
        print(f"Loading model: [bold cyan]{model}[/bold cyan]...")

    try:
        loaded_model = from_pretrained(
            model, dtype=bfloat16 if not fp32 else float32, cache_dir=cache_dir
        )

        if local_attention:
            loaded_model.encoder.set_attention_model(
                "rel_pos_local_attn",
                (local_attention_context_size, local_attention_context_size),
            )
        if verbose:
            print("[green]Model loaded successfully.[/green]")
    except Exception as e:
        print(f"[bold red]Error loading model {model}:[/bold red] {e}")
        raise typer.Exit(code=1)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[bold red]Error creating output directory {output_dir}:[/bold red] {e}")
        raise typer.Exit(code=1)

    if verbose:
        print(f"Output directory: [bold cyan]{output_dir.resolve()}[/bold cyan]")
        print(f"Output format(s): [bold cyan]{output_format}[/bold cyan]")
        if output_format in ["srt", "vtt", "all"] and highlight_words:
            print("Highlight words: [bold cyan]Enabled[/bold cyan]")

    formatters = {
        "txt": to_txt,
        "srt": lambda r: to_srt(r, highlight_words=highlight_words),
        "vtt": lambda r: to_vtt(r, highlight_words=highlight_words),
        "json": to_json,
    }

    formats_to_generate = []
    if output_format == "all":
        formats_to_generate = list(formatters.keys())
    elif output_format in formatters:
        formats_to_generate = [output_format]
    else:
        print(
            f"[bold red]Error: Invalid output format '{output_format}'. Choose from {list(formatters.keys()) + ['all']}.[/bold red]"
        )
        raise typer.Exit(code=1)

    total_files = len(audios)
    if verbose:
        print(f"Transcribing {total_files} file(s)...")

    decoding_config = DecodingConfig(
        decoding=Beam(
            beam_size=beam_size,
            length_penalty=length_penalty,
            patience=patience,
            duration_reward=duration_reward,
        )
        if decoding == "beam"
        else Greedy(),
        sentence=SentenceConfig(
            max_words=max_words, silence_gap=silence_gap, max_duration=max_duration
        ),
    )
    if verbose:
        print("Decoding config being used:", decoding_config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Transcribing...", total=total_files)

        for i, audio_path in enumerate(audios):
            if verbose:
                print(
                    f"\nProcessing file {i + 1}/{total_files}: [bold cyan]{audio_path.name}[/bold cyan]"
                )
            else:
                progress.update(
                    task, description=f"Processing [cyan]{audio_path.name}[/cyan]..."
                )

            try:
                result: AlignedResult = loaded_model.transcribe(
                    audio_path,
                    dtype=bfloat16 if not fp32 else float32,
                    chunk_duration=chunk_duration if chunk_duration != 0 else None,
                    overlap_duration=overlap_duration,
                    chunk_callback=lambda current, full: progress.update(
                        task, total=total_files * full, completed=full * i + current
                    ),
                    decoding_config=decoding_config,
                )

                if verbose:
                    for sentence in result.sentences:
                        start, end, text, conf = (
                            sentence.start,
                            sentence.end,
                            sentence.text,
                            sentence.confidence,
                        )
                        conf_str = f" [dim](confidence: {conf:.2%})[/dim]"
                        line = f"[blue][{format_timestamp(start)} --> {format_timestamp(end)}][/blue]{conf_str} {text.strip()}"
                        print(line)

                base_filename = audio_path.stem
                template_vars = {
                    "filename": base_filename,
                    "parent": str(audio_path.parent),
                    "date": datetime.datetime.now().strftime("%Y%m%d"),
                    "index": str(i + 1),
                }

                output_basename = output_template.format(**template_vars)

                for fmt in formats_to_generate:
                    formatter = formatters[fmt]
                    output_content = formatter(result)
                    output_filename = Path(f"{output_basename}.{fmt}")
                    if output_filename.is_absolute():
                        output_filepath = output_filename
                    else:
                        output_filepath = output_dir / output_filename

                    try:
                        with open(output_filepath, "w", encoding="utf-8") as f:
                            f.write(output_content)
                        if verbose:
                            print(
                                f"[green]Saved {fmt.upper()}:[/green] {output_filepath.absolute()}"
                            )
                    except Exception as e:
                        print(
                            f"[bold red]Error writing output file {output_filepath}:[/bold red] {e}"
                        )

            except Exception as e:
                print(f"[bold red]Error transcribing file {audio_path}:[/bold red] {e}")

            progress.update(task, total=total_files, completed=i + 1)

    print(
        f"\n[bold green]{model.removeprefix('mlx-community/')} transcription complete.[/bold green] Outputs saved in '{output_dir.resolve()}'."
    )


if __name__ == "__main__":
    app()
