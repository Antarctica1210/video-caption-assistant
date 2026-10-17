import shutil
from pathlib import Path

import static_ffmpeg
import typer

from src.video_caption.config import load_config
from src.video_caption.logger import get_logger, setup_logging
from src.video_caption.paths import to_display_path, to_local_path, work_dir_for
from src.video_caption.pipeline import PipelineError, run_pipeline

setup_logging()
log = get_logger("video_caption.main")

static_ffmpeg.add_paths()  # makes bundled ffmpeg binary available on PATH

app = typer.Typer(help="Video caption assistant — transcribe, translate, and export captions.")


@app.command()
def serve(
    host: str | None = typer.Option(None, "--host", help="Bind address (default from config.toml [web])"),
    port: int | None = typer.Option(None, "--port", "-p", help="Port (default from config.toml [web])"),
    config_path: str = typer.Option("config.toml", "--config", "-c", help="Path to config.toml"),
):
    """Start the web page for picking a video and generating captions."""
    import uvicorn

    from src.video_caption.web.server import create_app

    cfg = load_config(config_path)
    host = host or cfg.web.host
    port = port or cfg.web.port
    typer.echo(f"Open http://localhost:{port} in your browser (Ctrl+C to stop)")
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="warning")


@app.command()
def run(
    video: str = typer.Argument(..., help="Path to the video. Windows paths (C:\\...) work on WSL."),
    lang: str = typer.Option("zh", "--lang", "-l", help="Target language code (e.g. zh, en, es, fr, ja)"),
    source: str | None = typer.Option(None, "--source", "-s", help="Source audio language hint for Whisper (e.g. ja, en, zh)"),
    fast_mode: bool = typer.Option(False, "--fast-mode", help="Use fast_model_size (whisper-large-v3-turbo) for quicker transcription"),
    fmt: str = typer.Option("both", "--format", "-f", help="Output format: srt | ass | both"),
    title: str | None = typer.Option(None, "--title", "-t", help="Video title to translate and prepend"),
    config_path: str = typer.Option("config.toml", "--config", "-c", help="Path to config.toml"),
):
    """Generate captions for one video. Captions are written next to the video."""
    cfg = load_config(config_path)
    video_path = to_local_path(video)
    typer.echo(f"Processing {to_display_path(video_path)} → {lang} ({fmt})")

    try:
        outputs = run_pipeline(
            cfg, video_path, lang=lang, source=source, fast_mode=fast_mode, fmt=fmt, title=title,
            on_phase=lambda phase: typer.echo("Phase 1: Transcribing..." if phase == "transcribing"
                                              else "Phase 2: Translating and exporting..."),
        )
    except PipelineError as e:
        log.error("%s", e)
        typer.echo(f"[error] {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo("Done. Captions written:")
    for path in outputs:
        typer.echo(f"  {to_display_path(Path(path))}")


@app.command()
def delete(
    video: str = typer.Argument(..., help="Path to the video whose temp files should be removed"),
    config_path: str = typer.Option("config.toml", "--config", "-c", help="Path to config.toml"),
):
    """Delete a video's temp folder (audio, transcript, checkpoints). Captions next to the video are kept."""
    cfg = load_config(config_path)
    target = work_dir_for(to_local_path(video), cfg.temp_dir)
    if not target.exists():
        typer.echo(f"[error] No temp folder found for '{video}' at {target}", err=True)
        raise typer.Exit(code=1)
    shutil.rmtree(target)
    typer.echo(f"Deleted {target}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
