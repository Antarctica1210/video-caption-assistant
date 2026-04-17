import typer

from src.video_caption.clients.minio_client import MinIOClient
from src.video_caption.config import AppConfig, load_config
from src.video_caption.graph import build_graph, CaptionState
from src.video_caption.logger import get_logger, setup_logging

setup_logging()
log = get_logger("video_caption.main")

_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".m4v"}

app = typer.Typer(help="Video caption assistant — transcribe, translate, and export captions.")


@app.command()
def run(
    lang: str = typer.Option("zh", "--lang", "-l", help="Target language code (e.g. zh, en, es, fr, ja)"),
    fmt: str = typer.Option("both", "--format", "-f", help="Output format: srt | ass | both"),
    title: str | None = typer.Option(None, "--title", "-t", help="Video title to translate and prepend"),
    config_path: str = typer.Option("config.toml", "--config", "-c", help="Path to config.toml"),
):
    cfg = load_config(config_path)

    lm_ok = _check_lm_studio(cfg.lm_studio.base_url)
    if not lm_ok:
        typer.echo(f"[error] Cannot reach LM Studio at {cfg.lm_studio.base_url}", err=True)
        raise typer.Exit(code=1)

    minio = MinIOClient(
        endpoint=cfg.minio.endpoint,
        access_key=cfg.minio.access_key,
        secret_key=cfg.minio.secret_key,
        secure=cfg.minio.secure,
    )
    video_key = _detect_video(minio, cfg)

    graph = build_graph(cfg)

    initial_state = CaptionState(
        video_key=video_key,
        target_lang=lang,
        output_format=fmt,
        title=title,
        audio_chunks=[],
        raw_segments=[],
        bilingual_segments=[],
        output_keys=[],
    )

    log.info("Starting pipeline: video=%r lang=%s format=%s", video_key, lang, fmt)
    typer.echo(f"Processing {video_key!r} → {lang} ({fmt})")
    result = graph.invoke(initial_state)

    if result.get("error"):
        log.error("Pipeline failed: %s", result["error"])
        typer.echo(f"[error] {result['error']}", err=True)
        raise typer.Exit(code=1)

    log.info("Pipeline complete — %d output file(s)", len(result.get("output_keys", [])))
    typer.echo("Done. Files uploaded to MinIO video-output:")
    for key in result.get("output_keys", []):
        typer.echo(f"  {key}")


def _detect_video(minio: MinIOClient, cfg: AppConfig) -> str:
    minio.ensure_bucket(cfg.minio.input_bucket)
    all_keys = minio.list_objects(cfg.minio.input_bucket)
    videos = [k for k in all_keys if any(k.lower().endswith(ext) for ext in _VIDEO_EXTENSIONS)]

    if not videos:
        typer.echo(
            f"[error] No video found in '{cfg.minio.input_bucket}' bucket. "
            f"Upload a video file ({', '.join(_VIDEO_EXTENSIONS)}) and try again.",
            err=True,
        )
        raise typer.Exit(code=1)

    if len(videos) > 1:
        typer.echo(
            f"[error] Multiple videos found in '{cfg.minio.input_bucket}' bucket:\n"
            + "\n".join(f"  {v}" for v in videos)
            + "\nKeep only one video and try again.",
            err=True,
        )
        raise typer.Exit(code=1)

    return videos[0]


def _check_lm_studio(base_url: str) -> bool:
    import httpx
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/models", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def main() -> None:
    app()


if __name__ == "__main__":
    main()
