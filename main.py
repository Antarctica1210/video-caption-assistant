import os
from pathlib import Path

import static_ffmpeg
import typer

from src.video_caption.clients.minio_client import MinIOClient
from src.video_caption.config import AppConfig, load_config
from src.video_caption.graph import build_transcription_graph, build_translation_graph
from src.video_caption.state import CaptionState
from src.video_caption.logger import get_logger, setup_logging

setup_logging()
log = get_logger("video_caption.main")

static_ffmpeg.add_paths()  # makes bundled ffmpeg binary available on PATH

_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".m4v", ".ts"}

app = typer.Typer(help="Video caption assistant — transcribe, translate, and export captions.")


@app.command()
def run(
    lang: str = typer.Option("zh", "--lang", "-l", help="Target language code (e.g. zh, en, es, fr, ja)"),
    source: str | None = typer.Option(None, "--source", "-s", help="Source audio language hint for Whisper (e.g. ja, en, zh)"),
    fast_mode: bool = typer.Option(False, "--fast-mode", help="Use fast_model_size (whisper-large-v3-turbo) for quicker transcription"),
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
    typer.echo(f"Processing {video_key!r} → {lang} ({fmt})")

    # --- Phase 1: Transcription ---
    log.info("Phase 1 — Transcription: video=%r", video_key)
    typer.echo("Phase 1: Transcribing...")
    t_result = build_transcription_graph(cfg).invoke(CaptionState(
        video_key=video_key,
        source_lang=source,
        fast_mode=fast_mode,
        audio_chunks=[],
        raw_segments=[],
        output_keys=[],
    ))
    if t_result.get("error"):
        log.error("Transcription failed: %s", t_result["error"])
        typer.echo(f"[error] {t_result['error']}", err=True)
        raise typer.Exit(code=1)

    # --- Phase 2: Translation & Export ---
    log.info("Phase 2 — Translation & Export: lang=%s format=%s", lang, fmt)
    typer.echo("Phase 2: Translating and exporting...")
    tr_result = build_translation_graph(cfg).invoke(CaptionState(
        video_key=video_key,
        target_lang=lang,
        output_format=fmt,
        title=title,
        local_video_path=t_result["local_video_path"],
        transcript_jsonl_path=t_result["transcript_jsonl_path"],
        bilingual_segments=[],
        output_keys=[],
    ))
    if tr_result.get("error"):
        log.error("Translation failed: %s", tr_result["error"])
        typer.echo(f"[error] {tr_result['error']}", err=True)
        raise typer.Exit(code=1)

    log.info("Pipeline complete — %d output file(s)", len(tr_result.get("output_keys", [])))
    typer.echo("Done. Files uploaded to MinIO video-output:")
    for key in tr_result.get("output_keys", []):
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


@app.command()
def delete(
    name: str = typer.Argument(..., help="Video stem to delete from tmp (e.g. 'my_video')"),
    config_path: str = typer.Option("config.toml", "--config", "-c", help="Path to config.toml"),
):
    """Delete a video's tmp folder and all its contents (input, audio, output)."""
    import shutil
    cfg = load_config(config_path)
    target = Path(cfg.temp_dir) / name
    if not target.exists():
        typer.echo(f"[error] No tmp folder found for '{name}' at {target}", err=True)
        raise typer.Exit(code=1)
    shutil.rmtree(target)
    typer.echo(f"Deleted {target}")


def _check_lm_studio(base_url: str) -> bool:
    import httpx
    url = f"{base_url.rstrip('/')}/models"
    log.debug("Checking LM Studio at %s", url)
    try:
        headers = {"Authorization": f"Bearer {os.getenv('LM_STUDIO_API_KEY')}"}
        r = httpx.get(url, timeout=5, headers=headers)
        log.debug("LM Studio response: HTTP %d", r.status_code)
        if r.status_code != 200:
            log.warning("LM Studio returned unexpected status HTTP %d — body: %s", r.status_code, r.text[:200])
        return r.status_code == 200
    except httpx.ConnectError as e:
        log.error("LM Studio connection refused at %s — is the server running? (%s)", url, e)
    except httpx.TimeoutException:
        log.error("LM Studio health check timed out at %s", url)
    except Exception as e:
        log.error("LM Studio health check failed: %s: %s", type(e).__name__, e)
    return False



def main() -> None:
    app()


if __name__ == "__main__":
    main()
