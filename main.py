import typer

from src.video_caption.config import load_config
from src.video_caption.graph import build_graph

app = typer.Typer(help="Video caption assistant — transcribe, translate, and export captions.")


@app.command()
def run(
    video: str = typer.Argument(..., help="MinIO object key in the video_input bucket (e.g. my_video.mp4)"),
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

    graph = build_graph(cfg)

    initial: dict = {
        "video_key": video,
        "target_lang": lang,
        "output_format": fmt,
        "title": title,
        "audio_chunks": [],
        "raw_segments": [],
        "bilingual_segments": [],
        "output_keys": [],
    }

    typer.echo(f"Processing {video!r} → {lang} ({fmt})")
    result = graph.invoke(initial)

    if result.get("error"):
        typer.echo(f"[error] {result['error']}", err=True)
        raise typer.Exit(code=1)

    typer.echo("Done. Files uploaded to MinIO video-output:")
    for key in result.get("output_keys", []):
        typer.echo(f"  {key}")


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
