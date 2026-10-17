from collections.abc import Callable
from pathlib import Path

import httpx

from .config import AppConfig
from .graph import build_transcription_graph, build_translation_graph
from .logger import get_logger
from .paths import is_video, work_dir_for
from .state import CaptionState

log = get_logger("video_caption.pipeline")


class PipelineError(Exception):
    pass


def validate_video(video_path: Path) -> None:
    if not video_path.is_file():
        raise PipelineError(f"Video not found: {video_path}")
    if not is_video(video_path):
        raise PipelineError(f"Not a supported video file: {video_path.name}")


def run_pipeline(
    cfg: AppConfig,
    video_path: Path,
    lang: str,
    source: str | None = None,
    fast_mode: bool = False,
    fmt: str = "both",
    title: str | None = None,
    on_phase: Callable[[str], None] = lambda _: None,
) -> list[str]:
    """Transcribe + translate one local video. Returns the caption file paths written next to it."""
    video_path = video_path.resolve()
    validate_video(video_path)
    if not check_lm_studio(cfg):
        raise PipelineError(f"Cannot reach LM Studio at {cfg.lm_studio.base_url}")

    work_dir = str(work_dir_for(video_path, cfg.temp_dir))
    log.info("Video: %s | work dir: %s", video_path, work_dir)

    # --- Phase 1: Transcription ---
    on_phase("transcribing")
    log.info("Phase 1 — Transcription: video=%s", video_path)
    t_result = build_transcription_graph(cfg).invoke(CaptionState(
        video_path=str(video_path),
        work_dir=work_dir,
        source_lang=source,
        fast_mode=fast_mode,
        audio_chunks=[],
        raw_segments=[],
    ))
    if t_result.get("error"):
        raise PipelineError(f"Transcription failed: {t_result['error']}")

    # --- Phase 2: Translation & Export ---
    on_phase("translating")
    log.info("Phase 2 — Translation & Export: lang=%s format=%s", lang, fmt)
    tr_result = build_translation_graph(cfg).invoke(CaptionState(
        video_path=str(video_path),
        work_dir=work_dir,
        target_lang=lang,
        output_format=fmt,
        title=title,
        transcript_jsonl_path=t_result["transcript_jsonl_path"],
        bilingual_segments=[],
    ))
    if tr_result.get("error"):
        raise PipelineError(f"Translation failed: {tr_result['error']}")

    outputs = [p for p in (tr_result.get("srt_path"), tr_result.get("ass_path")) if p]
    log.info("Pipeline complete — %d caption file(s) written next to the video", len(outputs))
    return outputs


def check_lm_studio(cfg: AppConfig) -> bool:
    url = f"{cfg.lm_studio.base_url.rstrip('/')}/models"
    log.debug("Checking LM Studio at %s", url)
    try:
        headers = {"Authorization": f"Bearer {cfg.lm_studio.api_key}"}
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
