import json
from pathlib import Path

from faster_whisper import WhisperModel

from ..config import AppConfig
from ..logger import get_logger
from ..state import CaptionState, Segment

log = get_logger("video_caption.transcriber")

_models: dict[str, WhisperModel] = {}


def _get_model(app_config: AppConfig, model_size: str) -> WhisperModel:
    if model_size not in _models:
        log.info(
            "Loading faster-whisper model '%s' on %s (%s), download_root=%s",
            model_size,
            app_config.whisper.device,
            app_config.whisper.compute_type,
            app_config.whisper.download_root,
        )
        download_root = Path(app_config.whisper.download_root)
        download_root.mkdir(parents=True, exist_ok=True)
        _models[model_size] = WhisperModel(
            model_size,
            device=app_config.whisper.device,
            compute_type=app_config.whisper.compute_type,
            download_root=str(download_root),
        )
        log.info("Model loaded: %s", model_size)
    return _models[model_size]


def transcribe_chunks(state: CaptionState, app_config: AppConfig) -> dict:
    fast_mode = state.get("fast_mode", False)
    model_size = app_config.whisper.fast_model_size if fast_mode else app_config.whisper.model_size
    model = _get_model(app_config, model_size)

    audio_path = state["local_audio_path"]
    log.info("Transcribing full audio: %s (model=%s)", audio_path, model_size)

    source_lang = state.get("source_lang") or None
    transcribe_kwargs: dict = {
        "word_timestamps": True,
        "beam_size": app_config.whisper.beam_size,
        "no_speech_threshold": app_config.whisper.no_speech_threshold,
        "log_prob_threshold": app_config.whisper.log_prob_threshold,
        "condition_on_previous_text": app_config.whisper.condition_on_previous_text,
        "vad_filter": app_config.whisper.vad_filter,
    }
    if source_lang:
        transcribe_kwargs["language"] = source_lang
        log.info("Source language hint: %s", source_lang)
    log.info(
        "Transcribe settings: beam_size=%d no_speech_threshold=%.2f log_prob_threshold=%.2f "
        "condition_on_previous_text=%s vad_filter=%s",
        app_config.whisper.beam_size,
        app_config.whisper.no_speech_threshold,
        app_config.whisper.log_prob_threshold,
        app_config.whisper.condition_on_previous_text,
        app_config.whisper.vad_filter,
    )

    stem = Path(state["local_video_path"]).stem
    out_dir = Path(app_config.temp_dir) / stem / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "transcript.jsonl"

    raw_segments, info = model.transcribe(audio_path, **transcribe_kwargs)
    log.info("Detected language: %s (%.0f%% confidence)", info.language, info.language_probability * 100)

    segments: list[Segment] = []
    i = 0
    with jsonl_path.open("w", encoding="utf-8") as f:
        for s in raw_segments:
            try:
                text = s.text.strip()
                if not text:
                    continue
                seg: Segment = {
                    "id": i,
                    "text": text,
                    "start": round(s.start, 3),
                    "end": round(s.end, 3),
                }
                f.write(json.dumps(seg, ensure_ascii=False) + "\n")
                f.flush()
                segments.append(seg)
                i += 1
            except Exception as e:
                log.warning("Skipping segment at ~%.1fs due to error: %s", getattr(s, "start", -1), e)

    audio_duration = info.duration or 0
    _check_coverage(segments, audio_duration)

    log.info("Transcription complete — %d segment(s) → %s", len(segments), jsonl_path)
    return {"raw_segments": segments, "transcript_jsonl_path": str(jsonl_path)}


def _check_coverage(segments: list[Segment], audio_duration: float) -> None:
    """Warn for every 1-minute window that has no transcribed speech."""
    if not audio_duration or not segments:
        return

    covered = {int(seg["start"] // 60) for seg in segments}
    total_minutes = int(audio_duration // 60)
    empty = [m for m in range(total_minutes + 1) if m not in covered]

    if empty:
        ranges = ", ".join(f"{m}:00–{m + 1}:00" for m in empty[:20])
        log.warning(
            "%d minute-window(s) with no speech detected: %s%s",
            len(empty),
            ranges,
            " ..." if len(empty) > 20 else "",
        )
    else:
        log.info("Coverage check OK — all %d minute-window(s) have speech", total_minutes + 1)
