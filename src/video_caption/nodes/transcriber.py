from pathlib import Path

from faster_whisper import WhisperModel

from ..config import AppConfig
from ..logger import get_logger
from ..state import CaptionState, Segment

log = get_logger("video_caption.transcriber")

_model: WhisperModel | None = None


def _get_model(app_config: AppConfig) -> WhisperModel:
    global _model
    if _model is None:
        log.info(
            "Loading faster-whisper model '%s' on %s (%s), download_root=%s",
            app_config.whisper.model_size,
            app_config.whisper.device,
            app_config.whisper.compute_type,
            app_config.whisper.download_root,
        )
        download_root = Path(app_config.whisper.download_root)
        download_root.mkdir(parents=True, exist_ok=True)
        _model = WhisperModel(
            app_config.whisper.model_size,
            device=app_config.whisper.device,
            compute_type=app_config.whisper.compute_type,
            download_root=str(download_root),
        )
        log.info("Model loaded")
    return _model


def transcribe_chunks(state: CaptionState, app_config: AppConfig) -> dict:
    model = _get_model(app_config)
    audio_path = state["local_audio_path"]
    log.info("Transcribing full audio: %s", audio_path)

    # Transcribe the full WAV in one pass — no chunking, no overlap, no duplicate
    # segments at boundaries. faster-whisper streams internally so VRAM usage is
    # bounded regardless of audio length.
    source_lang = state.get("source_lang") or None
    transcribe_kwargs: dict = {"word_timestamps": True}
    if source_lang:
        transcribe_kwargs["language"] = source_lang
        log.info("Source language hint: %s", source_lang)
    raw_segments, info = model.transcribe(audio_path, **transcribe_kwargs)
    log.info("Detected language: %s (%.0f%% confidence)", info.language, info.language_probability * 100)

    segments: list[Segment] = [
        {
            "id": i,
            "text": s.text.strip(),
            "start": round(s.start, 3),
            "end": round(s.end, 3),
        }
        for i, s in enumerate(s for s in raw_segments if s.text.strip())
    ]

    log.info("Transcription complete — %d segment(s)", len(segments))
    return {"raw_segments": segments}
