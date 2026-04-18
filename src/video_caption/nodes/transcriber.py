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
    transcribe_kwargs: dict = {
        "word_timestamps": True,
        "beam_size": app_config.whisper.beam_size,
        "no_speech_threshold": app_config.whisper.no_speech_threshold,
        "log_prob_threshold": app_config.whisper.log_prob_threshold,
        "condition_on_previous_text": app_config.whisper.condition_on_previous_text,
    }
    if source_lang:
        transcribe_kwargs["language"] = source_lang
        log.info("Source language hint: %s", source_lang)
    log.info(
        "Transcribe settings: beam_size=%d no_speech_threshold=%.2f log_prob_threshold=%.2f condition_on_previous_text=%s",
        app_config.whisper.beam_size,
        app_config.whisper.no_speech_threshold,
        app_config.whisper.log_prob_threshold,
        app_config.whisper.condition_on_previous_text,
    )
    raw_segments, info = model.transcribe(audio_path, **transcribe_kwargs)
    log.info("Detected language: %s (%.0f%% confidence)", info.language, info.language_probability * 100)

    # Consume the generator eagerly with explicit error handling — a silent exception
    # inside a lazy generator would truncate results without any visible error.
    segments: list[Segment] = []
    i = 0
    for s in raw_segments:
        try:
            if not s.text.strip():
                continue
            segments.append({
                "id": i,
                "text": s.text.strip(),
                "start": round(s.start, 3),
                "end": round(s.end, 3),
            })
            i += 1
        except Exception as e:
            log.warning("Skipping segment at ~%.1fs due to error: %s", getattr(s, "start", -1), e)

    if segments:
        last_end = segments[-1]["end"]
        audio_duration = info.duration or 0
        gap = audio_duration - last_end
        if gap > 30:
            log.warning(
                "Last segment ends at %.1fs but audio is %.1fs — %.0fs of audio may be missing from transcription",
                last_end, audio_duration, gap,
            )

    log.info("Transcription complete — %d segment(s)", len(segments))
    return {"raw_segments": segments}
