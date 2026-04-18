from pathlib import Path

from faster_whisper import WhisperModel

from ..config import AppConfig
from ..logger import get_logger
from ..state import AudioChunk, CaptionState, Segment

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
    chunks = state["audio_chunks"]
    total = len(chunks)
    log.info("Transcribing %d chunk(s) sequentially", total)

    # Sequential — CTranslate2 is not thread-safe; the GPU runs one session at a
    # time anyway so parallelism here gives no throughput benefit and risks
    # corrupting internal model state when chunks share the same model instance.
    ordered: list[Segment] = []
    for chunk in sorted(chunks, key=lambda c: c["index"]):
        log.debug("Transcribing chunk %04d/%04d (%.1fs–%.1fs)", chunk["index"] + 1, total, chunk["start"], chunk["end"])
        segments, _ = model.transcribe(chunk["path"], word_timestamps=True)
        segs: list[Segment] = [
            {
                "text": s.text.strip(),
                "start": round(s.start + chunk["start"], 3),
                "end": round(s.end + chunk["start"], 3),
            }
            for s in segments
            if s.text.strip()
        ]
        log.debug("Chunk %04d → %d segment(s)", chunk["index"] + 1, len(segs))
        ordered.extend(segs)

    log.info("Transcription complete — %d segment(s) total", len(ordered))
    return {"raw_segments": ordered}
