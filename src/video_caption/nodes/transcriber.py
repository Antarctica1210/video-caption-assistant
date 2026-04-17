from concurrent.futures import ThreadPoolExecutor, as_completed

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
            "Loading faster-whisper model '%s' on %s (%s)",
            app_config.whisper.model_size,
            app_config.whisper.device,
            app_config.whisper.compute_type,
        )
        _model = WhisperModel(
            app_config.whisper.model_size,
            device=app_config.whisper.device,
            compute_type=app_config.whisper.compute_type,
        )
        log.info("Model loaded")
    return _model


def transcribe_chunks(state: CaptionState, app_config: AppConfig) -> dict:
    model = _get_model(app_config)
    chunks = state["audio_chunks"]
    log.info("Transcribing %d chunk(s) in parallel", len(chunks))

    results: dict[int, list[Segment]] = {}

    def _transcribe_one(chunk: AudioChunk) -> tuple[int, list[Segment]]:
        log.debug("Transcribing chunk %04d (%.1fs–%.1fs)", chunk["index"], chunk["start"], chunk["end"])
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
        log.debug("Chunk %04d → %d segment(s)", chunk["index"], len(segs))
        return chunk["index"], segs

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(_transcribe_one, c): c for c in chunks}
        for future in as_completed(futures):
            idx, segs = future.result()
            results[idx] = segs

    ordered = [seg for idx in sorted(results) for seg in results[idx]]
    log.info("Transcription complete — %d segment(s) total", len(ordered))
    return {"raw_segments": ordered}
