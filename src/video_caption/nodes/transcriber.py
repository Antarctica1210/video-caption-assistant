from concurrent.futures import ThreadPoolExecutor, as_completed

from faster_whisper import WhisperModel

from ..config import AppConfig
from ..state import AudioChunk, CaptionState, Segment

_model: WhisperModel | None = None


def _get_model(config: AppConfig) -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(
            config.whisper.model_size,
            device=config.whisper.device,
            compute_type=config.whisper.compute_type,
        )
    return _model


def transcribe_chunks(state: CaptionState, config: AppConfig) -> dict:
    model = _get_model(config)
    chunks = state["audio_chunks"]

    results: dict[int, list[Segment]] = {}

    def _transcribe_one(chunk: AudioChunk) -> tuple[int, list[Segment]]:
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
        return chunk["index"], segs

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(_transcribe_one, c): c for c in chunks}
        for future in as_completed(futures):
            idx, segs = future.result()
            results[idx] = segs

    ordered = [seg for idx in sorted(results) for seg in results[idx]]
    return {"raw_segments": ordered}
