from pathlib import Path

import ffmpeg

from ..config import AppConfig
from ..logger import get_logger
from ..state import AudioChunk, CaptionState

log = get_logger("video_caption.chunker")


def chunk_audio(state: CaptionState, app_config: AppConfig) -> dict:
    audio_path = Path(state["local_audio_path"])
    chunk_dir = audio_path.parent / "chunks"
    chunk_dir.mkdir(exist_ok=True)

    duration = _get_duration(audio_path)
    chunk_size = float(app_config.chunk_duration)
    overlap = float(app_config.chunk_overlap)
    if chunk_size <= 0:
        raise ValueError(f"chunk_duration must be > 0, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"chunk_overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({overlap}) must be smaller than chunk_duration ({chunk_size})"
        )

    log.info("Audio duration: %.1fs — splitting into %ds chunks (overlap %ds)", duration, chunk_size, overlap)

    chunks: list[AudioChunk] = []
    start = 0.0
    idx = 0

    while start < duration:
        end = min(start + chunk_size, duration)
        if end <= start:
            log.warning("Stopping chunking because end (%.3f) <= start (%.3f)", end, start)
            break

        chunk_path = chunk_dir / f"chunk_{idx:04d}.wav"

        (
            ffmpeg
            .input(str(audio_path), ss=start, to=end)
            .output(str(chunk_path), ac=1, ar=16000)
            .overwrite_output()
            .run(quiet=True)
        )

        log.debug("Chunk %04d: %.1fs → %.1fs", idx, start, end)
        chunks.append({"path": str(chunk_path), "start": start, "end": end, "index": idx})
        if end >= duration:
            break

        next_start = max(0.0, end - overlap)
        if next_start <= start:
            log.warning(
                "Stopping chunking because next_start (%.3f) did not advance past start (%.3f)",
                next_start,
                start,
            )
            break
        start = next_start
        idx += 1

    log.info("Created %d chunk(s)", len(chunks))
    return {"audio_chunks": chunks}


def _get_duration(path: Path) -> float:
    probe = ffmpeg.probe(str(path))
    return float(probe["format"]["duration"])
