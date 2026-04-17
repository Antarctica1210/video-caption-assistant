from pathlib import Path

import ffmpeg

from ..config import AppConfig
from ..state import AudioChunk, CaptionState


def chunk_audio(state: CaptionState, config: AppConfig) -> dict:
    audio_path = Path(state["local_audio_path"])
    chunk_dir = audio_path.parent / "chunks"
    chunk_dir.mkdir(exist_ok=True)

    duration = _get_duration(audio_path)
    chunk_size = float(config.chunk_duration)
    overlap = float(config.chunk_overlap)

    chunks: list[AudioChunk] = []
    start = 0.0
    idx = 0

    while start < duration:
        end = min(start + chunk_size, duration)
        chunk_path = chunk_dir / f"chunk_{idx:04d}.wav"

        (
            ffmpeg
            .input(str(audio_path), ss=start, to=end)
            .output(str(chunk_path), ac=1, ar=16000)
            .overwrite_output()
            .run(quiet=True)
        )

        chunks.append({"path": str(chunk_path), "start": start, "end": end, "index": idx})
        start = end - overlap
        idx += 1

    return {"audio_chunks": chunks}


def _get_duration(path: Path) -> float:
    probe = ffmpeg.probe(str(path))
    return float(probe["format"]["duration"])
