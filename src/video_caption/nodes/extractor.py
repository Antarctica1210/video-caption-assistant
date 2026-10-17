from pathlib import Path

import ffmpeg

from ..logger import get_logger
from ..state import CaptionState

log = get_logger("video_caption.extractor")


def extract_audio(state: CaptionState) -> dict:
    video_path = Path(state["video_path"])
    local_audio = Path(state["work_dir"]) / "audio" / f"{video_path.stem}.wav"
    local_audio.parent.mkdir(parents=True, exist_ok=True)

    log.info("Extracting audio from %s → %s", video_path, local_audio)
    (
        ffmpeg
        .input(str(video_path))
        .output(str(local_audio), ac=1, ar=16000, format="wav")
        .overwrite_output()
        .run(quiet=True)
    )
    log.info("Audio extraction complete")

    return {"local_audio_path": str(local_audio)}
