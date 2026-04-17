from pathlib import Path

import ffmpeg

from ..clients.minio_client import MinIOClient
from ..config import AppConfig
from ..logger import get_logger
from ..state import CaptionState

log = get_logger("video_caption.extractor")


def fetch_and_extract(state: CaptionState, config: AppConfig, minio: MinIOClient) -> dict:
    temp_dir = Path(config.temp_dir)
    video_key = state["video_key"]
    stem = Path(video_key).stem

    local_video = temp_dir / "input" / Path(video_key).name
    local_audio = temp_dir / "audio" / f"{stem}.wav"
    local_audio.parent.mkdir(parents=True, exist_ok=True)

    log.info("Downloading '%s' from bucket '%s'", video_key, config.minio.input_bucket)
    minio.download(config.minio.input_bucket, video_key, local_video)
    log.info("Downloaded to %s", local_video)

    log.info("Extracting audio → %s", local_audio)
    (
        ffmpeg
        .input(str(local_video))
        .output(str(local_audio), ac=1, ar=16000, format="wav")
        .overwrite_output()
        .run(quiet=True)
    )
    log.info("Audio extraction complete")

    return {"local_video_path": str(local_video), "local_audio_path": str(local_audio)}
