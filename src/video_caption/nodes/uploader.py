from pathlib import Path

from ..clients.minio_client import MinIOClient
from ..config import AppConfig
from ..logger import get_logger
from ..state import CaptionState

log = get_logger("video_caption.uploader")


def upload_outputs(state: CaptionState, config: AppConfig, minio: MinIOClient) -> dict:
    stem = Path(state["local_video_path"]).stem
    bucket = config.minio.output_bucket
    keys: list[str] = []

    for field in ("srt_path", "ass_path", "transcript_json_path", "transcript_csv_path"):
        path_str = state.get(field)  # type: ignore[literal-required]
        if path_str:
            src = Path(path_str)
            key = f"{stem}/{src.name}"
            log.info("Uploading %s → %s/%s", src.name, bucket, key)
            minio.upload(bucket, key, src)
            keys.append(key)

    log.info("Uploaded %d file(s) to bucket '%s'", len(keys), bucket)
    return {"output_keys": keys}
