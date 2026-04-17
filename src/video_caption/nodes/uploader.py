from pathlib import Path

from ..clients.minio_client import MinIOClient
from ..config import AppConfig
from ..state import CaptionState


def upload_outputs(state: CaptionState, config: AppConfig, minio: MinIOClient) -> dict:
    stem = Path(state["local_video_path"]).stem
    bucket = config.minio.output_bucket
    keys: list[str] = []

    for field in ("srt_path", "ass_path", "transcript_json_path", "transcript_csv_path"):
        path_str = state.get(field)  # type: ignore[literal-required]
        if path_str:
            src = Path(path_str)
            key = f"{stem}/{src.name}"
            minio.upload(bucket, key, src)
            keys.append(key)

    return {"output_keys": keys}
