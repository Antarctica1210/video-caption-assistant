import json
from pathlib import Path

from ..config import AppConfig
from ..logger import get_logger
from ..state import CaptionState, Segment

log = get_logger("video_caption.cache_check")


def check_cache(state: CaptionState, app_config: AppConfig) -> dict:
    stem = Path(state["video_key"]).stem
    out_dir = Path(app_config.temp_dir) / "output" / stem
    json_path = out_dir / "transcript.json"
    csv_path = out_dir / "transcript.csv"

    if json_path.exists() and csv_path.exists():
        log.info("Cache hit — loading existing transcript from %s", out_dir)
        segments: list[Segment] = json.loads(json_path.read_text(encoding="utf-8"))
        log.info("Loaded %d segment(s) from cache — skipping extraction and transcription", len(segments))
        return {
            "cache_hit": True,
            "raw_segments": segments,
            "transcript_json_path": str(json_path),
            "transcript_csv_path": str(csv_path),
            # set a synthetic local_video_path so downstream nodes can derive the stem
            "local_video_path": str(Path(app_config.temp_dir) / "input" / Path(state["video_key"]).name),
        }

    log.info("No cache found at %s — running full pipeline", out_dir)
    return {"cache_hit": False}
