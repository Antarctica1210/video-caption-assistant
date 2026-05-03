import json
from pathlib import Path

from ..config import AppConfig
from ..logger import get_logger
from ..state import CaptionState, Segment

log = get_logger("video_caption.transcript_loader")


def load_transcript(state: CaptionState, app_config: AppConfig) -> dict:
    jsonl_path = Path(state["transcript_jsonl_path"])
    segments: list[Segment] = [
        json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    log.info("Loaded %d segment(s) from %s", len(segments), jsonl_path)
    return {"raw_segments": segments}
