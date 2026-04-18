import json
from pathlib import Path

from ..config import AppConfig
from ..logger import get_logger
from ..state import CaptionState, Segment

log = get_logger("video_caption.assembler")


def merge_and_save(state: CaptionState, app_config: AppConfig) -> dict:
    before = len(state["raw_segments"])
    segments = _deduplicate(state["raw_segments"])
    for i, seg in enumerate(segments):
        seg["id"] = i
    after = len(segments)

    if before != after:
        log.info("Deduplicated segments: %d → %d", before, after)

    stem = Path(state["local_video_path"]).stem
    out_dir = Path(app_config.temp_dir) / stem / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "transcript.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(seg, ensure_ascii=False) + "\n")
    log.info("Saved transcript JSONL → %s (%d segments)", jsonl_path, len(segments))

    return {
        "raw_segments": segments,
        "transcript_jsonl_path": str(jsonl_path),
    }


def _deduplicate(segments: list[Segment]) -> list[Segment]:
    if not segments:
        return segments
    result: list[Segment] = [segments[0]]
    for seg in segments[1:]:
        prev = result[-1]
        if seg["text"] == prev["text"]:
            if seg["end"] > prev["end"]:
                result[-1] = {**prev, "end": seg["end"]}
            continue
        if seg["start"] >= prev["end"]:
            result.append(seg)
        elif seg["end"] > prev["end"]:
            result[-1] = {"text": prev["text"] + " " + seg["text"], "start": prev["start"], "end": seg["end"]}
    return result
