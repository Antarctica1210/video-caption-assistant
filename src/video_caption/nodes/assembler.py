import csv
import json
from pathlib import Path

from ..config import AppConfig
from ..state import CaptionState, Segment


def merge_and_save(state: CaptionState, config: AppConfig) -> dict:
    segments = _deduplicate(state["raw_segments"])

    stem = Path(state["local_video_path"]).stem
    out_dir = Path(config.temp_dir) / "output" / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "transcript.json"
    csv_path = out_dir / "transcript.csv"

    json_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["start", "end", "text"])
        writer.writeheader()
        writer.writerows(segments)

    return {
        "raw_segments": segments,
        "transcript_json_path": str(json_path),
        "transcript_csv_path": str(csv_path),
    }


def _deduplicate(segments: list[Segment]) -> list[Segment]:
    if not segments:
        return segments
    result: list[Segment] = [segments[0]]
    for seg in segments[1:]:
        prev = result[-1]
        if seg["start"] >= prev["end"]:
            result.append(seg)
        elif seg["end"] > prev["end"]:
            result[-1] = {"text": prev["text"] + " " + seg["text"], "start": prev["start"], "end": seg["end"]}
    return result
