from pathlib import Path

from ...logger import get_logger
from ...state import CaptionState
from ..title import TITLE_END, TITLE_START

log = get_logger("video_caption.export_srt")


def export_srt(state: CaptionState) -> dict:
    if state.get("output_format", "both") not in ("srt", "both"):
        return {"srt_path": None}

    # Captions go next to the source video so players pick them up automatically
    video_path = Path(state["video_path"])
    srt_path = video_path.with_suffix(".srt")

    blocks: list[str] = []
    idx = 1

    if state.get("translated_title"):
        blocks.append(_block(idx, TITLE_START, TITLE_END, state["translated_title"], state["translated_title"]))
        idx += 1

    for seg in state["bilingual_segments"]:
        blocks.append(_block(idx, seg["start"], seg["end"], seg["original"], seg["translated"]))
        idx += 1

    srt_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    log.info("Wrote %s", srt_path)
    return {"srt_path": str(srt_path)}


def _block(idx: int, start: float, end: float, original: str, translated: str) -> str:
    return f"{idx}\n{_ts(start)} --> {_ts(end)}\n{original}\n{translated}"


def _ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
