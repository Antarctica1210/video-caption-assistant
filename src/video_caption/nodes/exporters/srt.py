from pathlib import Path

from ...config import AppConfig
from ...state import CaptionState


def export_srt(state: CaptionState, app_config: AppConfig) -> dict:
    stem = Path(state["local_video_path"]).stem
    out_dir = Path(app_config.temp_dir) / "output" / stem
    srt_path = out_dir / f"{stem}.srt"

    blocks: list[str] = []
    idx = 1

    if state.get("translated_title"):
        blocks.append(_block(idx, 0.0, 5.0, state["translated_title"], state["translated_title"]))
        idx += 1

    for seg in state["bilingual_segments"]:
        blocks.append(_block(idx, seg["start"], seg["end"], seg["original"], seg["translated"]))
        idx += 1

    srt_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return {"srt_path": str(srt_path)}


def _block(idx: int, start: float, end: float, original: str, translated: str) -> str:
    return f"{idx}\n{_ts(start)} --> {_ts(end)}\n{original}\n{translated}"


def _ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
