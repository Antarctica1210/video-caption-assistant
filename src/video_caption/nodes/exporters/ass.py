from pathlib import Path

from ...config import AppConfig
from ...state import CaptionState

_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Original,Arial,52,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,30,1
Style: Translated,Arial,46,&H0000FFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def export_ass(state: CaptionState, app_config: AppConfig) -> dict:
    stem = Path(state["local_video_path"]).stem
    out_dir = Path(app_config.temp_dir) / "output" / stem
    ass_path = out_dir / f"{stem}.ass"

    events: list[str] = []

    if state.get("translated_title"):
        t = state["translated_title"]
        events.append(f"Dialogue: 0,{_ts(0.0)},{_ts(5.0)},Original,,0,0,0,,{t}")

    for seg in state["bilingual_segments"]:
        s, e = _ts(seg["start"]), _ts(seg["end"])
        events.append(f"Dialogue: 0,{s},{e},Original,,0,0,0,,{seg['original']}")
        events.append(f"Dialogue: 0,{s},{e},Translated,,0,0,0,,{seg['translated']}")

    ass_path.write_text(_HEADER + "\n".join(events) + "\n", encoding="utf-8")
    return {"ass_path": str(ass_path)}


def _ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"
