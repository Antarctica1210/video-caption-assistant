from ..state import BilingualSegment, CaptionState

MAX_CAPTION_DURATION = 10.0  # seconds


def validate_timeline(state: CaptionState) -> dict:
    segments = state["bilingual_segments"]
    fixed: list[BilingualSegment] = []

    for seg in segments:
        start = seg["start"]
        end = min(seg["end"], start + MAX_CAPTION_DURATION)

        if fixed and start < fixed[-1]["end"]:
            start = fixed[-1]["end"]

        if end > start:
            fixed.append({**seg, "start": round(start, 3), "end": round(end, 3)})

    return {"bilingual_segments": fixed}
