from ..logger import get_logger
from ..state import BilingualSegment, CaptionState

log = get_logger("video_caption.validator")

MAX_CAPTION_DURATION = 10.0


def validate_timeline(state: CaptionState) -> dict:
    segments = state["bilingual_segments"]
    fixed: list[BilingualSegment] = []
    clamped = 0
    overlaps = 0

    for seg in segments:
        start = seg["start"]
        end = seg["end"]

        if end - start > MAX_CAPTION_DURATION:
            end = start + MAX_CAPTION_DURATION
            clamped += 1

        if fixed and start < fixed[-1]["end"]:
            start = fixed[-1]["end"]
            overlaps += 1

        if end > start:
            fixed.append({**seg, "start": round(start, 3), "end": round(end, 3)})

    if clamped:
        log.warning("Clamped %d segment(s) exceeding max duration (%.1fs)", clamped, MAX_CAPTION_DURATION)
    if overlaps:
        log.warning("Fixed %d overlapping segment(s)", overlaps)

    log.info("Timeline validated — %d segment(s) OK", len(fixed))
    return {"bilingual_segments": fixed}
