from ..logger import get_logger
from ..state import BilingualSegment, CaptionState

log = get_logger("video_caption.validator")

MAX_CAPTION_DURATION = 10.0
MAX_CPS = 15.0  # characters per second — above this subtitles are too fast to read


def validate_timeline(state: CaptionState) -> dict:
    segments = state["bilingual_segments"]
    fixed: list[BilingualSegment] = []
    clamped = 0
    overlaps = 0
    fast_read = 0

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
            cps = len(seg["translated"]) / max(end - start, 0.1)
            if cps > MAX_CPS:
                fast_read += 1
                log.debug(
                    "High CPS %.1f at %.1fs–%.1fs: %r",
                    cps, start, end, seg["translated"][:50],
                )

    if clamped:
        log.warning("Clamped %d segment(s) exceeding max duration (%.1fs)", clamped, MAX_CAPTION_DURATION)
    if overlaps:
        log.warning("Fixed %d overlapping segment(s)", overlaps)
    if fast_read:
        log.warning("%d segment(s) exceed %.0f CPS — may be too fast to read", fast_read, MAX_CPS)

    log.info("Timeline validated — %d segment(s) OK", len(fixed))
    return {"bilingual_segments": fixed}
