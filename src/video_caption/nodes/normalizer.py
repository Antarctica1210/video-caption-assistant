import re

from ..logger import get_logger
from ..state import CaptionState, Segment

log = get_logger("video_caption.normalizer")

# Matches strings that are entirely punctuation/whitespace — no translatable content
_NOISE_ONLY = re.compile(r'^[.,!?;:—\-–…()\[\]{}\'"「」『』♪♫\s]+$')


def normalize_segments(state: CaptionState) -> dict:
    segments = state["raw_segments"]
    normalized: list[Segment] = []
    skipped = 0

    for seg in segments:
        text = _normalize(seg["text"])
        if _should_skip(text):
            skipped += 1
            continue
        normalized.append({**seg, "text": text})

    # Reassign contiguous ids after filtering
    for i, seg in enumerate(normalized):
        seg["id"] = i

    if skipped:
        log.info("Normalization: %d segment(s) skipped (noise/empty), %d remaining", skipped, len(normalized))
    else:
        log.debug("Normalization: all %d segment(s) kept", len(normalized))

    return {"raw_segments": normalized}


def _normalize(text: str) -> str:
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)      # collapse multiple spaces
    text = re.sub(r'\.{3,}', '…', text)   # normalize ellipsis sequences
    return text


def _should_skip(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if _NOISE_ONLY.match(stripped):
        return True
    return False
