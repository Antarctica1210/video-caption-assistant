from pathlib import Path

from ..clients.lm_studio import LMStudioClient
from ..logger import get_logger
from ..state import CaptionState

log = get_logger("video_caption.title")

TITLE_START = 3.0  # seconds into the video when title card appears
TITLE_END   = 10.0  # seconds when title card disappears (7s duration)


def translate_title(state: CaptionState, lm: LMStudioClient) -> dict:
    # Use explicitly provided title, or fall back to the video filename stem
    raw_title = state.get("title") or Path(state["video_key"]).stem
    log.info("Translating title: %r", raw_title)
    translated = lm.translate(raw_title, state["target_lang"]).strip()
    combined = f"{raw_title} | {translated}"
    log.info("Title result: %r", combined)
    return {"translated_title": translated}
