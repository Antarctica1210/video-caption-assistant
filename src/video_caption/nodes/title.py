from ..clients.lm_studio import LMStudioClient
from ..logger import get_logger
from ..state import CaptionState

log = get_logger("video_caption.title")


def translate_title(state: CaptionState, lm: LMStudioClient) -> dict:
    raw_title = state.get("title")
    if not raw_title:
        log.debug("No title provided — skipping title translation")
        return {"translated_title": None}

    log.info("Translating title: %r", raw_title)
    translated = lm.translate(raw_title, state["target_lang"])
    combined = f"{raw_title} | {translated}"
    log.info("Title result: %r", combined)
    return {"translated_title": combined}
