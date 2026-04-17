from ..clients.lm_studio import LMStudioClient
from ..config import AppConfig
from ..logger import get_logger
from ..state import BilingualSegment, CaptionState

log = get_logger("video_caption.translator")


def translate_segments(state: CaptionState, config: AppConfig, lm: LMStudioClient) -> dict:
    target_lang = state["target_lang"]
    total = len(state["raw_segments"])
    log.info("Translating %d segment(s) to '%s'", total, target_lang)

    bilingual: list[BilingualSegment] = []

    for i, seg in enumerate(state["raw_segments"], start=1):
        log.debug("Translating segment %d/%d", i, total)
        translated = lm.translate(seg["text"], target_lang)
        bilingual.append({
            "original": seg["text"],
            "translated": translated,
            "start": seg["start"],
            "end": seg["end"],
        })

    log.info("Translation complete")
    return {"bilingual_segments": bilingual}
