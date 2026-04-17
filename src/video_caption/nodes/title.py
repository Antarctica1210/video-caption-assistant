from ..clients.lm_studio import LMStudioClient
from ..state import CaptionState


def translate_title(state: CaptionState, lm: LMStudioClient) -> dict:
    raw_title = state.get("title")
    if not raw_title:
        return {"translated_title": None}

    translated = lm.translate(raw_title, state["target_lang"])
    return {"translated_title": f"{raw_title} | {translated}"}
