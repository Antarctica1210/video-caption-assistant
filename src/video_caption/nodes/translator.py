from ..clients.lm_studio import LMStudioClient
from ..config import AppConfig
from ..state import BilingualSegment, CaptionState


def translate_segments(state: CaptionState, config: AppConfig, lm: LMStudioClient) -> dict:
    target_lang = state["target_lang"]
    bilingual: list[BilingualSegment] = []

    for seg in state["raw_segments"]:
        translated = lm.translate(seg["text"], target_lang)
        bilingual.append({
            "original": seg["text"],
            "translated": translated,
            "start": seg["start"],
            "end": seg["end"],
        })

    return {"bilingual_segments": bilingual}
