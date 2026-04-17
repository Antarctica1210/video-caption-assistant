from ..clients.lm_studio import LMStudioClient
from ..config import AppConfig
from ..logger import get_logger
from ..state import BilingualSegment, CaptionState, Segment

log = get_logger("video_caption.translator")

BATCH_SIZE = 4  # segments per LM Studio request


def translate_segments(state: CaptionState, _app_config: AppConfig, lm: LMStudioClient) -> dict:
    target_lang = state["target_lang"]
    segments = state["raw_segments"]
    total = len(segments)
    log.info("Translating %d segment(s) to '%s' in batches of %d", total, target_lang, BATCH_SIZE)

    bilingual: list[BilingualSegment] = []
    batches = [segments[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches, start=1):
        log.debug("Translating batch %d/%d (%d segments)", batch_idx, len(batches), len(batch))
        translations = _translate_batch(lm, batch, target_lang)

        for seg, translated in zip(batch, translations):
            bilingual.append({
                "original": seg["text"],
                "translated": translated,
                "start": seg["start"],
                "end": seg["end"],
            })

    log.info("Translation complete — %d segment(s) translated in %d batch(es)", total, len(batches))
    return {"bilingual_segments": bilingual}


def _translate_batch(lm: LMStudioClient, batch: list[Segment], target_lang: str) -> list[str]:
    numbered = "\n".join(f"{i + 1}. {seg['text']}" for i, seg in enumerate(batch))
    system = (
        f"You are a professional subtitle translator. "
        f"Translate each numbered line to {target_lang}. "
        f"Keep the same number prefix. "
        f"Output only the translated numbered lines, no extra text."
    )
    response = lm.translate(numbered, target_lang, system_prompt=system)
    translations = _parse_numbered(response, expected=len(batch))

    if len(translations) != len(batch):
        log.warning(
            "Batch parse mismatch: expected %d, got %d — falling back to per-segment",
            len(batch), len(translations),
        )
        return [lm.translate(seg["text"], target_lang) for seg in batch]

    return translations


def _parse_numbered(text: str, expected: int) -> list[str]:
    lines: list[str] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # strip leading "N." or "N) " prefix
        if line[0].isdigit():
            dot = line.find(".")
            paren = line.find(")")
            sep = min(p for p in (dot, paren) if p != -1) if any(p != -1 for p in (dot, paren)) else -1
            if sep != -1 and sep < 5:
                line = line[sep + 1:].strip()
        lines.append(line)
    return lines if len(lines) == expected else lines
