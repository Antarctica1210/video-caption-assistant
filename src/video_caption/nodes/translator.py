from concurrent.futures import ThreadPoolExecutor, as_completed

from ..clients.lm_studio import LMStudioClient
from ..config import AppConfig
from ..logger import get_logger
from ..state import BilingualSegment, CaptionState, Segment

log = get_logger("video_caption.translator")

BATCH_SIZE = 10   # segments combined into one LM Studio request
MAX_WORKERS = 4   # concurrent requests sent in parallel
DELIMITER = "|||" # separator between translated lines in LLM response


def translate_segments(state: CaptionState, _app_config: AppConfig, lm: LMStudioClient) -> dict:
    target_lang = state["target_lang"]
    segments = state["raw_segments"]
    total = len(segments)
    batches = [segments[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    log.info(
        "Translating %d segment(s) → '%s' | %d batch(es) of %d | %d parallel workers",
        total, target_lang, len(batches), BATCH_SIZE, MAX_WORKERS,
    )

    results: dict[int, list[str]] = {}

    def _run(idx: int, batch: list[Segment]) -> tuple[int, list[str]]:
        log.debug("Batch %d/%d started (%d segments)", idx + 1, len(batches), len(batch))
        translations = _translate_batch(lm, batch, target_lang)
        log.debug("Batch %d/%d done", idx + 1, len(batches))
        return idx, translations

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_run, i, batch): i for i, batch in enumerate(batches)}
        for future in as_completed(futures):
            idx, translations = future.result()
            results[idx] = translations

    bilingual: list[BilingualSegment] = []
    for i, batch in enumerate(batches):
        for seg, translated in zip(batch, results[i]):
            bilingual.append({
                "original": seg["text"],
                "translated": translated,
                "start": seg["start"],
                "end": seg["end"],
            })

    log.info("Translation complete — %d segment(s) in %d batch(es)", total, len(batches))
    return {"bilingual_segments": bilingual}


def _translate_batch(lm: LMStudioClient, batch: list[Segment], target_lang: str) -> list[str]:
    combined = f"\n{DELIMITER}\n".join(seg["text"] for seg in batch)
    system = (
        f"You are a professional subtitle translator. "
        f"The input contains {len(batch)} subtitle lines separated by '{DELIMITER}'. "
        f"Translate each line to {target_lang}. "
        f"Return exactly {len(batch)} translated lines separated by '{DELIMITER}'. "
        f"Preserve the order. Output only the translated lines, no extra text."
    )
    response = lm.translate(combined, target_lang, system_prompt=system)
    translations = _split_response(response, expected=len(batch))

    if len(translations) != len(batch):
        log.warning(
            "Delimiter split mismatch: expected %d, got %d — falling back to per-segment",
            len(batch), len(translations),
        )
        return [lm.translate(seg["text"], target_lang) for seg in batch]

    return translations


def _split_response(text: str, expected: int) -> list[str]:
    parts = [p.strip() for p in text.split(DELIMITER)]
    parts = [p for p in parts if p]
    return parts if len(parts) == expected else parts
