from concurrent.futures import ThreadPoolExecutor, as_completed

from ..clients.lm_studio import LMStudioClient
from ..config import AppConfig
from ..logger import get_logger
from ..state import BilingualSegment, CaptionState, Segment

log = get_logger("video_caption.translator")

BATCH_SIZE = 10   # number of segments combined into one LM Studio request
MAX_WORKERS = 4   # number of batches sent concurrently to LM Studio
DELIMITER = "|||" # token used to separate lines inside the LLM response


def translate_segments(state: CaptionState, _app_config: AppConfig, lm: LMStudioClient) -> dict:
    target_lang = state["target_lang"]
    segments = state["raw_segments"]
    total = len(segments)

    # Split the full segment list into fixed-size batches.
    # Each batch is sent as one LM Studio request to reduce round-trips.
    batches = [segments[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    log.info(
        "Translating %d segment(s) → '%s' | %d batch(es) of %d | %d parallel workers",
        total, target_lang, len(batches), BATCH_SIZE, MAX_WORKERS,
    )

    # results keyed by batch index so we can reassemble in order
    # regardless of which futures complete first
    results: dict[int, list[str]] = {}

    def _run(idx: int, batch: list[Segment]) -> tuple[int, list[str]]:
        log.debug("Batch %d/%d started (%d segments)", idx + 1, len(batches), len(batch))
        translations = _translate_batch(lm, batch, target_lang)
        log.debug("Batch %d/%d done", idx + 1, len(batches))
        return idx, translations

    # Dispatch all batches concurrently up to MAX_WORKERS at a time.
    # as_completed yields futures in arrival order (fastest first),
    # but we store by idx so reassembly is always in original order.
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_run, i, batch): i for i, batch in enumerate(batches)}
        for future in as_completed(futures):
            idx, translations = future.result()
            results[idx] = translations

    # Reassemble: iterate batches in original order, zip each segment with
    # its translation, and attach the original timestamps.
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
    # Join all segment texts with the delimiter so the LLM receives them
    # as one coherent block — gives better translation context than
    # sending each line in isolation.
    combined = f"\n{DELIMITER}\n".join(seg["text"] for seg in batch)

    system = (
        f"You are a professional subtitle translator. "
        f"The input contains {len(batch)} subtitle lines separated by '{DELIMITER}'. "
        f"Translate each line to {target_lang}. "
        f"Return exactly {len(batch)} translated lines separated by '{DELIMITER}'. "
        f"Preserve the order. Output only the translated lines, no extra text."
    )

    response = lm.translate(combined, target_lang, system_prompt=system)

    # Split the response back into individual translations and validate count.
    # If the LLM returns a different number of lines (e.g. it merged or split
    # a sentence), fall back to translating each segment individually so no
    # timestamp is left without a translation.
    translations = _split_response(response, expected=len(batch))

    if len(translations) != len(batch):
        log.warning(
            "Delimiter split mismatch: expected %d, got %d — falling back to per-segment\nRaw response: %r",
            len(batch), len(translations), response[:300],
        )
        results = []
        for seg in batch:
            try:
                results.append(lm.translate(seg["text"], target_lang))
            except Exception as e:
                log.error(
                    "Per-segment translation failed, keeping original text: %r — %s",
                    seg["text"][:80], e,
                )
                results.append(seg["text"])
        return results

    return translations


def _split_response(text: str, expected: int) -> list[str]:
    # Strip whitespace around each part and discard any empty strings that
    # result from leading/trailing delimiters in the LLM output.
    parts = [p.strip() for p in text.split(DELIMITER)]
    parts = [p for p in parts if p]
    return parts if len(parts) == expected else parts
