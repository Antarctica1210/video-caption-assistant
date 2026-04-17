from concurrent.futures import ThreadPoolExecutor, as_completed

from ..clients.lm_studio import LMStudioClient
from ..config import AppConfig
from ..logger import get_logger
from ..state import BilingualSegment, CaptionState, Segment

log = get_logger("video_caption.translator")

BATCH_SIZE = 4    # segments per LM Studio request
MAX_WORKERS = 4   # concurrent requests to LM Studio


def translate_segments(state: CaptionState, _app_config: AppConfig, lm: LMStudioClient) -> dict:
    target_lang = state["target_lang"]
    segments = state["raw_segments"]
    total = len(segments)
    batches = [segments[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    log.info(
        "Translating %d segment(s) to '%s' — %d batch(es) x %d segments, %d parallel workers",
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
