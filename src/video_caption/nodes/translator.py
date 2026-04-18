import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..clients.lm_studio import LMStudioClient
from ..config import AppConfig
from ..logger import get_logger
from ..state import BilingualSegment, CaptionState, Segment

log = get_logger("video_caption.translator")

BATCH_SIZE = 3    # segments per JSON batch — small enough to reduce omissions
MAX_WORKERS = 4   # concurrent batch requests


def translate_segments(state: CaptionState, _app_config: AppConfig, lm: LMStudioClient) -> dict:
    target_lang = state["target_lang"]
    segments = state["raw_segments"]
    total = len(segments)

    checkpoint = _checkpoint_path(state["transcript_json_path"], target_lang)
    results: dict[int, str] = _load_checkpoint(checkpoint)

    pending = [seg for seg in segments if seg.get("id", -1) not in results]
    batches = [pending[i:i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]

    log.info(
        "Translating %d segment(s) → '%s' | %d pending in %d batch(es) of %d | %d workers",
        total, target_lang, len(pending), len(batches), BATCH_SIZE, MAX_WORKERS,
    )

    def _run_batch(batch: list[Segment]) -> list[tuple[int, str]]:
        items = [{"id": seg["id"], "text": seg["text"]} for seg in batch]
        translated = _translate_batch(lm, items, target_lang)
        print(translated)
        return list(translated.items())

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(_run_batch, batch) for batch in batches]
            for future in as_completed(futures):
                for seg_id, text in future.result():
                    results[seg_id] = text
                    _save_checkpoint(checkpoint, results)
    except KeyboardInterrupt:
        log.warning("Translation interrupted — %d/%d segment(s) saved to checkpoint", len(results), total)
        _save_checkpoint(checkpoint, results)
        raise

    # Fill any gaps (safety net)
    for seg in segments:
        seg_id = seg.get("id", -1)
        if seg_id not in results:
            log.warning("No translation for segment id=%d — keeping original", seg_id)
            results[seg_id] = seg["text"]

    bilingual: list[BilingualSegment] = [
        {
            "id": seg.get("id"),
            "original": seg["text"],
            "translated": results[seg.get("id", -1)],
            "start": seg["start"],
            "end": seg["end"],
        }
        for seg in segments
    ]

    try:
        os.remove(checkpoint)
        log.debug("Checkpoint removed: %s", checkpoint)
    except OSError:
        pass

    log.info("Translation complete — %d segment(s)", total)
    return {"bilingual_segments": bilingual}


def _translate_batch(lm: LMStudioClient, items: list[dict], target_lang: str) -> dict[int, str]:
    """Returns {id: translated_text} for all items.
    Strategy: normal batch → strict batch → per-item fallback."""
    expected_ids = {item["id"] for item in items}

    # Attempt 1 — normal prompt
    result = _try_json_batch(lm, items, target_lang, strict=False)
    if result is not None and result.keys() == expected_ids:
        return result
    log.warning("Batch attempt 1 failed (ids: expected=%s got=%s) — retrying strict",
                sorted(expected_ids), sorted(result.keys()) if result else [])

    # Attempt 2 — strict prompt
    result = _try_json_batch(lm, items, target_lang, strict=True)
    if result is not None and result.keys() == expected_ids:
        return result
    log.warning("Batch attempt 2 failed — falling back to per-item")

    # Attempt 3 — per-item fallback
    out: dict[int, str] = {}
    for item in items:
        try:
            translated = lm.translate(item["text"], target_lang).strip()
            out[item["id"]] = translated or item["text"]
        except Exception as e:
            log.error("Per-item fallback failed for id=%d: %s — keeping original", item["id"], e)
            out[item["id"]] = item["text"]
    return out


def _try_json_batch(lm: LMStudioClient, items: list[dict], target_lang: str, strict: bool) -> dict[int, str] | None:
    try:
        response = lm.translate_batch_json(items, target_lang, strict=strict)
    except Exception as e:
        log.warning("translate_batch_json error (strict=%s): %s", strict, e)
        return None

    out: dict[int, str] = {}
    for r in response:
        seg_id = r.get("id")
        translation = str(r.get("translation", "")).strip()
        if seg_id is None or not translation:
            log.warning("Invalid batch entry: %r", r)
            return None
        src = next((x["text"] for x in items if x["id"] == seg_id), "")
        if _suspicious_length(src, translation):
            log.warning("Suspicious length ratio id=%d: src=%r tgt=%r", seg_id, src[:50], translation[:50])
        out[seg_id] = translation

    return out


def _suspicious_length(src: str, tgt: str) -> bool:
    if not src.strip():
        return True
    ratio = len(tgt.strip()) / max(len(src.strip()), 1)
    return ratio > 3.0 or ratio < 0.2


def _checkpoint_path(transcript_json_path: str, target_lang: str) -> Path:
    return Path(transcript_json_path).parent / f"translation_checkpoint_{target_lang}.json"


def _load_checkpoint(path: Path) -> dict[int, str]:
    if path.exists():
        try:
            with open(path) as f:
                raw = json.load(f)
            results = {int(k): v for k, v in raw.items()}
            log.info("Loaded translation checkpoint: %d segment(s) already done", len(results))
            return results
        except Exception as e:
            log.warning("Could not load checkpoint — starting fresh: %s", e)
    return {}


def _save_checkpoint(path: Path, results: dict[int, str]) -> None:
    try:
        with open(path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("Could not save checkpoint: %s", e)
