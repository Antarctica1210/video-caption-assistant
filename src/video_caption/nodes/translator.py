import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..clients.lm_studio import LMStudioClient
from ..config import AppConfig
from ..logger import get_logger
from ..state import BilingualSegment, CaptionState, Segment

log = get_logger("video_caption.translator")

MAX_WORKERS = 4  # segments translated concurrently


def _checkpoint_path(transcript_json_path: str, target_lang: str) -> Path:
    base = Path(transcript_json_path).parent
    return base / f"translation_checkpoint_{target_lang}.json"


def _load_checkpoint(path: Path) -> dict[int, str]:
    if path.exists():
        try:
            with open(path) as f:
                raw = json.load(f)
            results = {int(k): v for k, v in raw.items()}
            log.info("Loaded translation checkpoint: %d segment(s) already done — %s", len(results), path)
            return results
        except Exception as e:
            log.warning("Could not load checkpoint %s — starting fresh: %s", path, e)
    return {}


def _save_checkpoint(path: Path, results: dict[int, str]) -> None:
    try:
        with open(path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("Could not save checkpoint: %s", e)


def translate_segments(state: CaptionState, _app_config: AppConfig, lm: LMStudioClient) -> dict:
    target_lang = state["target_lang"]
    segments = state["raw_segments"]
    total = len(segments)

    checkpoint = _checkpoint_path(state["transcript_json_path"], target_lang)
    results: dict[int, str] = _load_checkpoint(checkpoint)

    pending = [(i, seg) for i, seg in enumerate(segments) if i not in results]
    log.info(
        "Translating %d segment(s) → '%s' | %d pending | %d already done | %d parallel workers",
        total, target_lang, len(pending), len(results), MAX_WORKERS,
    )

    def _run(idx: int, seg: Segment) -> tuple[int, str]:
        log.debug("Segment %d/%d started", idx + 1, total)
        try:
            translated = lm.translate(seg["text"], target_lang)[0].get("text", "").strip()
        except Exception as e:
            log.error(
                "Segment %d/%d translation failed, keeping original: %r — %s",
                idx + 1, total, seg["text"][:80], e,
            )
            translated = seg["text"]
        log.debug("Segment %d/%d done", idx + 1, total)
        return idx, translated

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_run, i, seg): i for i, seg in pending}
            for future in as_completed(futures):
                idx, translated = future.result()
                results[idx] = translated
                # flush to disk after every completed segment so progress survives interruption
                _save_checkpoint(checkpoint, results)
    except KeyboardInterrupt:
        log.warning("Translation interrupted — %d/%d segment(s) saved to checkpoint", len(results), total)
        _save_checkpoint(checkpoint, results)
        raise

    bilingual: list[BilingualSegment] = [
        {
            "original": seg["text"],
            "translated": results[i],
            "start": seg["start"],
            "end": seg["end"],
        }
        for i, seg in enumerate(segments)
    ]

    # remove checkpoint once translation is fully complete
    try:
        os.remove(checkpoint)
        log.debug("Checkpoint removed: %s", checkpoint)
    except OSError:
        pass

    log.info("Translation complete — %d segment(s)", total)
    return {"bilingual_segments": bilingual}
