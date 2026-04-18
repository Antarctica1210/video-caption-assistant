from typing import NotRequired, TypedDict


class AudioChunk(TypedDict):
    path: str
    start: float
    end: float
    index: int


class Segment(TypedDict):
    id: NotRequired[int]  # sequential index; assigned in transcriber, reassigned after dedup
    text: str
    start: float
    end: float


class BilingualSegment(TypedDict):
    id: NotRequired[int]
    original: str
    translated: str
    start: float
    end: float


class CaptionState(TypedDict, total=False):
    # --- inputs (required at invocation) ---
    video_key: str          # MinIO object key inside video_input bucket
    target_lang: str        # e.g. "zh", "es", "fr"
    output_format: str      # "srt" | "ass" | "both"
    title: str | None       # optional video title

    # --- pipeline intermediates ---
    local_video_path: str
    local_audio_path: str
    audio_chunks: list[AudioChunk]
    raw_segments: list[Segment]
    bilingual_segments: list[BilingualSegment]
    translated_title: str | None

    # --- local output paths ---
    srt_path: str | None
    ass_path: str | None
    transcript_json_path: str | None
    transcript_csv_path: str | None

    # --- cache ---
    cache_hit: bool             # True if transcript already exists; skips extraction

    # --- final ---
    output_keys: list[str]
    error: str | None
