from typing import TypedDict


class AudioChunk(TypedDict):
    path: str
    start: float
    end: float
    index: int


class Segment(TypedDict):
    text: str
    start: float
    end: float


class BilingualSegment(TypedDict):
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

    # --- final ---
    output_keys: list[str]
    error: str | None
