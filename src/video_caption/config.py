import os
import tomllib
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LMStudioConfig:
    base_url: str
    api_key: str
    model: str
    timeout: int
    max_retries: int


@dataclass
class MinIOConfig:
    endpoint: str
    access_key: str
    secret_key: str
    secure: bool
    input_bucket: str
    output_bucket: str


@dataclass
class WhisperConfig:
    model_size: str
    device: str
    compute_type: str
    download_root: str
    no_speech_threshold: float
    log_prob_threshold: float
    beam_size: int
    condition_on_previous_text: bool
    fast_model_size: str


@dataclass
class AppConfig:
    lm_studio: LMStudioConfig
    minio: MinIOConfig
    whisper: WhisperConfig
    chunk_duration: int
    chunk_overlap: int
    temp_dir: str


def _whisper_device_settings() -> dict:
    if os.getenv("DEVICE", "cpu").lower() == "gpu":
        return {"device": "cuda", "compute_type": "float16"}
    return {"device": "cpu", "compute_type": "int8"}


def load_config(path: str = "config.toml") -> AppConfig:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    ls = data["lm_studio"]
    m = data["minio"]
    w = data["whisper"]
    p = data.get("pipeline", {})

    return AppConfig(
        lm_studio=LMStudioConfig(
            base_url=os.getenv("LM_STUDIO_BASE_URL", ls["base_url"]),
            api_key=os.getenv("LM_STUDIO_API_KEY", ls.get("api_key")),
            model=ls.get("model", "qwen/qwen3.5-9b"),
            timeout=ls.get("timeout", 60),
            max_retries=ls.get("max_retries", 3),
        ),
        minio=MinIOConfig(
            endpoint=os.getenv("MINIO_ENDPOINT", m["endpoint"]),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            secure=m.get("secure", False),
            input_bucket=m.get("input_bucket", "video-input"),
            output_bucket=m.get("output_bucket", "video-output"),
        ),
        whisper=WhisperConfig(
            model_size=os.getenv("WHISPER_MODEL_SIZE", w.get("model_size", "medium")),
            download_root=os.getenv("WHISPER_DOWNLOAD_ROOT", w.get("download_root", "./models/faster-whisper")),
            no_speech_threshold=w.get("no_speech_threshold", 0.6),
            log_prob_threshold=w.get("log_prob_threshold", -1.0),
            beam_size=w.get("beam_size", 5),
            condition_on_previous_text=w.get("condition_on_previous_text", True),
            fast_model_size=w.get("fast_model_size", "Systran/faster-whisper-large-v3-turbo"),
            **_whisper_device_settings(),
        ),
        chunk_duration=p.get("chunk_duration", 300),
        chunk_overlap=p.get("chunk_overlap", 5),
        temp_dir=p.get("temp_dir", "/tmp/video-caption"),
    )
