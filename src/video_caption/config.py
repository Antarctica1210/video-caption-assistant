import os
import tomllib
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LMStudioConfig:
    base_url: str
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
            model=ls["model"],
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
            **_whisper_device_settings(),
        ),
        chunk_duration=p.get("chunk_duration", 300),
        chunk_overlap=p.get("chunk_overlap", 5),
        temp_dir=p.get("temp_dir", "/tmp/video-caption"),
    )
