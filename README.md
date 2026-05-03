# Video Caption Assistant

Offline-first pipeline that transcribes video audio with faster-whisper, translates captions via a self-hosted LM Studio instance on your LAN, and exports bilingual SRT/ASS subtitle files.

---

## Requirements

- Python 3.12
- [uv](https://github.com/astral-sh/uv) package manager
- [Docker](https://docs.docker.com/get-docker/) (for MinIO)
- [LM Studio](https://lmstudio.ai/) running on your LAN with a model loaded

---

## Setup

```bash
# Install uv if not already installed
bash setup.sh

# Install dependencies
uv sync

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your LM Studio IP, API key, and MinIO credentials
```

Start MinIO:

```bash
docker compose up -d
# S3 API:      http://localhost:9000
# Web console: http://localhost:9001  (minioadmin / minioadmin)
```

---

## Usage

Upload one video to the `video-input` bucket via the MinIO console (`http://localhost:9001`), then run:

### Transcribe and translate

```bash
uv run main.py run --lang zh
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--lang` | `-l` | `zh` | Target translation language (`zh`, `en`, `es`, `fr`, `ja`, …) |
| `--source` | `-s` | *(auto-detect)* | Source audio language hint for Whisper (`ja`, `en`, `zh`, …) |
| `--fast-mode` | | `false` | Use `large-v3-turbo` (~4× faster, slightly lower accuracy) |
| `--format` | `-f` | `both` | Output format: `srt`, `ass`, or `both` |
| `--title` | `-t` | *(auto)* | Override video title (defaults to filename stem) |
| `--config` | `-c` | `config.toml` | Path to config file |

> **`--source`**: If you already know the video's spoken language, providing it skips Whisper's auto-detection and can improve accuracy. Omit it to let Whisper detect the language automatically.

**Examples:**

```bash
# Translate to Chinese, export both SRT and ASS (default)
uv run main.py run --lang zh

# Source is Japanese — skip auto-detection, translate to Chinese
uv run main.py run --lang zh --source ja

# Fast mode — use large-v3-turbo for quicker transcription
uv run main.py run --lang zh --fast-mode

# Translate to Spanish, SRT only
uv run main.py run --lang es --format srt

# Override the title shown in the caption file
uv run main.py run --lang zh --title "My Video Title"

# Use a custom config file
uv run main.py run --lang zh --config /path/to/config.toml
```

Output files are uploaded to the `video-output` bucket:
```
video-output/
└── my_video/
    ├── my_video.srt
    └── my_video.ass
```

---

### Clean up tmp files

```bash
uv run main.py delete "my_video"
```

Deletes `{temp_dir}/my_video/` and all its contents (downloaded video, extracted audio, chunks, transcripts, caption files).

---

## Configuration

Edit `config.toml` to set LM Studio, MinIO, Whisper, and pipeline options:

```toml
[lm_studio]
base_url = "http://192.168.1.100:1234/v1"
model    = "qwen/qwen3.5-9b"
timeout  = 300
max_retries = 3

[whisper]
model_size   = "large-v3"   # tiny | base | small | medium | large-v3
device       = "cuda"       # cpu | cuda
compute_type = "float16"    # int8 | float16 | float32

[minio]
endpoint      = "localhost:9000"
input_bucket  = "video-input"
output_bucket = "video-output"

[pipeline]
chunk_duration = 300   # seconds per audio chunk
chunk_overlap  = 5     # overlap between chunks in seconds
temp_dir       = "./tmp/video-caption"
```

Environment variables in `.env` override `config.toml` values. See `.env.example` for all available overrides.
