# Video Caption Assistant

Offline-first pipeline that transcribes video audio with faster-whisper, translates captions via a self-hosted LM Studio instance on your LAN, and exports bilingual SRT/ASS subtitle files **next to the source video**.

Pick the video from a local web page (or pass its path on the command line). No upload step and no object storage.

---

## Requirements

- Python 3.12
- [uv](https://github.com/astral-sh/uv) package manager
- [LM Studio](https://lmstudio.ai/) running on your LAN with a model loaded

---

## Setup

```bash
# Install uv if not already installed, then install dependencies
bash setup.sh

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your LM Studio IP and API key
```

---

## Usage

### Web page (recommended)

```bash
uv run main.py serve
# Open http://localhost:8000 in your browser
```

1. **Choose a video** — browse your drives and folders, or paste a path (a folder path opens that folder).
2. **Confirm** — the page shows the full video path and the folder the captions will be saved to, and warns if existing `.srt`/`.ass` files will be overwritten.
3. **Set options and generate** — target language, spoken language, format, title, fast mode. Progress and log lines appear on the page.

Captions are written to the same folder as the video:

```
D:\Videos\
├── my_video.mp4
├── my_video.srt
└── my_video.ass
```

Only one video is processed at a time. A second request waits until the first finishes.

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `127.0.0.1` | Bind address (from `config.toml [web]`) |
| `--port`, `-p` | `8000` | Port (from `config.toml [web]`) |

### Running on WSL with videos on Windows

Run the app inside WSL and use your normal Windows browser:

```bash
# inside WSL
uv run main.py serve
# then open http://localhost:8000 in Edge/Chrome on Windows
```

- **Computer** in the folder browser lists your Windows drives (`C:\`, `D:\`, …, mounted in WSL at `/mnt/c`, `/mnt/d`, …) and your WSL home folder.
- You can paste a Windows path straight from Explorer (Shift + right-click → **Copy as path**), for example `"C:\Users\me\Videos\talk.mp4"`. Quotes are fine. It is converted to `/mnt/c/Users/me/Videos/talk.mp4` for you.
- `\\wsl.localhost\Ubuntu\home\me\...` paths also work.
- Captions are written back into the Windows folder, so they show up in Explorer next to the video.

Notes:
- Reading videos from `/mnt/c` is slower than from the WSL disk. The audio is extracted once and cached, so this cost only applies to the first run.
- The page can list any folder your WSL user can read. Keep `host = "127.0.0.1"` (the default) so only your own machine can open it. WSL2 still forwards `localhost` to Windows.
- For GPU transcription on WSL you need the NVIDIA Windows driver with WSL support, plus CUDA 12 cuBLAS/cuDNN libraries inside WSL. Otherwise set `DEVICE=cpu` in `.env`.

### Command line

```bash
uv run main.py run "C:\Users\me\Videos\my_video.mp4" --lang zh
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `VIDEO` (argument) | | *(required)* | Path to the video. Windows paths work on WSL. |
| `--lang` | `-l` | `zh` | Target translation language (`zh`, `en`, `es`, `fr`, `ja`, …) |
| `--source` | `-s` | *(auto-detect)* | Source audio language hint for Whisper (`ja`, `en`, `zh`, …) |
| `--fast-mode` | | `false` | Use `large-v3-turbo` (~4× faster, slightly lower accuracy) |
| `--format` | `-f` | `both` | Output format: `srt`, `ass`, or `both` |
| `--title` | `-t` | *(auto)* | Override video title (defaults to filename stem) |
| `--config` | `-c` | `config.toml` | Path to config file |

> **`--source`**: If you already know the video's spoken language, providing it skips Whisper's auto-detection and can improve accuracy. Omit it to let Whisper detect the language automatically.

**Examples:**

```bash
# Source is Japanese — skip auto-detection, translate to Chinese
uv run main.py run ~/videos/talk.mp4 --lang zh --source ja

# Fast mode, Spanish, SRT only
uv run main.py run ~/videos/talk.mp4 --lang es --format srt --fast-mode
```

---

### Clean up temp files

Intermediate files (extracted audio, transcript, translation checkpoint) are kept in `temp_dir` so re-runs can skip transcription. Each video gets its own folder named `{stem}-{hash}`. To remove one:

```bash
uv run main.py delete "C:\Users\me\Videos\my_video.mp4"
```

This does not touch the captions next to the video.

---

## Configuration

Edit `config.toml` to set LM Studio, Whisper, web, and pipeline options:

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

[web]
host = "127.0.0.1"
port = 8000

[pipeline]
chunk_duration = 300   # seconds per audio chunk
chunk_overlap  = 5     # overlap between chunks in seconds
temp_dir       = "./tmp/video-caption"
```

Environment variables in `.env` override `config.toml` values. See `.env.example` for all available overrides.
