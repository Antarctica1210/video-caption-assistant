# Video Caption Assistant — Project Roadmap

## Overview

A fully **offline-first**, LangGraph-orchestrated pipeline that extracts audio from video, transcribes it locally with faster-whisper, translates captions via a **LM Studio instance running on the LAN** (OpenAI-compatible REST API), and exports SRT/ASS caption files. A secondary feature auto-translates the video title and prepends it inside the caption file.

---

## Architecture Decision

| Concern | Choice | Reason |
|---------|--------|--------|
| Orchestration | **LangGraph** (two graphs) | Stateful graph; split into transcription + translation for memory isolation |
| Transcription | **faster-whisper (local)** | Fully offline; CTranslate2 engine; word-level timestamps |
| Translation | **LM Studio via LAN REST API** | Self-hosted LLM; OpenAI-compatible `/v1/chat/completions`; zero cloud cost |
| LLM client | **langchain-openai (ChatOpenAI)** | OpenAI-compatible wrapper; `enable_thinking=False` for speed |
| Audio extraction | **static-ffmpeg** (Python package) | Bundled static binary; no system install required |
| Caption export | **custom SRT + ASS writers** | SRT standard; ASS for bilingual styled subtitles |
| Object storage | **MinIO** (Docker Compose) | S3-compatible self-hosted; `video-input` + `video-output` buckets |
| Transcript cache | **JSON on disk** | Skip re-transcription if `transcript.json` already exists |

---

## Feature Scope

| # | Feature | Status |
|---|---------|--------|
| 1 | Audio extraction | ✅ Done |
| 2 | Audio chunking with overlap | ✅ Done |
| 3 | Parallel faster-whisper transcription | ✅ Done |
| 4 | Chunk merging + deduplication | ✅ Done |
| 5 | Batch + parallel translation (10 segments, 4 workers) | ✅ Done |
| 6 | Bilingual captions (original + translated) | ✅ Done |
| 7 | Timeline validation | ✅ Done |
| 8 | SRT + ASS export | ✅ Done |
| 9 | Transcript cache (JSON/CSV) — skip re-transcription | ✅ Done |
| 10 | Title translation + injection | ✅ Done |
| 11 | MinIO input/output storage | ✅ Done |
| 12 | Auto-detect single video from `video-input` bucket | ✅ Done |
| 13 | Logging to console + `logs/app.log` | ✅ Done |
| 14 | Batch video support | 🔲 Pending |
| 15 | Tests | 🔲 Pending |

---

## Roadmap Phases

### Phase 1 — Infrastructure & LangGraph Setup ✅
- [x] Set up LangGraph state schema (`CaptionState`)
- [x] Two-graph architecture: `build_transcription_graph` + `build_translation_graph`
- [x] LM Studio LAN client via `langchain-openai` (`ChatOpenAI`) with `enable_thinking=False`
- [x] faster-whisper model loading — `DEVICE` env var selects cpu/cuda + compute type
- [x] MinIO client — bucket provisioning, upload, download
- [x] TOML config + `.env` overrides for all services
- [x] `static-ffmpeg` Python package — no system binary needed
- [x] Logging system — `LOG_LEVEL` / `LOG_DIR` env vars; console + file output

### Phase 2 — Core Transcription Pipeline ✅
- [x] `check_cache` node: skip transcription if `transcript.json` + `transcript.csv` exist
- [x] `fetch_and_extract` node: download from MinIO `video-input` + ffmpeg → mono 16kHz WAV
- [x] `chunk_audio` node: split with configurable chunk size + overlap
- [x] `transcribe_chunks` node: parallel faster-whisper inference (word-level timestamps)
- [x] `merge_and_save` node: deduplicate by text + timestamp, save JSON + CSV

### Phase 3 — Translation via LM Studio ✅
- [x] `load_transcript` node: Graph 2 entry — always reads `raw_segments` from `transcript.json`
- [x] `translate_title` node: translate title via LM Studio, format as `original | translated`
- [x] `translate_segments` node:
  - Combine 10 segments into one request using `|||` delimiter
  - Send up to 4 batches concurrently via `ThreadPoolExecutor`
  - Parse response by splitting on `|||`; fall back to per-segment on mismatch
  - Reassemble in original order with original timestamps

### Phase 4 — Caption Export & Validation ✅
- [x] `validate_timeline` node: clamp max duration, fix overlaps, log warnings
- [x] `export_srt` node: bilingual SRT (original line + translated line per block)
- [x] `export_ass` node: bilingual ASS with `Original` + `Translated` styles
- [x] `upload_outputs` node: push SRT, ASS, JSON, CSV to MinIO `video-output/<stem>/`

### Phase 5 — CLI & Polish ✅
- [x] CLI via `typer`: `uv run main.py --lang zh --format srt`
- [x] Auto-detect single video from MinIO `video-input` (error if 0 or 2+)
- [x] LM Studio health check with `LM_STUDIO_API_KEY` Bearer token
- [x] `setup.sh`: auto-install `uv` if missing, then `uv sync`
- [x] Phase labels in terminal output (Phase 1 / Phase 2)

### Phase 6 — Pending
- [ ] Batch video support (process multiple videos sequentially)
- [ ] Unit + integration tests (`pytest`)
- [ ] Progress bar per node

---

## Translation Strategy

Segments are translated in **batches of 10** sent **4 at a time** in parallel:

```
All segments split into batches of 10
      │
      ├── Batch 0 (seg 0–9)  ─┐
      ├── Batch 1 (seg 10–19) ├─ dispatched concurrently (max 4 workers)
      ├── Batch 2 (seg 20–29) ┘
      └── ...

Each batch request:
  Input:  "Line 1 ||| Line 2 ||| ... ||| Line 10"
  Output: "译文1 ||| 译文2 ||| ... ||| 译文10"
  Split on ||| → mapped back to original timestamps

Fallback: if split count mismatches, retry per-segment
```

---

## Models

### Speech Recognition — Offline Only

| Model | Disk | CPU speed | GPU speed | VRAM |
|-------|------|-----------|-----------|------|
| tiny | ~150 MB | ~30× RT | ~100× RT | <1 GB |
| base | ~290 MB | ~15× RT | ~70× RT | <1 GB |
| small | ~960 MB | ~5× RT | ~35× RT | ~1 GB |
| medium | ~3 GB | ~1.5× RT | ~15× RT | ~2 GB |
| **large-v3** | ~6 GB | ~0.3× RT | ~8× RT | ~4 GB |

Set via `DEVICE=gpu` + `WHISPER_MODEL_SIZE=large-v3` in `.env`.

### Translation — LM Studio (LAN)

| Model | Size | Notes |
|-------|------|-------|
| **Qwen2.5-7B-Instruct** | ~5 GB | Excellent Chinese ↔ English; recommended |
| Qwen2.5-3B-Instruct | ~2 GB | ~2.5× faster; good quality for subtitles |
| Mistral-7B-Instruct-v0.3 | ~5 GB | Strong multilingual |
| Llama-3.1-8B-Instruct | ~6 GB | Broad language support |
| Gemma-2-9B-Instruct | ~7 GB | High quality; formal register |

---

## Tech Stack

| Layer | Choice | Purpose |
|-------|--------|---------|
| Language | Python 3.12 | Runtime |
| Package manager | `uv` | Dependency + venv management |
| Orchestration | `langgraph` | Two-graph stateful pipeline |
| Audio extraction | `static-ffmpeg` + `ffmpeg-python` | Bundled ffmpeg binary; video → WAV |
| Transcription | `faster-whisper` | Offline CTranslate2 Whisper inference |
| LLM client | `langchain-openai` (`ChatOpenAI`) | LM Studio REST via OpenAI-compat interface |
| HTTP | `httpx` | Health check calls |
| Parallelism | `ThreadPoolExecutor` | Parallel transcription chunks + translation batches |
| Object storage | `minio` Python SDK | S3-compatible MinIO client |
| Caption export | custom SRT + ASS writers | Bilingual subtitle files |
| Data storage | `json`, `csv` (stdlib) | Transcript cache |
| CLI | `typer` | Command-line interface |
| Config | `tomllib` + `python-dotenv` | TOML config + `.env` overrides |
| Logging | stdlib `logging` | Console + `logs/app.log` |
| Testing | `pytest` | Unit + integration tests (pending) |

---

## LM Studio LAN Configuration

```
Settings → Local Server → Start Server
Expose on local network: ON
Port: 1234 (default)
API Keys → generate a token → set as LM_STUDIO_API_KEY in .env
```

`config.toml`:
```toml
[lm_studio]
base_url = "http://192.168.1.100:1234/v1"
model = "qwen2.5-7b-instruct"
timeout = 60
max_retries = 3
```

---

## MinIO Storage Layout

```
MinIO
├── video-input/          ← drop ONE video here before running
│   └── my_video.mp4
└── video-output/
    └── my_video/
        ├── my_video.srt      ← bilingual SRT
        ├── my_video.ass      ← bilingual ASS (styled)
        ├── transcript.json   ← raw segments with timestamps
        └── transcript.csv    ← same as CSV
```

Start MinIO:
```bash
docker compose up -d
# S3 API:      http://localhost:9000
# Web console: http://localhost:9001  (minioadmin / minioadmin)
```

---

## Workflow

```mermaid
flowchart TD
    MI[(MinIO\nvideo-input)] -->|auto-detected| A

    subgraph Graph 1 — Transcription
        A[check_cache]
        A -->|cache hit — transcript.json exists| DONE1([END])
        A -->|cache miss| B[fetch_and_extract\ndownload + ffmpeg → WAV]
        B --> C[chunk_audio\noverlapping chunks]
        C --> D1[transcribe chunk 1\nfaster-whisper]
        C --> D2[transcribe chunk 2\nfaster-whisper]
        C --> D3[transcribe chunk N\nfaster-whisper]
        D1 & D2 & D3 --> E[merge_and_save\ndeduplicate + write JSON/CSV]
        E --> DONE1
    end

    DONE1 -->|transcript.json| F

    subgraph Graph 2 — Translation and Export
        F[load_transcript\nread segments from JSON]
        F --> G[translate_title\nLM Studio LAN]
        F --> H[translate_segments\nbatch 10 segs x 4 parallel]
        G & H --> I[validate_timeline\noverlap and duration check]
        I --> J[export_srt]
        I --> K[export_ass]
        J & K --> L[upload_outputs\npush to MinIO video-output]
    end

    subgraph LAN Infrastructure
        Q[(LM Studio\n192.168.x.x:1234)]
        MO[(MinIO\nvideo-output)]
    end

    H -->|10 segs as ||| delimited block| Q
    G -->|title text| Q
    L -->|SRT, ASS, JSON, CSV| MO

    style Graph 1 — Transcription fill:#f0fdf4,stroke:#16a34a
    style Graph 2 — Translation and Export fill:#f0f4ff,stroke:#6b7280
    style LAN Infrastructure fill:#fff7ed,stroke:#ea580c
```

---

## Directory Structure

```
video-caption-assistant/
├── main.py                       # CLI entry — runs Graph 1 then Graph 2
├── pyproject.toml
├── .python-version               # Python 3.12
├── config.toml                   # LM Studio, MinIO, Whisper, pipeline config
├── .env.example                  # env var template
├── docker-compose.yml            # MinIO service
├── setup.sh                      # install uv + uv sync
├── road_map/
│   └── ROADMAP.md
└── src/
    └── video_caption/
        ├── config.py             # load_config — TOML + env overrides
        ├── state.py              # CaptionState TypedDict
        ├── logger.py             # setup_logging + get_logger
        ├── graph.py              # build_transcription_graph + build_translation_graph
        ├── nodes/
        │   ├── cache_check.py    # check if transcript.json exists → skip transcription
        │   ├── extractor.py      # download from MinIO + ffmpeg → WAV
        │   ├── chunker.py        # split WAV into overlapping chunks
        │   ├── transcriber.py    # parallel faster-whisper inference
        │   ├── assembler.py      # deduplicate + merge + save JSON/CSV
        │   ├── transcript_loader.py  # Graph 2 entry — load segments from JSON
        │   ├── translator.py     # batch(10) + parallel(4) LM Studio translation
        │   ├── validator.py      # timeline overlap + duration fix
        │   ├── title.py          # translate title via LM Studio
        │   ├── uploader.py       # upload outputs to MinIO video-output
        │   └── exporters/
        │       ├── srt.py        # bilingual SRT writer
        │       └── ass.py        # bilingual ASS writer (Original + Translated styles)
        └── clients/
            ├── lm_studio.py      # ChatOpenAI wrapper for LM Studio REST
            └── minio_client.py   # MinIO upload / download / bucket init
```
