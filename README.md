# CogniStream

Recorded Demo and further insights : https://drive.google.com/drive/folders/1oyUfRPiHtsRozdE1VZ-S2bfkCsBHaDXs 

Privacy-preserving multimodal video retrieval engine. Search video content with natural language queries like *"Show me when the red car arrived"* — runs fully offline on edge hardware, with optional NVIDIA cloud acceleration.

## Features

- **Natural language video search** — query with text, find matching video moments
- **Live video feeds** — RTSP cameras, webcams, phone browsers, screen share
- **Multi-model pipeline** — VLM scene analysis + speech transcription + visual embeddings
- **Knowledge graph** — entity relationships + temporal event detection
- **Real-time processing** — streaming chunk pipeline, segments searchable within seconds
- **Multi-vector retrieval** — text + visual + audio embeddings with configurable weights
- **NVIDIA NIM cloud** — optional higher-quality models via API (Llama-3.2-11B, NV-Embed, NVCLIP)
- **Dashboard** — stats panel, live feed monitor, video reports
- **Security** — API key auth, rate limiting, multi-user support

## How It Works

Videos are processed through a multi-stage pipeline with parallel execution:

```
Video → Shot Detection ──→ Frame Sampling ──┐
        Audio Extraction ───────────────────┤ (parallel)
                                            ├→ VLM Analysis (N workers) ──┐
                                            └→ Whisper STT ──────────────┤ (parallel or sequential GPU swap)
                                                                         ├→ SigLIP Frame Embeddings
                                                                         ├→ Multimodal Fusion
                                                                         ├→ Text Embedding
                                                                         ├→ Knowledge Graph + Events
                                                                         └→ ChromaDB Storage
```

Search uses a 4-stage retrieval pipeline: embed query → multi-vector search (text + visual) → temporal re-ranking → hybrid re-ranking.

## Models

| Component | Local (Default) | NVIDIA Cloud (Optional) |
|-----------|----------------|------------------------|
| **VLM** | moondream (1.8B, GPU) | Llama-3.2-11B-Vision |
| **STT** | Whisper large-v3-turbo (GPU) | Parakeet ASR |
| **Text Embeddings** | all-MiniLM-L6-v2 (384-dim) | NV-EmbedQA-E5 (1024-dim) |
| **Visual Embeddings** | SigLIP 2 (768-dim) | NVCLIP (1024-dim) |
| **Object Detection** | — | NV-Grounding-DINO |

The pipeline auto-detects model capabilities and adjusts (4-pass prompts for small VLMs, single-pass for larger ones).

## Quick Start

### Prerequisites

- Python 3.11+, Node.js 20+, FFmpeg
- [Ollama](https://ollama.com/) installed
- NVIDIA GPU recommended (RTX 3050+ with 6GB VRAM)

### Setup

```bash
# Clone
git clone https://github.com/Zayed024/cognistream.git && cd cognistream

# Install Python deps
pip install -r requirements.txt

# Pull VLM model
ollama pull moondream

# Install frontend
cd frontend && npm install && cd ..

# Copy and edit config
cp .env.example .env
```

### Run

```bash
# Terminal 1: Ollama
ollama serve

# Terminal 2: Backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Terminal 3: Frontend
cd frontend && npm run dev
```

Open http://localhost:3000

### Docker

```bash
cd docker && docker compose up --build
```

## API Reference

### Core

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest-video` | Upload video (streaming, max 2 GB) |
| `POST` | `/process-video` | Process video (`standard` or `streaming` mode) |
| `POST` | `/search` | Natural language search |
| `GET` | `/videos` | List all videos |
| `GET` | `/video/{id}` | Video metadata + status |
| `GET` | `/video/{id}/stream` | Stream video file |
| `GET` | `/video/{id}/report` | Full video summary report |
| `DELETE` | `/video/{id}` | Delete video + all data |

### Live Video

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/live/start` | Start RTSP/webcam/HTTP feed |
| `POST` | `/live/stop` | Stop a live feed |
| `GET` | `/live/status` | List active feeds |
| `WS` | `/ws/live/{id}` | Real-time events + live search |
| `POST` | `/live/browser-chunk` | Upload browser camera chunk |
| `POST` | `/live/browser-stop` | Finalize browser feed |

### Features

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/similar` | Find similar segments |
| `POST` | `/video/{id}/clip` | Export video clip (FFmpeg) |
| `GET` | `/video/{id}/graph` | Knowledge graph (nodes + edges) |
| `GET` | `/video/{id}/events` | Detected events timeline |
| `GET` | `/video/{id}/thumbnail` | Video thumbnail |
| `POST` | `/annotations` | Create annotation/bookmark |
| `GET` | `/video/{id}/annotations` | List annotations |
| `POST` | `/process-batch` | Queue multiple videos |
| `GET` | `/stats` | Dashboard analytics |
| `GET` | `/health` | Health check + provider status |

## Project Structure

```
cognistream/
├── backend/
│   ├── main.py                 # FastAPI app + middleware (auth, rate limit)
│   ├── config.py               # Central config + hardware auto-tuning
│   ├── api/router.py           # 32 REST + WebSocket endpoints
│   ├── ingestion/              # Video loader, shot detector, frame sampler
│   ├── visual/                 # VLM runner, caption processor, SigLIP embedder
│   ├── audio/                  # FFmpeg extractor, Whisper runner (GPU)
│   ├── fusion/                 # Multimodal embedder (fusion + encoding)
│   ├── knowledge/              # Knowledge graph, event detector
│   ├── retrieval/              # Query engine (multi-vector + temporal rerank)
│   ├── pipeline/               # Orchestrator + streaming pipeline
│   ├── providers/nvidia.py     # NVIDIA NIM cloud (NVCLIP, NV-Embed, VLM, ASR)
│   ├── webhooks.py             # Event notifications
│   ├── db/                     # SQLite + ChromaDB wrappers
│   └── tests/                  # 331 tests (21 test files)
├── frontend/
│   └── src/
│       ├── components/         # 10 components + 3 test files
│       │   ├── SearchBar, VideoPlayer, ResultsPanel, TimelineMarkers
│       │   ├── VideoList, VideoUpload, KnowledgeGraph, EventTimeline
│       │   ├── LiveView (URL/camera/screen), StatsPanel
│       │   └── __tests__/      # Vitest + React Testing Library
│       ├── hooks/              # useSearch, useVideo
│       ├── api/client.ts       # Typed API client (fetch-based)
│       └── types/index.ts      # TypeScript interfaces
├── docker/                     # Docker Compose (4 services)
├── scripts/                    # Model pull + benchmark scripts
├── requirements.txt
└── .env.example
```

### Phase 0 Baseline Runner

Run repeatable baseline measurements across one or more videos. This will:
- trigger processing repeatedly
- capture per-run benchmark payloads
- compute retrieval diversity per run
- aggregate elapsed time, estimated VLM frames/min, and process RSS peak

Run with explicit video IDs:

```bash
python scripts/phase0_baseline.py --video-id <VIDEO_ID> --runs 3
```

Or auto-discover videos from the API:

```bash
python scripts/phase0_baseline.py --discover-videos --max-videos 2 --runs 3
```

Outputs:
- `reports/phase0/raw/*` per-run JSON payloads
- `reports/phase0/phase0_summary.json` aggregate machine-readable summary
- `reports/phase0/phase0_summary.md` quick human-readable summary

### Phase 1 Worker Saturation Benchmark

Compare local VLM throughput at worker counts 1, 2, and 4 on a fixed keyframe subset:

```bash
python scripts/phase1_worker_saturation.py --video-id <VIDEO_ID> --sample-size 24
```

Outputs:
- `reports/phase1/worker_saturation_<VIDEO_ID>.json` with per-worker elapsed time, frames/min, and novelty stats

### Phase 2 Semantic Reuse Benchmark

Run a single end-to-end pass and extract semantic reuse hit metrics:

```bash
python scripts/phase2_reuse_benchmark.py --video-id <VIDEO_ID>
```

Outputs:
- `reports/phase2/<VIDEO_ID>/phase0_summary.json` full run payload
- `reports/phase2/<VIDEO_ID>/phase2_summary.json` reuse hit ratio and reuse counters
## Configuration

All settings via environment variables. See `.env.example` for the full list.

### Key Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `moondream` | Vision language model (1.8B, fits 6GB VRAM) |
| `WHISPER_MODEL_SIZE` | `large-v3-turbo` | Whisper model (GPU accelerated) |
| `WHISPER_DEVICE` | `cuda` | `cuda` or `cpu` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Text embedding model (384-dim) |
| `PIPELINE_MODE` | `auto` | `auto`, `quality` (4-pass), `fast` (single-pass) |
| `VLM_WORKERS` | `0` | 0=auto, 1=sequential, N=concurrent |
| `NVIDIA_API_KEY` | `""` | Set to enable NVIDIA cloud models |
| `COGNISTREAM_API_KEY` | `""` | API key auth (comma-separated for multi-user) |
| `RATE_LIMIT_RPM` | `120` | Requests per minute per IP |

### GPU Memory Management

The pipeline automatically swaps GPU memory between VLM and Whisper when using large models:
- **Small Whisper** (small, 2GB): VLM + Whisper run in parallel
- **Large Whisper** (large-v3-turbo, 6GB): VLM runs first → unloads → Whisper gets full VRAM

### NVIDIA Cloud Mode

Set `NVIDIA_API_KEY` in `.env` to enable. No downloads needed — API-based. Falls back to local models on failure. Get a free key at https://build.nvidia.com/

## Performance

Benchmarked on RTX 3050 Laptop (6 GB VRAM, 16 GB RAM) with a 3.8-min video (156 keyframes).

### VLM Frame Analysis

| Mode | Per Frame | 156 Keyframes | Caption Quality |
|------|-----------|---------------|-----------------|
| **Moondream (GPU)** | **0.6s** | **~1.6 min** | Good — reads text, identifies people/objects |
| Moondream (CPU) | 17.6s | ~46 min | Same quality, 29x slower |
| NVIDIA cloud (1 worker) | 3.1s | ~8 min | Excellent — detailed scenes, spatial reasoning |
| **NVIDIA cloud (4 workers)** | **0.8s effective** | **~2 min** | Excellent (Llama-3.2-11B-Vision) |

### Whisper Speech-to-Text (226s audio)

| Mode | Time | Speedup |
|------|------|---------|
| **GPU (large-v3-turbo, float16)** | **9.7s** | **7.2x** |
| CPU (small, int8) | 70s | baseline |

### Full Pipeline

| Configuration | Total Time |
|---------------|-----------|
| **Local GPU (moondream + whisper-turbo)** | **~2 min** |
| Local CPU only | ~47 min |
| NVIDIA cloud VLM + local whisper GPU | ~2.5 min |

## Tests

```bash
# Backend (342 tests)
python -m pytest --tb=short -q

# Frontend (16 tests)
cd frontend && npx vitest run
```

## Tech Stack

**Backend**: Python 3.11, FastAPI, OpenCV, FFmpeg, Ollama, Faster-Whisper (CUDA), SentenceTransformers, SigLIP 2, ChromaDB, NetworkX, spaCy, httpx

**Frontend**: React 19, TypeScript, Vite 7, Vitest

**Infrastructure**: Docker Compose, Ollama, nginx, NVIDIA NIM (optional)

## License

MIT
