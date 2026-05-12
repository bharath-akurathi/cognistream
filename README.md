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
| **Text Embeddings** | cognistream-embedder (fine-tuned, 384-dim) | NV-EmbedQA-E5 (1024-dim) |
| **Visual Embeddings** | SigLIP 2 (768-dim) | NVCLIP (1024-dim) |
| **CV Pre-filter** | YOLOv8n via `ultralytics` (optional, ~6 MB) | NV-Grounding-DINO |

The pipeline auto-detects model capabilities and adjusts (4-pass prompts for small VLMs, single-pass for larger ones).

## Quick Start

CogniStream supports three deployment modes. Pick the one that matches your hardware:

| Mode | Best for | GPU? | VLM | STT |
|---|---|---|---|---|
| **Host with GPU** | Dev machines / workstations with NVIDIA GPU | ✅ CUDA | Local Ollama (GPU) or NVIDIA cloud | Faster-Whisper (GPU) |
| **Host CPU-only** | Laptops without NVIDIA, CI runners | ❌ | NVIDIA cloud (recommended) or Ollama (CPU, slow) | Faster-Whisper (CPU) or NVIDIA cloud |
| **Docker** | Servers / edge boxes / reproducible deploys | ❌ (CPU image) | NVIDIA cloud or sibling Ollama container | Whisper (CPU, base model) |

### Option 1: Host with GPU (recommended for dev)

**Prerequisites**: Python 3.11+, Node.js 20+, FFmpeg, NVIDIA GPU + CUDA 12 drivers, [Ollama](https://ollama.com/)

```bash
# Clone
git clone https://github.com/Zayed024/cognistream.git && cd cognistream

# Install Python deps (will pull CUDA torch)
pip install -r requirements.txt

# Optional: install YOLO for the CV pre-filter (~30% faster pipeline)
pip install ultralytics

# Pull the local VLM
ollama pull moondream

# Frontend deps
cd frontend && npm install && cd ..

# Config — set NVIDIA_API_KEY if you want cloud VLM/embeddings
cp .env.example .env
```

Run with three terminals:

```bash
# Terminal 1
ollama serve

# Terminal 2
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Terminal 3
cd frontend && npm run dev
```

Open http://localhost:3000

### Option 2: Host CPU-only

Same as Option 1 but skip the Ollama install and force CPU Whisper:

```bash
# In .env:
WHISPER_DEVICE=cpu
WHISPER_MODEL_SIZE=base   # or "tiny" if RAM-constrained
NVIDIA_API_KEY=nvapi-...  # strongly recommended — local Ollama on CPU is very slow
```

Then run just two terminals (skip `ollama serve`):

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
cd frontend && npm run dev
```

The pipeline will use NVIDIA cloud for the VLM and embeddings, and Faster-Whisper on CPU for transcription.

### Option 3: Docker (CPU-only)

The Docker image is intentionally CPU-only — no `nvidia-cu12-*` packages, no CUDA runtime. Image size is ~3.5 GB. If you need GPU acceleration, use Option 1 instead.

```bash
cd docker
docker compose up -d --build
```

This brings up three services: `backend`, `frontend`, `chromadb`. Ollama is disabled by default (CPU-only container is too slow). Set `NVIDIA_API_KEY` in `docker-compose.yml` so the backend uses NVIDIA cloud for the VLM. To re-enable Ollama as a sibling container, uncomment the `ollama` service block in `docker/docker-compose.yml`.

Open http://localhost:3000.

To rebuild after pulling new code: `docker compose up -d --build backend`. To stop: `docker compose down`.

#### Why is the Docker image CPU-only?
The `python:3.11-slim` base image has no CUDA runtime, the compose file doesn't request GPU devices, and CUDA libraries can't talk to a GPU without the host's NVIDIA Container Toolkit installed. Bundling them anyway adds 6+ GB of unusable bloat — which is exactly the trap the previous 14.6 GB image fell into. If you need GPU inside Docker, you'd need a different base image (`nvidia/cuda:12.x-runtime`), GPU torch (`--index-url https://download.pytorch.org/whl/cu121`), and a `devices: [{ driver: nvidia }]` block in `docker-compose.yml`. Happy to add that as an opt-in `Dockerfile.backend.gpu` if anyone needs it.

## API Reference

### Core

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest-video` | Upload video (streaming, max 2 GB) |
| `POST` | `/process-video` | Process video (`standard` or `streaming` mode) |
| `POST` | `/search` | Natural language search (see Search below) |
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

### Search

The `/search` endpoint accepts a JSON body with these parameters:

```json
{
  "query": "person wearing sunglasses",
  "top_k": 10,
  "search_mode": "hybrid",
  "min_score": 0.0,
  "agentic": false
}
```

| Parameter | Default | Description |
|---|---|---|
| `query` | *(required)* | Natural language text, or `"quoted phrase"` for exact transcript match |
| `top_k` | `10` | Max results to return |
| `search_mode` | `hybrid` | `visual` (vector only), `speech` (FTS5 transcript only), `hybrid` (both) |
| `min_score` | `0.0` | Minimum score threshold — results below this are filtered out. Returns empty instead of low-confidence noise. |
| `agentic` | `false` | Enable query decomposition + VLM reflection rerank (slower, better for compound queries) |
| `video_id` | `null` | Scope search to one video |
| `source_filter` | `null` | Restrict to source type: `visual`, `audio`, `fused`, `speech`, `event` |

**Search modes:**

- **`hybrid`** (default) — runs both vector similarity search (visual/caption embeddings) and FTS5 full-text search over transcripts, then merges results. Speech matches get injected with highlighted snippets.
- **`visual`** — vector search only. Skips transcript matching. Fastest mode.
- **`speech`** — FTS5 transcript search only. Skips visual embeddings. Returns transcript segments with `<mark>` highlighted snippets.

**Quoted-phrase exact match:**

Wrapping the query in quotes (`"you know the rules"`) routes directly to FTS5 for verbatim transcript search, bypassing the vector pipeline entirely. Supports both straight quotes and smart quotes.

**Response includes:**

| Field | Description |
|---|---|
| `speech_snippet` | FTS5 highlighted match with `<mark>` tags (e.g. `You know the <mark>rules</mark>`) |
| `related_count` | Number of adjacent frames collapsed into this moment (0 = standalone) |
| `result_count` | Total results returned (top-level field) |

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
│   └── tests/                  # 342 tests (21 test files)
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

Benchmarked on RTX 3050 Laptop (6 GB VRAM, 16 GB RAM).

### Full Pipeline (7 standard test videos, NVIDIA cloud VLM)

| Configuration | Total Time | Per video avg | Notes |
|---|---:|---:|---|
| **+ local YOLO CV pre-filter** | **109.7s** | **15.7s** | 7/7 success, 117 segments, 0% empty |
| Apr 6 baseline (no CV filter) | 157.6s | 19.7s | 8/8 success, 80 segments, 0% empty |
| All-MiniLM baseline (no fine-tuning) | 411.0s | 51.4s | 8/8 success, 80 segments, 0% empty |

The CV pre-filter (YOLOv8n via `ultralytics`, ~6 MB local model) drops keyframes with no interesting objects before they reach the slow VLM. Cuts average `vlm_sec` per video from 15.91s → **8.59s (-46%)**. Frame drop rate varies by content:

| Video | Keyframes kept | Dropped |
|---|---:|---:|
| `outdoor_nature.mp4` | 11 / 46 | **76%** |
| `lecture_clip.mp4` | 2 / 10 | **80%** |
| `xiph_foreman.mp4` | 7 / 10 | 30% |
| `cooking_demo.mp4` | 9 / 10 | 10% |
| `xiph_bus.mp4` | 8 / 10 | 20% |
| `traffic_cam.mp4`, `xiph_news.mp4` | 10 / 10 | 0% |

Install with `pip install ultralytics` (uncomment in `requirements.txt`). Without it the pipeline still runs — VLM just sees every keyframe.

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

## Fine-tuned Models

### cognistream-embedder (included in repo)

A fine-tuned `all-MiniLM-L6-v2` (384-dim) trained on 1,181 query-passage pairs generated from NVIDIA Llama-3.2-11B-Vision captions via knowledge distillation. Improves retrieval precision for video-specific queries.

- **Location**: `models/cognistream-embedder/` (88 MB, included in repo)
- **Auto-detected**: `config.py` uses it if the directory exists, falls back to `all-MiniLM-L6-v2`
- **Training**: 3 epochs, MultipleNegativesRankingLoss, 11 min on CPU
- **Training data**: 315 NVIDIA-distilled captions → 1,181 synthetic query-passage pairs

### cognistream-moondream-lora (included in repo)

A LoRA adapter (rank 16) for moondream2's Phi-2 text decoder, trained on 315 NVIDIA-distilled captions. Teaches moondream to produce the structured SCENE/OBJECTS/ACTIVITY/ANOMALY format reliably.

- **Location**: `models/cognistream-moondream-lora/` (52 MB, included in repo)
- **Training**: 3 epochs on Colab T4, 2.1 min, loss 1.45 → 0.40
- **Trainable params**: 12.6M (0.88% of 1.4B base)
- **Base model**: `vikhyatk/moondream2` revision `2024-08-26`
- **Load with**:
  ```python
  from peft import PeftModel
  base.text_model = PeftModel.from_pretrained(base.text_model, "models/cognistream-moondream-lora")
  ```

#### Benchmark: 3-Way VLM Comparison (14 real keyframes from standard test videos)

| Model | Empty responses | Follows SCENE/OBJECTS/ACTIVITY/ANOMALY format | Avg speed |
|-------|----------------|----------------------------------------------|-----------|
| **Base moondream (Ollama)** | 0% | **0%** — generates free-form prose, ignores the format | ~1.5s |
| **NVIDIA Llama-3.2-11B (cloud)** | 7% | **93%** — reliably follows the structured format | ~4s |
| **LoRA fine-tuned moondream** | 0%* | **100%*** — learned the format via distillation | ~1.5s |

*Tested on 5 sample images via Colab (the LoRA runs in PyTorch transformers, not Ollama GGUF).

**The key insight**: base moondream generates nice prose but completely ignores the structured format the pipeline expects. That means the parser falls back to storing everything as scene description, losing object extraction, activity parsing, and anomaly detection. NVIDIA's Llama-3.2-11B follows the format naturally because it's a much larger model. The LoRA adapter teaches moondream to produce the same structured output locally — closing the quality gap without needing cloud API calls.

#### Qualitative improvements (5 sample images, base vs LoRA)

| Improvement | Base moondream | LoRA fine-tuned |
|-------------|---------------|-----------------|
| **Anomaly detection** | Never mentions anomalies | Adds "unusual aspect" notes (3/5 images) |
| **Hallucinations** | Invents details (fake flags, festive atmosphere) | Sticks to visible content |
| **Spatial detail** | Generic "desks, chairs" | Notes "map on the right side", "red traffic lights" |
| **Factual accuracy** | "black cutting board" | "large black board with white writing" (correctly identifies it as signage) |

### Knowledge Distillation Pipeline

Re-train or improve the models with your own data:

```bash
# Step 1: Generate training data from NVIDIA cloud VLM (needs API key)
python scripts/finetune/distill.py

# Step 2: Fine-tune embeddings (11 min on CPU, no GPU needed)
python scripts/finetune/train_embeddings.py

# Step 3: Fine-tune moondream via Colab (2 min on T4 GPU)
# Upload scripts/finetune/CogniStream_Moondream_Finetune.ipynb to Colab
```

## Tests

```bash
# Backend (342 tests)
python -m pytest --tb=short -q

# Frontend (16 tests)
cd frontend && npx vitest run

# Benchmark on standard test videos
python scripts/benchmark_test_videos.py --tag my_run
```

## Tech Stack

**Backend**: Python 3.11, FastAPI, OpenCV, FFmpeg, Ollama, Faster-Whisper (CUDA), SentenceTransformers, SigLIP 2, ChromaDB, NetworkX, spaCy, httpx

**Frontend**: React 19, TypeScript, Vite 7, Vitest

**Infrastructure**: Docker Compose, Ollama, nginx, NVIDIA NIM (optional)

**Fine-tuning**: PyTorch, PEFT/LoRA, sentence-transformers, knowledge distillation

## License

MIT
