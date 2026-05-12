"""
CogniStream — Central Configuration

All tunable parameters live here. Values are read from environment variables
with sensible defaults for edge deployment (2-4 CPU cores, 4-6 GB RAM).
"""

import os
import platform
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass

from dotenv import load_dotenv

from dotenv import load_dotenv

# Load .env file from project root (if it exists)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

# ──────────────────────────────────────────────
# Path roots
# ──────────────────────────────────────────────
_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Load local .env for non-Docker runs (uvicorn from workspace root).
load_dotenv(_DEFAULT_PROJECT_ROOT / ".env", override=False)

PROJECT_ROOT = Path(os.getenv("COGNISTREAM_ROOT", _DEFAULT_PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = DATA_DIR / "logs"

VIDEO_DIR = DATA_DIR / "videos"
FRAME_DIR = DATA_DIR / "frames"
AUDIO_DIR = DATA_DIR / "audio"
GRAPH_DIR = DATA_DIR / "graphs"
DB_DIR = DATA_DIR / "db"

SQLITE_PATH = DB_DIR / "cognistream.db"
CHROMA_DIR = DB_DIR / "chroma"

# Ensure all directories exist at import time
for _d in (VIDEO_DIR, FRAME_DIR, AUDIO_DIR, GRAPH_DIR, DB_DIR, CHROMA_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Video ingestion
# ──────────────────────────────────────────────
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
MAX_VIDEO_SIZE_MB = int(os.getenv("MAX_VIDEO_SIZE_MB", "2048"))

# Shot detection: histogram-difference threshold (0-1 scale, lower = more sensitive)
SHOT_THRESHOLD = float(os.getenv("SHOT_THRESHOLD", "0.45"))
# Minimum segment length in frames (prevents micro-segments from noise)
MIN_SEGMENT_FRAMES = int(os.getenv("MIN_SEGMENT_FRAMES", "15"))

# Frame sampling (Phase 1 conservative reduction defaults)
MAX_KEYFRAMES_PER_VIDEO = int(os.getenv("MAX_KEYFRAMES", "140"))
MIN_KEYFRAMES_PER_SEGMENT = 1
MAX_KEYFRAMES_PER_SEGMENT = int(os.getenv("MAX_KEYFRAMES_PER_SEGMENT", "10"))
KEYFRAME_IMAGE_FORMAT = "jpg"
KEYFRAME_JPEG_QUALITY = 85

# Audio extraction
AUDIO_SAMPLE_RATE = 16000  # 16 kHz mono — required by Whisper models
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")

# ──────────────────────────────────────────────
# Ollama (Visual Narrative Engine)
# ──────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "moondream")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
# Local best-performing strategy from Phase 2.5 benchmarking on Windows CPU-only:
# keep backend VLM workers low and raise Ollama request parallelism.
# Note: OLLAMA_NUM_PARALLEL is consumed by `ollama serve` process, not backend.
LOCAL_OLLAMA_NUM_PARALLEL_RECOMMENDED = int(os.getenv("LOCAL_OLLAMA_NUM_PARALLEL_RECOMMENDED", "4"))

# ──────────────────────────────────────────────
# Faster-Whisper (Audio Transcriber)
# ──────────────────────────────────────────────
# Whisper model: "small" (fast, 2GB GPU), "large-v3-turbo" (best quality, 6GB GPU)
# When using large-v3-turbo, the VLM is unloaded from GPU first to free VRAM.
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3-turbo")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

# ──────────────────────────────────────────────
# Embeddings (Multimodal Fusion)
# ──────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # local model dimension

# SigLIP visual embedding — embeds frames directly into a text-searchable vector space.
# Enables visual similarity search without VLM captioning (much faster).
# Set to empty string to disable (avoids dimension mismatch with text embeddings).
SIGLIP_MODEL = os.getenv("SIGLIP_MODEL", "google/siglip2-base-patch16-224")
SIGLIP_DIM = 768
# Easy toggle: set SIGLIP_ENABLED=false in .env to skip visual embeddings altogether
SIGLIP_ENABLED = os.getenv("SIGLIP_ENABLED", "true").lower() in {"1", "true", "yes", "on"}

# When NVIDIA is enabled, embeddings are 1024-dim (NV-EmbedQA-E5-V5).
# ChromaDB collections are dimension-locked, so we use a separate
# collection name to avoid conflicts when switching providers.
NVIDIA_EMBEDDING_DIM = 1024

# ──────────────────────────────────────────────
# ChromaDB
# ──────────────────────────────────────────────
# Collection name auto-switches when NVIDIA is enabled to avoid dimension conflicts.
# Local embeddings (384-dim) and NVIDIA embeddings (1024-dim) can't share a collection.
CHROMA_COLLECTION = os.getenv(
    "CHROMA_COLLECTION",
    "cognistream_nvidia" if os.getenv("NVIDIA_API_KEY", "") else "cognistream_segments",
)
# When running in Docker, ChromaDB is a separate service.
# Set CHROMA_HOST to enable HTTP client mode instead of embedded.
CHROMA_HOST = os.getenv("CHROMA_HOST", "")  # empty = embedded/persistent mode
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

# ──────────────────────────────────────────────
# Multimodal fusion
# ──────────────────────────────────────────────
# Temporal overlap window (seconds) for merging visual captions with transcripts.
FUSION_WINDOW_SEC = float(os.getenv("FUSION_WINDOW_SEC", "2.0"))

# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────
DEFAULT_TOP_K = 10

# Multi-vector retrieval weights — how much each modality contributes to
# the final search score.  Set to 0 to disable a modality.
RETRIEVAL_WEIGHT_TEXT = float(os.getenv("RETRIEVAL_WEIGHT_TEXT", "0.5"))
RETRIEVAL_WEIGHT_VISUAL = float(os.getenv("RETRIEVAL_WEIGHT_VISUAL", "0.3"))
RETRIEVAL_WEIGHT_AUDIO = float(os.getenv("RETRIEVAL_WEIGHT_AUDIO", "0.2"))

# ──────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────
# Only one video processes at a time on edge hardware.
# Additional submissions are rejected while processing.
MAX_CONCURRENT_JOBS = 1

# Processing mode: "auto" (default) = picks best strategy per model
#   moondream → 4-pass quality (small model, needs focused prompts)
#   gemma3n/gemma4/qwen → single-pass fast (larger model, handles structured output)
#   nvidia cloud → single-pass fast
# Can also be forced to "quality" (always 4-pass) or "fast" (always single-pass).
PIPELINE_MODE = os.getenv("PIPELINE_MODE", "auto")

# Known small models that need 4-pass quality mode (can't follow combined prompts)
_SMALL_VLMS = {"moondream", "moondream2", "moondream:latest"}

# Number of parallel workers for VLM frame analysis.
# Local Ollama strategy default is 1 worker (np4_w1 profile).
# NVIDIA cloud: set to 4-8 for concurrent API calls.
# Use explicit default of 1 locally; override with env for experiments.
VLM_WORKERS = int(os.getenv("VLM_WORKERS", "1"))

# Number of parallel workers for shot detection (splits video into chunks).
SHOT_DETECTION_WORKERS = int(os.getenv("SHOT_DETECTION_WORKERS", "2"))

# Enable adaptive worker sizing based on detected hardware.
AUTO_TUNE_PIPELINE = os.getenv("AUTO_TUNE_PIPELINE", "1").strip().lower() in {
    "1", "true", "yes", "on"
}

# Drop near-duplicate keyframes before VLM to reduce local inference cost.
KEYFRAME_NOVELTY_FILTER = os.getenv("KEYFRAME_NOVELTY_FILTER", "1").strip().lower() in {
    "1", "true", "yes", "on"
}
# Mean absolute grayscale difference threshold (0-255) between kept frames.
KEYFRAME_NOVELTY_DIFF_THRESHOLD = float(os.getenv("KEYFRAME_NOVELTY_DIFF_THRESHOLD", "12.0"))
# Force-keep a frame after this many consecutive skips to preserve coverage.
KEYFRAME_NOVELTY_MAX_SKIP = int(os.getenv("KEYFRAME_NOVELTY_MAX_SKIP", "10"))
# Never reduce below this many frames in a video during novelty filtering.
KEYFRAME_NOVELTY_MIN_KEEP = int(os.getenv("KEYFRAME_NOVELTY_MIN_KEEP", "32"))

# Phase 2 semantic reuse cache (conservative, in-run only).
# Reuses captions for near-identical frames after novelty filtering.
KEYFRAME_SEMANTIC_REUSE = os.getenv("KEYFRAME_SEMANTIC_REUSE", "1").strip().lower() in {
    "1", "true", "yes", "on"
}
# Signature diff threshold (0-255) for semantic reuse candidates.
KEYFRAME_SEMANTIC_DIFF_THRESHOLD = float(os.getenv("KEYFRAME_SEMANTIC_DIFF_THRESHOLD", "20.0"))
# Stricter threshold for exact/near-duplicate reuse tracking.
KEYFRAME_SEMANTIC_EXACT_THRESHOLD = float(os.getenv("KEYFRAME_SEMANTIC_EXACT_THRESHOLD", "1.5"))
# Compare only against the most recent representative leaders.
KEYFRAME_SEMANTIC_LOOKBACK = int(os.getenv("KEYFRAME_SEMANTIC_LOOKBACK", "10"))
# Maximum frame-index gap allowed for reuse to limit cross-scene leakage.
KEYFRAME_SEMANTIC_MAX_FRAME_GAP = int(os.getenv("KEYFRAME_SEMANTIC_MAX_FRAME_GAP", "180"))

# Streaming / live-video chunk size in seconds
STREAM_CHUNK_SEC = int(os.getenv("STREAM_CHUNK_SEC", "30"))

# Live feed segment TTL — auto-delete segments older than this (hours).
# 0 = keep forever (default for file-based processing).
LIVE_SEGMENT_TTL_HOURS = int(os.getenv("LIVE_SEGMENT_TTL_HOURS", "24"))

# ──────────────────────────────────────────────
# NVIDIA NIM Cloud (optional — set API key to enable)
# When enabled, NVIDIA models are used for higher quality.
# When disabled (default), local models (Ollama/Whisper) are used.
# ──────────────────────────────────────────────
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# Which NVIDIA models to use (only active when NVIDIA_API_KEY is set)
NVIDIA_VLM_MODEL = os.getenv("NVIDIA_VLM_MODEL", "meta/llama-3.2-11b-vision-instruct")
NVIDIA_EMBED_MODEL = os.getenv("NVIDIA_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
NVIDIA_ASR_MODEL = os.getenv("NVIDIA_ASR_MODEL", "nvidia/parakeet-ctc-1_1b-asr")
NVIDIA_ASR_FUNCTION_ID = os.getenv("NVIDIA_ASR_FUNCTION_ID", "1598d209-5e27-4d3c-8079-4751568b1081")
NVIDIA_CLIP_MODEL = os.getenv("NVIDIA_CLIP_MODEL", "nvidia/nvclip")
NVIDIA_GROUNDING_MODEL = os.getenv("NVIDIA_GROUNDING_MODEL", "nvidia/nv-grounding-dino")
NVIDIA_GROUNDING_URL = os.getenv("NVIDIA_GROUNDING_URL", "https://ai.api.nvidia.com/v1/cv/nvidia/nv-grounding-dino")

def is_nvidia_enabled() -> bool:
    """True if NVIDIA cloud mode is active (API key provided)."""
    return bool(NVIDIA_API_KEY)


@dataclass(frozen=True)
class HardwareProfile:
    cpu_cores: int
    total_ram_gb: float
    has_cuda: bool
    cuda_vram_gb: float
    platform_name: str


def _detect_total_ram_gb() -> float:
    """Best-effort RAM detection without hard dependency on psutil."""
    # Prefer psutil if present.
    try:
        import psutil  # type: ignore

        return round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        pass

    # Windows fallback via ctypes.
    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
            return round(mem.ullTotalPhys / (1024**3), 2)
        except Exception:
            pass

    # Conservative fallback when detection fails.
    return 8.0


def _detect_cuda_profile() -> tuple[bool, float]:
    """Best-effort CUDA + VRAM detection."""
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return False, 0.0

        total = 0
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            total += int(getattr(props, "total_memory", 0))
        return True, round(total / (1024**3), 2)
    except Exception:
        return False, 0.0


@lru_cache(maxsize=1)
def get_hardware_profile() -> HardwareProfile:
    """Detect host capabilities once and reuse for worker sizing."""
    has_cuda, cuda_vram = _detect_cuda_profile()
    return HardwareProfile(
        cpu_cores=max(1, os.cpu_count() or 1),
        total_ram_gb=_detect_total_ram_gb(),
        has_cuda=has_cuda,
        cuda_vram_gb=cuda_vram,
        platform_name=platform.system().lower(),
    )


def resolve_shot_detection_workers() -> int:
    """Return safe worker count for shot detection on this hardware."""
    if SHOT_DETECTION_WORKERS > 0 or not AUTO_TUNE_PIPELINE:
        return max(1, SHOT_DETECTION_WORKERS)

    hw = get_hardware_profile()
    if hw.total_ram_gb < 6 or hw.cpu_cores <= 2:
        return 1
    if hw.total_ram_gb < 12 or hw.cpu_cores <= 4:
        return 2
    return min(6, max(2, hw.cpu_cores // 2))


def resolve_vlm_workers(cloud_mode: bool) -> int:
    """Return safe VLM worker count for local/cloud inference."""
    if VLM_WORKERS > 0 or not AUTO_TUNE_PIPELINE:
        return max(1, VLM_WORKERS)

    hw = get_hardware_profile()

    # Cloud mode is mostly network-bound and can benefit from moderate fan-out.
    if cloud_mode:
        if hw.total_ram_gb < 6:
            return 1
        return min(8, max(2, hw.cpu_cores // 2))

    # Local Ollama tends to be single-request bottleneck on CPU systems.
    if not hw.has_cuda:
        return 1

    # CUDA local setup can sometimes sustain light concurrency.
    if hw.cuda_vram_gb >= 12 and hw.total_ram_gb >= 16 and hw.cpu_cores >= 8:
        return 2
    return 1


def resolve_pipeline_stage_workers() -> int:
    """Workers for orchestrator's parallel stage executor."""
    if not AUTO_TUNE_PIPELINE:
        return 2
    hw = get_hardware_profile()
    if hw.total_ram_gb < 6 or hw.cpu_cores <= 2:
        return 1
    return 2


def get_pipeline_tuning_summary() -> dict[str, object]:
    """Return detected hardware and resolved worker settings for logging/UI."""
    hw = get_hardware_profile()
    cloud_mode = is_nvidia_enabled()
    return {
        "auto_tune": AUTO_TUNE_PIPELINE,
        "local_strategy_profile": "np4_w1",
        "ollama_num_parallel_recommended": LOCAL_OLLAMA_NUM_PARALLEL_RECOMMENDED,
        "platform": hw.platform_name,
        "cpu_cores": hw.cpu_cores,
        "ram_gb": hw.total_ram_gb,
        "cuda": hw.has_cuda,
        "cuda_vram_gb": hw.cuda_vram_gb,
        "nvidia_cloud_mode": cloud_mode,
        "resolved_workers": {
            "shot_detection": resolve_shot_detection_workers(),
            "vlm": resolve_vlm_workers(cloud_mode=cloud_mode),
            "pipeline_stage": resolve_pipeline_stage_workers(),
        },
        "env_overrides": {
            "SHOT_DETECTION_WORKERS": SHOT_DETECTION_WORKERS,
            "VLM_WORKERS": VLM_WORKERS,
        },
    }

# ──────────────────────────────────────────────
# Security
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# Webhooks — notify external systems on events
# ──────────────────────────────────────────────
# Comma-separated list of URLs to POST event payloads to.
# Empty = disabled.  Events: video_processed, event_detected, live_chunk_ready
WEBHOOK_URLS = [u.strip() for u in os.getenv("WEBHOOK_URLS", "").split(",") if u.strip()]

# Optional API key authentication.  Set to enable — requests must include
# header "X-API-Key: <key>".  Empty = no auth (default for local dev).
# Supports multiple comma-separated keys for multi-user setups.
API_KEY = os.getenv("COGNISTREAM_API_KEY", "")
API_KEYS: set[str] = {k.strip() for k in API_KEY.split(",") if k.strip()}

# Rate limiting (requests per minute per IP)
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "120"))

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "1").strip().lower() in {"1", "true", "yes", "on"}
LOG_FILE_LEVEL = os.getenv("LOG_FILE_LEVEL", "DEBUG")
