"""
CogniStream — API Routes

All FastAPI endpoints in a single router.  Kept in one file because
edge deployments have a small API surface — splitting into five files
would add complexity without benefit.

Endpoints:
    POST /ingest-video     Upload a video file
    POST /process-video    Trigger processing pipeline
    POST /search           Natural language search
    GET  /video/{id}       Video metadata & status
    GET  /video/{id}/stream   Stream video file (range requests)
    GET  /video/{id}/frame/{name}  Serve a keyframe image
    GET  /video/{id}/progress      Polling progress
    GET  /video/{id}/progress/stream  SSE progress stream
    GET  /videos           List all videos
    DELETE /video/{id}     Delete a video and all its data
    GET  /health           Liveness check
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import shutil
import subprocess
import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.config import (
    ALLOWED_VIDEO_EXTENSIONS,
    FFMPEG_PATH,
    FRAME_DIR,
    GRAPH_DIR,
    MAX_VIDEO_SIZE_MB,
    VIDEO_DIR,
)
from backend.db.chroma_store import ChromaStore
from backend.db.models import VideoStatus
from backend.db.sqlite import SQLiteDB
from backend.fusion.multimodal_embedder import MultimodalEmbedder
from backend.ingestion.loader import VideoLoadError, VideoLoader
from backend.pipeline.orchestrator import PipelineOrchestrator, PipelineProgress
from backend.pipeline.streaming import LiveEvent, StreamingPipeline
from backend.retrieval.query_engine import QueryEngine

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Progress tracking (in-memory, single-instance edge deployment) ──
_progress_store: dict[str, PipelineProgress] = {}
_progress_events: dict[str, asyncio.Event] = {}
_progress_lock = threading.Lock()
# Captured at startup by init_event_loop() so background threads can
# safely schedule asyncio.Event.set() via call_soon_threadsafe.
_event_loop: asyncio.AbstractEventLoop | None = None


def init_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Store the main event loop reference (called from lifespan handler)."""
    global _event_loop
    _event_loop = loop


def _on_progress(progress: PipelineProgress) -> None:
    """Callback to store progress updates from the orchestrator."""
    with _progress_lock:
        _progress_store[progress.video_id] = progress
        # Signal any waiting SSE clients
        if progress.video_id in _progress_events and _event_loop is not None:
            _event_loop.call_soon_threadsafe(
                _progress_events[progress.video_id].set
            )


def cleanup_progress(video_id: str) -> None:
    """Remove progress data for a completed/failed video."""
    with _progress_lock:
        _progress_store.pop(video_id, None)
        _progress_events.pop(video_id, None)


def _save_benchmark(video_id: str, payload: dict[str, Any]) -> None:
    run = {
        "id": uuid.uuid4().hex,
        "video_id": video_id,
        "captured_at": payload.get("captured_at") or datetime.now(timezone.utc).isoformat(),
        "success": bool(payload.get("success", False)),
        "elapsed_sec": float(payload.get("elapsed_sec", 0.0)),
        "segments_stored": int(payload.get("segments_stored", 0)),
        "events_detected": int(payload.get("events_detected", 0)),
        "stage_timings": payload.get("stage_timings") or {},
        "quality_metrics": payload.get("quality_metrics") or {},
        "warnings": payload.get("warnings") or [],
        "errors": payload.get("errors") or [],
    }
    _db.save_benchmark_run(run)


# ── Batch processing queue ──
_process_queue: deque[str] = deque()
_queue_lock = threading.Lock()


def _queue_worker() -> None:
    """Background thread that processes videos from the queue one at a time."""
    while True:
        with _queue_lock:
            if not _process_queue:
                return
            video_id = _process_queue[0]

        meta = _db.get_video(video_id)
        if meta is not None:
            try:
                _orchestrator.process(meta)
            except Exception:
                logger.exception("Queue processing failed for video %s", video_id)
            finally:
                cleanup_progress(video_id)

        with _queue_lock:
            if _process_queue and _process_queue[0] == video_id:
                _process_queue.popleft()


# ── Shared singletons (created once, reused across requests) ──
_db = SQLiteDB()
_store = ChromaStore()
_embedder = MultimodalEmbedder()
_query_engine = QueryEngine(embedder=_embedder, store=_store)
_orchestrator = PipelineOrchestrator(db=_db, store=_store, on_progress=_on_progress)

# ── Live feed WebSocket management ──
_ws_clients: dict[str, set[WebSocket]] = {}  # video_id → connected clients
_ws_lock = threading.Lock()


def _on_live_event(event: LiveEvent) -> None:
    """Broadcast a live event to all WebSocket clients subscribed to that feed."""
    with _ws_lock:
        clients = _ws_clients.get(event.video_id, set()).copy()
    if not clients or _event_loop is None:
        return

    payload = json.dumps({
        "video_id": event.video_id,
        "event_type": event.event_type,
        "data": event.data,
        "timestamp": event.timestamp,
    })
    for ws in clients:
        try:
            _event_loop.call_soon_threadsafe(
                asyncio.ensure_future,
                ws.send_text(payload),
            )
        except Exception:
            pass  # Client may have disconnected


_streaming_pipeline = StreamingPipeline(
    db=_db, store=_store, on_live_event=_on_live_event,
)
_loader = VideoLoader()

# Maximum allowed top_k to prevent ChromaDB abuse
_MAX_TOP_K = 100

# Chunk size for streaming uploads (1 MB)
_UPLOAD_CHUNK = 1024 * 1024


# ── Request / Response models ──────────────────────────────────

class ProcessRequest(BaseModel):
    video_id: str
    mode: str = Field(default="standard", description="'standard' or 'streaming' (chunked, near-RT)")
    chunk_sec: int = Field(default=30, ge=5, le=120)


class BatchProcessRequest(BaseModel):
    video_ids: list[str]


class SearchRequest(BaseModel):
    query: str
    video_id: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=_MAX_TOP_K)
    source_filter: Optional[str] = None


class SimilarRequest(BaseModel):
    segment_id: str
    top_k: int = Field(default=10, ge=1, le=_MAX_TOP_K)
    video_id: Optional[str] = None


class ClipRequest(BaseModel):
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)


class AnnotationRequest(BaseModel):
    video_id: str
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    label: str
    note: str = ""
    color: str = "#3b82f6"


class LiveStartRequest(BaseModel):
    url: str = Field(description="RTSP/RTMP/HTTP stream URL or webcam index ('0', '1')")
    video_id: str = Field(description="Unique identifier for this live feed")
    chunk_sec: int = Field(default=15, ge=5, le=120)


class LiveStopRequest(BaseModel):
    video_id: str


# ── Browser camera chunk management ──
# Tracks browser-based camera feeds where chunks arrive via HTTP upload
_browser_feeds: dict[str, dict] = {}  # video_id → {chunk_idx, embedder, captions, transcripts}
_browser_feeds_lock = threading.Lock()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Health
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/health")
async def health():
    """Liveness check for Docker health probes + provider status."""
    from backend.providers.nvidia import nvidia
    from backend.config import OLLAMA_MODEL, WHISPER_MODEL_SIZE, EMBEDDING_MODEL

    return {
        "status": "ok",
        "service": "cognistream-backend",
        "providers": {
            "nvidia_cloud": nvidia.available,
            "vlm": "nvidia" if nvidia.available else f"ollama/{OLLAMA_MODEL}",
            "stt": "nvidia/parakeet" if nvidia.available else f"whisper/{WHISPER_MODEL_SIZE}",
            "embeddings": "nvidia/nv-embed" if nvidia.available else EMBEDDING_MODEL,
        },
    }


@router.get("/stats")
async def get_stats():
    """Dashboard analytics — processing stats, model info, system status."""
    from backend.providers.nvidia import nvidia
    from backend.config import (
        OLLAMA_MODEL, WHISPER_MODEL_SIZE, EMBEDDING_MODEL,
        PIPELINE_MODE, LIVE_SEGMENT_TTL_HOURS,
    )

    videos = _db.list_videos()
    total_segments = _store.count()
    live_feeds = _streaming_pipeline.get_live_status()

    status_counts = {}
    for v in videos:
        s = v.status.value
        status_counts[s] = status_counts.get(s, 0) + 1

    total_duration = sum(v.duration_sec or 0 for v in videos)

    return {
        "videos": {
            "total": len(videos),
            "by_status": status_counts,
            "total_duration_sec": round(total_duration, 1),
            "total_duration_human": f"{total_duration / 3600:.1f}h" if total_duration > 3600 else f"{total_duration / 60:.1f}m",
        },
        "segments": {
            "total": total_segments,
        },
        "live_feeds": {
            "active": len([f for f in live_feeds if f.state == "running"]),
            "total": len(live_feeds),
        },
        "config": {
            "pipeline_mode": PIPELINE_MODE,
            "nvidia_cloud": nvidia.available,
            "vlm_model": "nvidia" if nvidia.available else OLLAMA_MODEL,
            "stt_model": "nvidia/parakeet" if nvidia.available else f"whisper-{WHISPER_MODEL_SIZE}",
            "embedding_model": "nvidia/nv-embed" if nvidia.available else EMBEDDING_MODEL,
            "live_ttl_hours": LIVE_SEGMENT_TTL_HOURS,
        },
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Video ingestion
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/ingest-video", status_code=201)
async def ingest_video(
    file: UploadFile = File(...),
    name: Optional[str] = Query(None),
):
    """Upload a video file for processing.

    Streams the upload to disk in chunks to avoid loading the entire
    file into memory (critical on edge hardware with 3 GB RAM limit).
    """
    if not file.filename:
        raise HTTPException(400, "No filename provided.")

    # Validate extension before reading any bytes
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported format '{ext}'. Allowed: {ALLOWED_VIDEO_EXTENSIONS}",
        )

    # Stream upload to a temp file, enforcing size limit
    max_bytes = MAX_VIDEO_SIZE_MB * 1024 * 1024
    tmp_path = VIDEO_DIR / f"_upload_{uuid.uuid4().hex}{ext}"

    try:
        bytes_written = 0
        with open(tmp_path, "wb") as f:
            while chunk := await file.read(_UPLOAD_CHUNK):
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    tmp_path.unlink(missing_ok=True)
                    raise HTTPException(
                        413,
                        f"File too large. Max allowed: {MAX_VIDEO_SIZE_MB} MB",
                    )
                f.write(chunk)

        meta = _loader.load(tmp_path)
    except VideoLoadError as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(400, str(exc))
    finally:
        # Loader copies the file; remove the temp upload
        tmp_path.unlink(missing_ok=True)

    _db.save_video(meta)

    return {
        "video_id": meta.id,
        "filename": meta.filename,
        "status": meta.status.value,
        "duration_sec": meta.duration_sec,
        "message": "Video uploaded. Call POST /process-video to begin processing.",
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Processing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _safe_process(meta):
    """Wrapper that catches and logs exceptions from the background thread."""
    try:
        result = _orchestrator.process(meta)
        _save_benchmark(
            meta.id,
            {
                "video_id": meta.id,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "success": result.success,
                "elapsed_sec": result.elapsed_sec,
                "segments_stored": result.segments_stored,
                "events_detected": result.events_detected,
                "stage_timings": result.stage_timings,
                "quality_metrics": result.quality_metrics,
                "warnings": result.warnings,
                "errors": result.errors,
            },
        )
    except Exception:
        logger.exception("Background processing failed for video %s", meta.id)
        _save_benchmark(
            meta.id,
            {
                "video_id": meta.id,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "success": False,
                "elapsed_sec": 0.0,
                "segments_stored": 0,
                "events_detected": 0,
                "stage_timings": {},
                "quality_metrics": {},
                "warnings": [],
                "errors": ["Background processing failed"],
            },
        )
    finally:
        cleanup_progress(meta.id)


def _safe_stream_process(meta, chunk_sec: int):
    """Streaming pipeline wrapper for background thread."""
    try:
        _db.update_status(meta.id, VideoStatus.PROCESSING)
        sp = StreamingPipeline(db=_db, store=_store, chunk_sec=chunk_sec)
        for progress in sp.process_streaming(meta):
            # Re-use the progress system so SSE works
            pp = PipelineProgress(
                video_id=meta.id,
                stage=f"Chunk {progress.chunk_index + 1}/{progress.total_chunks}",
                stage_number=progress.chunk_index + 1,
                total_stages=progress.total_chunks,
                detail=f"{progress.segments_stored} segments stored",
                started_at=0,
                elapsed_sec=progress.elapsed_sec,
            )
            _on_progress(pp)
        _db.update_status(meta.id, VideoStatus.PROCESSED)
    except Exception:
        logger.exception("Streaming pipeline failed for video %s", meta.id)
        _db.update_status(meta.id, VideoStatus.FAILED, error_message="Streaming pipeline failed")
        _save_benchmark(
            meta.id,
            {
                "video_id": meta.id,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "success": False,
                "elapsed_sec": 0.0,
                "segments_stored": 0,
                "events_detected": 0,
                "stage_timings": {},
                "quality_metrics": {},
                "warnings": [],
                "errors": ["Streaming pipeline failed"],
            },
        )
    finally:
        cleanup_progress(meta.id)


@router.post("/process-video", status_code=202)
async def process_video(req: ProcessRequest):
    """Trigger the processing pipeline (runs in background thread).

    Modes:
        - ``standard``: Full sequential pipeline (parallel VLM+Whisper).
        - ``streaming``: Chunked pipeline — segments become searchable as
          each chunk finishes.  Better for long videos and near-RT use.
    """
    meta = _db.get_video(req.video_id)
    if meta is None:
        raise HTTPException(404, f"Video not found: {req.video_id}")

    if meta.status == VideoStatus.PROCESSING:
        raise HTTPException(409, "Video is already being processed.")

    if _orchestrator.is_busy:
        raise HTTPException(
            429,
            "Pipeline is busy processing another video. "
            "Edge hardware supports one job at a time.",
        )

    if req.mode == "streaming":
        thread = threading.Thread(
            target=_safe_stream_process,
            args=(meta, req.chunk_sec),
            daemon=True,
        )
    else:
        thread = threading.Thread(
            target=_safe_process,
            args=(meta,),
            daemon=True,
        )
    thread.start()

    return {
        "video_id": meta.id,
        "status": "PROCESSING",
        "mode": req.mode,
        "message": f"Processing started ({req.mode} mode).",
    }


@router.get("/video/{video_id}/progress")
async def get_progress(video_id: str):
    """Get processing progress for a video."""
    progress = _progress_store.get(video_id)
    if progress is None:
        meta = _db.get_video(video_id)
        if meta is None:
            raise HTTPException(404, f"Video not found: {video_id}")
        # Not processing or already done
        return {
            "video_id": video_id,
            "stage": meta.status.value,
            "stage_number": 10 if meta.status == VideoStatus.PROCESSED else 0,
            "total_stages": 10,
            "percent": 100 if meta.status == VideoStatus.PROCESSED else 0,
        }

    return {
        "video_id": progress.video_id,
        "stage": progress.stage,
        "stage_number": progress.stage_number,
        "total_stages": progress.total_stages,
        "percent": round((progress.stage_number / progress.total_stages) * 100),
        "elapsed_sec": progress.elapsed_sec,
    }


@router.get("/video/{video_id}/benchmark")
async def get_benchmark(video_id: str):
    """Get latest benchmark metrics captured for this video processing run."""
    meta = _db.get_video(video_id)
    if meta is None:
        raise HTTPException(404, f"Video not found: {video_id}")

    benchmark = _db.get_latest_benchmark(video_id)

    if benchmark is None:
        raise HTTPException(
            404,
            "No benchmark metrics available yet for this video. Process it first.",
        )

    return benchmark


@router.get("/video/{video_id}/benchmark/history")
async def get_benchmark_history(video_id: str, limit: int = Query(default=20, ge=1, le=200)):
    """Get historical benchmark runs (newest first)."""
    meta = _db.get_video(video_id)
    if meta is None:
        raise HTTPException(404, f"Video not found: {video_id}")

    runs = _db.list_benchmark_runs(video_id, limit=limit)
    return {
        "video_id": video_id,
        "count": len(runs),
        "runs": runs,
    }


@router.get("/video/{video_id}/benchmark/trend")
async def get_benchmark_trend(video_id: str, limit: int = Query(default=20, ge=2, le=200)):
    """Summarise benchmark trends across recent runs for regression tracking."""
    meta = _db.get_video(video_id)
    if meta is None:
        raise HTTPException(404, f"Video not found: {video_id}")

    runs = _db.list_benchmark_runs(video_id, limit=limit)
    if len(runs) < 2:
        raise HTTPException(404, "Need at least 2 benchmark runs for trend analysis.")

    newest = runs[0]
    oldest = runs[-1]
    elapsed_values = [float(r.get("elapsed_sec", 0.0)) for r in runs]
    elapsed_avg = sum(elapsed_values) / max(1, len(elapsed_values))

    stage_keys: set[str] = set()
    for run in runs:
        stage_keys.update((run.get("stage_timings") or {}).keys())

    stage_delta: dict[str, float] = {}
    for key in sorted(stage_keys):
        new_v = float((newest.get("stage_timings") or {}).get(key, 0.0))
        old_v = float((oldest.get("stage_timings") or {}).get(key, 0.0))
        stage_delta[key] = round(new_v - old_v, 3)

    def _safe_ratio(run: dict, key: str) -> float:
        val = (run.get("quality_metrics") or {}).get(key, 0.0)
        try:
            f = float(val)
        except Exception:
            return 0.0
        return 0.0 if math.isnan(f) else f

    quality_delta = {
        "captions_fallback_ratio_delta": round(
            _safe_ratio(newest, "captions_fallback_ratio") - _safe_ratio(oldest, "captions_fallback_ratio"),
            4,
        ),
        "captions_static_ratio_delta": round(
            _safe_ratio(newest, "captions_static_ratio") - _safe_ratio(oldest, "captions_static_ratio"),
            4,
        ),
        "keyframes_kept_delta": round(
            _safe_ratio(newest, "keyframes_kept") - _safe_ratio(oldest, "keyframes_kept"),
            2,
        ),
    }

    return {
        "video_id": video_id,
        "run_count": len(runs),
        "elapsed_sec": {
            "latest": newest.get("elapsed_sec", 0.0),
            "oldest": oldest.get("elapsed_sec", 0.0),
            "average": round(elapsed_avg, 3),
            "delta_latest_minus_oldest": round(float(newest.get("elapsed_sec", 0.0)) - float(oldest.get("elapsed_sec", 0.0)), 3),
        },
        "stage_timing_delta": stage_delta,
        "quality_delta": quality_delta,
    }


@router.get("/video/{video_id}/progress/stream")
async def stream_progress(video_id: str):
    """Stream processing progress via Server-Sent Events (SSE).

    Much more efficient than polling - client receives updates in real-time.
    """
    meta = _db.get_video(video_id)
    if meta is None:
        raise HTTPException(404, f"Video not found: {video_id}")

    async def event_generator():
        # Create an event for this video
        event = asyncio.Event()
        _progress_events[video_id] = event
        last_stage = -1

        try:
            while True:
                # Check current status
                progress = _progress_store.get(video_id)
                current_meta = _db.get_video(video_id)

                if current_meta is None:
                    # Video was deleted
                    yield f"data: {json.dumps({'done': True, 'deleted': True})}\n\n"
                    break

                if current_meta.status == VideoStatus.PROCESSED:
                    yield f"data: {json.dumps({'video_id': video_id, 'stage': 'Complete', 'stage_number': 10, 'total_stages': 10, 'percent': 100, 'done': True})}\n\n"
                    break

                if current_meta.status == VideoStatus.FAILED:
                    yield f"data: {json.dumps({'video_id': video_id, 'stage': 'Failed', 'percent': 0, 'done': True, 'error': True})}\n\n"
                    break

                if progress and progress.stage_number != last_stage:
                    last_stage = progress.stage_number
                    yield f"data: {json.dumps({'video_id': progress.video_id, 'stage': progress.stage, 'stage_number': progress.stage_number, 'total_stages': progress.total_stages, 'percent': round((progress.stage_number / progress.total_stages) * 100), 'elapsed_sec': progress.elapsed_sec})}\n\n"

                # Wait for next update or timeout after 2 seconds
                event.clear()
                try:
                    await asyncio.wait_for(event.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            # Server shutting down or client disconnected
            return
        finally:
            # Cleanup
            _progress_events.pop(video_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Search
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/search")
async def search(req: SearchRequest):
    """Natural language search across processed videos."""
    results = _query_engine.search(
        query=req.query,
        top_k=req.top_k,
        video_id=req.video_id,
        source_filter=req.source_filter,
    )

    return {
        "query": req.query,
        "results": [
            {
                "video_id": r.video_id,
                "segment_id": r.segment_id,
                "start_time": r.start_time,
                "end_time": r.end_time,
                "text": r.text,
                "source_type": r.source_type,
                "score": r.score,
                "event_type": r.event_type,
                "frame_url": r.frame_url,
            }
            for r in results
        ],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Video metadata & streaming
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/videos")
async def list_videos():
    """List all ingested videos."""
    videos = _db.list_videos()
    return {
        "videos": [
            {
                "video_id": v.id,
                "filename": v.filename,
                "duration_sec": v.duration_sec,
                "status": v.status.value,
                "created_at": v.created_at,
            }
            for v in videos
        ]
    }


@router.get("/video/{video_id}")
async def get_video(video_id: str):
    """Get video metadata and processing status."""
    meta = _db.get_video(video_id)
    if meta is None:
        raise HTTPException(404, f"Video not found: {video_id}")

    return {
        "video_id": meta.id,
        "filename": meta.filename,
        "duration_sec": meta.duration_sec,
        "fps": meta.fps,
        "resolution": f"{meta.width}x{meta.height}",
        "status": meta.status.value,
        "segment_count": _db.segment_count(video_id),
        "event_count": _db.event_count(video_id),
        "created_at": meta.created_at,
        "processed_at": meta.processed_at,
    }


@router.get("/video/{video_id}/stream")
async def stream_video(video_id: str):
    """Stream the video file (supports range requests via FileResponse)."""
    meta = _db.get_video(video_id)
    if meta is None:
        raise HTTPException(404, f"Video not found: {video_id}")

    path = Path(meta.file_path)
    if not path.is_file():
        raise HTTPException(404, "Video file not found on disk.")

    return FileResponse(
        path=str(path),
        media_type="video/mp4",
        filename=meta.filename,
    )


@router.get("/video/{video_id}/frame/{frame_name}")
async def get_frame(video_id: str, frame_name: str):
    """Serve a keyframe image.

    Sanitises frame_name to prevent path traversal attacks.
    Only the filename component is used — directory separators are stripped.
    """
    # SECURITY: strip directory components to prevent traversal
    safe_name = Path(frame_name).name
    if not safe_name or safe_name != frame_name:
        raise HTTPException(400, "Invalid frame name.")

    frame_path = FRAME_DIR / video_id / safe_name

    # SECURITY: verify the resolved path is still inside FRAME_DIR
    try:
        frame_path.resolve().relative_to(FRAME_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "Invalid frame path.")

    if not frame_path.is_file():
        raise HTTPException(404, f"Frame not found: {safe_name}")

    return FileResponse(
        path=str(frame_path),
        media_type="image/jpeg",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Thumbnail
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/video/{video_id}/thumbnail")
async def get_thumbnail(video_id: str):
    """Serve the first keyframe as a thumbnail for the video card."""
    frame_dir = FRAME_DIR / video_id
    if not frame_dir.is_dir():
        raise HTTPException(404, "No keyframes available for this video.")

    frames = sorted(frame_dir.glob("*.jpg"))
    if not frames:
        frames = sorted(frame_dir.glob("*.jpeg")) + sorted(frame_dir.glob("*.png"))
    if not frames:
        raise HTTPException(404, "No keyframes found.")

    return FileResponse(path=str(frames[0]), media_type="image/jpeg")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Find similar segments
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/similar")
async def find_similar(req: SimilarRequest):
    """Find segments similar to a given segment using its embedding."""
    segment = _store.get_segment(req.segment_id)
    if segment is None:
        raise HTTPException(404, f"Segment not found: {req.segment_id}")

    embedding = segment.get("embedding")
    if embedding is None:
        raise HTTPException(400, "Segment has no embedding.")

    raw_results = _store.query(
        embedding=embedding,
        top_k=req.top_k + 1,  # +1 to exclude self
        video_id=req.video_id,
    )

    # Exclude the source segment itself
    results = [r for r in raw_results if r["id"] != req.segment_id][:req.top_k]

    return {
        "source_segment_id": req.segment_id,
        "results": [
            {
                "video_id": r["video_id"],
                "segment_id": r["id"],
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "text": r["text"],
                "source_type": r["source_type"],
                "score": r["score"],
                "frame_url": f"/video/{r['video_id']}/frame/{Path(r['frame_path']).name}"
                if r.get("frame_path") else None,
            }
            for r in results
        ],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Clip export
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/video/{video_id}/clip")
async def export_clip(video_id: str, req: ClipRequest):
    """Extract a video clip using FFmpeg and return it as a download."""
    meta = _db.get_video(video_id)
    if meta is None:
        raise HTTPException(404, f"Video not found: {video_id}")

    src = Path(meta.file_path)
    if not src.is_file():
        raise HTTPException(404, "Video file not found on disk.")

    if req.end_time <= req.start_time:
        raise HTTPException(400, "end_time must be greater than start_time.")

    clip_name = f"clip_{video_id[:8]}_{req.start_time:.1f}-{req.end_time:.1f}.mp4"
    clip_path = VIDEO_DIR / f"_clip_{uuid.uuid4().hex}.mp4"

    try:
        cmd = [
            str(FFMPEG_PATH), "-y",
            "-ss", str(req.start_time),
            "-to", str(req.end_time),
            "-i", str(src),
            "-c", "copy",
            "-movflags", "+faststart",
            str(clip_path),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            raise HTTPException(500, f"FFmpeg failed: {result.stderr.decode()[:200]}")

        return FileResponse(
            path=str(clip_path),
            media_type="video/mp4",
            filename=clip_name,
            background=None,  # Don't delete before sending
        )
    except subprocess.TimeoutExpired:
        clip_path.unlink(missing_ok=True)
        raise HTTPException(504, "Clip export timed out.")
    except HTTPException:
        raise
    except Exception as exc:
        clip_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Clip export failed: {exc}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Video summary report
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/video/{video_id}/report")
async def get_video_report(video_id: str):
    """Generate a structured summary report for a processed video.

    Returns metadata, segment stats, top entities, detected events,
    and a timeline of key moments — useful for dashboards and exports.
    """
    meta = _db.get_video(video_id)
    if meta is None:
        raise HTTPException(404, f"Video not found: {video_id}")

    # Segment stats from ChromaDB
    segments = _store.get_by_video(video_id)
    source_counts = {}
    for seg in segments:
        st = seg.get("source_type", "unknown")
        source_counts[st] = source_counts.get(st, 0) + 1

    # Events from DB
    events = _db.list_events(video_id)

    # Knowledge graph summary
    graph_path = GRAPH_DIR / f"{video_id}.graphml"
    entities = []
    if graph_path.is_file():
        import networkx as nx
        G = nx.read_graphml(str(graph_path))
        # Top entities by occurrence count
        for node_id, data in G.nodes(data=True):
            entities.append({
                "label": data.get("label", node_id),
                "type": data.get("type", "object"),
                "count": int(float(data.get("count", 1))),
            })
        entities.sort(key=lambda e: e["count"], reverse=True)

    # Annotations
    annotations = _db.list_annotations(video_id)

    return {
        "video_id": video_id,
        "filename": meta.filename,
        "duration_sec": meta.duration_sec,
        "status": meta.status.value,
        "created_at": meta.created_at,
        "processed_at": meta.processed_at,
        "segments": {
            "total": len(segments),
            "by_source": source_counts,
        },
        "events": {
            "total": len(events),
            "items": events[:20],  # Top 20 events
        },
        "entities": {
            "total": len(entities),
            "top": entities[:30],  # Top 30 entities
        },
        "annotations": {
            "total": len(annotations),
            "items": annotations,
        },
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Knowledge graph
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/video/{video_id}/graph")
async def get_graph(video_id: str):
    """Return the knowledge graph as JSON nodes and edges for visualization."""
    meta = _db.get_video(video_id)
    if meta is None:
        raise HTTPException(404, f"Video not found: {video_id}")

    graph_path = GRAPH_DIR / f"{video_id}.graphml"
    if not graph_path.is_file():
        return {"nodes": [], "edges": []}

    import networkx as nx

    G = nx.read_graphml(str(graph_path))

    nodes = []
    for node_id, data in G.nodes(data=True):
        nodes.append({
            "id": node_id,
            "label": data.get("label", node_id),
            "type": data.get("type", "object"),
            "count": int(float(data.get("count", 1))),
            "first_seen": float(data.get("first_seen", 0)),
            "last_seen": float(data.get("last_seen", 0)),
        })

    edges = []
    for src, tgt, data in G.edges(data=True):
        edges.append({
            "source": src,
            "target": tgt,
            "action": data.get("action", ""),
            "timestamp": float(data.get("timestamp", 0)),
        })

    # Older graphs may contain entity nodes without persisted edges.
    # Synthesize lightweight co-occurrence links from shared timestamps so
    # the frontend can still show relationships for already-processed videos.
    if not edges and len(nodes) > 1:
        grouped_nodes: dict[float, list[str]] = {}
        for node in nodes:
            ts = round(float(node.get("first_seen", 0)), 3)
            grouped_nodes.setdefault(ts, []).append(node["id"])

        for timestamp, node_ids in grouped_nodes.items():
            unique_ids = list(dict.fromkeys(node_ids))
            if len(unique_ids) < 2:
                continue
            for i, src in enumerate(unique_ids[:-1]):
                for tgt in unique_ids[i + 1:]:
                    edges.append({
                        "source": src,
                        "target": tgt,
                        "action": "co_occurs_with",
                        "timestamp": timestamp,
                    })

    return {"nodes": nodes, "edges": edges}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Events & timeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/video/{video_id}/events")
async def get_events(video_id: str):
    """Return all detected events for a video timeline."""
    meta = _db.get_video(video_id)
    if meta is None:
        raise HTTPException(404, f"Video not found: {video_id}")

    events = _db.list_events(video_id)
    return {"video_id": video_id, "events": events}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Batch processing queue
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/process-batch", status_code=202)
async def process_batch(req: BatchProcessRequest):
    """Queue multiple videos for sequential processing."""
    queued = []
    errors = []

    for vid in req.video_ids:
        meta = _db.get_video(vid)
        if meta is None:
            errors.append({"video_id": vid, "error": "Not found"})
            continue
        if meta.status == VideoStatus.PROCESSING:
            errors.append({"video_id": vid, "error": "Already processing"})
            continue

        with _queue_lock:
            if vid not in _process_queue:
                _process_queue.append(vid)
                queued.append(vid)

    # Start the worker if pipeline isn't busy
    if queued and not _orchestrator.is_busy:
        thread = threading.Thread(target=_queue_worker, daemon=True)
        thread.start()

    return {
        "queued": queued,
        "queue_size": len(_process_queue),
        "errors": errors,
    }


@router.get("/process-queue")
async def get_queue():
    """Return the current processing queue."""
    with _queue_lock:
        return {
            "queue": list(_process_queue),
            "queue_size": len(_process_queue),
            "is_busy": _orchestrator.is_busy,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Annotations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/annotations", status_code=201)
async def create_annotation(req: AnnotationRequest):
    """Create a new annotation (bookmark/tag) for a video segment."""
    meta = _db.get_video(req.video_id)
    if meta is None:
        raise HTTPException(404, f"Video not found: {req.video_id}")

    ann = {
        "id": uuid.uuid4().hex,
        "video_id": req.video_id,
        "start_time": req.start_time,
        "end_time": req.end_time,
        "label": req.label,
        "note": req.note,
        "color": req.color,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _db.save_annotation(ann)
    return ann


@router.get("/video/{video_id}/annotations")
async def list_annotations(video_id: str):
    """List all annotations for a video."""
    meta = _db.get_video(video_id)
    if meta is None:
        raise HTTPException(404, f"Video not found: {video_id}")

    annotations = _db.list_annotations(video_id)
    return {"video_id": video_id, "annotations": annotations}


@router.delete("/annotations/{annotation_id}")
async def delete_annotation(annotation_id: str):
    """Delete an annotation by ID."""
    deleted = _db.delete_annotation(annotation_id)
    if not deleted:
        raise HTTPException(404, f"Annotation not found: {annotation_id}")
    return {"message": "Annotation deleted.", "id": annotation_id}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Live video feeds
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/live/start", status_code=201)
async def start_live_feed(req: LiveStartRequest):
    """Start a live video feed from an RTSP/RTMP/HTTP stream or webcam.

    The feed continuously captures and processes video in real-time chunks.
    Connect to the WebSocket at ``/ws/live/{video_id}`` to receive events.
    """
    try:
        status = _streaming_pipeline.start_live(
            url=req.url,
            video_id=req.video_id,
            chunk_sec=req.chunk_sec,
        )
        return {
            "video_id": status.video_id,
            "url": status.url,
            "state": status.state,
            "chunk_sec": req.chunk_sec,
            "message": f"Live feed started. Connect to /ws/live/{req.video_id} for real-time events.",
        }
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@router.post("/live/stop")
async def stop_live_feed(req: LiveStopRequest):
    """Stop an active live feed."""
    stopped = _streaming_pipeline.stop_live(req.video_id)
    if not stopped:
        raise HTTPException(404, f"No active live feed: {req.video_id}")
    return {"video_id": req.video_id, "message": "Live feed stopping."}


@router.get("/live/status")
async def live_feed_status(video_id: Optional[str] = Query(None)):
    """Get status of all active live feeds, or a specific one."""
    feeds = _streaming_pipeline.get_live_status(video_id)
    return {
        "feeds": [
            {
                "video_id": f.video_id,
                "url": f.url,
                "state": f.state,
                "chunks_processed": f.chunks_processed,
                "total_segments": f.total_segments,
                "fps": f.fps,
                "started_at": f.started_at,
                "last_chunk_at": f.last_chunk_at,
                "error": f.error,
            }
            for f in feeds
        ],
    }


@router.websocket("/ws/live/{video_id}")
async def live_websocket(websocket: WebSocket, video_id: str):
    """WebSocket endpoint for real-time live feed events.

    Clients receive JSON messages with fields:
        - video_id: str
        - event_type: "chunk_ready" | "segment_indexed" | "event_detected" | "status_change" | "error"
        - data: dict (event-specific payload)
        - timestamp: ISO 8601

    Send ``{"action": "search", "query": "..."}`` to search indexed segments
    from this live feed in real-time.
    """
    await websocket.accept()

    # Register client
    with _ws_lock:
        if video_id not in _ws_clients:
            _ws_clients[video_id] = set()
        _ws_clients[video_id].add(websocket)

    try:
        # Send current status immediately
        feeds = _streaming_pipeline.get_live_status(video_id)
        if feeds:
            f = feeds[0]
            await websocket.send_json({
                "video_id": video_id,
                "event_type": "status_change",
                "data": {"state": f.state, "chunks_processed": f.chunks_processed},
                "timestamp": f.last_chunk_at or f.started_at,
            })

        # Listen for client messages (search queries, keepalives)
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send keepalive ping
                await websocket.send_json({"event_type": "ping"})
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # Handle live search request
            if msg.get("action") == "search" and msg.get("query"):
                results = _query_engine.search(
                    query=msg["query"],
                    top_k=msg.get("top_k", 10),
                    video_id=video_id,
                )
                await websocket.send_json({
                    "video_id": video_id,
                    "event_type": "search_results",
                    "data": {
                        "query": msg["query"],
                        "results": [
                            {
                                "segment_id": r.segment_id,
                                "start_time": r.start_time,
                                "end_time": r.end_time,
                                "text": r.text,
                                "score": r.score,
                                "source_type": r.source_type,
                            }
                            for r in results
                        ],
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug("WebSocket error for %s: %s", video_id, exc)
    finally:
        # Unregister client
        with _ws_lock:
            clients = _ws_clients.get(video_id)
            if clients:
                clients.discard(websocket)
                if not clients:
                    del _ws_clients[video_id]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Browser camera feed (phone / screen share via getUserMedia)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/live/browser-chunk", status_code=200)
async def upload_browser_chunk(
    video_id: str = Query(...),
    chunk_index: int = Query(...),
    chunk_start: float = Query(0),
    file: UploadFile = File(...),
):
    """Receive a video chunk from a browser-based camera (phone, webcam, screen share).

    The frontend captures via getUserMedia / getDisplayMedia, encodes chunks
    using MediaRecorder, and POSTs each chunk here.  The backend saves the
    chunk to disk, extracts keyframes, runs VLM + Whisper, and indexes it.

    This enables phone cameras and screen shares without needing RTSP.
    """
    import tempfile
    import cv2 as _cv2
    from concurrent.futures import ThreadPoolExecutor as _TPE
    from backend.pipeline.streaming import StreamingPipeline

    # Save uploaded chunk to temp file
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    try:
        # Initialize or retrieve feed state
        with _browser_feeds_lock:
            if video_id not in _browser_feeds:
                _browser_feeds[video_id] = {
                    "chunk_idx": 0,
                    "embedder": MultimodalEmbedder(),
                    "all_captions": [],
                    "all_transcripts": [],
                }
            feed = _browser_feeds[video_id]

        # WebM from MediaRecorder may not be seekable. Convert to mp4 via
        # FFmpeg first so OpenCV can reliably extract frames.
        mp4_path = tmp_path.replace(".webm", ".mp4")
        convert_cmd = [
            str(FFMPEG_PATH), "-y", "-i", tmp_path,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-c:a", "aac", "-b:a", "64k",
            "-movflags", "+faststart",
            mp4_path,
        ]
        convert_result = subprocess.run(convert_cmd, capture_output=True, timeout=60)
        if convert_result.returncode == 0 and Path(mp4_path).stat().st_size > 500:
            video_chunk_path = mp4_path
        else:
            # Fallback: try the raw webm
            video_chunk_path = tmp_path
            mp4_path = None
            logger.warning("FFmpeg webm→mp4 conversion failed, using raw webm")

        # Extract keyframes from the chunk video
        cap = _cv2.VideoCapture(video_chunk_path)
        if not cap.isOpened():
            raise HTTPException(400, f"Could not read video chunk {chunk_index}")

        fps = cap.get(_cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            # Some containers don't report frame count — read until EOF
            total_frames = int(fps * 30)  # assume max 30s chunk
        chunk_dir = FRAME_DIR / video_id / f"browser_{chunk_index:06d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        max_kf = 5
        step = max(1, total_frames // max_kf)
        keyframes = []
        from backend.db.models import Keyframe as _Keyframe

        for i in range(0, total_frames, step):
            if len(keyframes) >= max_kf:
                break
            cap.set(_cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                if len(keyframes) > 0:
                    break  # We got some frames, that's enough
                continue
            ts = chunk_start + (i / fps)
            fp = str(chunk_dir / f"frame_{i:06d}.jpg")
            _cv2.imwrite(fp, frame, [_cv2.IMWRITE_JPEG_QUALITY, 85])
            keyframes.append(_Keyframe(
                video_id=video_id,
                segment_index=chunk_index,
                frame_number=i,
                timestamp=round(ts, 3),
                file_path=fp,
            ))
        cap.release()

        # Use the chunk video path for whisper (it has audio)
        audio_source = video_chunk_path

        # Parallel VLM + Whisper
        from backend.visual.vlm_runner import OllamaClient as _OC
        from backend.db.models import FusedSegment as _FusedSegment
        import uuid as _uuid

        vlm_client = _OC()
        vlm_available = vlm_client.is_available()

        chunk_duration = total_frames / fps if fps > 0 else 15.0

        def _browser_whisper(src_path: str) -> list:
            """Extract audio from browser chunk and transcribe."""
            import tempfile as _tmpmod
            from backend.audio.whisper_runner import WhisperRunner as _WR
            from backend.db.models import TranscriptSegment  # noqa: F811

            with _tmpmod.NamedTemporaryFile(suffix=".wav", delete=False) as wav_tmp:
                wav_path = wav_tmp.name
            try:
                # Extract all audio from the chunk (no seeking needed)
                cmd = [
                    str(FFMPEG_PATH), "-y", "-i", src_path,
                    "-vn", "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1",
                    wav_path,
                ]
                proc = subprocess.run(cmd, capture_output=True, timeout=60)
                if proc.returncode != 0:
                    logger.debug("Browser chunk audio extraction failed: %s",
                                 proc.stderr.decode(errors="replace")[-200:])
                    return []
                if Path(wav_path).stat().st_size < 1000:
                    return []

                wr = _WR()
                segs = wr.transcribe(wav_path)
                wr.unload_model()
                return segs
            except Exception as exc:
                logger.debug("Browser whisper failed: %s", exc)
                return []
            finally:
                Path(wav_path).unlink(missing_ok=True)

        with _TPE(max_workers=2, thread_name_prefix="browser") as pool:
            vlm_future = pool.submit(
                StreamingPipeline._vlm_on_keyframes,
                keyframes,
                vlm_client if vlm_available else None,
            )
            whisper_future = pool.submit(_browser_whisper, audio_source)
            captions = vlm_future.result()
            transcripts = whisper_future.result()

        # Offset whisper timestamps
        for seg in transcripts:
            seg.start_time = round(seg.start_time + chunk_start, 3)
            seg.end_time = round(seg.end_time + chunk_start, 3)

        feed["all_captions"].extend(captions)
        feed["all_transcripts"].extend(transcripts)

        # Fuse + embed + store
        embedder = feed["embedder"]
        fused = embedder.fuse(video_id, captions, transcripts)

        logger.info(
            "Browser chunk %d: %d keyframes, %d captions, %d transcripts, vlm=%s",
            chunk_index, len(keyframes), len(captions), len(transcripts),
            "on" if vlm_available else "off",
        )

        # Fallback: if VLM and Whisper both produced nothing, create a
        # basic segment per keyframe so the chunk is still searchable.
        if not fused and keyframes:
            for kf in keyframes:
                fused.append(_FusedSegment(
                    id=_uuid.uuid4().hex,
                    video_id=video_id,
                    start_time=kf.timestamp,
                    end_time=kf.timestamp + chunk_duration / max(len(keyframes), 1),
                    text=f"Live frame at {kf.timestamp:.1f}s from {video_id}",
                    source_type="visual",
                    frame_path=kf.file_path,
                ))

        segments_stored = 0
        if fused:
            embedder.embed(fused)
            segments_stored = _store.add_segments(fused)

        feed["chunk_idx"] = chunk_index + 1

        # Broadcast via WebSocket if clients connected
        if _on_live_event:
            from backend.pipeline.streaming import LiveEvent
            _on_live_event(LiveEvent(
                video_id=video_id,
                event_type="chunk_ready",
                data={
                    "chunk_index": chunk_index,
                    "segments_stored": segments_stored,
                    "start_time": chunk_start,
                },
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))

        return {
            "video_id": video_id,
            "chunk_index": chunk_index,
            "segments_stored": segments_stored,
            "keyframes_extracted": len(keyframes),
        }

    finally:
        Path(tmp_path).unlink(missing_ok=True)
        # Clean up converted mp4 if it exists
        mp4_cleanup = tmp_path.replace(".webm", ".mp4")
        Path(mp4_cleanup).unlink(missing_ok=True)


@router.post("/live/browser-stop")
async def stop_browser_feed(video_id: str = Query(...)):
    """Finalize a browser camera feed — build knowledge graph and clean up."""
    with _browser_feeds_lock:
        feed = _browser_feeds.pop(video_id, None)

    if feed is None:
        raise HTTPException(404, f"No browser feed: {video_id}")

    # Build knowledge graph in background
    captions = feed["all_captions"]
    transcripts = feed["all_transcripts"]
    embedder = feed["embedder"]

    if captions or transcripts:
        from backend.knowledge.graph import KnowledgeGraph
        from backend.knowledge.event_detector import EventDetector

        kg = KnowledgeGraph(video_id)
        kg.build_from_captions(captions, transcripts)
        kg.save()

        detector = EventDetector()
        events = detector.detect(kg)
        if events:
            from backend.pipeline.streaming import StreamingPipeline
            event_segments = StreamingPipeline._events_to_segments(video_id, events)
            embedder.embed(event_segments)
            _store.add_segments(event_segments)

    embedder.unload_model()

    return {
        "video_id": video_id,
        "total_chunks": feed["chunk_idx"],
        "total_captions": len(captions),
        "total_transcripts": len(transcripts),
        "message": "Browser feed finalized.",
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Video deletion
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.delete("/video/{video_id}")
async def delete_video(video_id: str):
    """Delete a video and all its associated data."""
    meta = _db.get_video(video_id)
    if meta is None:
        raise HTTPException(404, f"Video not found: {video_id}")

    if meta.status == VideoStatus.PROCESSING:
        raise HTTPException(409, "Cannot delete a video that is currently processing.")

    _store.purge_video(video_id)

    video_path = Path(meta.file_path)
    if video_path.is_file():
        video_path.unlink()
    frame_dir = FRAME_DIR / video_id
    if frame_dir.is_dir():
        shutil.rmtree(frame_dir)

    _db.delete_video(video_id)

    return {"message": f"Video {video_id} deleted.", "video_id": video_id}
