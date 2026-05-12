"""
CogniStream — Streaming Pipeline

Processes video in rolling time-window chunks for near-real-time indexing.
Each chunk is independently processed through the full pipeline and becomes
searchable immediately — no need to wait for the entire video to finish.

Two modes:

1. **File-based streaming** — ``process_streaming(meta)``
   Splits a pre-uploaded video into N-second windows and processes each
   chunk, storing results incrementally.

2. **Live feed** — ``start_live(url, video_id)`` / ``stop_live(video_id)``
   Continuously captures from an RTSP / RTMP / HTTP / webcam URL,
   assembles frames into time-windowed chunks, and processes each chunk
   in real-time.  Supports reconnection on stream interruption.

Usage (file-based):
    sp = StreamingPipeline(db, store, chunk_sec=30)
    for progress in sp.process_streaming(video_meta):
        print(f"Chunk {progress.chunk_index}: {progress.segments_stored} segments")

Usage (live feed):
    sp = StreamingPipeline(db, store, chunk_sec=15)
    sp.start_live("rtsp://cam:554/stream", video_id="cam01")
    # ... later ...
    sp.stop_live("cam01")
"""

from __future__ import annotations

import logging
import math
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Generator, Optional

import cv2

from backend.audio.whisper_runner import WhisperRunner
from backend.config import (
    AUDIO_DIR,
    FFMPEG_PATH,
    FRAME_DIR,
    KEYFRAME_JPEG_QUALITY,
    STREAM_CHUNK_SEC,
)
from backend.db.chroma_store import ChromaStore
from backend.db.models import (
    FusedSegment,
    Keyframe,
    TranscriptSegment,
    VideoMeta,
    VideoStatus,
    VisualCaption,
)
from backend.db.sqlite import SQLiteDB
from backend.fusion.multimodal_embedder import MultimodalEmbedder
from backend.knowledge.event_detector import EventDetector
from backend.knowledge.graph import KnowledgeGraph
from backend.visual.vlm_runner import OllamaClient, VLMRunner

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data classes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ChunkProgress:
    """Progress report for one processed chunk."""
    video_id: str
    chunk_index: int
    total_chunks: int
    start_time: float
    end_time: float
    segments_stored: int = 0
    elapsed_sec: float = 0.0
    searchable_now: bool = True


@dataclass
class LiveFeedStatus:
    """Status of an active live feed."""
    video_id: str
    url: str
    state: str  # "connecting", "running", "reconnecting", "stopped", "error"
    chunks_processed: int = 0
    total_segments: int = 0
    fps: float = 0.0
    started_at: str = ""
    last_chunk_at: str = ""
    error: str = ""


@dataclass
class LiveEvent:
    """Real-time event pushed from a live feed for WebSocket consumers."""
    video_id: str
    event_type: str  # "chunk_ready", "segment_indexed", "event_detected", "status_change", "error"
    data: dict = field(default_factory=dict)
    timestamp: str = ""


ChunkCallback = Callable[[ChunkProgress], None]
LiveEventCallback = Callable[[LiveEvent], None]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Streaming Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class StreamingPipeline:
    """Process video in time-windowed chunks for incremental indexing.

    Supports both file-based streaming and live RTSP/webcam feeds.
    """

    # Maximum reconnection attempts before giving up
    MAX_RECONNECT_ATTEMPTS = 10
    RECONNECT_DELAY_SEC = 3

    def __init__(
        self,
        db: SQLiteDB | None = None,
        store: ChromaStore | None = None,
        chunk_sec: int | None = None,
        on_chunk: ChunkCallback | None = None,
        on_live_event: LiveEventCallback | None = None,
    ):
        self.db = db or SQLiteDB()
        self.store = store or ChromaStore()
        self.chunk_sec = chunk_sec or STREAM_CHUNK_SEC
        self._on_chunk = on_chunk
        self._on_live_event = on_live_event
        self._stop_event = threading.Event()

        # Live feed management
        self._live_feeds: dict[str, _LiveFeedWorker] = {}
        self._live_lock = threading.Lock()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # File-based streaming (pre-uploaded videos)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def process_streaming(
        self, meta: VideoMeta
    ) -> Generator[ChunkProgress, None, None]:
        """Process a video file in chunks, yielding progress after each."""
        duration = meta.duration_sec or self._probe_duration(meta.file_path)
        if duration <= 0:
            logger.error("Cannot determine video duration: %s", meta.file_path)
            return

        total_chunks = max(1, math.ceil(duration / self.chunk_sec))
        logger.info(
            "Streaming pipeline: %s → %d chunks of %ds (duration=%.1fs)",
            meta.filename, total_chunks, self.chunk_sec, duration,
        )

        fps = meta.fps or 30.0
        all_captions: list[VisualCaption] = []
        all_transcripts: list[TranscriptSegment] = []
        embedder = MultimodalEmbedder()

        vlm_client = OllamaClient()
        vlm_available = vlm_client.is_available()
        if not vlm_available:
            logger.warning("Ollama not available — visual analysis disabled for stream")

        for chunk_idx in range(total_chunks):
            if self._stop_event.is_set():
                logger.info("Streaming pipeline stopped at chunk %d", chunk_idx)
                break

            t_chunk_start = time.monotonic()
            chunk_start_sec = chunk_idx * self.chunk_sec
            chunk_end_sec = min((chunk_idx + 1) * self.chunk_sec, duration)
            start_frame = int(chunk_start_sec * fps)
            end_frame = int(chunk_end_sec * fps)

            logger.info(
                "Processing chunk %d/%d [%.1f–%.1fs]",
                chunk_idx + 1, total_chunks, chunk_start_sec, chunk_end_sec,
            )

            keyframes = self._extract_chunk_keyframes(
                meta, start_frame, end_frame, fps, chunk_idx,
            )

            chunk_captions: list[VisualCaption] = []
            chunk_transcripts: list[TranscriptSegment] = []

            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="stream") as pool:
                vlm_future = pool.submit(
                    self._vlm_on_keyframes, keyframes, vlm_client if vlm_available else None
                )
                whisper_future = pool.submit(
                    self._whisper_on_chunk, meta.file_path, chunk_start_sec, chunk_end_sec
                )
                chunk_captions = vlm_future.result()
                chunk_transcripts = whisper_future.result()

            all_captions.extend(chunk_captions)
            all_transcripts.extend(chunk_transcripts)

            fused = embedder.fuse(meta.id, chunk_captions, chunk_transcripts)
            if fused:
                embedder.embed(fused)
                segments_stored = self.store.add_segments(fused)
            else:
                segments_stored = 0

            elapsed = time.monotonic() - t_chunk_start
            progress = ChunkProgress(
                video_id=meta.id,
                chunk_index=chunk_idx,
                total_chunks=total_chunks,
                start_time=chunk_start_sec,
                end_time=chunk_end_sec,
                segments_stored=segments_stored,
                elapsed_sec=round(elapsed, 1),
            )

            if self._on_chunk:
                self._on_chunk(progress)
            yield progress

        # Post-stream: knowledge graph + events
        if all_captions or all_transcripts:
            logger.info("Building knowledge graph from %d captions + %d transcripts",
                        len(all_captions), len(all_transcripts))
            kg = KnowledgeGraph(meta.id)
            kg.build_from_captions(all_captions, all_transcripts)
            kg.save()

            detector = EventDetector()
            events = detector.detect(kg)
            if events:
                event_segments = self._events_to_segments(meta.id, events)
                embedder.embed(event_segments)
                self.store.add_segments(event_segments)
                logger.info("Stored %d event segments", len(events))

        embedder.unload_model()

    def stop(self) -> None:
        """Signal the file-based streaming pipeline to stop."""
        self._stop_event.set()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Live feed management
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def start_live(
        self,
        url: str,
        video_id: str,
        chunk_sec: int | None = None,
    ) -> LiveFeedStatus:
        """Start continuous capture and processing from a live video source.

        Supports RTSP, RTMP, HTTP streams, and local webcams (integer index
        passed as string, e.g. "0" for /dev/video0).

        Args:
            url:       Stream URL or webcam index ("0", "1", ...).
            video_id:  Unique identifier for this feed.
            chunk_sec: Override chunk duration for this feed.

        Returns:
            Initial LiveFeedStatus.

        Raises:
            ValueError: If video_id is already active.
        """
        with self._live_lock:
            if video_id in self._live_feeds:
                existing = self._live_feeds[video_id]
                if existing.is_alive():
                    raise ValueError(f"Live feed '{video_id}' is already running.")
                # Dead worker — clean up
                del self._live_feeds[video_id]

            worker = _LiveFeedWorker(
                url=url,
                video_id=video_id,
                chunk_sec=chunk_sec or self.chunk_sec,
                db=self.db,
                store=self.store,
                on_live_event=self._on_live_event,
            )
            self._live_feeds[video_id] = worker
            worker.start()

            return LiveFeedStatus(
                video_id=video_id,
                url=url,
                state="connecting",
                started_at=datetime.now(timezone.utc).isoformat(),
            )

    def stop_live(self, video_id: str) -> bool:
        """Stop a live feed by video_id. Returns True if it was running."""
        with self._live_lock:
            worker = self._live_feeds.get(video_id)
            if worker is None:
                return False
            worker.stop()
            return True

    def get_live_status(self, video_id: str | None = None) -> list[LiveFeedStatus]:
        """Get status of live feeds. If video_id given, returns just that one."""
        with self._live_lock:
            if video_id:
                worker = self._live_feeds.get(video_id)
                if worker is None:
                    return []
                return [worker.get_status()]

            statuses = []
            dead_ids = []
            for vid, worker in self._live_feeds.items():
                if not worker.is_alive() and worker.status.state not in ("stopped", "error"):
                    dead_ids.append(vid)
                    continue
                statuses.append(worker.get_status())
            # Clean up dead workers
            for vid in dead_ids:
                del self._live_feeds[vid]
            return statuses

    def stop_all_live(self) -> int:
        """Stop all active live feeds. Returns count stopped."""
        with self._live_lock:
            count = 0
            for worker in self._live_feeds.values():
                if worker.is_alive():
                    worker.stop()
                    count += 1
            return count

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Shared helpers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _probe_duration(file_path: str) -> float:
        """Get video duration via OpenCV."""
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frames / fps if fps > 0 else 0.0

    @staticmethod
    def _extract_chunk_keyframes(
        meta: VideoMeta,
        start_frame: int,
        end_frame: int,
        fps: float,
        chunk_idx: int,
        max_frames: int = 5,
    ) -> list[Keyframe]:
        """Extract evenly-spaced keyframes from a chunk of the video."""
        cap = cv2.VideoCapture(meta.file_path)
        if not cap.isOpened():
            return []

        frame_span = end_frame - start_frame
        if frame_span <= 0:
            cap.release()
            return []

        step = max(1, frame_span // max_frames)
        target_frames = list(range(start_frame, end_frame, step))[:max_frames]

        chunk_dir = Path(FRAME_DIR) / meta.id / f"chunk_{chunk_idx:04d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        keyframes: list[Keyframe] = []
        for frame_num in target_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue

            file_path = str(chunk_dir / f"frame_{frame_num:06d}.jpg")
            cv2.imwrite(file_path, frame, [cv2.IMWRITE_JPEG_QUALITY, KEYFRAME_JPEG_QUALITY])

            keyframes.append(Keyframe(
                video_id=meta.id,
                segment_index=chunk_idx,
                frame_number=frame_num,
                timestamp=round(frame_num / fps, 3),
                file_path=file_path,
            ))

        cap.release()
        return keyframes

    @staticmethod
    def _vlm_on_keyframes(
        keyframes: list[Keyframe],
        client: OllamaClient | None,
    ) -> list[VisualCaption]:
        """Run VLM on keyframes (always fast mode for streaming)."""
        if not client or not keyframes:
            return []
        try:
            runner = VLMRunner(client, fast_mode=True)
            return runner.analyse_keyframes(keyframes)
        except Exception as exc:
            logger.error("Streaming VLM failed: %s", exc)
            return []

    @staticmethod
    def _whisper_on_chunk(
        video_path: str,
        start_sec: float,
        end_sec: float,
    ) -> list[TranscriptSegment]:
        """Extract and transcribe audio for a time window."""
        duration = end_sec - start_sec
        if duration <= 0:
            return []

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cmd = [
                FFMPEG_PATH, "-y",
                "-ss", str(start_sec),
                "-t", str(duration),
                "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                tmp_path,
            ]
            proc = subprocess.run(cmd, capture_output=True, timeout=60)
            if proc.returncode != 0:
                logger.warning("FFmpeg chunk audio extraction failed: %s", proc.stderr[:200])
                return []

            if Path(tmp_path).stat().st_size < 1000:
                return []

            whisper = WhisperRunner()
            segments = whisper.transcribe(tmp_path)
            whisper.unload_model()

            for seg in segments:
                seg.start_time = round(seg.start_time + start_sec, 3)
                seg.end_time = round(seg.end_time + start_sec, 3)

            return segments
        except Exception as exc:
            logger.error("Streaming whisper failed: %s", exc)
            return []
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @staticmethod
    def _events_to_segments(video_id: str, events: list) -> list[FusedSegment]:
        """Convert detected events into embeddable FusedSegments."""
        segments = []
        for ev in events:
            segments.append(FusedSegment(
                id=uuid.uuid4().hex,
                video_id=video_id,
                start_time=ev.start_time,
                end_time=ev.end_time,
                text=f"Event: {ev.event_type}. {ev.description}. Entities: {', '.join(ev.entities)}",
                source_type="event",
            ))
        return segments


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Live Feed Worker (background thread per feed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _LiveFeedWorker(threading.Thread):
    """Background thread that continuously captures from a live source.

    Frame capture loop:
        1. Open cv2.VideoCapture(url)
        2. Read frames into a buffer for ``chunk_sec`` seconds
        3. Save keyframes + extract audio for the chunk
        4. Process chunk: VLM + Whisper (parallel) → fuse → embed → store
        5. Emit LiveEvent for each indexed chunk
        6. Repeat until stopped or stream dies after max reconnect attempts

    Audio for live feeds:
        FFmpeg reads directly from the URL for the chunk duration, so audio
        transcription works for RTSP/RTMP/HTTP sources.  For webcam indices,
        audio is skipped (webcam audio requires OS-specific capture).
    """

    def __init__(
        self,
        url: str,
        video_id: str,
        chunk_sec: int,
        db: SQLiteDB,
        store: ChromaStore,
        on_live_event: LiveEventCallback | None = None,
    ):
        super().__init__(daemon=True, name=f"live-{video_id}")
        self.url = url
        self.video_id = video_id
        self.chunk_sec = chunk_sec
        self.db = db
        self.store = store
        self._on_live_event = on_live_event
        self._halt = threading.Event()

        self.status = LiveFeedStatus(
            video_id=video_id,
            url=url,
            state="connecting",
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        # Resolve webcam index
        self._capture_source: str | int = url
        if url.isdigit():
            self._capture_source = int(url)

    def stop(self) -> None:
        """Signal the worker to stop after the current chunk."""
        self._halt.set()

    def get_status(self) -> LiveFeedStatus:
        return self.status

    def run(self) -> None:
        """Main capture + processing loop."""
        logger.info("Live feed starting: %s → video_id=%s", self.url, self.video_id)
        self._emit_event("status_change", {"state": "connecting"})

        embedder = MultimodalEmbedder()
        vlm_client = OllamaClient()
        vlm_available = vlm_client.is_available()
        if not vlm_available:
            logger.warning("Ollama not available — live VLM analysis disabled")

        all_captions: list[VisualCaption] = []
        all_transcripts: list[TranscriptSegment] = []
        chunk_idx = 0
        reconnect_attempts = 0

        while not self._halt.is_set():
            # ── Open / reconnect to stream ────────────────────
            cap = cv2.VideoCapture(self._capture_source)
            if not cap.isOpened():
                reconnect_attempts += 1
                if reconnect_attempts > StreamingPipeline.MAX_RECONNECT_ATTEMPTS:
                    self.status.state = "error"
                    self.status.error = f"Failed to connect after {reconnect_attempts} attempts"
                    self._emit_event("error", {"message": self.status.error})
                    logger.error("Live feed %s: %s", self.video_id, self.status.error)
                    break

                self.status.state = "reconnecting"
                self._emit_event("status_change", {
                    "state": "reconnecting",
                    "attempt": reconnect_attempts,
                })
                logger.warning(
                    "Live feed %s: connection failed, retry %d/%d in %ds",
                    self.video_id, reconnect_attempts,
                    StreamingPipeline.MAX_RECONNECT_ATTEMPTS,
                    StreamingPipeline.RECONNECT_DELAY_SEC,
                )
                self._halt.wait(StreamingPipeline.RECONNECT_DELAY_SEC)
                continue

            # Connection established
            reconnect_attempts = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 120:
                fps = 25.0  # Sensible default for RTSP streams
            self.status.fps = fps
            self.status.state = "running"
            self._emit_event("status_change", {"state": "running", "fps": fps})
            logger.info("Live feed %s: connected (fps=%.1f)", self.video_id, fps)

            # ── Capture loop ──────────────────────────────────
            try:
                while not self._halt.is_set():
                    chunk_start_wall = time.monotonic()
                    chunk_start_sec = chunk_idx * self.chunk_sec
                    chunk_end_sec = chunk_start_sec + self.chunk_sec

                    # Capture frames for chunk_sec duration
                    keyframes = self._capture_chunk_frames(
                        cap, fps, chunk_idx, chunk_start_sec,
                    )

                    if not keyframes:
                        # Stream may have ended or dropped
                        logger.warning("Live feed %s: no frames captured, reconnecting", self.video_id)
                        break

                    # Process chunk: parallel VLM + Whisper
                    chunk_captions: list[VisualCaption] = []
                    chunk_transcripts: list[TranscriptSegment] = []

                    with ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"live-{self.video_id}") as pool:
                        vlm_future = pool.submit(
                            StreamingPipeline._vlm_on_keyframes,
                            keyframes,
                            vlm_client if vlm_available else None,
                        )
                        whisper_future = pool.submit(
                            self._live_audio_chunk,
                            chunk_start_sec, chunk_end_sec,
                        )
                        chunk_captions = vlm_future.result()
                        chunk_transcripts = whisper_future.result()

                    all_captions.extend(chunk_captions)
                    all_transcripts.extend(chunk_transcripts)

                    # Fuse + embed + store
                    fused = embedder.fuse(self.video_id, chunk_captions, chunk_transcripts)
                    segments_stored = 0
                    if fused:
                        embedder.embed(fused)
                        segments_stored = self.store.add_segments(fused)

                    self.status.chunks_processed = chunk_idx + 1
                    self.status.total_segments += segments_stored
                    self.status.last_chunk_at = datetime.now(timezone.utc).isoformat()

                    elapsed = time.monotonic() - chunk_start_wall
                    self._emit_event("chunk_ready", {
                        "chunk_index": chunk_idx,
                        "start_time": chunk_start_sec,
                        "end_time": chunk_end_sec,
                        "segments_stored": segments_stored,
                        "elapsed_sec": round(elapsed, 1),
                        "total_segments": self.status.total_segments,
                    })

                    logger.info(
                        "Live feed %s chunk %d: %d segments in %.1fs",
                        self.video_id, chunk_idx, segments_stored, elapsed,
                    )

                    chunk_idx += 1

                    # Run knowledge graph every 10 chunks
                    if chunk_idx % 10 == 0 and (all_captions or all_transcripts):
                        self._build_knowledge_graph(
                            all_captions, all_transcripts, embedder,
                        )

            except Exception as exc:
                logger.exception("Live feed %s error: %s", self.video_id, exc)
                self.status.error = str(exc)
            finally:
                cap.release()

            # If stopped intentionally, don't reconnect
            if self._halt.is_set():
                break

            # Otherwise attempt reconnection
            reconnect_attempts += 1
            if reconnect_attempts > StreamingPipeline.MAX_RECONNECT_ATTEMPTS:
                self.status.state = "error"
                self.status.error = "Stream lost after max reconnect attempts"
                self._emit_event("error", {"message": self.status.error})
                break

            self.status.state = "reconnecting"
            self._emit_event("status_change", {
                "state": "reconnecting",
                "attempt": reconnect_attempts,
            })
            self._halt.wait(StreamingPipeline.RECONNECT_DELAY_SEC)

        # ── Cleanup ───────────────────────────────────────────
        # Final knowledge graph build
        if all_captions or all_transcripts:
            self._build_knowledge_graph(all_captions, all_transcripts, embedder)

        embedder.unload_model()

        if self.status.state != "error":
            self.status.state = "stopped"
        self._emit_event("status_change", {"state": self.status.state})
        logger.info(
            "Live feed %s stopped: %d chunks, %d segments",
            self.video_id, self.status.chunks_processed, self.status.total_segments,
        )

    def _capture_chunk_frames(
        self,
        cap: cv2.VideoCapture,
        fps: float,
        chunk_idx: int,
        chunk_start_sec: float,
        max_keyframes: int = 5,
    ) -> list[Keyframe]:
        """Capture frames from a live stream for one chunk duration.

        Reads frames in real-time for ``chunk_sec`` seconds, saving
        evenly-spaced keyframes to disk.
        """
        total_frames_needed = int(fps * self.chunk_sec)
        save_interval = max(1, total_frames_needed // max_keyframes)

        chunk_dir = Path(FRAME_DIR) / self.video_id / f"live_{chunk_idx:06d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        keyframes: list[Keyframe] = []
        frames_read = 0
        chunk_wall_start = time.monotonic()

        while frames_read < total_frames_needed:
            if self._halt.is_set():
                break

            ret, frame = cap.read()
            if not ret:
                if frames_read == 0:
                    return []  # Stream dead from the start
                break  # Partial chunk is OK

            frames_read += 1

            # Save keyframe at regular intervals
            if frames_read % save_interval == 0 and len(keyframes) < max_keyframes:
                timestamp = chunk_start_sec + (frames_read / fps)
                file_path = str(chunk_dir / f"live_{chunk_idx:06d}_{frames_read:06d}.jpg")
                cv2.imwrite(file_path, frame, [cv2.IMWRITE_JPEG_QUALITY, KEYFRAME_JPEG_QUALITY])

                keyframes.append(Keyframe(
                    video_id=self.video_id,
                    segment_index=chunk_idx,
                    frame_number=frames_read,
                    timestamp=round(timestamp, 3),
                    file_path=file_path,
                ))

        elapsed = time.monotonic() - chunk_wall_start
        logger.debug(
            "Live capture chunk %d: %d frames in %.1fs, %d keyframes saved",
            chunk_idx, frames_read, elapsed, len(keyframes),
        )
        return keyframes

    def _live_audio_chunk(
        self,
        start_sec: float,
        end_sec: float,
    ) -> list[TranscriptSegment]:
        """Capture audio from the live source for a time window.

        For network streams (RTSP/RTMP/HTTP), FFmpeg reads directly from the URL.
        For webcam indices, audio capture is skipped.
        """
        # Webcams don't reliably provide audio via FFmpeg
        if isinstance(self._capture_source, int):
            return []

        duration = end_sec - start_sec
        if duration <= 0:
            return []

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # FFmpeg reads live audio directly from the stream URL
            cmd = [
                FFMPEG_PATH, "-y",
                "-i", self.url,
                "-t", str(duration),
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                tmp_path,
            ]
            proc = subprocess.run(
                cmd, capture_output=True,
                timeout=int(duration) + 30,
            )
            if proc.returncode != 0:
                logger.debug("Live audio extraction skipped: %s", proc.stderr[:100])
                return []

            if Path(tmp_path).stat().st_size < 1000:
                return []

            whisper = WhisperRunner()
            segments = whisper.transcribe(tmp_path)
            whisper.unload_model()

            # Offset timestamps to absolute stream time
            for seg in segments:
                seg.start_time = round(seg.start_time + start_sec, 3)
                seg.end_time = round(seg.end_time + start_sec, 3)

            return segments
        except subprocess.TimeoutExpired:
            logger.warning("Live audio extraction timed out for chunk at %.1fs", start_sec)
            return []
        except Exception as exc:
            logger.error("Live whisper failed: %s", exc)
            return []
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _build_knowledge_graph(
        self,
        captions: list[VisualCaption],
        transcripts: list[TranscriptSegment],
        embedder: MultimodalEmbedder,
    ) -> None:
        """Build/update the knowledge graph for this live feed."""
        try:
            kg = KnowledgeGraph(self.video_id)
            kg.build_from_captions(captions, transcripts)
            kg.save()

            detector = EventDetector()
            events = detector.detect(kg)
            if events:
                event_segments = StreamingPipeline._events_to_segments(self.video_id, events)
                embedder.embed(event_segments)
                self.store.add_segments(event_segments)

                for ev in events:
                    self._emit_event("event_detected", {
                        "event_type": ev.event_type,
                        "description": ev.description,
                        "start_time": ev.start_time,
                        "end_time": ev.end_time,
                        "entities": ev.entities,
                    })
        except Exception as exc:
            logger.error("Knowledge graph update failed: %s", exc)

    def _emit_event(self, event_type: str, data: dict) -> None:
        """Push a real-time event to any registered callback."""
        event = LiveEvent(
            video_id=self.video_id,
            event_type=event_type,
            data=data,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        if self._on_live_event:
            try:
                self._on_live_event(event)
            except Exception:
                pass  # Don't let callback errors kill the capture loop
