"""
CogniStream — Pipeline Orchestrator

Wires all processing modules into a sequential pipeline with:
    - Status tracking (persisted to SQLite)
    - Graceful degradation (VLM failure → audio-only mode)
    - Model lifecycle management (unload between stages)
    - Progress callbacks for real-time frontend updates
    - Single-job concurrency guard for edge hardware

Pipeline stages:
    1. Load & validate video         (VideoLoader)
    2. Detect shot boundaries        (ShotDetector)
    3. Extract keyframes             (FrameSampler)
    4. Extract audio                 (AudioExtractor)
    5. Visual analysis via VLM       (VLMRunner)       — skippable
    6. Audio transcription           (WhisperRunner)   — skippable
    7. Multimodal fusion & embedding (MultimodalEmbedder)
    8. Knowledge graph construction  (KnowledgeGraph)
    9. Event detection               (EventDetector)
   10. Store to ChromaDB             (ChromaStore)

Usage:
    orchestrator = PipelineOrchestrator()
    result = orchestrator.process(video_meta)
"""

from __future__ import annotations

import csv
import logging
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from backend.audio.audio_extractor import AudioExtractor
from backend.audio.whisper_runner import WhisperRunner
from backend.db.chroma_store import ChromaStore
from backend.db.models import (
    FusedSegment,
    TranscriptSegment,
    VideoMeta,
    VideoStatus,
    VisualCaption,
)
from backend.db.sqlite import SQLiteDB
from backend.fusion.multimodal_embedder import MultimodalEmbedder
from backend.ingestion.frame_sampler import FrameSampler
from backend.ingestion.shot_detector import ShotDetector
from backend.knowledge.event_detector import EventDetector
from backend.knowledge.graph import KnowledgeGraph
from backend.visual.vlm_runner import OllamaClient, VLMRunner
from backend.config import resolve_pipeline_stage_workers

logger = logging.getLogger(__name__)


def _get_process_rss_mb() -> float:
    """Best-effort current process RSS in MiB (works without psutil)."""
    try:
        import psutil  # type: ignore

        rss = psutil.Process(os.getpid()).memory_info().rss
        return round(float(rss) / (1024 * 1024), 2)
    except Exception:
        pass

    if os.name == "nt":
        try:
            output = subprocess.check_output(
                ["tasklist", "/fi", f"PID eq {os.getpid()}", "/fo", "csv", "/nh"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if output and """No tasks are running which match the specified criteria.""" not in output:
                rows = list(csv.reader([output]))
                if rows:
                    row = rows[0]
                    if len(row) >= 5:
                        mem_field = row[4]
                        mem_kb_text = re.sub(r"[^0-9]", "", mem_field)
                        if mem_kb_text:
                            return round(int(mem_kb_text) / 1024.0, 2)
        except Exception:
            pass

    try:
        import resource

        # Linux reports KiB, macOS reports bytes.
        ru = resource.getrusage(resource.RUSAGE_SELF)
        scale = 1.0 if os.uname().sysname.lower() == "darwin" else 1024.0  # type: ignore[attr-defined]
        return round(float(ru.ru_maxrss) * scale / (1024 * 1024), 2)
    except Exception:
        return 0.0


@dataclass
class PipelineProgress:
    """Tracks progress through the pipeline stages."""
    video_id: str
    stage: str = ""
    stage_number: int = 0
    total_stages: int = 10
    detail: str = ""
    started_at: float = 0.0
    elapsed_sec: float = 0.0


@dataclass
class PipelineResult:
    """Final result of a pipeline run."""
    video_id: str
    success: bool
    segments_stored: int = 0
    events_detected: int = 0
    elapsed_sec: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stage_timings: dict[str, float] = field(default_factory=dict)
    quality_metrics: dict[str, float] = field(default_factory=dict)


# Type alias for progress callback
ProgressCallback = Callable[[PipelineProgress], None]


class PipelineOrchestrator:
    """End-to-end video processing pipeline with fault tolerance."""

    def __init__(
        self,
        db: SQLiteDB | None = None,
        store: ChromaStore | None = None,
        on_progress: ProgressCallback | None = None,
    ):
        self.db = db or SQLiteDB()
        self.store = store or ChromaStore()
        self._on_progress = on_progress
        self._lock = threading.Lock()
        self._active_video: Optional[str] = None

    @property
    def is_busy(self) -> bool:
        """True if a video is currently being processed."""
        return self._active_video is not None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Main entry point
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def process(self, meta: VideoMeta) -> PipelineResult:
        """Run the full processing pipeline for a video.

        Thread-safe: only one video can process at a time.
        Additional calls while busy raise RuntimeError.
        """
        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            raise RuntimeError(
                f"Pipeline busy processing {self._active_video}. "
                "Only one video at a time on edge hardware."
            )

        self._active_video = meta.id
        t_start = time.monotonic()
        result = PipelineResult(video_id=meta.id, success=False)
        progress = PipelineProgress(
            video_id=meta.id,
            started_at=t_start,
        )
        embedder: Optional[MultimodalEmbedder] = None
        timing_starts: dict[str, float] = {}
        start_rss_mb = _get_process_rss_mb()
        peak_rss_mb = start_rss_mb
        memory_stop = threading.Event()

        def _watch_memory() -> None:
            nonlocal peak_rss_mb
            while not memory_stop.wait(0.25):
                current = _get_process_rss_mb()
                if current > peak_rss_mb:
                    peak_rss_mb = current

        memory_thread = threading.Thread(target=_watch_memory, daemon=True)
        memory_thread.start()

        def _start_timer(name: str) -> None:
            timing_starts[name] = time.monotonic()

        def _stop_timer(name: str) -> None:
            t0 = timing_starts.pop(name, None)
            if t0 is None:
                return
            result.stage_timings[name] = round(time.monotonic() - t0, 3)

        try:
            # Mark as processing
            self.db.update_status(meta.id, VideoStatus.PROCESSING)

            # ── Stages 1-3: Ingestion (audio runs in parallel with visual) ─
            self._emit(progress, 1, "Shot detection + Audio extraction (parallel)")

            # Audio extraction is independent of shot detection — run in parallel
            audio_result = None
            audio_elapsed_sec = 0.0
            def _extract_audio():
                nonlocal audio_result
                nonlocal audio_elapsed_sec
                t0 = time.monotonic()
                ext = AudioExtractor()
                audio_result = ext.extract(meta)
                audio_elapsed_sec = time.monotonic() - t0

            audio_thread = threading.Thread(target=_extract_audio, daemon=True)
            audio_thread.start()

            # Shot detection + frame sampling (sequential, frame sampling needs shots)
            _start_timer("shot_detection_sec")
            detector = ShotDetector()
            segments = detector.detect(meta)
            _stop_timer("shot_detection_sec")

            self._emit(progress, 2, "Frame sampling")
            _start_timer("frame_sampling_sec")
            sampler = FrameSampler()
            keyframes = sampler.sample(meta, segments)
            _stop_timer("frame_sampling_sec")

            # Wait for audio extraction to finish
            audio_thread.join()
            result.stage_timings["audio_extraction_sec"] = round(audio_elapsed_sec, 3)
            self._emit(progress, 3, "Audio extraction complete")

            # ── Stages 4+5: VLM + Whisper ─────────────────────
            # GPU memory strategy:
            #   - If whisper uses "small" model (~2GB), run VLM + Whisper in PARALLEL
            #   - If whisper uses "large-v3-turbo" (~6GB), run SEQUENTIALLY with GPU swap
            #     (unload VLM → run Whisper → reload VLM is not needed since VLM runs first)
            from backend.config import WHISPER_MODEL_SIZE
            large_whisper = WHISPER_MODEL_SIZE in ("large-v3", "large-v3-turbo", "distil-large-v3", "large")

            captions: list[VisualCaption] = []
            transcripts: list[TranscriptSegment] = []
            vlm_client: OllamaClient | None = None
            vlm_runner_ref: VLMRunner | None = None

            def _run_vlm() -> list[VisualCaption]:
                nonlocal vlm_client, vlm_runner_ref
                try:
                    from backend.providers.nvidia import nvidia
                    vlm_client = OllamaClient()
                    # NVIDIA cloud VLM does not require local Ollama.
                    if nvidia.available:
                        runner = VLMRunner(vlm_client)
                        vlm_runner_ref = runner
                        return runner.analyse_keyframes(keyframes)
                    if vlm_client.is_available():
                        runner = VLMRunner(vlm_client)
                        vlm_runner_ref = runner
                        return runner.analyse_keyframes(keyframes)
                    else:
                        result.warnings.append("Ollama not available — skipping VLM analysis.")
                        return []
                except Exception as exc:
                    result.warnings.append(f"VLM analysis failed: {exc}")
                    logger.error("VLM analysis failed: %s", exc)
                    return []

            def _run_whisper() -> list[TranscriptSegment]:
                if not audio_result or audio_result.is_silent:
                    if audio_result and audio_result.is_silent:
                        result.warnings.append("Audio track is silent — skipping transcription.")
                    else:
                        result.warnings.append("No audio stream — skipping transcription.")
                    return []
                try:
                    whisper = WhisperRunner()
                    segs = whisper.transcribe(audio_result.audio_path)
                    whisper.unload_model()
                    return segs
                except Exception as exc:
                    result.warnings.append(f"Transcription failed: {exc}")
                    logger.error("Transcription failed: %s", exc)
                    return []

            stage_workers = resolve_pipeline_stage_workers()

            if large_whisper:
                # Sequential: VLM first → unload from GPU → Whisper gets full VRAM
                self._emit(progress, 4, "Visual analysis (VLM)")
                _start_timer("vlm_sec")
                captions = _run_vlm()
                _stop_timer("vlm_sec")
                self._emit(progress, 5, "Unloading VLM for Whisper")
                if vlm_client:
                    vlm_client.unload()  # Free GPU VRAM
                self._emit(progress, 5, "Audio transcription (large model)")
                _start_timer("transcription_sec")
                transcripts = _run_whisper()
                _stop_timer("transcription_sec")
            else:
                # Parallel: both fit in VRAM simultaneously
                self._emit(progress, 4, "Visual + Audio analysis (parallel)")
                _start_timer("vlm_sec")
                _start_timer("transcription_sec")
                with ThreadPoolExecutor(max_workers=stage_workers, thread_name_prefix="pipeline") as pool:
                    vlm_future: Future = pool.submit(_run_vlm)
                    whisper_future: Future = pool.submit(_run_whisper)
                    captions = vlm_future.result()
                    _stop_timer("vlm_sec")
                    self._emit(progress, 5, "Visual analysis complete")
                    transcripts = whisper_future.result()
                    _stop_timer("transcription_sec")
                    self._emit(progress, 5, "Audio transcription complete")

            # ── Stage 6b: Visual frame embeddings (NVCLIP or SigLIP) ─
            from backend.providers.nvidia import nvidia
            from backend.config import SIGLIP_ENABLED
            import uuid as _uuid
            visual_segments: list[FusedSegment] = []

            if keyframes and SIGLIP_ENABLED:
                image_paths = [kf.file_path for kf in keyframes]
                clip_vectors = None

                if nvidia.available:
                    self._emit(progress, 5, "NVCLIP image embeddings")
                    clip_vectors = nvidia.embed_images(image_paths)
                    if clip_vectors:
                        logger.info("NVCLIP: %d image embeddings", len(clip_vectors))
                else:
                    # Local SigLIP fallback
                    from backend.visual.siglip_embedder import SigLIPEmbedder
                    siglip = SigLIPEmbedder()
                    if siglip.enabled:
                        self._emit(progress, 5, "SigLIP image embeddings")
                        clip_vectors = siglip.embed_images(image_paths)
                        if clip_vectors:
                            logger.info("SigLIP: %d image embeddings", len(clip_vectors))
                        siglip.unload()

                if clip_vectors:
                    for kf, vec in zip(keyframes, clip_vectors):
                        visual_segments.append(FusedSegment(
                            id=_uuid.uuid4().hex,
                            video_id=meta.id,
                            start_time=kf.timestamp,
                            end_time=kf.timestamp + 1.0,
                            text=f"[image] keyframe at {kf.timestamp:.1f}s",
                            source_type="visual",
                            frame_path=kf.file_path,
                            embedding=vec,
                        ))

            # ── Stage 6c: Grounding DINO object detection (when NVIDIA available) ─
            if nvidia.available and keyframes:
                self._emit(progress, 5, "Object detection (Grounding DINO)")
                common_objects = ["person", "car", "vehicle", "building", "animal", "bag", "phone"]
                for kf in keyframes[:20]:  # Limit to first 20 keyframes for speed
                    detections = nvidia.detect_objects(kf.file_path, common_objects)
                    if detections:
                        det_text = ", ".join(
                            f"{d['label']} ({d['confidence']:.0%})"
                            for d in detections
                        )
                        # Enrich captions with detection info
                        for cap in captions:
                            if cap.keyframe.frame_number == kf.frame_number:
                                cap.objects = list(set(cap.objects + [d["label"] for d in detections]))
                                break

            # ── Stage 7: Fusion & embedding ────────────────────
            self._emit(progress, 6, "Multimodal fusion")
            _start_timer("fusion_embedding_sec")
            fused: list = []
            if not captions and not transcripts:
                msg = "No captions or transcripts — skipping fusion (video metadata still saved)."
                logger.warning(msg)
                result.warnings.append(msg)
            else:
                embedder = MultimodalEmbedder()
                fused = embedder.fuse_and_embed(meta.id, captions, transcripts)
            _stop_timer("fusion_embedding_sec")

            # Add NVCLIP image segments (already have embeddings, skip re-embedding)
            if visual_segments:
                fused.extend(visual_segments)

            # ── Stage 8: Knowledge graph ───────────────────────
            self._emit(progress, 7, "Knowledge graph")
            _start_timer("knowledge_graph_sec")
            kg = KnowledgeGraph(meta.id)
            kg.build_from_captions(captions, transcripts)
            kg.save()
            _stop_timer("knowledge_graph_sec")

            # ── Stage 9: Event detection ───────────────────────
            self._emit(progress, 8, "Event detection")
            _start_timer("event_detection_sec")
            event_detector = EventDetector()
            events = event_detector.detect(kg)
            result.events_detected = len(events)
            _stop_timer("event_detection_sec")

            # Add events as searchable segments
            event_segments = self._events_to_segments(meta.id, events)
            if event_segments:
                if embedder is None:
                    embedder = MultimodalEmbedder()
                embedder.embed(event_segments)
                fused.extend(event_segments)

            # Free embedding model memory
            if embedder is not None:
                embedder.unload_model()

            # ── Stage 10: Store to ChromaDB ────────────────────
            self._emit(progress, 9, "Storing embeddings")
            _start_timer("store_sec")
            # Purge old data for this video (idempotent reprocessing)
            self.store.purge_video(meta.id)
            if fused:
                result.segments_stored = self.store.add_segments(fused)
            else:
                result.segments_stored = 0
                logger.info("No segments to store — video processed without searchable content.")
            _stop_timer("store_sec")

            # ── Finalise ───────────────────────────────────────
            self._emit(progress, 10, "Complete")
            elapsed = time.monotonic() - t_start
            result.elapsed_sec = round(elapsed, 1)
            result.success = True

            # Quality/efficiency diagnostics for benchmarking.
            novelty = getattr(vlm_runner_ref, "last_novelty_stats", {}) if vlm_runner_ref else {}
            reuse = getattr(vlm_runner_ref, "last_reuse_stats", {}) if vlm_runner_ref else {}
            captions_with_fallback = sum(
                1 for c in captions if "no scene description available" in (c.scene_description or "").lower()
            )
            static_activity = sum(
                1 for c in captions if (c.activity or "").strip().lower() == "static scene"
            )
            total_caps = max(1, len(captions))
            result.quality_metrics = {
                "keyframes_input": float(novelty.get("input", len(keyframes))),
                "keyframes_kept": float(novelty.get("kept", len(keyframes))),
                "keyframes_dropped": float(novelty.get("dropped", 0)),
                "reuse_hits_total": float(reuse.get("reuse_hits_total", 0)),
                "reuse_hits_exact": float(reuse.get("reuse_hits_exact", 0)),
                "reuse_hits_semantic": float(reuse.get("reuse_hits_semantic", 0)),
                "reuse_misses": float(reuse.get("reuse_misses", 0)),
                "reuse_candidates_checked": float(reuse.get("reuse_candidates_checked", 0)),
                "captions_count": float(len(captions)),
                "transcripts_count": float(len(transcripts)),
                "captions_fallback_ratio": round(captions_with_fallback / total_caps, 4),
                "captions_static_ratio": round(static_activity / total_caps, 4),
            }

            self.db.update_status(
                meta.id,
                VideoStatus.PROCESSED,
                processed_at=datetime.now(timezone.utc).isoformat(),
            )

            logger.info(
                "Pipeline complete: video=%s, segments=%d, events=%d, time=%.1fs",
                meta.id, result.segments_stored, result.events_detected, elapsed,
            )

            # Fire webhook notification
            from backend.webhooks import fire_webhook
            fire_webhook("video_processed", {
                "video_id": meta.id,
                "filename": meta.filename,
                "segments_stored": result.segments_stored,
                "events_detected": result.events_detected,
                "elapsed_sec": result.elapsed_sec,
            })

        except Exception as exc:
            msg = f"Pipeline failed: {exc}"
            logger.exception(msg)
            result.errors.append(msg)
            self._mark_failed(meta.id, str(exc))

        finally:
            memory_stop.set()
            memory_thread.join(timeout=1.0)
            final_rss_mb = _get_process_rss_mb()
            peak_rss_mb = max(peak_rss_mb, final_rss_mb)
            result.quality_metrics.setdefault("process_rss_start_mb", round(start_rss_mb, 2))
            result.quality_metrics.setdefault("process_rss_peak_mb", round(peak_rss_mb, 2))
            result.quality_metrics.setdefault(
                "process_rss_delta_mb",
                round(max(0.0, peak_rss_mb - start_rss_mb), 2),
            )

            # Release models that may still be loaded on exception path
            if embedder is not None:
                try:
                    embedder.unload_model()
                except Exception:
                    pass
            self._active_video = None
            self._lock.release()

        return result

    # ── helpers ─────────────────────────────────────────────────

    def _emit(self, progress: PipelineProgress, stage: int, name: str) -> None:
        """Update and emit progress."""
        progress.stage_number = stage
        progress.stage = name
        progress.elapsed_sec = round(time.monotonic() - progress.started_at, 1)
        progress.detail = f"Stage {stage}/{progress.total_stages}: {name}"
        logger.info(progress.detail)
        if self._on_progress:
            self._on_progress(progress)

    def _mark_failed(self, video_id: str, error: str) -> None:
        self.db.update_status(video_id, VideoStatus.FAILED, error_message=error[:500])

    @staticmethod
    def _events_to_segments(
        video_id: str, events: list
    ) -> list[FusedSegment]:
        """Convert detected events into embeddable FusedSegments."""
        import uuid as _uuid

        segments = []
        for ev in events:
            segments.append(FusedSegment(
                id=_uuid.uuid4().hex,
                video_id=video_id,
                start_time=ev.start_time,
                end_time=ev.end_time,
                text=f"Event: {ev.event_type}. {ev.description}. Entities: {', '.join(ev.entities)}",
                source_type="event",
            ))
        return segments
