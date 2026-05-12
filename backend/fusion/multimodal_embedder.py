"""
CogniStream — Multimodal Embedder

Fuses visual captions and audio transcripts into unified segments,
then encodes them as dense vectors using SentenceTransformers.

Two responsibilities in one module:

1. **Fusion** — Merge VisualCaption + TranscriptSegment lists by temporal
   overlap.  A caption at t=34s and a transcript spanning [32, 36] are
   combined into one FusedSegment with concatenated text.  Segments
   without a partner are emitted as single-source entries so nothing
   is dropped.

2. **Embedding** — Encode segment text into 384-dim vectors using
   ``all-MiniLM-L6-v2``.  The model is loaded once (lazy singleton,
   ~80 MB RAM) and encodes in batches for throughput.

Usage:
    embedder = MultimodalEmbedder()

    # Fuse visual + audio
    fused = embedder.fuse(video_id, captions, transcripts)

    # Embed all segments
    fused = embedder.embed(fused)

    # Or do both in one call
    fused = embedder.fuse_and_embed(video_id, captions, transcripts)
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

from backend.config import EMBEDDING_MODEL, FUSION_WINDOW_SEC
from backend.db.models import (
    FusedSegment,
    TranscriptSegment,
    VisualCaption,
)

logger = logging.getLogger(__name__)

# Batch size for SentenceTransformer encoding.
# Larger batches are faster but use more RAM.  32 is conservative for edge.
_ENCODE_BATCH_SIZE = 32


class MultimodalEmbedder:
    """Fuse multimodal sources and encode them as dense vectors."""

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or EMBEDDING_MODEL
        self._model = None  # lazy loaded

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Public API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def fuse(
        self,
        video_id: str,
        captions: list[VisualCaption],
        transcripts: list[TranscriptSegment],
    ) -> list[FusedSegment]:
        """Merge visual captions and transcripts by temporal overlap.

        Algorithm:
            1. Build a flat list of visual segments with their time ranges.
            2. For each visual segment, find all transcript segments whose
               time range overlaps (any intersection counts).
            3. Concatenate the visual text and matched transcript text.
            4. Any transcript segments that matched NO visual segment are
               emitted as audio-only entries.

        Returns:
            Sorted list of FusedSegment (by start_time).  The ``.embedding``
            field is ``None`` — call :meth:`embed` to populate it.
        """
        if not captions and not transcripts:
            return []

        fused: list[FusedSegment] = []
        consumed_transcript_indices: set[int] = set()

        # ── assign each transcript to its nearest visual keyframe ──
        # This prevents a transcript from being duplicated across
        # multiple fused segments (inflating its retrieval score).
        transcript_assignments: dict[int, list[int]] = {}  # cap_idx → [t_idx, ...]
        for t_idx, tseg in enumerate(transcripts):
            t_mid = (tseg.start_time + tseg.end_time) / 2.0
            best_cap_idx: int | None = None
            best_dist = float("inf")

            for c_idx, cap in enumerate(captions):
                kf = cap.keyframe
                vis_start = max(0.0, kf.timestamp - FUSION_WINDOW_SEC)
                vis_end = kf.timestamp + FUSION_WINDOW_SEC

                if self._overlaps(vis_start, vis_end, tseg.start_time, tseg.end_time):
                    dist = abs(kf.timestamp - t_mid)
                    if dist < best_dist:
                        best_dist = dist
                        best_cap_idx = c_idx

            if best_cap_idx is not None:
                transcript_assignments.setdefault(best_cap_idx, []).append(t_idx)
                consumed_transcript_indices.add(t_idx)

        # ── visual → fused (with assigned transcripts) ─────────
        for c_idx, cap in enumerate(captions):
            kf = cap.keyframe
            vis_start = max(0.0, kf.timestamp - FUSION_WINDOW_SEC)
            vis_end = kf.timestamp + FUSION_WINDOW_SEC

            # Build the visual text block
            parts: list[str] = []
            if cap.scene_description:
                parts.append(cap.scene_description)
            if cap.objects:
                parts.append(f"Objects: {', '.join(cap.objects)}")
            if cap.activity:
                parts.append(f"Activity: {cap.activity}")
            if cap.anomaly:
                parts.append(f"Anomaly: {cap.anomaly}")
            visual_text = " ".join(parts)

            # Merge only the transcripts assigned to this keyframe
            assigned = transcript_assignments.get(c_idx, [])
            transcript_text = " ".join(transcripts[i].text for i in assigned)

            # Combine
            combined = visual_text
            if transcript_text.strip():
                combined += f" [Speech: {transcript_text.strip()}]"

            source_type = "fused" if transcript_text.strip() else "visual"

            fused.append(
                FusedSegment(
                    id=uuid.uuid4().hex,
                    video_id=video_id,
                    start_time=round(vis_start, 3),
                    end_time=round(vis_end, 3),
                    text=combined,
                    source_type=source_type,
                    frame_path=kf.file_path,
                )
            )

        # ── unmatched transcripts → audio-only segments ────────
        for t_idx, tseg in enumerate(transcripts):
            if t_idx not in consumed_transcript_indices:
                fused.append(
                    FusedSegment(
                        id=uuid.uuid4().hex,
                        video_id=video_id,
                        start_time=tseg.start_time,
                        end_time=tseg.end_time,
                        text=tseg.text,
                        source_type="audio",
                    )
                )

        fused.sort(key=lambda s: s.start_time)

        logger.info(
            "Fusion complete: %d visual + %d audio → %d fused segments "
            "(%d audio-only)",
            len(captions),
            len(transcripts),
            len(fused),
            len(transcripts) - len(consumed_transcript_indices),
        )
        return fused

    def embed(self, segments: list[FusedSegment]) -> list[FusedSegment]:
        """Encode all segment texts as dense vectors (in-place).

        Uses NVIDIA NV-Embed when available, otherwise falls back to
        local SentenceTransformers.
        """
        if not segments:
            return segments

        from backend.providers.nvidia import nvidia

        texts = [seg.text for seg in segments]
        t_start = time.monotonic()

        # Try NVIDIA cloud embeddings first
        if nvidia.available:
            logger.info("Embedding %d segments via NVIDIA NV-Embed", len(texts))
            vectors = nvidia.embed_texts(texts, input_type="passage")
            if vectors and len(vectors) == len(segments):
                for seg, vec in zip(segments, vectors):
                    seg.embedding = vec
                elapsed = time.monotonic() - t_start
                logger.info(
                    "NVIDIA embedding complete: %d vectors in %.1fs",
                    len(segments), elapsed,
                )
                return segments
            logger.warning("NVIDIA embed failed or partial, falling back to local")

        # Local SentenceTransformers fallback
        model = self._get_model()
        logger.info("Embedding %d segments locally (batch_size=%d)", len(texts), _ENCODE_BATCH_SIZE)

        vectors = model.encode(
            texts,
            batch_size=_ENCODE_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        for seg, vec in zip(segments, vectors):
            seg.embedding = vec.tolist()

        elapsed = time.monotonic() - t_start
        logger.info(
            "Embedding complete: %d vectors in %.1fs (%.0f seg/s)",
            len(segments),
            elapsed,
            len(segments) / elapsed if elapsed > 0 else 0,
        )
        return segments

    def embed_query(self, query: str) -> list[float]:
        """Encode a single query string into a vector.

        Uses NVIDIA when available, otherwise local SentenceTransformers.
        """
        from backend.providers.nvidia import nvidia

        if nvidia.available:
            vec = nvidia.embed_text(query, input_type="query")
            if vec:
                return vec

        model = self._get_model()
        vec = model.encode(
            query,
            normalize_embeddings=True,
        )
        return vec.tolist()

    def embed_query_local(self, query: str) -> list[float]:
        """Encode a query with the local SentenceTransformer only.

        Useful when the active Chroma collection stores local embeddings
        and the cloud provider returns a different vector dimension.
        """
        model = self._get_model()
        vec = model.encode(
            query,
            normalize_embeddings=True,
        )
        return vec.tolist()

    def fuse_and_embed(
        self,
        video_id: str,
        captions: list[VisualCaption],
        transcripts: list[TranscriptSegment],
    ) -> list[FusedSegment]:
        """Convenience method: fuse, then embed in one call."""
        segments = self.fuse(video_id, captions, transcripts)
        return self.embed(segments)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Model management
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _get_model(self):
        """Lazy-load the SentenceTransformer model."""
        if self._model is not None:
            return self._model

        logger.info("Loading SentenceTransformer: %s", self._model_name)
        t_start = time.monotonic()

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_name)

        logger.info(
            "SentenceTransformer loaded in %.1fs",
            time.monotonic() - t_start,
        )
        return self._model

    def unload_model(self) -> None:
        """Release the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("SentenceTransformer model unloaded.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Helpers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _overlaps(
        a_start: float, a_end: float,
        b_start: float, b_end: float,
    ) -> bool:
        """Return True if intervals [a_start, a_end] and [b_start, b_end] overlap."""
        return a_start < b_end and b_start < a_end
