"""
CogniStream — ChromaDB Store

Persistent vector storage for fused segments.  Wraps a single ChromaDB
collection with typed helpers for upserting, querying, and managing
video lifecycle.

Design decisions:

* **Upsert semantics** — Reprocessing a video overwrites previous segments
  instead of duplicating them, keyed by segment UUID.

* **Rich metadata** — ``video_id``, ``start_time``, ``end_time``,
  ``source_type``, and ``frame_path`` are stored as metadata so
  ChromaDB's ``where`` filter can scope searches without post-filtering.

* **Persistent client** — ChromaDB runs in embedded mode with a
  persistent directory (``data/db/chroma/``).  No server process needed.

Usage:
    store = ChromaStore()
    store.add_segments(fused_segments)
    results = store.query(embedding, top_k=10, video_id="abc")
    store.purge_video("abc")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

# Keep local logs clean and avoid posthog compatibility warnings.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

import chromadb

from backend.config import CHROMA_COLLECTION, CHROMA_DIR, CHROMA_HOST, CHROMA_PORT
from backend.db.models import FusedSegment

logger = logging.getLogger(__name__)


class ChromaStore:
    """Typed wrapper around a persistent ChromaDB collection."""

    def __init__(
        self,
        persist_dir: Path | None = None,
        collection_name: str | None = None,
        host: str | None = None,
        port: int | None = None,
    ):
        self._persist_dir = str(persist_dir or CHROMA_DIR)
        self._collection_name = collection_name or CHROMA_COLLECTION
        self._host = host if host is not None else CHROMA_HOST
        self._port = port if port is not None else CHROMA_PORT
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Connection
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _get_collection(self) -> chromadb.Collection:
        """Lazy-initialise the ChromaDB client and collection.

        Uses HTTP client when CHROMA_HOST is set (Docker deployment),
        otherwise falls back to embedded PersistentClient (local dev).
        """
        if self._collection is not None:
            return self._collection

        if self._host:
            logger.info(
                "Initialising ChromaDB HTTP client: %s:%d, collection=%s",
                self._host,
                self._port,
                self._collection_name,
            )
            try:
                self._client = chromadb.HttpClient(
                    host=self._host,
                    port=self._port,
                )
                # Validate connectivity so local dev can fall back cleanly.
                self._client.heartbeat()
            except Exception as exc:
                logger.warning(
                    "ChromaDB HTTP unavailable at %s:%d (%s). Falling back to embedded storage at %s.",
                    self._host,
                    self._port,
                    exc,
                    self._persist_dir,
                )
                self._client = chromadb.PersistentClient(path=self._persist_dir)
        else:
            logger.info(
                "Initialising ChromaDB persistent client: dir=%s, collection=%s",
                self._persist_dir,
                self._collection_name,
            )
            self._client = chromadb.PersistentClient(path=self._persist_dir)

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        count = self._collection.count()
        logger.info("ChromaDB ready: %d existing documents.", count)
        return self._collection

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Write operations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def add_segments(self, segments: list[FusedSegment]) -> int:
        """Upsert a batch of fused segments into ChromaDB.

        Segments without an embedding are skipped with a warning.

        Args:
            segments: List of FusedSegment with populated ``.embedding``.

        Returns:
            Number of segments actually stored.
        """
        collection = self._get_collection()

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for seg in segments:
            if seg.embedding is None:
                logger.warning("Segment %s has no embedding — skipping.", seg.id)
                continue

            ids.append(seg.id)
            embeddings.append(seg.embedding)
            documents.append(seg.text)
            metadatas.append(self._segment_metadata(seg))

        # Validate all embeddings have the same dimension
        if embeddings:
            dimensions = set(len(e) for e in embeddings)
            if len(dimensions) > 1:
                dim_counts = {}
                for e in embeddings:
                    dim = len(e)
                    dim_counts[dim] = dim_counts.get(dim, 0) + 1
                logger.error(
                    "Embedding dimension mismatch detected! Found %d different dimensions: %s. "
                    "This usually means SigLIP (768-dim) is being mixed with text embeddings (384-dim). "
                    "Fix: Set SIGLIP_MODEL='' in .env to disable SigLIP, or use separate collections.",
                    len(dimensions),
                    dim_counts,
                )
                raise ValueError(
                    f"Inconsistent embedding dimensions: {dim_counts}. "
                    "See logs for troubleshooting. Likely SigLIP + text embedding mismatch."
                )

        if not ids:
            logger.warning("No valid segments to store.")
            return 0

        # Upsert in chunks of 500 (ChromaDB batch limit)
        stored = 0
        for i in range(0, len(ids), 500):
            end = i + 500
            collection.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )
            stored += len(ids[i:end])

        logger.info(
            "Stored %d segments in ChromaDB (total: %d).",
            stored,
            collection.count(),
        )
        return stored

    def purge_video(self, video_id: str) -> int:
        """Delete all segments for a given video.

        Uses filter-based delete to avoid the TOCTOU race condition
        of get-then-delete (where new segments could be inserted
        between the two calls).

        Returns:
            Approximate number of documents deleted.
        """
        collection = self._get_collection()

        # Count before delete for logging (best-effort)
        existing = collection.get(
            where={"video_id": video_id},
            include=[],
        )
        count = len(existing["ids"])

        if count == 0:
            logger.debug("No segments found for video %s.", video_id)
            return 0

        # Atomic filter-based delete — no gap between read and write
        collection.delete(where={"video_id": video_id})
        logger.info("Purged %d segments for video %s.", count, video_id)
        return count

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Read operations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def query(
        self,
        embedding: list[float],
        top_k: int = 10,
        video_id: Optional[str] = None,
        source_filter: Optional[str] = None,
    ) -> list[dict]:
        """Search for the nearest segments to a query embedding.

        Args:
            embedding:     Query vector (384-dim, normalised).
            top_k:         Number of results to return.
            video_id:      Optional — scope search to a single video.
            source_filter: Optional — one of "visual", "audio", "fused", "event".

        Returns:
            List of dicts, each containing:
                id, text, score (0–1, higher = more similar),
                video_id, start_time, end_time, source_type, frame_path.
        """
        collection = self._get_collection()

        where = self._build_where_filter(video_id, source_filter)

        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        return self._format_results(results)

    def get_by_video(self, video_id: str) -> list[dict]:
        """Retrieve all stored segments for a video (no similarity search)."""
        collection = self._get_collection()

        results = collection.get(
            where={"video_id": video_id},
            include=["documents", "metadatas"],
        )

        items = []
        for i, doc_id in enumerate(results["ids"]):
            meta = results["metadatas"][i] if results["metadatas"] else {}
            items.append({
                "id": doc_id,
                "text": results["documents"][i] if results["documents"] else "",
                **meta,
            })

        items.sort(key=lambda x: x.get("start_time", 0.0))
        return items

    def get_segment(self, segment_id: str) -> dict | None:
        """Retrieve a single segment by ID, including its embedding."""
        collection = self._get_collection()
        result = collection.get(
            ids=[segment_id],
            include=["documents", "metadatas", "embeddings"],
        )
        if not result["ids"]:
            return None
        meta = result["metadatas"][0] if result["metadatas"] else {}
        return {
            "id": result["ids"][0],
            "text": result["documents"][0] if result["documents"] else "",
            "embedding": result["embeddings"][0] if result["embeddings"] else None,
            **meta,
        }

    def count(self, video_id: Optional[str] = None) -> int:
        """Return the number of stored segments, optionally filtered."""
        collection = self._get_collection()
        if video_id is None:
            return collection.count()

        results = collection.get(
            where={"video_id": video_id},
            include=[],
        )
        return len(results["ids"])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Helpers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def purge_expired(self, ttl_hours: int) -> int:
        """Delete segments older than ``ttl_hours``.

        Uses the ``indexed_at`` metadata field (epoch seconds).
        Returns the number of segments deleted.
        """
        if ttl_hours <= 0:
            return 0

        import time
        collection = self._get_collection()
        cutoff = time.time() - (ttl_hours * 3600)

        try:
            old = collection.get(
                where={"indexed_at": {"$lt": cutoff}},
                include=[],
            )
            if not old["ids"]:
                return 0

            collection.delete(ids=old["ids"])
            logger.info("Purged %d expired segments (TTL=%dh).", len(old["ids"]), ttl_hours)
            return len(old["ids"])
        except Exception as exc:
            logger.debug("TTL purge skipped: %s", exc)
            return 0

    @staticmethod
    def _segment_metadata(seg: FusedSegment) -> dict:
        """Build the metadata dict stored alongside each vector."""
        import time
        meta = {
            "video_id": seg.video_id,
            "start_time": seg.start_time,
            "end_time": seg.end_time,
            "source_type": seg.source_type,
            "indexed_at": time.time(),
        }
        if seg.frame_path:
            meta["frame_path"] = seg.frame_path
        return meta

    @staticmethod
    def _build_where_filter(
        video_id: Optional[str],
        source_filter: Optional[str],
    ) -> Optional[dict]:
        """Build a ChromaDB ``where`` filter from optional constraints."""
        conditions: list[dict] = []

        if video_id:
            conditions.append({"video_id": {"$eq": video_id}})
        if source_filter:
            conditions.append({"source_type": {"$eq": source_filter}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @staticmethod
    def _format_results(raw: dict) -> list[dict]:
        """Convert ChromaDB query output into a flat list of result dicts.

        ChromaDB returns distances (lower = closer).  For cosine space,
        distance ∈ [0, 2].  We convert to a similarity score ∈ [0, 1]
        via ``score = 1 - (distance / 2)``.
        """
        if not raw["ids"] or not raw["ids"][0]:
            return []

        ids = raw["ids"][0]
        documents = raw["documents"][0] if raw["documents"] else [""] * len(ids)
        metadatas = raw["metadatas"][0] if raw["metadatas"] else [{}] * len(ids)
        distances = raw["distances"][0] if raw["distances"] else [0.0] * len(ids)

        results = []
        for i, doc_id in enumerate(ids):
            meta = metadatas[i] if metadatas[i] else {}
            distance = distances[i]
            score = 1.0 - (distance / 2.0)  # cosine distance → similarity

            results.append({
                "id": doc_id,
                "text": documents[i],
                "score": round(max(0.0, min(1.0, score)), 4),
                "video_id": meta.get("video_id", ""),
                "start_time": meta.get("start_time", 0.0),
                "end_time": meta.get("end_time", 0.0),
                "source_type": meta.get("source_type", ""),
                "frame_path": meta.get("frame_path"),
            })

        return results
