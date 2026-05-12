"""
CogniStream — SQLite Persistence

Stores video metadata and processing status so they survive restarts.
Uses raw sqlite3 (no ORM) for minimal dependencies and edge-friendly
memory footprint.

Tables:
    videos   — one row per ingested video (status, metadata, timestamps)
    segments — relational mirror of ChromaDB entries (for listing/counts
               without hitting the vector DB)
    events   — higher-level events detected by the knowledge graph engine

All timestamps are ISO 8601 strings.  IDs are hex UUIDs.
"""

from __future__ import annotations

import logging
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from backend.config import SQLITE_PATH
from backend.db.models import VideoMeta, VideoStatus

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id            TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    file_path     TEXT NOT NULL,
    duration_sec  REAL DEFAULT 0,
    fps           REAL DEFAULT 0,
    width         INTEGER DEFAULT 0,
    height        INTEGER DEFAULT 0,
    total_frames  INTEGER DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'UPLOADED',
    created_at    TEXT NOT NULL,
    processed_at  TEXT,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS segments (
    id          TEXT PRIMARY KEY,
    video_id    TEXT NOT NULL REFERENCES videos(id),
    start_time  REAL NOT NULL,
    end_time    REAL NOT NULL,
    text        TEXT NOT NULL,
    source_type TEXT NOT NULL,
    frame_path  TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id          TEXT PRIMARY KEY,
    video_id    TEXT NOT NULL REFERENCES videos(id),
    event_type  TEXT NOT NULL,
    start_time  REAL NOT NULL,
    end_time    REAL NOT NULL,
    description TEXT,
    entities    TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS annotations (
    id          TEXT PRIMARY KEY,
    video_id    TEXT NOT NULL REFERENCES videos(id),
    start_time  REAL NOT NULL,
    end_time    REAL NOT NULL,
    label       TEXT NOT NULL,
    note        TEXT DEFAULT '',
    color       TEXT DEFAULT '#3b82f6',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS benchmark_runs (
    id              TEXT PRIMARY KEY,
    video_id        TEXT NOT NULL REFERENCES videos(id),
    captured_at     TEXT NOT NULL,
    success         INTEGER NOT NULL,
    elapsed_sec     REAL NOT NULL DEFAULT 0,
    segments_stored INTEGER NOT NULL DEFAULT 0,
    events_detected INTEGER NOT NULL DEFAULT 0,
    stage_timings   TEXT NOT NULL DEFAULT '{}',
    quality_metrics TEXT NOT NULL DEFAULT '{}',
    warnings        TEXT NOT NULL DEFAULT '[]',
    errors          TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_segments_video     ON segments(video_id);
CREATE INDEX IF NOT EXISTS idx_events_video       ON events(video_id);
CREATE INDEX IF NOT EXISTS idx_annotations_video  ON annotations(video_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_video_time ON benchmark_runs(video_id, captured_at DESC);
"""


class SQLiteDB:
    """Thin wrapper around sqlite3 for video metadata persistence."""

    def __init__(self, db_path: Path | None = None):
        self._db_path = str(db_path or SQLITE_PATH)
        self._initialised = False

    # ── connection ──────────────────────────────────────────────

    @contextmanager
    def _connect(self):
        """Yield a sqlite3 connection with WAL mode and foreign keys."""
        if not self._initialised:
            self._init_schema()

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
            logger.info("SQLite schema initialised: %s", self._db_path)
        finally:
            conn.close()
        self._initialised = True

    # ── video CRUD ──────────────────────────────────────────────

    def save_video(self, meta: VideoMeta) -> None:
        """Insert or replace a video record."""
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO videos
                   (id, filename, file_path, duration_sec, fps, width, height,
                    total_frames, status, created_at, processed_at, error_message)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    meta.id, meta.filename, meta.file_path,
                    meta.duration_sec, meta.fps, meta.width, meta.height,
                    meta.total_frames, meta.status.value, meta.created_at,
                    meta.processed_at, meta.error_message,
                ),
            )

    def get_video(self, video_id: str) -> Optional[VideoMeta]:
        """Fetch a single video by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM videos WHERE id = ?", (video_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_meta(row)

    def list_videos(self) -> list[VideoMeta]:
        """Return all videos ordered by creation time (newest first)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM videos ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_meta(r) for r in rows]

    def update_status(
        self,
        video_id: str,
        status: VideoStatus,
        processed_at: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update video processing status."""
        with self._connect() as conn:
            conn.execute(
                """UPDATE videos
                   SET status = ?, processed_at = ?, error_message = ?
                   WHERE id = ?""",
                (status.value, processed_at, error_message, video_id),
            )
        logger.debug("Video %s → %s", video_id, status.value)

    def delete_video(self, video_id: str) -> None:
        """Delete a video and all its associated segments and events."""
        with self._connect() as conn:
            conn.execute("DELETE FROM benchmark_runs WHERE video_id = ?", (video_id,))
            conn.execute("DELETE FROM events WHERE video_id = ?", (video_id,))
            conn.execute("DELETE FROM segments WHERE video_id = ?", (video_id,))
            conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))
        logger.info("Deleted video %s from SQLite.", video_id)

    # ── benchmark runs ─────────────────────────────────────────

    def save_benchmark_run(self, benchmark: dict) -> None:
        """Persist one benchmark run payload for historical tracking."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO benchmark_runs
                   (id, video_id, captured_at, success, elapsed_sec,
                    segments_stored, events_detected, stage_timings,
                    quality_metrics, warnings, errors)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    benchmark["id"],
                    benchmark["video_id"],
                    benchmark["captured_at"],
                    1 if benchmark.get("success") else 0,
                    float(benchmark.get("elapsed_sec", 0.0)),
                    int(benchmark.get("segments_stored", 0)),
                    int(benchmark.get("events_detected", 0)),
                    json.dumps(benchmark.get("stage_timings", {}), ensure_ascii=True),
                    json.dumps(benchmark.get("quality_metrics", {}), ensure_ascii=True),
                    json.dumps(benchmark.get("warnings", []), ensure_ascii=True),
                    json.dumps(benchmark.get("errors", []), ensure_ascii=True),
                ),
            )

    def get_latest_benchmark(self, video_id: str) -> Optional[dict]:
        """Get the most recent benchmark run for a video."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM benchmark_runs
                   WHERE video_id = ?
                   ORDER BY captured_at DESC
                   LIMIT 1""",
                (video_id,),
            ).fetchone()
        if row is None:
            return None
        return self._benchmark_row_to_dict(row)

    def list_benchmark_runs(self, video_id: str, limit: int = 20) -> list[dict]:
        """List benchmark runs (newest first) for a video."""
        safe_limit = max(1, min(200, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM benchmark_runs
                   WHERE video_id = ?
                   ORDER BY captured_at DESC
                   LIMIT ?""",
                (video_id, safe_limit),
            ).fetchall()
        return [self._benchmark_row_to_dict(r) for r in rows]

    def segment_count(self, video_id: str) -> int:
        """Count segments for a video."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM segments WHERE video_id = ?",
                (video_id,),
            ).fetchone()
        return row[0] if row else 0

    def reset_stale_processing(self) -> list[str]:
        """Reset any videos stuck in PROCESSING state back to UPLOADED.

        This handles recovery from interrupted processing (e.g., server crash).
        Returns list of video IDs that were reset.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM videos WHERE status = ?",
                (VideoStatus.PROCESSING.value,),
            ).fetchall()
            if rows:
                conn.execute(
                    "UPDATE videos SET status = ?, error_message = ? WHERE status = ?",
                    (VideoStatus.UPLOADED.value, "Reset after server restart", VideoStatus.PROCESSING.value),
                )
        reset_ids = [r[0] for r in rows]
        if reset_ids:
            logger.warning("Reset %d stale PROCESSING videos: %s", len(reset_ids), reset_ids)
        return reset_ids

    def event_count(self, video_id: str) -> int:
        """Count events for a video."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM events WHERE video_id = ?",
                (video_id,),
            ).fetchone()
        return row[0] if row else 0

    # ── events ───────────────────────────────────────────────────

    def list_events(self, video_id: str) -> list[dict]:
        """Return all events for a video, sorted by start_time."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE video_id = ? ORDER BY start_time",
                (video_id,),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "video_id": r["video_id"],
                "event_type": r["event_type"],
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "description": r["description"],
                "entities": r["entities"].split(",") if r["entities"] else [],
            }
            for r in rows
        ]

    # ── annotations ──────────────────────────────────────────────

    def save_annotation(self, ann: dict) -> None:
        """Insert or replace an annotation."""
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO annotations
                   (id, video_id, start_time, end_time, label, note, color, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    ann["id"], ann["video_id"], ann["start_time"], ann["end_time"],
                    ann["label"], ann.get("note", ""), ann.get("color", "#3b82f6"),
                    ann["created_at"],
                ),
            )

    def list_annotations(self, video_id: str) -> list[dict]:
        """Return all annotations for a video, sorted by start_time."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM annotations WHERE video_id = ? ORDER BY start_time",
                (video_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_annotation(self, annotation_id: str) -> bool:
        """Delete a single annotation. Returns True if deleted."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM annotations WHERE id = ?", (annotation_id,),
            )
        return cursor.rowcount > 0

    # ── helpers ─────────────────────────────────────────────────

    @staticmethod
    def _row_to_meta(row: sqlite3.Row) -> VideoMeta:
        return VideoMeta(
            id=row["id"],
            filename=row["filename"],
            file_path=row["file_path"],
            duration_sec=row["duration_sec"],
            fps=row["fps"],
            width=row["width"],
            height=row["height"],
            total_frames=row["total_frames"],
            status=VideoStatus(row["status"]),
            created_at=row["created_at"],
            processed_at=row["processed_at"],
            error_message=row["error_message"],
        )

    @staticmethod
    def _benchmark_row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "video_id": row["video_id"],
            "captured_at": row["captured_at"],
            "success": bool(row["success"]),
            "elapsed_sec": row["elapsed_sec"],
            "segments_stored": row["segments_stored"],
            "events_detected": row["events_detected"],
            "stage_timings": json.loads(row["stage_timings"] or "{}"),
            "quality_metrics": json.loads(row["quality_metrics"] or "{}"),
            "warnings": json.loads(row["warnings"] or "[]"),
            "errors": json.loads(row["errors"] or "[]"),
        }
