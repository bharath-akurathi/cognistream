"""
CogniStream — Video Loader

Entry point for the ingestion pipeline.  Accepts a raw video file,
validates it, copies it into managed storage, and extracts metadata
(duration, fps, resolution) using OpenCV.

Usage:
    loader = VideoLoader()
    meta = loader.load("/path/to/lecture.mp4")
    # meta is a VideoMeta with status=UPLOADED, ready for processing
"""

from __future__ import annotations

import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2

from backend.config import (
    ALLOWED_VIDEO_EXTENSIONS,
    FRAME_DIR,
    MAX_VIDEO_SIZE_MB,
    VIDEO_DIR,
)
from backend.db.models import VideoMeta, VideoStatus

logger = logging.getLogger(__name__)


class VideoLoadError(Exception):
    """Raised when a video file cannot be loaded or validated."""


class VideoLoader:
    """Validate, store, and extract metadata from a raw video file."""

    def __init__(self, video_dir: Path | None = None):
        self.video_dir = video_dir or VIDEO_DIR
        self.video_dir.mkdir(parents=True, exist_ok=True)

    # ── public API ──────────────────────────────────────────────

    def load(self, source_path: str | Path) -> VideoMeta:
        """Load a video from *source_path* into managed storage.

        Steps:
            1. Validate file extension and size.
            2. Assign a UUID-based video_id.
            3. Copy file to ``data/videos/{video_id}{ext}``.
            4. Probe metadata with OpenCV.
            5. Create a per-video frame directory.

        Returns:
            A populated :class:`VideoMeta` with status ``UPLOADED``.

        Raises:
            VideoLoadError: If the file is missing, too large, or unreadable.
        """
        source = Path(source_path)
        self._validate_file(source)

        video_id = uuid.uuid4().hex
        dest_path = self._copy_to_storage(source, video_id)

        logger.info("Probing video metadata: %s", dest_path)
        meta = self._probe_metadata(video_id, source.name, dest_path)

        # Create a directory for this video's keyframes
        (FRAME_DIR / video_id).mkdir(parents=True, exist_ok=True)

        logger.info(
            "Video loaded: id=%s  duration=%.1fs  fps=%.1f  resolution=%dx%d",
            meta.id,
            meta.duration_sec,
            meta.fps,
            meta.width,
            meta.height,
        )
        return meta

    def load_from_bytes(
        self, data: bytes, filename: str
    ) -> VideoMeta:
        """Load a video from raw bytes (e.g. from an upload endpoint).

        Writes the bytes to a temporary file, then delegates to :meth:`load`.
        """
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            raise VideoLoadError(
                f"Unsupported format '{ext}'. "
                f"Allowed: {ALLOWED_VIDEO_EXTENSIONS}"
            )

        tmp_path = self.video_dir / f"_upload_tmp{ext}"
        try:
            tmp_path.write_bytes(data)
            return self.load(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    # ── internal helpers ────────────────────────────────────────

    def _validate_file(self, path: Path) -> None:
        """Check existence, extension, and file size."""
        if not path.exists():
            raise VideoLoadError(f"File not found: {path}")
        if not path.is_file():
            raise VideoLoadError(f"Not a regular file: {path}")

        ext = path.suffix.lower()
        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            raise VideoLoadError(
                f"Unsupported format '{ext}'. "
                f"Allowed: {ALLOWED_VIDEO_EXTENSIONS}"
            )

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_VIDEO_SIZE_MB:
            raise VideoLoadError(
                f"File too large ({size_mb:.0f} MB). "
                f"Max allowed: {MAX_VIDEO_SIZE_MB} MB"
            )

        logger.debug("Validation passed: %s (%.1f MB)", path.name, size_mb)

    def _copy_to_storage(self, source: Path, video_id: str) -> Path:
        """Copy the source file into managed storage with a deterministic name."""
        ext = source.suffix.lower()
        dest = self.video_dir / f"{video_id}{ext}"
        shutil.copy2(source, dest)
        logger.debug("Copied %s → %s", source, dest)
        return dest

    def _probe_metadata(
        self, video_id: str, original_name: str, file_path: Path
    ) -> VideoMeta:
        """Open the video with OpenCV to read duration, fps, and resolution."""
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise VideoLoadError(f"OpenCV cannot open video: {file_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0.0
        finally:
            cap.release()

        return VideoMeta(
            id=video_id,
            filename=original_name,
            file_path=str(file_path),
            duration_sec=round(duration, 3),
            fps=round(fps, 3),
            width=width,
            height=height,
            total_frames=total_frames,
            status=VideoStatus.UPLOADED,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
