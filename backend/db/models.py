"""
CogniStream — Data Models

Pure dataclasses that represent domain objects flowing through the pipeline.
No ORM coupling — these are passed between modules and serialised to DB/ChromaDB
by the db layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class VideoStatus(str, Enum):
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"


@dataclass
class VideoMeta:
    """Metadata for an ingested video file."""
    id: str
    filename: str
    file_path: str
    duration_sec: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    total_frames: int = 0
    status: VideoStatus = VideoStatus.UPLOADED
    created_at: str = ""
    processed_at: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class ShotSegment:
    """A contiguous shot detected by the shot boundary detector."""
    segment_index: int
    start_frame: int
    end_frame: int
    start_time: float   # seconds
    end_time: float     # seconds
    frame_count: int = 0


@dataclass
class Keyframe:
    """A single extracted keyframe."""
    video_id: str
    segment_index: int
    frame_number: int
    timestamp: float      # seconds
    file_path: str        # saved image path


@dataclass
class VisualCaption:
    """Output from the Visual Narrative Engine for one keyframe."""
    keyframe: Keyframe
    scene_description: str = ""
    objects: list[str] = field(default_factory=list)
    activity: str = ""
    anomaly: Optional[str] = None
    reused_from_frame: Optional[int] = None
    reuse_type: Optional[str] = None  # exact | semantic
    reuse_similarity: Optional[float] = None


@dataclass
class TranscriptSegment:
    """A chunk of transcribed speech."""
    start_time: float
    end_time: float
    text: str
    keywords: list[str] = field(default_factory=list)


@dataclass
class FusedSegment:
    """Merged visual + audio segment ready for embedding."""
    id: str
    video_id: str
    start_time: float
    end_time: float
    text: str                       # combined caption + transcript
    source_type: str = "fused"      # visual | audio | fused | event
    frame_path: Optional[str] = None
    embedding: Optional[list[float]] = None


@dataclass
class Event:
    """A higher-level event inferred from action sequences."""
    id: str
    video_id: str
    event_type: str
    start_time: float
    end_time: float
    description: str = ""
    entities: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """A single result returned by the retrieval engine."""
    video_id: str
    segment_id: str
    start_time: float
    end_time: float
    text: str
    source_type: str
    score: float
    event_type: Optional[str] = None
    frame_url: Optional[str] = None
