"""Extended tests for backend.api.router — covers all untested API endpoints.

Each test function patches the module-level singletons in the router module
(_db, _store, _query_engine, _streaming_pipeline, _orchestrator) so that
no real database, ChromaDB, or model inference is needed.
"""

from __future__ import annotations

import io
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

from backend.db.models import (
    FusedSegment,
    SearchResult,
    VideoMeta,
    VideoStatus,
)
from backend.pipeline.orchestrator import PipelineProgress
from backend.pipeline.streaming import LiveFeedStatus


# ── Helpers ────────────────────────────────────────────────────────

def _make_meta(
    video_id: str = "vid1",
    status: VideoStatus = VideoStatus.UPLOADED,
    file_path: str = "/fake/video.mp4",
) -> VideoMeta:
    """Build a lightweight VideoMeta for testing."""
    return VideoMeta(
        id=video_id,
        filename="video.mp4",
        file_path=file_path,
        duration_sec=30.0,
        fps=30.0,
        width=320,
        height=240,
        total_frames=900,
        status=status,
        created_at="2026-01-01T00:00:00Z",
    )


def _make_search_result(**overrides) -> SearchResult:
    defaults = dict(
        video_id="vid1",
        segment_id="seg1",
        start_time=1.0,
        end_time=5.0,
        text="A red car on the road.",
        source_type="fused",
        score=0.95,
        event_type=None,
        frame_url=None,
    )
    defaults.update(overrides)
    return SearchResult(**defaults)


# ── Fixture: TestClient with all singletons mocked ────────────────

@pytest.fixture
def mocks(tmp_path, monkeypatch):
    """Patch config paths and reload modules so the TestClient uses temp dirs
    and mocked singletons.  Returns a dict of the mock objects for assertions.
    """
    monkeypatch.setenv("COGNISTREAM_ROOT", str(tmp_path))

    import importlib
    import backend.config
    importlib.reload(backend.config)

    # Ensure data directories exist
    for d in [
        backend.config.VIDEO_DIR,
        backend.config.FRAME_DIR,
        backend.config.AUDIO_DIR,
        backend.config.GRAPH_DIR,
        backend.config.DB_DIR,
        backend.config.CHROMA_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    import backend.db.sqlite
    importlib.reload(backend.db.sqlite)
    import backend.db.chroma_store
    importlib.reload(backend.db.chroma_store)
    import backend.ingestion.loader
    importlib.reload(backend.ingestion.loader)
    import backend.api.router as router_mod
    importlib.reload(router_mod)
    import backend.main
    importlib.reload(backend.main)

    # Build mocks
    mock_db = MagicMock(name="SQLiteDB")
    mock_store = MagicMock(name="ChromaStore")
    mock_query_engine = MagicMock(name="QueryEngine")
    mock_orchestrator = MagicMock(name="PipelineOrchestrator")
    mock_streaming = MagicMock(name="StreamingPipeline")

    # Defaults: orchestrator is not busy
    type(mock_orchestrator).is_busy = PropertyMock(return_value=False)

    # Inject into the router module
    router_mod._db = mock_db
    router_mod._store = mock_store
    router_mod._query_engine = mock_query_engine
    router_mod._orchestrator = mock_orchestrator
    router_mod._streaming_pipeline = mock_streaming

    # Also clear any leftover queue / progress / browser-feed state
    router_mod._process_queue.clear()
    router_mod._progress_store.clear()
    router_mod._browser_feeds.clear()

    from fastapi.testclient import TestClient
    client = TestClient(backend.main.app)

    return {
        "client": client,
        "db": mock_db,
        "store": mock_store,
        "qe": mock_query_engine,
        "orch": mock_orchestrator,
        "streaming": mock_streaming,
        "tmp_path": tmp_path,
        "config": backend.config,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. POST /search
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_search_basic(mocks):
    """Search with a plain query returns results from QueryEngine."""
    mocks["qe"].search.return_value = [
        _make_search_result(video_id="v1", segment_id="s1", score=0.9),
        _make_search_result(video_id="v1", segment_id="s2", score=0.7),
    ]

    resp = mocks["client"].post("/search", json={"query": "red car"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "red car"
    assert len(body["results"]) == 2
    assert body["results"][0]["segment_id"] == "s1"

    mocks["qe"].search.assert_called_once_with(
        query="red car", top_k=10, video_id=None, source_filter=None,
    )


def test_search_with_video_filter(mocks):
    """Search scoped to a specific video_id."""
    mocks["qe"].search.return_value = []

    resp = mocks["client"].post(
        "/search",
        json={"query": "person walking", "video_id": "v42", "top_k": 5},
    )
    assert resp.status_code == 200
    assert resp.json()["results"] == []

    mocks["qe"].search.assert_called_once_with(
        query="person walking", top_k=5, video_id="v42", source_filter=None,
    )


def test_search_with_source_filter(mocks):
    """Search restricted to a source_type (e.g. 'visual')."""
    mocks["qe"].search.return_value = [
        _make_search_result(source_type="visual"),
    ]

    resp = mocks["client"].post(
        "/search",
        json={"query": "dog", "source_filter": "visual"},
    )
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 1

    mocks["qe"].search.assert_called_once_with(
        query="dog", top_k=10, video_id=None, source_filter="visual",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. GET /video/{id}/progress
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_progress_existing_video_not_processing(mocks):
    """Progress for a video that exists but is not currently processing."""
    mocks["db"].get_video.return_value = _make_meta(
        video_id="v1", status=VideoStatus.UPLOADED,
    )

    resp = mocks["client"].get("/video/v1/progress")
    assert resp.status_code == 200
    body = resp.json()
    assert body["video_id"] == "v1"
    assert body["stage"] == "UPLOADED"
    assert body["percent"] == 0


def test_progress_processed_video(mocks):
    """Progress returns 100% for a fully processed video."""
    mocks["db"].get_video.return_value = _make_meta(
        video_id="v1", status=VideoStatus.PROCESSED,
    )

    resp = mocks["client"].get("/video/v1/progress")
    assert resp.status_code == 200
    body = resp.json()
    assert body["percent"] == 100
    assert body["stage_number"] == 10


def test_progress_actively_processing(mocks):
    """Progress with an in-flight PipelineProgress entry."""
    import backend.api.router as router_mod

    router_mod._progress_store["vp"] = PipelineProgress(
        video_id="vp",
        stage="Visual Analysis",
        stage_number=4,
        total_stages=10,
        started_at=0,
        elapsed_sec=12.5,
    )

    resp = mocks["client"].get("/video/vp/progress")
    assert resp.status_code == 200
    body = resp.json()
    assert body["stage"] == "Visual Analysis"
    assert body["percent"] == 40
    assert body["elapsed_sec"] == 12.5


def test_progress_404(mocks):
    """Progress for a nonexistent video returns 404."""
    mocks["db"].get_video.return_value = None

    resp = mocks["client"].get("/video/no-such/progress")
    assert resp.status_code == 404


def test_benchmark_success(mocks):
    """Benchmark endpoint returns stored metrics for an existing video."""
    mocks["db"].get_video.return_value = _make_meta(video_id="vbench", status=VideoStatus.PROCESSED)
    mocks["db"].get_latest_benchmark.return_value = {
        "id": "run1",
        "video_id": "vbench",
        "success": True,
        "elapsed_sec": 12.3,
        "stage_timings": {"vlm_sec": 4.2},
        "quality_metrics": {"keyframes_kept": 10.0},
    }

    resp = mocks["client"].get("/video/vbench/benchmark")
    assert resp.status_code == 200
    body = resp.json()
    assert body["video_id"] == "vbench"
    assert body["success"] is True
    assert body["stage_timings"]["vlm_sec"] == 4.2


def test_benchmark_404_when_missing(mocks):
    """Benchmark endpoint returns 404 when video exists but no metrics are captured."""
    mocks["db"].get_video.return_value = _make_meta(video_id="vbench", status=VideoStatus.PROCESSED)
    mocks["db"].get_latest_benchmark.return_value = None

    resp = mocks["client"].get("/video/vbench/benchmark")
    assert resp.status_code == 404
    assert "No benchmark metrics available" in resp.json()["detail"]


def test_benchmark_history_success(mocks):
    """Benchmark history returns a bounded list of benchmark runs."""
    mocks["db"].get_video.return_value = _make_meta(video_id="vbench", status=VideoStatus.PROCESSED)
    mocks["db"].list_benchmark_runs.return_value = [
        {"id": "r2", "video_id": "vbench", "elapsed_sec": 8.0},
        {"id": "r1", "video_id": "vbench", "elapsed_sec": 9.1},
    ]

    resp = mocks["client"].get("/video/vbench/benchmark/history?limit=2")
    assert resp.status_code == 200
    body = resp.json()
    assert body["video_id"] == "vbench"
    assert body["count"] == 2
    assert body["runs"][0]["id"] == "r2"


def test_benchmark_trend_success(mocks):
    """Benchmark trend computes deltas between latest and oldest runs."""
    mocks["db"].get_video.return_value = _make_meta(video_id="vbench", status=VideoStatus.PROCESSED)
    mocks["db"].list_benchmark_runs.return_value = [
        {
            "id": "new",
            "video_id": "vbench",
            "elapsed_sec": 8.0,
            "stage_timings": {"vlm_sec": 3.5, "store_sec": 0.8},
            "quality_metrics": {
                "captions_fallback_ratio": 0.1,
                "captions_static_ratio": 0.2,
                "keyframes_kept": 30,
            },
        },
        {
            "id": "old",
            "video_id": "vbench",
            "elapsed_sec": 10.0,
            "stage_timings": {"vlm_sec": 4.0, "store_sec": 0.7},
            "quality_metrics": {
                "captions_fallback_ratio": 0.3,
                "captions_static_ratio": 0.25,
                "keyframes_kept": 40,
            },
        },
    ]

    resp = mocks["client"].get("/video/vbench/benchmark/trend?limit=2")
    assert resp.status_code == 200
    body = resp.json()
    assert body["video_id"] == "vbench"
    assert body["run_count"] == 2
    assert body["elapsed_sec"]["delta_latest_minus_oldest"] == -2.0
    assert body["stage_timing_delta"]["vlm_sec"] == -0.5


def test_benchmark_trend_404_when_insufficient_runs(mocks):
    """Trend endpoint requires at least two runs."""
    mocks["db"].get_video.return_value = _make_meta(video_id="vbench", status=VideoStatus.PROCESSED)
    mocks["db"].list_benchmark_runs.return_value = [{"id": "only", "video_id": "vbench"}]

    resp = mocks["client"].get("/video/vbench/benchmark/trend?limit=2")
    assert resp.status_code == 404
    assert "at least 2 benchmark runs" in resp.json()["detail"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. GET /video/{id}/thumbnail
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_thumbnail_no_frame_dir(mocks):
    """Thumbnail returns 404 when no frames directory exists."""
    resp = mocks["client"].get("/video/v_no_frames/thumbnail")
    assert resp.status_code == 404


def test_thumbnail_empty_frame_dir(mocks):
    """Thumbnail returns 404 when the frame dir exists but is empty."""
    frame_dir = mocks["config"].FRAME_DIR / "v_empty"
    frame_dir.mkdir(parents=True, exist_ok=True)

    resp = mocks["client"].get("/video/v_empty/thumbnail")
    assert resp.status_code == 404


def test_thumbnail_success(mocks):
    """Thumbnail returns the first .jpg file when frames exist."""
    import cv2

    frame_dir = mocks["config"].FRAME_DIR / "v_thumb"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Create a tiny JPEG file
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imwrite(str(frame_dir / "frame_000001.jpg"), img)
    cv2.imwrite(str(frame_dir / "frame_000002.jpg"), img)

    resp = mocks["client"].get("/video/v_thumb/thumbnail")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert len(resp.content) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. POST /similar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_similar_segment_not_found(mocks):
    """Similar returns 404 when the segment does not exist in ChromaDB."""
    mocks["store"].get_segment.return_value = None

    resp = mocks["client"].post(
        "/similar", json={"segment_id": "missing_seg"},
    )
    assert resp.status_code == 404


def test_similar_no_embedding(mocks):
    """Similar returns 400 when the segment has no embedding."""
    mocks["store"].get_segment.return_value = {
        "id": "seg_no_emb",
        "text": "something",
        "embedding": None,
    }

    resp = mocks["client"].post(
        "/similar", json={"segment_id": "seg_no_emb"},
    )
    assert resp.status_code == 400


def test_similar_success(mocks):
    """Similar returns filtered results excluding the source segment."""
    embedding = [0.1] * 384
    mocks["store"].get_segment.return_value = {
        "id": "seg_src",
        "text": "source",
        "embedding": embedding,
    }
    mocks["store"].query.return_value = [
        {
            "id": "seg_src",
            "video_id": "v1",
            "start_time": 1.0,
            "end_time": 3.0,
            "text": "source",
            "source_type": "fused",
            "score": 1.0,
            "frame_path": None,
        },
        {
            "id": "seg_2",
            "video_id": "v1",
            "start_time": 5.0,
            "end_time": 8.0,
            "text": "similar content",
            "source_type": "fused",
            "score": 0.85,
            "frame_path": "/frames/v1/frame_001.jpg",
        },
    ]

    resp = mocks["client"].post(
        "/similar", json={"segment_id": "seg_src", "top_k": 5},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source_segment_id"] == "seg_src"
    # The source segment should be excluded
    assert len(body["results"]) == 1
    assert body["results"][0]["segment_id"] == "seg_2"
    assert body["results"][0]["frame_url"] == "/video/v1/frame/frame_001.jpg"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. POST /video/{id}/clip
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_clip_video_not_found(mocks):
    """Clip export returns 404 for nonexistent video."""
    mocks["db"].get_video.return_value = None

    resp = mocks["client"].post(
        "/video/no_vid/clip",
        json={"start_time": 0, "end_time": 5},
    )
    assert resp.status_code == 404


def test_clip_bad_time_range(mocks):
    """Clip export returns 400 when end_time <= start_time."""
    mocks["db"].get_video.return_value = _make_meta(video_id="v1")

    # The file_path needs to exist for the check to pass before time range check
    # But the check order in the router is: meta lookup -> file check -> time check
    # We need the file to exist too.
    meta = _make_meta(video_id="v1", file_path=str(mocks["tmp_path"] / "fake.mp4"))
    mocks["db"].get_video.return_value = meta
    # Create the fake file so the file-exists check passes
    (mocks["tmp_path"] / "fake.mp4").write_bytes(b"fake")

    resp = mocks["client"].post(
        "/video/v1/clip",
        json={"start_time": 10, "end_time": 5},
    )
    assert resp.status_code == 400
    assert "end_time" in resp.json()["detail"].lower()


def test_clip_file_not_on_disk(mocks):
    """Clip export returns 404 when the video file doesn't exist on disk."""
    meta = _make_meta(video_id="v1", file_path="/nonexistent/video.mp4")
    mocks["db"].get_video.return_value = meta

    resp = mocks["client"].post(
        "/video/v1/clip",
        json={"start_time": 0, "end_time": 5},
    )
    assert resp.status_code == 404


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. GET /video/{id}/graph
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_graph_video_not_found(mocks):
    """Graph endpoint returns 404 for nonexistent video."""
    mocks["db"].get_video.return_value = None

    resp = mocks["client"].get("/video/no_vid/graph")
    assert resp.status_code == 404


def test_graph_no_file(mocks):
    """Graph returns empty nodes/edges when no .graphml file exists."""
    mocks["db"].get_video.return_value = _make_meta(video_id="v1")

    resp = mocks["client"].get("/video/v1/graph")
    assert resp.status_code == 200
    body = resp.json()
    assert body["nodes"] == []
    assert body["edges"] == []


def test_graph_with_graphml(mocks):
    """Graph endpoint parses a real GraphML file and returns nodes/edges."""
    import networkx as nx

    mocks["db"].get_video.return_value = _make_meta(video_id="v_graph")

    # Create a small graph and save it
    G = nx.DiGraph()
    G.add_node("car", label="car", type="object", count=3, first_seen=1.0, last_seen=10.0)
    G.add_node("person", label="person", type="object", count=2, first_seen=2.0, last_seen=8.0)
    G.add_edge("person", "car", action="approaches", timestamp=5.0)

    graph_path = mocks["config"].GRAPH_DIR / "v_graph.graphml"
    nx.write_graphml(G, str(graph_path))

    resp = mocks["client"].get("/video/v_graph/graph")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["nodes"]) == 2
    assert len(body["edges"]) == 1

    # Verify node attributes
    node_ids = {n["id"] for n in body["nodes"]}
    assert "car" in node_ids
    assert "person" in node_ids

    edge = body["edges"][0]
    assert edge["source"] == "person"
    assert edge["target"] == "car"
    assert edge["action"] == "approaches"


def test_graph_synthesizes_relationships_when_graph_has_only_nodes(mocks):
    """Graph endpoint infers lightweight edges for legacy node-only GraphML files."""
    import networkx as nx

    mocks["db"].get_video.return_value = _make_meta(video_id="v_legacy_graph")

    G = nx.DiGraph()
    G.add_node("car", label="car", type="vehicle", count=2, first_seen=1.0, last_seen=4.0)
    G.add_node("traffic_light", label="traffic_light", type="object", count=1, first_seen=1.0, last_seen=1.0)
    G.add_node("pedestrian", label="pedestrian", type="person", count=1, first_seen=5.0, last_seen=5.0)

    graph_path = mocks["config"].GRAPH_DIR / "v_legacy_graph.graphml"
    nx.write_graphml(G, str(graph_path))

    resp = mocks["client"].get("/video/v_legacy_graph/graph")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["edges"]) == 1
    assert body["edges"][0]["source"] == "car"
    assert body["edges"][0]["target"] == "traffic_light"
    assert body["edges"][0]["action"] == "co_occurs_with"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. GET /video/{id}/events
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_events_valid_video(mocks):
    """Events endpoint returns the list from db.list_events."""
    mocks["db"].get_video.return_value = _make_meta(video_id="v1")
    mocks["db"].list_events.return_value = [
        {"event_type": "person_enters", "start_time": 1.0, "end_time": 3.0},
        {"event_type": "vehicle_moves", "start_time": 5.0, "end_time": 8.0},
    ]

    resp = mocks["client"].get("/video/v1/events")
    assert resp.status_code == 200
    body = resp.json()
    assert body["video_id"] == "v1"
    assert len(body["events"]) == 2


def test_events_404(mocks):
    """Events endpoint returns 404 for nonexistent video."""
    mocks["db"].get_video.return_value = None

    resp = mocks["client"].get("/video/nope/events")
    assert resp.status_code == 404


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. POST /process-batch
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_process_batch_queues_multiple(mocks):
    """Batch processing queues multiple valid videos."""
    mocks["db"].get_video.side_effect = lambda vid: (
        _make_meta(video_id=vid) if vid in ("v1", "v2") else None
    )
    # Mark orchestrator as busy so it won't start the worker thread
    type(mocks["orch"]).is_busy = PropertyMock(return_value=True)

    resp = mocks["client"].post(
        "/process-batch", json={"video_ids": ["v1", "v2"]},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert set(body["queued"]) == {"v1", "v2"}
    assert body["errors"] == []
    assert body["queue_size"] >= 2


def test_process_batch_with_nonexistent_video(mocks):
    """Batch processing reports errors for missing videos."""
    mocks["db"].get_video.side_effect = lambda vid: (
        _make_meta(video_id=vid) if vid == "v1" else None
    )
    type(mocks["orch"]).is_busy = PropertyMock(return_value=True)

    resp = mocks["client"].post(
        "/process-batch", json={"video_ids": ["v1", "missing"]},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert "v1" in body["queued"]
    assert len(body["errors"]) == 1
    assert body["errors"][0]["video_id"] == "missing"
    assert "Not found" in body["errors"][0]["error"]


def test_process_batch_already_processing(mocks):
    """Batch processing reports errors for videos already being processed."""
    mocks["db"].get_video.return_value = _make_meta(
        video_id="vbusy", status=VideoStatus.PROCESSING,
    )
    type(mocks["orch"]).is_busy = PropertyMock(return_value=True)

    resp = mocks["client"].post(
        "/process-batch", json={"video_ids": ["vbusy"]},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["queued"] == []
    assert len(body["errors"]) == 1
    assert "Already processing" in body["errors"][0]["error"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. GET /process-queue
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_process_queue_empty(mocks):
    """Empty processing queue."""
    type(mocks["orch"]).is_busy = PropertyMock(return_value=False)

    resp = mocks["client"].get("/process-queue")
    assert resp.status_code == 200
    body = resp.json()
    assert body["queue"] == []
    assert body["queue_size"] == 0
    assert body["is_busy"] is False


def test_process_queue_with_items(mocks):
    """Processing queue with items from a previous batch."""
    import backend.api.router as router_mod

    router_mod._process_queue.append("v1")
    router_mod._process_queue.append("v2")
    type(mocks["orch"]).is_busy = PropertyMock(return_value=True)

    resp = mocks["client"].get("/process-queue")
    assert resp.status_code == 200
    body = resp.json()
    assert body["queue"] == ["v1", "v2"]
    assert body["queue_size"] == 2
    assert body["is_busy"] is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. POST /annotations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_create_annotation_success(mocks):
    """Creating an annotation for a valid video returns 201."""
    mocks["db"].get_video.return_value = _make_meta(video_id="v1")

    resp = mocks["client"].post("/annotations", json={
        "video_id": "v1",
        "start_time": 2.5,
        "end_time": 7.0,
        "label": "Interesting scene",
        "note": "A person crossing the road",
        "color": "#ff0000",
    })
    assert resp.status_code == 201
    body = resp.json()
    assert body["video_id"] == "v1"
    assert body["label"] == "Interesting scene"
    assert body["start_time"] == 2.5
    assert body["end_time"] == 7.0
    assert body["note"] == "A person crossing the road"
    assert body["color"] == "#ff0000"
    assert "id" in body
    assert "created_at" in body

    # Verify db.save_annotation was called
    mocks["db"].save_annotation.assert_called_once()
    saved = mocks["db"].save_annotation.call_args[0][0]
    assert saved["video_id"] == "v1"


def test_create_annotation_video_not_found(mocks):
    """Creating an annotation for a nonexistent video returns 404."""
    mocks["db"].get_video.return_value = None

    resp = mocks["client"].post("/annotations", json={
        "video_id": "missing",
        "start_time": 0,
        "end_time": 5,
        "label": "test",
    })
    assert resp.status_code == 404


def test_create_annotation_default_color(mocks):
    """Annotation uses the default color if none is provided."""
    mocks["db"].get_video.return_value = _make_meta(video_id="v1")

    resp = mocks["client"].post("/annotations", json={
        "video_id": "v1",
        "start_time": 0,
        "end_time": 1,
        "label": "tag",
    })
    assert resp.status_code == 201
    assert resp.json()["color"] == "#3b82f6"  # default from AnnotationRequest


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. GET /video/{id}/annotations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_list_annotations_success(mocks):
    """List annotations for a valid video."""
    mocks["db"].get_video.return_value = _make_meta(video_id="v1")
    mocks["db"].list_annotations.return_value = [
        {"id": "a1", "video_id": "v1", "label": "scene1", "start_time": 0, "end_time": 5},
        {"id": "a2", "video_id": "v1", "label": "scene2", "start_time": 10, "end_time": 15},
    ]

    resp = mocks["client"].get("/video/v1/annotations")
    assert resp.status_code == 200
    body = resp.json()
    assert body["video_id"] == "v1"
    assert len(body["annotations"]) == 2


def test_list_annotations_404(mocks):
    """List annotations returns 404 for nonexistent video."""
    mocks["db"].get_video.return_value = None

    resp = mocks["client"].get("/video/missing/annotations")
    assert resp.status_code == 404


def test_list_annotations_empty(mocks):
    """List annotations for a video with no annotations returns empty list."""
    mocks["db"].get_video.return_value = _make_meta(video_id="v1")
    mocks["db"].list_annotations.return_value = []

    resp = mocks["client"].get("/video/v1/annotations")
    assert resp.status_code == 200
    assert resp.json()["annotations"] == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 12. DELETE /annotations/{id}
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_delete_annotation_success(mocks):
    """Deleting an existing annotation returns 200."""
    mocks["db"].delete_annotation.return_value = True

    resp = mocks["client"].delete("/annotations/abc123")
    assert resp.status_code == 200
    assert resp.json()["id"] == "abc123"
    mocks["db"].delete_annotation.assert_called_once_with("abc123")


def test_delete_annotation_not_found(mocks):
    """Deleting a nonexistent annotation returns 404."""
    mocks["db"].delete_annotation.return_value = False

    resp = mocks["client"].delete("/annotations/no_such")
    assert resp.status_code == 404


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 13. POST /live/start
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_live_start_success(mocks):
    """Starting a live feed returns 201 with feed info."""
    mocks["streaming"].start_live.return_value = LiveFeedStatus(
        video_id="cam1",
        url="rtsp://192.168.1.100/stream",
        state="connecting",
        started_at="2026-01-01T00:00:00Z",
    )

    resp = mocks["client"].post("/live/start", json={
        "url": "rtsp://192.168.1.100/stream",
        "video_id": "cam1",
        "chunk_sec": 15,
    })
    assert resp.status_code == 201
    body = resp.json()
    assert body["video_id"] == "cam1"
    assert body["state"] == "connecting"
    assert body["chunk_sec"] == 15
    assert "/ws/live/cam1" in body["message"]


def test_live_start_already_running(mocks):
    """Starting a live feed that is already active returns 409."""
    mocks["streaming"].start_live.side_effect = ValueError(
        "Live feed 'cam1' is already running."
    )

    resp = mocks["client"].post("/live/start", json={
        "url": "rtsp://192.168.1.100/stream",
        "video_id": "cam1",
    })
    assert resp.status_code == 409


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 14. POST /live/stop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_live_stop_nonexistent(mocks):
    """Stopping a feed that doesn't exist returns 404."""
    mocks["streaming"].stop_live.return_value = False

    resp = mocks["client"].post("/live/stop", json={"video_id": "ghost"})
    assert resp.status_code == 404


def test_live_stop_success(mocks):
    """Stopping an active live feed returns 200."""
    mocks["streaming"].stop_live.return_value = True

    resp = mocks["client"].post("/live/stop", json={"video_id": "cam1"})
    assert resp.status_code == 200
    assert resp.json()["video_id"] == "cam1"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 15. GET /live/status
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_live_status_empty(mocks):
    """Live status returns empty feeds when nothing is running."""
    mocks["streaming"].get_live_status.return_value = []

    resp = mocks["client"].get("/live/status")
    assert resp.status_code == 200
    assert resp.json()["feeds"] == []


def test_live_status_with_feeds(mocks):
    """Live status returns details for active feeds."""
    mocks["streaming"].get_live_status.return_value = [
        LiveFeedStatus(
            video_id="cam1",
            url="rtsp://example.com/stream",
            state="running",
            chunks_processed=12,
            total_segments=48,
            fps=29.97,
            started_at="2026-01-01T00:00:00Z",
            last_chunk_at="2026-01-01T00:06:00Z",
            error="",
        ),
    ]

    resp = mocks["client"].get("/live/status")
    assert resp.status_code == 200
    feeds = resp.json()["feeds"]
    assert len(feeds) == 1
    assert feeds[0]["video_id"] == "cam1"
    assert feeds[0]["state"] == "running"
    assert feeds[0]["chunks_processed"] == 12


def test_live_status_specific_feed(mocks):
    """Live status filtered by video_id query param."""
    mocks["streaming"].get_live_status.return_value = [
        LiveFeedStatus(
            video_id="cam2",
            url="rtsp://example.com/cam2",
            state="running",
        ),
    ]

    resp = mocks["client"].get("/live/status?video_id=cam2")
    assert resp.status_code == 200
    assert len(resp.json()["feeds"]) == 1
    mocks["streaming"].get_live_status.assert_called_with("cam2")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 16. POST /live/browser-chunk
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_browser_chunk_missing_video_data(mocks):
    """Browser chunk with empty video data should fail gracefully."""
    # Sending an empty file — the endpoint will try to read/process it
    # and should either return 400 or fail on the video decode
    resp = mocks["client"].post(
        "/live/browser-chunk?video_id=bc1&chunk_index=0&chunk_start=0",
        files={"file": ("chunk.webm", io.BytesIO(b""), "video/webm")},
    )
    # The endpoint attempts ffmpeg conversion and cv2.VideoCapture on garbage data.
    # It should return 400 because cv2 can't open the file.
    assert resp.status_code == 400


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 17. POST /live/browser-stop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_browser_stop_nonexistent_feed(mocks):
    """Stopping a browser feed that doesn't exist returns 404."""
    resp = mocks["client"].post("/live/browser-stop?video_id=ghost")
    assert resp.status_code == 404


def test_browser_stop_existing_feed(mocks):
    """Stopping an existing browser feed finalizes it."""
    import backend.api.router as router_mod

    # Manually inject a browser feed state
    with router_mod._browser_feeds_lock:
        router_mod._browser_feeds["bc_stop"] = {
            "chunk_idx": 3,
            "embedder": MagicMock(),
            "all_captions": [],
            "all_transcripts": [],
        }

    resp = mocks["client"].post("/live/browser-stop?video_id=bc_stop")
    assert resp.status_code == 200
    body = resp.json()
    assert body["video_id"] == "bc_stop"
    assert body["total_chunks"] == 3
    assert body["message"] == "Browser feed finalized."


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 18. POST /process-video
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_process_video_standard_mode(mocks):
    """Processing a video in standard mode returns 202."""
    mocks["db"].get_video.return_value = _make_meta(
        video_id="v1", status=VideoStatus.UPLOADED,
    )
    type(mocks["orch"]).is_busy = PropertyMock(return_value=False)

    resp = mocks["client"].post(
        "/process-video", json={"video_id": "v1"},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["video_id"] == "v1"
    assert body["status"] == "PROCESSING"
    assert body["mode"] == "standard"


def test_process_video_streaming_mode(mocks):
    """Processing a video in streaming mode returns 202."""
    mocks["db"].get_video.return_value = _make_meta(
        video_id="v1", status=VideoStatus.UPLOADED,
    )
    type(mocks["orch"]).is_busy = PropertyMock(return_value=False)

    resp = mocks["client"].post(
        "/process-video",
        json={"video_id": "v1", "mode": "streaming", "chunk_sec": 30},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["mode"] == "streaming"


def test_process_video_404(mocks):
    """Processing a nonexistent video returns 404."""
    mocks["db"].get_video.return_value = None

    resp = mocks["client"].post(
        "/process-video", json={"video_id": "missing"},
    )
    assert resp.status_code == 404


def test_process_video_409_already_processing(mocks):
    """Processing a video that is already being processed returns 409."""
    mocks["db"].get_video.return_value = _make_meta(
        video_id="v1", status=VideoStatus.PROCESSING,
    )

    resp = mocks["client"].post(
        "/process-video", json={"video_id": "v1"},
    )
    assert resp.status_code == 409


def test_process_video_429_pipeline_busy(mocks):
    """Processing returns 429 when the pipeline is busy with another video."""
    mocks["db"].get_video.return_value = _make_meta(
        video_id="v1", status=VideoStatus.UPLOADED,
    )
    type(mocks["orch"]).is_busy = PropertyMock(return_value=True)

    resp = mocks["client"].post(
        "/process-video", json={"video_id": "v1"},
    )
    assert resp.status_code == 429


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 19. GET /video/{id}/frame/{name}
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_frame_success(mocks):
    """Serving an existing keyframe image returns 200 with JPEG content."""
    import cv2

    frame_dir = mocks["config"].FRAME_DIR / "v_frame"
    frame_dir.mkdir(parents=True, exist_ok=True)

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imwrite(str(frame_dir / "frame_000001.jpg"), img)

    resp = mocks["client"].get("/video/v_frame/frame/frame_000001.jpg")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert len(resp.content) > 0


def test_frame_not_found(mocks):
    """Requesting a frame that doesn't exist returns 404."""
    # Create the directory but not the file
    frame_dir = mocks["config"].FRAME_DIR / "v_noframe"
    frame_dir.mkdir(parents=True, exist_ok=True)

    resp = mocks["client"].get("/video/v_noframe/frame/nonexistent.jpg")
    assert resp.status_code == 404


def test_frame_path_traversal_blocked(mocks):
    """Path traversal attempts in frame names are rejected."""
    resp = mocks["client"].get("/video/v1/frame/../../etc/passwd")
    assert resp.status_code in (400, 404)


def test_frame_directory_separator_blocked(mocks):
    """Frame names containing directory separators are rejected."""
    resp = mocks["client"].get("/video/v1/frame/sub%2Fdir%2Ffile.jpg")
    assert resp.status_code in (400, 404)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Additional edge-case tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_search_empty_query(mocks):
    """An empty query should still call the engine (which may return [])."""
    mocks["qe"].search.return_value = []

    resp = mocks["client"].post("/search", json={"query": ""})
    assert resp.status_code == 200
    assert resp.json()["results"] == []


def test_similar_with_video_filter(mocks):
    """Similar endpoint respects video_id filter."""
    mocks["store"].get_segment.return_value = {
        "id": "seg1",
        "text": "content",
        "embedding": [0.1] * 384,
    }
    mocks["store"].query.return_value = []

    resp = mocks["client"].post(
        "/similar",
        json={"segment_id": "seg1", "top_k": 3, "video_id": "v_specific"},
    )
    assert resp.status_code == 200
    # Verify the video_id was forwarded to the store query
    mocks["store"].query.assert_called_once()
    call_kwargs = mocks["store"].query.call_args
    assert call_kwargs.kwargs.get("video_id") == "v_specific" or (
        len(call_kwargs.args) > 2 and call_kwargs.args[2] == "v_specific"
    ) or call_kwargs[1].get("video_id") == "v_specific"


def test_events_empty_list(mocks):
    """Events endpoint with no events returns an empty list."""
    mocks["db"].get_video.return_value = _make_meta(video_id="v1")
    mocks["db"].list_events.return_value = []

    resp = mocks["client"].get("/video/v1/events")
    assert resp.status_code == 200
    assert resp.json()["events"] == []


def test_process_batch_empty_list(mocks):
    """Batch processing with empty video_ids list."""
    type(mocks["orch"]).is_busy = PropertyMock(return_value=False)

    resp = mocks["client"].post(
        "/process-batch", json={"video_ids": []},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["queued"] == []
    assert body["errors"] == []
