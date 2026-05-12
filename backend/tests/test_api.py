"""Tests for backend.api.router — FastAPI endpoint integration tests.

Uses TestClient (synchronous) to test the API without starting a server.
Heavy operations (processing, VLM, Whisper) are mocked.
"""

import io
import pytest
import numpy as np
import cv2

from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create a TestClient with isolated DB and storage."""
    # Override config paths before importing the app
    monkeypatch.setenv("COGNISTREAM_ROOT", str(tmp_path))

    # Force re-import of config with new env var
    import importlib
    import backend.config
    importlib.reload(backend.config)

    # Ensure directories exist
    for d in [
        backend.config.VIDEO_DIR,
        backend.config.FRAME_DIR,
        backend.config.AUDIO_DIR,
        backend.config.GRAPH_DIR,
        backend.config.DB_DIR,
        backend.config.CHROMA_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # Re-import modules that depend on config
    import backend.db.sqlite
    importlib.reload(backend.db.sqlite)
    import backend.db.chroma_store
    importlib.reload(backend.db.chroma_store)
    import backend.ingestion.loader
    importlib.reload(backend.ingestion.loader)

    # Import and reload the router module to pick up new singletons
    import backend.api.router as router_mod
    importlib.reload(router_mod)

    import backend.main
    importlib.reload(backend.main)

    return TestClient(backend.main.app)


def _create_video_bytes(fps=30.0, width=160, height=120, frames=15):
    """Create a minimal MP4 video in memory and return its bytes."""
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
    for _ in range(frames):
        writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    writer.release()

    data = Path(tmp_path).read_bytes()
    Path(tmp_path).unlink()
    return data


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestIngestEndpoint:
    def test_ingest_valid_video(self, client):
        video_data = _create_video_bytes()
        resp = client.post(
            "/ingest-video",
            files={"file": ("test.mp4", io.BytesIO(video_data), "video/mp4")},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert "video_id" in body
        assert body["status"] == "UPLOADED"

    def test_ingest_bad_extension(self, client):
        resp = client.post(
            "/ingest-video",
            files={"file": ("test.txt", io.BytesIO(b"not a video"), "text/plain")},
        )
        assert resp.status_code == 400

    def test_ingest_no_filename(self, client):
        resp = client.post(
            "/ingest-video",
            files={"file": ("", io.BytesIO(b""), "application/octet-stream")},
        )
        assert resp.status_code in (400, 422)


class TestVideoEndpoints:
    def _ingest(self, client):
        video_data = _create_video_bytes()
        resp = client.post(
            "/ingest-video",
            files={"file": ("test.mp4", io.BytesIO(video_data), "video/mp4")},
        )
        return resp.json()["video_id"]

    def test_get_video(self, client):
        vid = self._ingest(client)
        resp = client.get(f"/video/{vid}")
        assert resp.status_code == 200
        assert resp.json()["video_id"] == vid

    def test_get_nonexistent_video(self, client):
        resp = client.get("/video/nonexistent")
        assert resp.status_code == 404

    def test_list_videos(self, client):
        self._ingest(client)
        resp = client.get("/videos")
        assert resp.status_code == 200
        assert len(resp.json()["videos"]) >= 1

    def test_delete_video(self, client):
        vid = self._ingest(client)
        resp = client.delete(f"/video/{vid}")
        assert resp.status_code == 200
        # Should be gone
        resp2 = client.get(f"/video/{vid}")
        assert resp2.status_code == 404

    def test_stream_video(self, client):
        vid = self._ingest(client)
        resp = client.get(f"/video/{vid}/stream")
        assert resp.status_code == 200
        assert len(resp.content) > 0


class TestFrameSecurityEndpoint:
    def test_path_traversal_blocked(self, client):
        resp = client.get("/video/v1/frame/../../../etc/passwd")
        # Returns 400 (invalid frame name) or 404 (video not found) depending on order
        assert resp.status_code in (400, 404)

    def test_directory_separator_blocked(self, client):
        resp = client.get("/video/v1/frame/sub%2Fdir%2Ffile.jpg")
        # URL-decoded this becomes sub/dir/file.jpg which should be rejected
        assert resp.status_code in (400, 404)


class TestProcessEndpoint:
    def test_process_nonexistent_video(self, client):
        resp = client.post("/process-video", json={"video_id": "nonexistent"})
        assert resp.status_code == 404
