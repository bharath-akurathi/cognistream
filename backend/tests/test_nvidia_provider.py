"""Tests for the NVIDIA NIM cloud provider (backend/providers/nvidia.py).

All tests mock httpx and config values — no real API calls are made.
"""

import base64
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from backend.providers.nvidia import NvidiaProvider


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def provider():
    """Fresh NvidiaProvider instance per test (no shared client state)."""
    p = NvidiaProvider()
    yield p
    p.close()


@pytest.fixture
def tiny_image(tmp_path):
    """Create a minimal 1x1 JPEG image on disk and return its path."""
    import numpy as np
    import cv2

    img = np.zeros((1, 1, 3), dtype=np.uint8)
    img[0, 0] = (128, 64, 32)
    path = tmp_path / "tiny.jpg"
    cv2.imwrite(str(path), img)
    return str(path)


def _mock_response(json_data, status_code=200):
    """Build a mock httpx.Response-like object."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(
            f"HTTP {status_code}"
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


# ──────────────────────────────────────────────
# Availability
# ──────────────────────────────────────────────


class TestAvailability:
    """NvidiaProvider.available should reflect the API key config."""

    @patch("backend.providers.nvidia.is_nvidia_enabled", return_value=False)
    def test_not_available_when_no_api_key(self, mock_enabled, provider):
        assert provider.available is False

    @patch("backend.providers.nvidia.is_nvidia_enabled", return_value=True)
    def test_available_when_api_key_set(self, mock_enabled, provider):
        assert provider.available is True


# ──────────────────────────────────────────────
# embed_text
# ──────────────────────────────────────────────


class TestEmbedText:
    @patch("backend.providers.nvidia.is_nvidia_enabled", return_value=False)
    def test_returns_none_when_not_available(self, mock_enabled, provider):
        result = provider.embed_text("hello world")
        assert result is None

    @patch("backend.providers.nvidia.is_nvidia_enabled", return_value=True)
    def test_makes_correct_request_and_returns_embedding(self, mock_enabled, provider):
        fake_embedding = [0.1, 0.2, 0.3]
        mock_resp = _mock_response({
            "data": [{"embedding": fake_embedding}]
        })

        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        provider._client = mock_client

        result = provider.embed_text("test query", input_type="query")

        assert result == fake_embedding
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/embeddings"
        body = call_args[1]["json"]
        assert body["input"] == ["test query"]
        assert body["input_type"] == "query"
        assert body["encoding_format"] == "float"

    @patch("backend.providers.nvidia.is_nvidia_enabled", return_value=True)
    def test_returns_none_on_api_failure(self, mock_enabled, provider):
        mock_client = MagicMock()
        mock_client.post.side_effect = Exception("connection failed")
        provider._client = mock_client

        result = provider.embed_text("hello")
        assert result is None


# ──────────────────────────────────────────────
# embed_texts (batch)
# ──────────────────────────────────────────────


class TestEmbedTexts:
    @patch("backend.providers.nvidia.is_nvidia_enabled", return_value=True)
    def test_batch_embedding(self, mock_enabled, provider):
        fake_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        mock_resp = _mock_response({
            "data": [
                {"embedding": fake_embeddings[0]},
                {"embedding": fake_embeddings[1]},
            ]
        })

        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        provider._client = mock_client

        result = provider.embed_texts(["hello", "world"], input_type="passage")

        assert result == fake_embeddings
        call_args = mock_client.post.call_args
        body = call_args[1]["json"]
        assert body["input"] == ["hello", "world"]
        assert body["input_type"] == "passage"

    @patch("backend.providers.nvidia.is_nvidia_enabled", return_value=True)
    def test_returns_none_for_empty_list(self, mock_enabled, provider):
        result = provider.embed_texts([])
        assert result is None

    @patch("backend.providers.nvidia.is_nvidia_enabled", return_value=False)
    def test_returns_none_when_not_available(self, mock_enabled, provider):
        result = provider.embed_texts(["hello"])
        assert result is None

    @patch("backend.providers.nvidia.is_nvidia_enabled", return_value=True)
    def test_returns_none_on_api_failure(self, mock_enabled, provider):
        mock_client = MagicMock()
        mock_client.post.side_effect = Exception("timeout")
        provider._client = mock_client

        result = provider.embed_texts(["hello"])
        assert result is None


# ──────────────────────────────────────────────
# embed_image
# ──────────────────────────────────────────────


class TestEmbedImage:
    @patch("backend.providers.nvidia.is_nvidia_enabled", return_value=True)
    def test_encodes_image_and_calls_nvclip(self, mock_enabled, provider, tiny_image):
        fake_embedding = [0.5, 0.6, 0.7]
        mock_resp = _mock_response({
            "data": [{"embedding": fake_embedding}]
        })

        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        provider._client = mock_client

        result = provider.embed_image(tiny_image)

        assert result == fake_embedding
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/embeddings"
        body = call_args[1]["json"]
        # The input should be a base64 data URI
        assert len(body["input"]) == 1
        assert body["input"][0].startswith("data:image/jpeg;base64,")
        # Verify the base64 payload decodes to the original file bytes
        b64_payload = body["input"][0].split(",", 1)[1]
        decoded = base64.b64decode(b64_payload)
        original = Path(tiny_image).read_bytes()
        assert decoded == original

    @patch("backend.providers.nvidia.is_nvidia_enabled", return_value=False)
    def test_returns_none_when_not_available(self, mock_enabled, provider, tiny_image):
        result = provider.embed_image(tiny_image)
        assert result is None

    @patch("backend.providers.nvidia.is_nvidia_enabled", return_value=True)
    def test_returns_none_on_api_failure(self, mock_enabled, provider, tiny_image):
        mock_client = MagicMock()
        mock_client.post.side_effect = Exception("server error")
        provider._client = mock_client

        result = provider.embed_image(tiny_image)
        assert result is None


# ──────────────────────────────────────────────
# caption_image
# ──────────────────────────────────────────────


class TestCaptionImage:
    @patch("backend.providers.nvidia.is_nvidia_enabled", return_value=True)
    def test_sends_correct_chat_completions_request(self, mock_enabled, provider, tiny_image):
        mock_resp = _mock_response({
            "choices": [
                {"message": {"content": "A small test image."}}
            ]
        })

        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        provider._client = mock_client

        result = provider.caption_image(tiny_image, "Describe this image.")

        assert result == "A small test image."
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/chat/completions"
        body = call_args[1]["json"]
        assert body["max_tokens"] == 512
        assert body["temperature"] == 0.2
        # Messages structure: one user message with text + image_url
        msg = body["messages"][0]
        assert msg["role"] == "user"
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][0]["text"] == "Describe this image."
        assert msg["content"][1]["type"] == "image_url"
        assert msg["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    @patch("backend.providers.nvidia.is_nvidia_enabled", return_value=False)
    def test_returns_none_when_not_available(self, mock_enabled, provider, tiny_image):
        result = provider.caption_image(tiny_image, "Describe.")
        assert result is None

    @patch("backend.providers.nvidia.is_nvidia_enabled", return_value=True)
    def test_returns_none_on_api_failure(self, mock_enabled, provider, tiny_image):
        mock_client = MagicMock()
        mock_client.post.side_effect = Exception("bad gateway")
        provider._client = mock_client

        result = provider.caption_image(tiny_image, "Describe.")
        assert result is None


# ──────────────────────────────────────────────
# detect_objects
# ──────────────────────────────────────────────


class TestDetectObjects:
    @patch("backend.providers.nvidia.is_nvidia_enabled", return_value=True)
    def test_calls_grounding_dino_endpoint(self, mock_enabled, provider, tiny_image):
        mock_resp = _mock_response({
            "detections": [
                {"label": "car", "confidence": 0.85, "bbox": [10, 20, 100, 200]},
                {"label": "person", "confidence": 0.72, "bbox": [50, 60, 120, 180]},
            ]
        })

        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        provider._grounding_client = mock_client

        result = provider.detect_objects(tiny_image, ["car", "person"], threshold=0.3)

        assert result is not None
        assert len(result) == 2
        assert result[0]["label"] == "car"
        assert result[0]["confidence"] == 0.85

        # Verify the prompt format (period-separated)
        call_args = mock_client.post.call_args
        body = call_args[1]["json"]
        assert body["prompt"] == "car. person."
        assert body["threshold"] == 0.3

    @patch("backend.providers.nvidia.is_nvidia_enabled", return_value=False)
    def test_returns_none_when_not_available(self, mock_enabled, provider, tiny_image):
        result = provider.detect_objects(tiny_image, ["car"])
        assert result is None

    @patch("backend.providers.nvidia.is_nvidia_enabled", return_value=True)
    def test_returns_none_on_api_failure(self, mock_enabled, provider, tiny_image):
        mock_client = MagicMock()
        mock_client.post.side_effect = Exception("timeout")
        provider._grounding_client = mock_client

        result = provider.detect_objects(tiny_image, ["car"])
        assert result is None


# ──────────────────────────────────────────────
# _parse_detections
# ──────────────────────────────────────────────


class TestParseDetections:
    def test_list_format(self):
        """When the API returns a flat list of detection dicts."""
        data = [
            {"label": "car", "confidence": 0.9, "bbox": [0, 0, 50, 50]},
            {"label": "person", "confidence": 0.2, "bbox": [10, 10, 30, 30]},
            {"label": "tree", "confidence": 0.6, "bbox": [20, 20, 40, 40]},
        ]
        result = NvidiaProvider._parse_detections(data, threshold=0.3)
        assert len(result) == 2
        labels = {d["label"] for d in result}
        assert "car" in labels
        assert "tree" in labels
        assert "person" not in labels  # below threshold

    def test_dict_detections_format(self):
        """When the API returns {'detections': [...]}."""
        data = {
            "detections": [
                {"label": "dog", "confidence": 0.8, "bbox": [5, 5, 60, 60]},
                {"label": "cat", "score": 0.4, "box": [15, 15, 45, 45]},
                {"label": "bird", "confidence": 0.1, "bbox": [0, 0, 10, 10]},
            ]
        }
        result = NvidiaProvider._parse_detections(data, threshold=0.3)
        assert len(result) == 2
        labels = {d["label"] for d in result}
        assert "dog" in labels
        assert "cat" in labels
        assert "bird" not in labels

    def test_dict_with_score_key(self):
        """Detections using 'score' instead of 'confidence'."""
        data = {
            "detections": [
                {"class": "truck", "score": 0.95, "box": [10, 10, 80, 80]},
            ]
        }
        result = NvidiaProvider._parse_detections(data, threshold=0.3)
        assert len(result) == 1
        assert result[0]["label"] == "truck"
        assert result[0]["confidence"] == 0.95

    def test_empty_detections(self):
        result = NvidiaProvider._parse_detections({"detections": []}, threshold=0.3)
        assert result == []

    def test_empty_list(self):
        result = NvidiaProvider._parse_detections([], threshold=0.3)
        assert result == []


# ──────────────────────────────────────────────
# transcribe (Riva gRPC)
# ──────────────────────────────────────────────


class TestTranscribe:
    @patch("backend.providers.nvidia.is_nvidia_enabled", return_value=False)
    def test_returns_none_when_not_available(self, mock_enabled, provider):
        result = provider.transcribe("/fake/audio.wav")
        assert result is None

    @patch("backend.providers.nvidia.is_nvidia_enabled", return_value=True)
    def test_returns_none_when_riva_not_installed(self, mock_enabled, provider):
        """When nvidia-riva-client is not installed, ImportError is caught."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "riva.client":
                raise ImportError("No module named 'riva'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = provider.transcribe("/fake/audio.wav")
        assert result is None

    @patch("backend.providers.nvidia.is_nvidia_enabled", return_value=True)
    def test_returns_none_on_general_failure(self, mock_enabled, provider):
        """Any exception in the transcribe path returns None gracefully."""
        # Patch Path.read_bytes to raise (simulates file read error)
        with patch("backend.providers.nvidia.Path.read_bytes", side_effect=Exception("read error")):
            # Also need riva.client to be "importable" so we get past the ImportError check
            mock_riva = MagicMock()
            import sys
            sys.modules["riva"] = MagicMock()
            sys.modules["riva.client"] = mock_riva
            try:
                result = provider.transcribe("/fake/audio.wav")
                assert result is None
            finally:
                sys.modules.pop("riva", None)
                sys.modules.pop("riva.client", None)


# ──────────────────────────────────────────────
# Graceful failure for all HTTP methods
# ──────────────────────────────────────────────


class TestGracefulFailures:
    """Every public method must return None when the HTTP call raises."""

    @patch("backend.providers.nvidia.is_nvidia_enabled", return_value=True)
    def test_embed_text_http_error(self, mock_enabled, provider):
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response({}, status_code=500)
        provider._client = mock_client
        assert provider.embed_text("test") is None

    @patch("backend.providers.nvidia.is_nvidia_enabled", return_value=True)
    def test_embed_texts_http_error(self, mock_enabled, provider):
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response({}, status_code=500)
        provider._client = mock_client
        assert provider.embed_texts(["a", "b"]) is None

    @patch("backend.providers.nvidia.is_nvidia_enabled", return_value=True)
    def test_embed_image_http_error(self, mock_enabled, provider, tiny_image):
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response({}, status_code=500)
        provider._client = mock_client
        assert provider.embed_image(tiny_image) is None

    @patch("backend.providers.nvidia.is_nvidia_enabled", return_value=True)
    def test_caption_image_http_error(self, mock_enabled, provider, tiny_image):
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response({}, status_code=500)
        provider._client = mock_client
        assert provider.caption_image(tiny_image, "describe") is None

    @patch("backend.providers.nvidia.is_nvidia_enabled", return_value=True)
    def test_detect_objects_http_error(self, mock_enabled, provider, tiny_image):
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response({}, status_code=500)
        provider._grounding_client = mock_client
        assert provider.detect_objects(tiny_image, ["car"]) is None


# ──────────────────────────────────────────────
# Client lifecycle
# ──────────────────────────────────────────────


class TestClientLifecycle:
    def test_close_cleans_up_clients(self, provider):
        mock_client = MagicMock()
        mock_grounding = MagicMock()
        provider._client = mock_client
        provider._grounding_client = mock_grounding

        provider.close()

        mock_client.close.assert_called_once()
        mock_grounding.close.assert_called_once()
        assert provider._client is None
        assert provider._grounding_client is None

    def test_close_is_idempotent(self, provider):
        """Calling close() twice should not raise."""
        provider.close()
        provider.close()
