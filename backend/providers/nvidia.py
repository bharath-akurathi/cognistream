"""
CogniStream — NVIDIA NIM Cloud Provider

Optional cloud backend that uses NVIDIA NIM APIs for higher-quality
inference when an API key is configured.  Falls back gracefully to
local models (Ollama, Whisper, SentenceTransformers) when disabled.

API formats (confirmed via research):
    - NVCLIP:          OpenAI-compatible /v1/embeddings (images as base64 data URIs)
    - NV-Embed:        OpenAI-compatible /v1/embeddings (input_type: query/passage)
    - Grounding DINO:  Custom REST at ai.api.nvidia.com (input/prompt/threshold)
    - VLM (Llama):     OpenAI-compatible /v1/chat/completions (vision messages)
    - Parakeet ASR:    gRPC via nvidia-riva-client (NOT REST)

All methods return None or empty lists when NVIDIA is not configured,
so callers can fall back to local models cleanly.
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Optional

import httpx

from backend.config import (
    NVIDIA_API_KEY,
    NVIDIA_ASR_FUNCTION_ID,
    NVIDIA_ASR_MODEL,
    NVIDIA_BASE_URL,
    NVIDIA_CLIP_MODEL,
    NVIDIA_EMBED_MODEL,
    NVIDIA_GROUNDING_MODEL,
    NVIDIA_GROUNDING_URL,
    NVIDIA_VLM_MODEL,
    is_nvidia_enabled,
)

logger = logging.getLogger(__name__)


class NvidiaProvider:
    """Unified client for NVIDIA NIM cloud APIs."""

    def __init__(self):
        self._client: Optional[httpx.Client] = None
        self._grounding_client: Optional[httpx.Client] = None
        self._failure_streak = 0
        self._disabled_until = 0.0

    _FAILURE_THRESHOLD = 3
    _DISABLE_SECONDS = 300

    @property
    def available(self) -> bool:
        if not is_nvidia_enabled():
            return False
        if time.monotonic() < self._disabled_until:
            return False
        return True

    def _record_success(self) -> None:
        self._failure_streak = 0

    def _record_failure(self, exc: Exception) -> None:
        self._failure_streak += 1
        msg = str(exc).lower()
        network_like = (
            "getaddrinfo" in msg
            or "name or service not known" in msg
            or "temporary failure in name resolution" in msg
            or "timed out" in msg
            or "connection" in msg
        )
        if network_like and self._failure_streak >= self._FAILURE_THRESHOLD:
            self._disabled_until = time.monotonic() + self._DISABLE_SECONDS
            logger.warning(
                "NVIDIA cloud circuit opened for %ds after %d consecutive failures; falling back to local providers.",
                self._DISABLE_SECONDS,
                self._failure_streak,
            )

    def _get_client(self) -> httpx.Client:
        """Client for integrate.api.nvidia.com (embeddings, chat)."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=NVIDIA_BASE_URL,
                headers={
                    "Authorization": f"Bearer {NVIDIA_API_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        return self._client

    def _get_grounding_client(self) -> httpx.Client:
        """Client for ai.api.nvidia.com (Grounding DINO)."""
        if self._grounding_client is None:
            self._grounding_client = httpx.Client(
                headers={
                    "Authorization": f"Bearer {NVIDIA_API_KEY}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=httpx.Timeout(60.0, connect=10.0),
            )
        return self._grounding_client

    @staticmethod
    def _encode_image(path: str) -> str:
        """Read image file and return base64 string."""
        return base64.b64encode(Path(path).read_bytes()).decode("ascii")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # NVCLIP — Multimodal image + text embeddings
    # Same vector space for images and text → direct cross-modal search
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def embed_image(self, image_path: str) -> list[float] | None:
        """Embed an image into vector space using NVCLIP.

        The vector lives in the same space as text embeddings,
        enabling direct text-to-image search.
        """
        if not self.available:
            return None

        try:
            b64 = self._encode_image(image_path)
            resp = self._get_client().post(
                "/embeddings",
                json={
                    "model": NVIDIA_CLIP_MODEL,
                    "input": [f"data:image/jpeg;base64,{b64}"],
                    "encoding_format": "float",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            self._record_success()
            return data["data"][0]["embedding"]
        except Exception as exc:
            self._record_failure(exc)
            logger.warning("NVCLIP image embed failed: %s", exc)
            return None

    def embed_images(self, image_paths: list[str]) -> list[list[float]] | None:
        """Batch embed multiple images. Max 64 per call."""
        if not self.available or not image_paths:
            return None

        try:
            inputs = [
                f"data:image/jpeg;base64,{self._encode_image(p)}"
                for p in image_paths
            ]
            resp = self._get_client().post(
                "/embeddings",
                json={
                    "model": NVIDIA_CLIP_MODEL,
                    "input": inputs,
                    "encoding_format": "float",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            self._record_success()
            return [item["embedding"] for item in data["data"]]
        except Exception as exc:
            self._record_failure(exc)
            logger.warning("NVCLIP batch image embed failed: %s", exc)
            return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # NV-Embed — Text embeddings (1024-dim, QA-optimized)
    # IMPORTANT: input_type must be "query" for searches, "passage" for indexing
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def embed_text(self, text: str, input_type: str = "query") -> list[float] | None:
        """Embed text using NV-Embed."""
        if not self.available:
            return None

        try:
            resp = self._get_client().post(
                "/embeddings",
                json={
                    "model": NVIDIA_EMBED_MODEL,
                    "input": [text],
                    "input_type": input_type,
                    "encoding_format": "float",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            self._record_success()
            return data["data"][0]["embedding"]
        except Exception as exc:
            self._record_failure(exc)
            logger.warning("NV-Embed text embed failed: %s", exc)
            return None

    def embed_texts(
        self, texts: list[str], input_type: str = "passage"
    ) -> list[list[float]] | None:
        """Batch embed multiple texts."""
        if not self.available or not texts:
            return None

        try:
            resp = self._get_client().post(
                "/embeddings",
                json={
                    "model": NVIDIA_EMBED_MODEL,
                    "input": texts,
                    "input_type": input_type,
                    "encoding_format": "float",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            self._record_success()
            return [item["embedding"] for item in data["data"]]
        except Exception as exc:
            self._record_failure(exc)
            logger.warning("NV-Embed batch embed failed: %s", exc)
            return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # NV-Grounding-DINO — Zero-shot object detection
    # Uses custom REST API at ai.api.nvidia.com (NOT OpenAI-compatible)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def detect_objects(
        self,
        image_path: str,
        labels: list[str],
        threshold: float = 0.3,
    ) -> list[dict] | None:
        """Detect objects in an image using NV-Grounding-DINO.

        Args:
            image_path: Path to the image.
            labels: Object categories to detect (e.g. ["car", "person"]).
            threshold: Confidence threshold (0-1).

        Returns:
            List of {label, confidence, bbox} dicts. None if unavailable.
        """
        if not self.available:
            return None

        try:
            b64 = self._encode_image(image_path)
            # Grounding DINO uses period-separated prompt
            prompt = ". ".join(labels) + "."

            resp = self._get_grounding_client().post(
                NVIDIA_GROUNDING_URL,
                json={
                    "input": [
                        {
                            "type": "image_url",
                            "url": f"data:image/jpeg;base64,{b64}",
                        }
                    ],
                    "prompt": prompt,
                    "threshold": threshold,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            self._record_success()
            return self._parse_detections(data, threshold)
        except Exception as exc:
            self._record_failure(exc)
            logger.warning("NV-Grounding-DINO detection failed: %s", exc)
            return None

    @staticmethod
    def _parse_detections(data: dict, threshold: float) -> list[dict]:
        """Parse Grounding DINO response into structured detections."""
        detections = []

        # The API returns pred_logits and pred_boxes tensors
        # or a structured list depending on the endpoint version
        if isinstance(data, list):
            return [d for d in data if d.get("confidence", 1.0) >= threshold]

        if "detections" in data:
            for det in data["detections"]:
                if det.get("confidence", det.get("score", 0)) >= threshold:
                    detections.append({
                        "label": det.get("label", det.get("class", "unknown")),
                        "confidence": det.get("confidence", det.get("score", 0)),
                        "bbox": det.get("bbox", det.get("box", [])),
                    })
            return detections

        # Fallback: try to extract from raw tensor format
        logits = data.get("pred_logits", [[]])
        boxes = data.get("pred_boxes", [[]])
        labels_map = data.get("labels", [])

        if logits and boxes:
            for i, score in enumerate(logits[0] if isinstance(logits[0], list) else logits):
                s = float(score) if not isinstance(score, (list, dict)) else 0
                if s >= threshold:
                    detections.append({
                        "label": labels_map[i] if i < len(labels_map) else f"object_{i}",
                        "confidence": s,
                        "bbox": boxes[0][i] if i < len(boxes[0]) else [],
                    })

        return detections

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # VLM — Vision-language caption / analysis
    # OpenAI-compatible chat/completions with vision messages
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def caption_image(self, image_path: str, prompt: str) -> str | None:
        """Generate a text description of an image using a cloud VLM."""
        if not self.available:
            return None

        try:
            b64 = self._encode_image(image_path)
            resp = self._get_client().post(
                "/chat/completions",
                json={
                    "model": NVIDIA_VLM_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{b64}",
                                    },
                                },
                            ],
                        }
                    ],
                    "max_tokens": 512,
                    "temperature": 0.2,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            self._record_success()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            self._record_failure(exc)
            logger.warning("NVIDIA VLM caption failed: %s", exc)
            return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Parakeet ASR — Speech-to-text via gRPC
    # Requires: pip install nvidia-riva-client
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def transcribe(self, audio_path: str) -> list[dict] | None:
        """Transcribe audio using NVIDIA Parakeet ASR via Riva gRPC.

        Returns list of {start_time, end_time, text} segments,
        or None if NVIDIA is not available or riva-client not installed.
        """
        if not self.available:
            return None

        try:
            import riva.client

            auth = riva.client.Auth(
                ssl_cert=None,
                use_ssl=True,
                uri="grpc.nvcf.nvidia.com:443",
                metadata_args=[
                    ("function-id", NVIDIA_ASR_FUNCTION_ID),
                    ("authorization", f"Bearer {NVIDIA_API_KEY}"),
                ],
            )

            asr_service = riva.client.ASRService(auth)

            audio_bytes = Path(audio_path).read_bytes()
            config = riva.client.RecognitionConfig(
                language_code="en-US",
                max_alternatives=1,
                enable_automatic_punctuation=True,
                enable_word_time_offsets=True,
            )

            response = asr_service.offline_recognize(audio_bytes, config)

            segments = []
            for result in response.results:
                if not result.alternatives:
                    continue
                alt = result.alternatives[0]
                text = alt.transcript.strip()
                if not text:
                    continue

                # Extract word-level timestamps to build segment boundaries
                words = alt.words
                if words:
                    start = words[0].start_time
                    end = words[-1].end_time
                    segments.append({
                        "start_time": start,
                        "end_time": end,
                        "text": text,
                    })
                else:
                    segments.append({
                        "start_time": 0,
                        "end_time": 0,
                        "text": text,
                    })

            return segments

        except ImportError:
            logger.info(
                "nvidia-riva-client not installed — "
                "falling back to local Whisper. "
                "Install with: pip install nvidia-riva-client"
            )
            return None
        except Exception as exc:
            self._record_failure(exc)
            logger.warning("NVIDIA Parakeet ASR failed: %s", exc)
            return None

    def close(self) -> None:
        """Close HTTP clients."""
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._grounding_client is not None:
            self._grounding_client.close()
            self._grounding_client = None


# Module-level singleton
nvidia = NvidiaProvider()
