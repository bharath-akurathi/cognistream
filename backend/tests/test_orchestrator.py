"""Tests for backend.pipeline.orchestrator — PipelineOrchestrator."""

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from backend.db.models import (
    Event,
    FusedSegment,
    Keyframe,
    ShotSegment,
    TranscriptSegment,
    VideoMeta,
    VideoStatus,
    VisualCaption,
)
from backend.pipeline.orchestrator import (
    PipelineOrchestrator,
    PipelineProgress,
    PipelineResult,
)


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.update_status = MagicMock()
    return db


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.purge_video = MagicMock()
    store.add_segments = MagicMock(return_value=3)
    return store


@pytest.fixture
def video_meta(tmp_path):
    return VideoMeta(
        id="vid001",
        filename="sample.mp4",
        file_path=str(tmp_path / "sample.mp4"),
        duration_sec=10.0,
        fps=30.0,
        width=320,
        height=240,
        total_frames=300,
        status=VideoStatus.UPLOADED,
        created_at="2026-01-01T00:00:00Z",
    )


@pytest.fixture
def sample_keyframes():
    return [
        Keyframe(video_id="vid001", segment_index=0, frame_number=30, timestamp=1.0, file_path="/tmp/f1.jpg"),
        Keyframe(video_id="vid001", segment_index=1, frame_number=90, timestamp=3.0, file_path="/tmp/f2.jpg"),
    ]


@pytest.fixture
def sample_segments():
    return [
        ShotSegment(segment_index=0, start_frame=0, end_frame=149, start_time=0.0, end_time=5.0, frame_count=150),
        ShotSegment(segment_index=1, start_frame=150, end_frame=299, start_time=5.0, end_time=10.0, frame_count=150),
    ]


@pytest.fixture
def sample_captions(sample_keyframes):
    return [
        VisualCaption(
            keyframe=sample_keyframes[0],
            scene_description="A street scene.",
            objects=["car", "tree"],
            activity="Traffic flowing.",
            anomaly=None,
        ),
    ]


@pytest.fixture
def sample_transcripts():
    return [
        TranscriptSegment(start_time=0.5, end_time=2.0, text="Hello world.", keywords=["hello"]),
    ]


@pytest.fixture
def sample_fused():
    return [
        FusedSegment(id="f1", video_id="vid001", start_time=0.0, end_time=2.0, text="A street scene. Hello world.", source_type="fused"),
        FusedSegment(id="f2", video_id="vid001", start_time=2.0, end_time=5.0, text="Traffic flowing.", source_type="visual"),
        FusedSegment(id="f3", video_id="vid001", start_time=0.5, end_time=2.0, text="Hello world.", source_type="audio"),
    ]


# ── Helper: build a fully-mocked orchestrator ──────────────────


def _build_patched_orchestrator(
    mock_db,
    mock_store,
    *,
    vlm_available=True,
    audio_silent=False,
    captions=None,
    transcripts=None,
    fused=None,
    events=None,
    on_progress=None,
    vlm_exception=None,
    stage_exception=None,
):
    """Return an orchestrator with all heavy dependencies mocked."""
    orch = PipelineOrchestrator(db=mock_db, store=mock_store, on_progress=on_progress)

    # Default values
    if captions is None:
        captions = []
    if transcripts is None:
        transcripts = []
    if fused is None:
        fused = []
    if events is None:
        events = []

    # We'll use patch as context managers in the actual tests.
    # This helper just returns the orchestrator configured.
    return orch


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Happy path
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestProcessHappyPath:
    @patch("backend.pipeline.orchestrator.EventDetector")
    @patch("backend.pipeline.orchestrator.KnowledgeGraph")
    @patch("backend.pipeline.orchestrator.MultimodalEmbedder")
    @patch("backend.pipeline.orchestrator.WhisperRunner")
    @patch("backend.pipeline.orchestrator.OllamaClient")
    @patch("backend.pipeline.orchestrator.VLMRunner")
    @patch("backend.pipeline.orchestrator.AudioExtractor")
    @patch("backend.pipeline.orchestrator.FrameSampler")
    @patch("backend.pipeline.orchestrator.ShotDetector")
    def test_all_stages_succeed(
        self,
        MockShotDetector,
        MockFrameSampler,
        MockAudioExtractor,
        MockVLMRunner,
        MockOllamaClient,
        MockWhisperRunner,
        MockEmbedder,
        MockKG,
        MockEventDetector,
        mock_db,
        mock_store,
        video_meta,
        sample_segments,
        sample_keyframes,
        sample_captions,
        sample_transcripts,
        sample_fused,
    ):
        # Configure mocks
        MockShotDetector.return_value.detect.return_value = sample_segments
        MockFrameSampler.return_value.sample.return_value = sample_keyframes

        audio_result = MagicMock()
        audio_result.is_silent = False
        audio_result.audio_path = "/tmp/audio.wav"
        MockAudioExtractor.return_value.extract.return_value = audio_result

        MockOllamaClient.return_value.is_available.return_value = True
        MockVLMRunner.return_value.analyse_keyframes.return_value = sample_captions

        MockWhisperRunner.return_value.transcribe.return_value = sample_transcripts
        MockWhisperRunner.return_value.unload_model = MagicMock()

        MockEmbedder.return_value.fuse_and_embed.return_value = sample_fused
        MockEmbedder.return_value.embed.return_value = sample_fused
        MockEmbedder.return_value.unload_model = MagicMock()

        MockKG.return_value.build_from_captions = MagicMock()
        MockKG.return_value.save = MagicMock()

        MockEventDetector.return_value.detect.return_value = []

        mock_store.add_segments.return_value = len(sample_fused)

        orch = PipelineOrchestrator(db=mock_db, store=mock_store)
        result = orch.process(video_meta)

        assert result.success is True
        assert result.video_id == "vid001"
        assert result.segments_stored == len(sample_fused)
        assert result.errors == []
        assert result.elapsed_sec >= 0  # may round to 0.0 on fast machines

        # DB should have been marked PROCESSING then PROCESSED
        calls = mock_db.update_status.call_args_list
        assert calls[0].args[1] == VideoStatus.PROCESSING
        assert calls[-1].args[1] == VideoStatus.PROCESSED

        # ChromaDB purge should have been called
        mock_store.purge_video.assert_called_once_with("vid001")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VLM unavailable — graceful degradation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestProcessVLMUnavailable:
    @patch("backend.pipeline.orchestrator.EventDetector")
    @patch("backend.pipeline.orchestrator.KnowledgeGraph")
    @patch("backend.pipeline.orchestrator.MultimodalEmbedder")
    @patch("backend.pipeline.orchestrator.WhisperRunner")
    @patch("backend.pipeline.orchestrator.OllamaClient")
    @patch("backend.pipeline.orchestrator.AudioExtractor")
    @patch("backend.pipeline.orchestrator.FrameSampler")
    @patch("backend.pipeline.orchestrator.ShotDetector")
    def test_skips_vlm_with_warning(
        self,
        MockShotDetector,
        MockFrameSampler,
        MockAudioExtractor,
        MockOllamaClient,
        MockWhisperRunner,
        MockEmbedder,
        MockKG,
        MockEventDetector,
        mock_db,
        mock_store,
        video_meta,
        sample_segments,
        sample_keyframes,
        sample_transcripts,
        sample_fused,
    ):
        MockShotDetector.return_value.detect.return_value = sample_segments
        MockFrameSampler.return_value.sample.return_value = sample_keyframes

        audio_result = MagicMock()
        audio_result.is_silent = False
        audio_result.audio_path = "/tmp/audio.wav"
        MockAudioExtractor.return_value.extract.return_value = audio_result

        # VLM not available
        MockOllamaClient.return_value.is_available.return_value = False

        MockWhisperRunner.return_value.transcribe.return_value = sample_transcripts
        MockWhisperRunner.return_value.unload_model = MagicMock()

        MockEmbedder.return_value.fuse_and_embed.return_value = sample_fused
        MockEmbedder.return_value.embed.return_value = sample_fused
        MockEmbedder.return_value.unload_model = MagicMock()

        MockKG.return_value.build_from_captions = MagicMock()
        MockKG.return_value.save = MagicMock()
        MockEventDetector.return_value.detect.return_value = []
        mock_store.add_segments.return_value = len(sample_fused)

        orch = PipelineOrchestrator(db=mock_db, store=mock_store)
        result = orch.process(video_meta)

        assert result.success is True
        assert any("Ollama" in w or "VLM" in w for w in result.warnings)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Silent audio — skip whisper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestProcessSilentAudio:
    @patch("backend.pipeline.orchestrator.EventDetector")
    @patch("backend.pipeline.orchestrator.KnowledgeGraph")
    @patch("backend.pipeline.orchestrator.MultimodalEmbedder")
    @patch("backend.pipeline.orchestrator.WhisperRunner")
    @patch("backend.pipeline.orchestrator.OllamaClient")
    @patch("backend.pipeline.orchestrator.VLMRunner")
    @patch("backend.pipeline.orchestrator.AudioExtractor")
    @patch("backend.pipeline.orchestrator.FrameSampler")
    @patch("backend.pipeline.orchestrator.ShotDetector")
    def test_silent_audio_skips_whisper(
        self,
        MockShotDetector,
        MockFrameSampler,
        MockAudioExtractor,
        MockVLMRunner,
        MockOllamaClient,
        MockWhisperRunner,
        MockEmbedder,
        MockKG,
        MockEventDetector,
        mock_db,
        mock_store,
        video_meta,
        sample_segments,
        sample_keyframes,
        sample_captions,
        sample_fused,
    ):
        MockShotDetector.return_value.detect.return_value = sample_segments
        MockFrameSampler.return_value.sample.return_value = sample_keyframes

        # Audio is silent
        audio_result = MagicMock()
        audio_result.is_silent = True
        MockAudioExtractor.return_value.extract.return_value = audio_result

        MockOllamaClient.return_value.is_available.return_value = True
        MockVLMRunner.return_value.analyse_keyframes.return_value = sample_captions

        MockEmbedder.return_value.fuse_and_embed.return_value = sample_fused
        MockEmbedder.return_value.embed.return_value = sample_fused
        MockEmbedder.return_value.unload_model = MagicMock()

        MockKG.return_value.build_from_captions = MagicMock()
        MockKG.return_value.save = MagicMock()
        MockEventDetector.return_value.detect.return_value = []
        mock_store.add_segments.return_value = len(sample_fused)

        orch = PipelineOrchestrator(db=mock_db, store=mock_store)
        result = orch.process(video_meta)

        assert result.success is True
        assert any("silent" in w.lower() for w in result.warnings)
        # WhisperRunner should not have been used for transcription
        MockWhisperRunner.return_value.transcribe.assert_not_called()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# No captions and no transcripts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestProcessNoCaptionsNoTranscripts:
    @patch("backend.pipeline.orchestrator.EventDetector")
    @patch("backend.pipeline.orchestrator.KnowledgeGraph")
    @patch("backend.pipeline.orchestrator.MultimodalEmbedder")
    @patch("backend.pipeline.orchestrator.WhisperRunner")
    @patch("backend.pipeline.orchestrator.OllamaClient")
    @patch("backend.pipeline.orchestrator.AudioExtractor")
    @patch("backend.pipeline.orchestrator.FrameSampler")
    @patch("backend.pipeline.orchestrator.ShotDetector")
    def test_warns_but_succeeds(
        self,
        MockShotDetector,
        MockFrameSampler,
        MockAudioExtractor,
        MockOllamaClient,
        MockWhisperRunner,
        MockEmbedder,
        MockKG,
        MockEventDetector,
        mock_db,
        mock_store,
        video_meta,
        sample_segments,
        sample_keyframes,
    ):
        MockShotDetector.return_value.detect.return_value = sample_segments
        MockFrameSampler.return_value.sample.return_value = sample_keyframes

        # No audio stream
        MockAudioExtractor.return_value.extract.return_value = None

        # VLM not available
        MockOllamaClient.return_value.is_available.return_value = False

        MockEmbedder.return_value.fuse_and_embed.return_value = []
        MockEmbedder.return_value.unload_model = MagicMock()

        MockKG.return_value.build_from_captions = MagicMock()
        MockKG.return_value.save = MagicMock()
        MockEventDetector.return_value.detect.return_value = []

        orch = PipelineOrchestrator(db=mock_db, store=mock_store)
        result = orch.process(video_meta)

        assert result.success is True
        # Should have a warning about no captions or transcripts
        assert any("caption" in w.lower() or "transcript" in w.lower() for w in result.warnings)
        assert result.segments_stored == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Concurrency guard
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestProcessConcurrency:
    @patch("backend.pipeline.orchestrator.EventDetector")
    @patch("backend.pipeline.orchestrator.KnowledgeGraph")
    @patch("backend.pipeline.orchestrator.MultimodalEmbedder")
    @patch("backend.pipeline.orchestrator.WhisperRunner")
    @patch("backend.pipeline.orchestrator.OllamaClient")
    @patch("backend.pipeline.orchestrator.AudioExtractor")
    @patch("backend.pipeline.orchestrator.FrameSampler")
    @patch("backend.pipeline.orchestrator.ShotDetector")
    def test_second_call_while_busy_raises(
        self,
        MockShotDetector,
        MockFrameSampler,
        MockAudioExtractor,
        MockOllamaClient,
        MockWhisperRunner,
        MockEmbedder,
        MockKG,
        MockEventDetector,
        mock_db,
        mock_store,
        video_meta,
        sample_segments,
        sample_keyframes,
    ):
        # Make shot detection block so the pipeline stays busy
        barrier = threading.Barrier(2, timeout=5)
        done_event = threading.Event()

        def slow_detect(meta):
            barrier.wait()  # sync with test thread
            done_event.wait(timeout=5)  # hold the lock until told to finish
            return sample_segments

        MockShotDetector.return_value.detect.side_effect = slow_detect
        MockFrameSampler.return_value.sample.return_value = sample_keyframes

        audio_result = MagicMock()
        audio_result.is_silent = True
        MockAudioExtractor.return_value.extract.return_value = audio_result
        MockOllamaClient.return_value.is_available.return_value = False
        MockWhisperRunner.return_value.transcribe.return_value = []
        MockWhisperRunner.return_value.unload_model = MagicMock()
        MockEmbedder.return_value.fuse_and_embed.return_value = []
        MockEmbedder.return_value.unload_model = MagicMock()
        MockKG.return_value.build_from_captions = MagicMock()
        MockKG.return_value.save = MagicMock()
        MockEventDetector.return_value.detect.return_value = []

        orch = PipelineOrchestrator(db=mock_db, store=mock_store)

        # Start first process in a background thread
        errors = []

        def run_first():
            try:
                orch.process(video_meta)
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=run_first)
        t.start()

        # Wait until the first call has acquired the lock and is inside slow_detect
        barrier.wait()

        # Now try a second call — it should raise RuntimeError
        video_meta2 = VideoMeta(
            id="vid002", filename="other.mp4", file_path="/tmp/other.mp4",
            status=VideoStatus.UPLOADED, created_at="2026-01-01T00:00:00Z",
        )
        with pytest.raises(RuntimeError, match="Pipeline busy"):
            orch.process(video_meta2)

        # Let the first pipeline finish
        done_event.set()
        t.join(timeout=10)
        assert not errors, f"First pipeline raised: {errors}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stage failure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestProcessFailure:
    @patch("backend.pipeline.orchestrator.ShotDetector")
    def test_exception_in_stage_leads_to_failure(
        self,
        MockShotDetector,
        mock_db,
        mock_store,
        video_meta,
    ):
        MockShotDetector.return_value.detect.side_effect = RuntimeError("corrupt video")

        orch = PipelineOrchestrator(db=mock_db, store=mock_store)
        result = orch.process(video_meta)

        assert result.success is False
        assert len(result.errors) >= 1
        assert "corrupt video" in result.errors[0]

        # Should have been marked FAILED
        failed_calls = [
            c for c in mock_db.update_status.call_args_list
            if len(c.args) >= 2 and c.args[1] == VideoStatus.FAILED
        ]
        assert len(failed_calls) >= 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _emit() progress callback
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEmit:
    def test_emit_calls_callback(self, mock_db, mock_store):
        callback = MagicMock()
        orch = PipelineOrchestrator(db=mock_db, store=mock_store, on_progress=callback)

        progress = PipelineProgress(video_id="vid001", started_at=time.monotonic())
        orch._emit(progress, 3, "Audio extraction")

        callback.assert_called_once()
        arg = callback.call_args[0][0]
        assert isinstance(arg, PipelineProgress)
        assert arg.stage_number == 3
        assert arg.stage == "Audio extraction"
        assert "Stage 3/" in arg.detail

    def test_emit_without_callback_does_not_raise(self, mock_db, mock_store):
        orch = PipelineOrchestrator(db=mock_db, store=mock_store, on_progress=None)
        progress = PipelineProgress(video_id="vid001", started_at=time.monotonic())
        # Should not raise
        orch._emit(progress, 1, "Shot detection")

    def test_emit_updates_elapsed(self, mock_db, mock_store):
        callback = MagicMock()
        orch = PipelineOrchestrator(db=mock_db, store=mock_store, on_progress=callback)

        started = time.monotonic() - 5.0  # pretend started 5s ago
        progress = PipelineProgress(video_id="vid001", started_at=started)
        orch._emit(progress, 7, "Knowledge graph")

        assert progress.elapsed_sec >= 4.9


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _events_to_segments()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEventsToSegments:
    def test_converts_events_to_fused_segments(self):
        events = [
            Event(
                id="e1",
                video_id="vid001",
                event_type="car_arrival",
                start_time=10.0,
                end_time=15.0,
                description="A car arrived at the entrance.",
                entities=["red_car", "entrance"],
            ),
            Event(
                id="e2",
                video_id="vid001",
                event_type="building_entry",
                start_time=20.0,
                end_time=25.0,
                description="Person entered the building.",
                entities=["person_1", "building"],
            ),
        ]

        segments = PipelineOrchestrator._events_to_segments("vid001", events)

        assert len(segments) == 2
        for seg in segments:
            assert isinstance(seg, FusedSegment)
            assert seg.video_id == "vid001"
            assert seg.source_type == "event"
            assert seg.id  # non-empty

        assert segments[0].start_time == 10.0
        assert segments[0].end_time == 15.0
        assert "car_arrival" in segments[0].text
        assert "red_car" in segments[0].text
        assert "entrance" in segments[0].text

        assert segments[1].start_time == 20.0
        assert "building_entry" in segments[1].text

    def test_empty_events_returns_empty(self):
        segments = PipelineOrchestrator._events_to_segments("vid001", [])
        assert segments == []

    def test_segment_ids_are_unique(self):
        events = [
            Event(id="e1", video_id="v", event_type="t", start_time=0, end_time=1, entities=[]),
            Event(id="e2", video_id="v", event_type="t", start_time=1, end_time=2, entities=[]),
        ]
        segments = PipelineOrchestrator._events_to_segments("v", events)
        ids = [s.id for s in segments]
        assert len(set(ids)) == len(ids), "Segment IDs should be unique"
