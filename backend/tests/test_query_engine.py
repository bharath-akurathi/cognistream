"""Tests for backend.retrieval.query_engine — search + temporal re-ranking."""

import pytest

from backend.db.models import SearchResult
from backend.retrieval.query_engine import QueryEngine


class TestTemporalReranking:
    def _make_engine(self, **kwargs):
        """Create a QueryEngine with mock embedder/store for re-ranking tests."""
        return QueryEngine(
            embedder=None,  # not used for re-ranking
            store=None,     # not used for re-ranking
            **kwargs,
        )

    def test_single_candidate_unchanged(self):
        engine = self._make_engine()
        candidates = [
            {"id": "a", "text": "t", "score": 0.9, "start_time": 0, "end_time": 2, "video_id": "v1", "source_type": "fused"},
        ]
        result = engine._temporal_rerank(candidates)
        assert result[0]["score"] == 0.9

    def test_clustered_segments_boosted(self):
        engine = self._make_engine(temporal_weight=0.25)
        candidates = [
            {"id": "a", "text": "t", "score": 0.8, "start_time": 10, "end_time": 12, "video_id": "v1", "source_type": "fused"},
            {"id": "b", "text": "t", "score": 0.7, "start_time": 11, "end_time": 13, "video_id": "v1", "source_type": "fused"},
            {"id": "c", "text": "t", "score": 0.6, "start_time": 100, "end_time": 102, "video_id": "v1", "source_type": "fused"},
        ]
        result = engine._temporal_rerank(candidates)

        # a and b should boost each other, c is isolated
        # a and b should score higher relative to c after re-ranking
        scores = {r["id"]: r["score"] for r in result}
        assert scores["a"] > scores["c"]
        assert scores["b"] > scores["c"]

    def test_different_videos_not_boosted(self):
        engine = self._make_engine(temporal_weight=0.25)
        candidates = [
            {"id": "a", "text": "t", "score": 0.8, "start_time": 10, "end_time": 12, "video_id": "v1", "source_type": "fused"},
            {"id": "b", "text": "t", "score": 0.7, "start_time": 11, "end_time": 13, "video_id": "v2", "source_type": "fused"},
        ]
        result = engine._temporal_rerank(candidates)

        # No boost should occur — different videos
        a_score = next(r for r in result if r["id"] == "a")
        # With temporal_weight=0.25, zero boost → score = 0.75 * 0.8 + 0.25 * 0 = 0.6
        assert a_score["score"] < 0.8

    def test_zero_weight_disables_reranking(self):
        engine = self._make_engine(temporal_weight=0.0)
        candidates = [
            {"id": "a", "text": "t", "score": 0.8, "start_time": 10, "end_time": 12, "video_id": "v1", "source_type": "fused"},
            {"id": "b", "text": "t", "score": 0.7, "start_time": 11, "end_time": 13, "video_id": "v1", "source_type": "fused"},
        ]
        result = engine._temporal_rerank(candidates)
        assert result[0]["score"] == 0.8
        assert result[1]["score"] == 0.7


class TestFormatResults:
    def test_format_basic(self):
        candidates = [
            {
                "id": "s1",
                "text": "A car stopped.",
                "score": 0.95,
                "video_id": "v1",
                "start_time": 10.0,
                "end_time": 12.0,
                "source_type": "fused",
                "frame_path": "/frames/v1/000030.jpg",
            },
        ]
        results = QueryEngine._format(candidates)
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, SearchResult)
        assert r.video_id == "v1"
        assert r.score == 0.95
        assert r.frame_url == "/video/v1/frame/000030.jpg"

    def test_format_no_frame_path(self):
        candidates = [
            {"id": "s1", "text": "t", "score": 0.5, "video_id": "v1", "start_time": 0, "end_time": 1, "source_type": "audio"},
        ]
        results = QueryEngine._format(candidates)
        assert results[0].frame_url is None

    def test_format_empty(self):
        assert QueryEngine._format([]) == []
