"""
CogniStream — Query Engine

End-to-end semantic retrieval pipeline.  Takes a natural-language query
and returns ranked video segments with timestamps.

Pipeline stages:
    1. **Embed**    — encode query with the same SentenceTransformer used
                      at index time.
    2. **Search**   — vector similarity search in ChromaDB with optional
                      video_id / source_type filters.
    3. **Re-rank**  — temporal proximity boost: segments near other
                      high-scoring segments are promoted.
    4. **Format**   — map raw results into SearchResult dataclasses.

Temporal re-ranking rationale:
    If a user asks "when did the red car arrive", three segments at
    t=33s, 35s, 37s all matching is stronger evidence than a lone match
    at t=200s.  A Gaussian decay kernel lets nearby segments contribute
    mutual score boosts, controlled by a configurable time window.

Usage:
    engine = QueryEngine(embedder, store)
    results = engine.search("when did the professor explain deadlocks")
    for r in results:
        print(f"[{r.start_time:.1f}s] {r.text} (score={r.score:.3f})")
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Optional

from backend.config import (
    DEFAULT_TOP_K,
    RETRIEVAL_WEIGHT_AUDIO,
    RETRIEVAL_WEIGHT_TEXT,
    RETRIEVAL_WEIGHT_VISUAL,
)
from backend.db.chroma_store import ChromaStore
from backend.db.models import SearchResult
from backend.fusion.multimodal_embedder import MultimodalEmbedder

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Re-ranking config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Temporal window (seconds): segments within this range of each other
# can exchange score boosts.
_TEMPORAL_WINDOW = 15.0

# Weight of the temporal boost relative to the base similarity score.
# 0.0 = no temporal effect, 1.0 = temporal signal equals similarity signal.
_TEMPORAL_WEIGHT = 0.25

# Gaussian sigma for temporal decay (in seconds).  Controls how quickly
# the boost falls off with distance.  sigma = window/3 gives near-zero
# contribution at the window edge.
_TEMPORAL_SIGMA = _TEMPORAL_WINDOW / 3.0

# Over-fetch factor: retrieve more candidates from ChromaDB than top_k
# so the re-ranker has room to promote temporally clustered results.
_OVERFETCH_FACTOR = 2
_LEXICAL_WEIGHT = 0.20
_SOURCE_PRIOR_WEIGHT = 0.08

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "it", "this", "that", "these", "those", "as", "if", "then", "than",
    "not", "no", "you", "your", "i", "we", "they", "he", "she", "them",
}


class QueryEngine:
    """Semantic search with temporal re-ranking."""

    def __init__(
        self,
        embedder: MultimodalEmbedder | None = None,
        store: ChromaStore | None = None,
        db=None,
        temporal_window: float = _TEMPORAL_WINDOW,
        temporal_weight: float = _TEMPORAL_WEIGHT,
    ):
        self.embedder = embedder or MultimodalEmbedder()
        self.store = store or ChromaStore()
        self.db = db  # SQLiteDB — used for FTS5 transcript search
        self.temporal_window = temporal_window
        self.temporal_weight = temporal_weight

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Public API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        video_id: Optional[str] = None,
        source_filter: Optional[str] = None,
        search_mode: str = "hybrid",
        agentic: bool = False,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Execute the full search pipeline.

        Args:
            query:         Natural-language search query.
            top_k:         Number of final results to return.
            video_id:      Scope search to one video (optional).
            source_filter: Restrict to a source type: "visual", "audio",
                           "fused", or "event" (optional).
            agentic:       If True, run query decomposition + VLM reflection rerank
                           (VSS 3 agentic search). Slower but better for compound queries.
            min_score:     Minimum score threshold. Results below this are
                           filtered out (returns empty instead of garbage).

        Returns:
            Up to *top_k* :class:`SearchResult` objects sorted by
            final score (descending).
        """
        if not query.strip():
            logger.warning("Empty query — returning no results.")
            return []

        # Quoted-phrase detection: "..." or \u201c...\u201d → FTS5 verbatim.
        # Borrowed from Moment Search's quoted-phrase exact-match routing.
        stripped = query.strip()
        quoted_match = re.match(r'^["\u201c](.+)["\u201d]$', stripped)
        if quoted_match and self.db:
            phrase = quoted_match.group(1)
            logger.info("Quoted phrase search: '%s'", phrase)
            speech_hits = self.db.search_transcripts(phrase, video_id=video_id, limit=top_k)
            results = [
                SearchResult(
                    video_id=h["video_id"],
                    segment_id=f"speech-{h['video_id']}-{h['start_time']:.1f}",
                    start_time=h["start_time"],
                    end_time=h["end_time"],
                    text=h["text"],
                    source_type="speech",
                    score=min(0.95, 0.7 + h["score"] * 0.05),
                    speech_snippet=h.get("snippet"),
                )
                for h in speech_hits
            ]
            if min_score > 0:
                results = [r for r in results if r.score >= min_score]
            logger.info("Quoted phrase returned %d results.", len(results))
            return results

        # Speech-only mode: skip vector search entirely, use FTS5 only.
        if search_mode == "speech" and self.db:
            logger.info("Speech-only search: '%s'", query)
            speech_hits = self.db.search_transcripts(query, video_id=video_id, limit=top_k)
            results = [
                SearchResult(
                    video_id=h["video_id"],
                    segment_id=f"speech-{h['video_id']}-{h['start_time']:.1f}",
                    start_time=h["start_time"],
                    end_time=h["end_time"],
                    text=h["text"],
                    source_type="speech",
                    score=min(0.95, 0.7 + h["score"] * 0.05),
                    speech_snippet=h.get("snippet"),
                )
                for h in speech_hits
            ]
            if min_score > 0:
                results = [r for r in results if r.score >= min_score]
            return results

        # Agentic mode: decompose, search each sub-query, fuse, then rerank with VLM
        if agentic:
            return self.search_agentic(query, top_k, video_id, source_filter)

        # Stage 1: embed the query (text embedding)
        logger.info("Search query: '%s'", query)
        query_embedding = self.embedder.embed_query(query)

        # Stage 2: vector search (over-fetch for re-ranking headroom)
        fetch_k = min(top_k * _OVERFETCH_FACTOR, top_k + 20)
        try:
            raw_results = self.store.query(
                embedding=query_embedding,
                top_k=fetch_k,
                video_id=video_id,
                source_filter=source_filter,
            )
        except Exception as exc:
            if not self._is_dimension_mismatch(exc):
                raise

            expected_dim, got_dim = self._parse_dims(str(exc))
            logger.warning(
                "Embedding dimension mismatch during search (expected=%s, got=%s). Retrying with compatible query embedding.",
                expected_dim,
                got_dim,
            )

            if expected_dim == 384:
                query_embedding = self.embedder.embed_query_local(query)
            elif expected_dim == 1024:
                from backend.providers.nvidia import nvidia

                vec = nvidia.embed_text(query, input_type="query") if nvidia.available else None
                if not vec:
                    # Fall back to local if NVIDIA is unavailable; this may still fail,
                    # but preserves graceful behavior in degraded environments.
                    query_embedding = self.embedder.embed_query_local(query)
                else:
                    query_embedding = vec
            else:
                # Unknown expected dimension; safest retry is local embedding.
                query_embedding = self.embedder.embed_query_local(query)

            raw_results = self.store.query(
                embedding=query_embedding,
                top_k=fetch_k,
                video_id=video_id,
                source_filter=source_filter,
            )

        # Stage 2b: visual embedding search (SigLIP/NVCLIP)
        # Search with the visual embedding of the query text to find
        # matching frames that were embedded with SigLIP/NVCLIP.
        if RETRIEVAL_WEIGHT_VISUAL > 0:
            visual_results = self._visual_search(query, fetch_k, video_id)
            if visual_results:
                raw_results = self._merge_multi_vector(
                    raw_results, visual_results,
                    RETRIEVAL_WEIGHT_TEXT, RETRIEVAL_WEIGHT_VISUAL,
                )

        if not raw_results:
            logger.info("No results found for query: '%s'", query)
            return []

        logger.debug("ChromaDB returned %d candidates.", len(raw_results))

        # Stage 3: temporal re-ranking
        reranked = self._temporal_rerank(raw_results)

        # Stage 3b: hybrid reranking for better query differentiation
        reranked = self._hybrid_rerank(reranked, query)

        # Stage 4: FTS5 speech search — merge transcript hits into the
        # candidate pool so speech-matching segments get promoted.
        # Skipped in visual-only mode.
        if search_mode != "visual":
            reranked = self._merge_speech_results(reranked, query, video_id)

        # Stage 5: per-video diversification — prevent caption-rich videos
        # from monopolising results when multiple videos match.
        reranked = self._diversify(reranked, top_k)

        # Stage 6: min-score threshold — return empty instead of noise
        if min_score > 0:
            reranked = [c for c in reranked if c.get("score", 0) >= min_score]

        # Stage 7: collapse adjacent same-video frames into moments
        reranked = self._collapse_moments(reranked)

        # Stage 8: format and trim to top_k
        results = self._format(reranked[:top_k])

        logger.info(
            "Search complete: %d results (top score=%.3f)",
            len(results),
            results[0].score if results else 0.0,
        )
        return results

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Stage 2b: Multi-vector visual search
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Agentic search (VSS 3): decompose → multi-search → fuse → VLM rerank
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def search_agentic(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        video_id: Optional[str] = None,
        source_filter: Optional[str] = None,
    ) -> list[SearchResult]:
        """Agentic search: query decomposition + multi-vector + VLM reflection rerank.

        Borrowed from NVIDIA Metropolis VSS 3.
        """
        logger.info("Agentic search: '%s'", query)

        # Step 1: Decompose the query into sub-queries
        sub_queries = self._decompose_query(query)
        logger.info("Decomposed into %d sub-queries: %s", len(sub_queries), sub_queries)

        # Step 2: Search each sub-query and collect results
        all_results: dict[str, dict] = {}  # segment_id → result with merged score
        for sub_q in sub_queries:
            try:
                emb = self.embedder.embed_query(sub_q)
                fetch_k = min(top_k * _OVERFETCH_FACTOR, top_k + 20)
                sub_results = self.store.query(
                    embedding=emb,
                    top_k=fetch_k,
                    video_id=video_id,
                    source_filter=source_filter,
                )
                # Merge into the global result set, keeping the max score per segment
                for r in sub_results:
                    rid = r["id"]
                    if rid not in all_results or r["score"] > all_results[rid]["score"]:
                        all_results[rid] = r
            except Exception as exc:
                logger.warning("Sub-query '%s' failed: %s", sub_q, exc)

        if not all_results:
            return []

        # Step 3: Visual search fusion (if visual embeddings exist)
        if RETRIEVAL_WEIGHT_VISUAL > 0:
            visual_results = self._visual_search(query, top_k * 2, video_id)
            if visual_results:
                merged_list = self._merge_multi_vector(
                    list(all_results.values()), visual_results,
                    RETRIEVAL_WEIGHT_TEXT, RETRIEVAL_WEIGHT_VISUAL,
                )
                all_results = {r["id"]: r for r in merged_list}

        # Step 4: Temporal re-ranking
        candidates = list(all_results.values())
        candidates.sort(key=lambda c: c["score"], reverse=True)
        candidates = candidates[: top_k * 2]
        candidates = self._temporal_rerank(candidates)

        # Step 5: VLM reflection rerank (top candidates only — expensive)
        candidates = self._vlm_reflect_rerank(query, candidates[: top_k * 2])

        # Step 6: Format
        return self._format(candidates[:top_k])

    def _decompose_query(self, query: str) -> list[str]:
        """Use an LLM to break a compound query into sub-queries.

        Falls back to simple keyword splitting if no LLM is available.
        """
        # Try NVIDIA cloud LLM first
        try:
            from backend.providers.nvidia import nvidia
            if nvidia.available:
                from backend.config import NVIDIA_API_KEY, NVIDIA_BASE_URL
                import httpx
                with httpx.Client(timeout=httpx.Timeout(15.0, connect=5.0)) as client:
                    resp = client.post(
                        f"{NVIDIA_BASE_URL}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {NVIDIA_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": "meta/llama-3.2-11b-vision-instruct",
                            "messages": [
                                {
                                    "role": "system",
                                    "content": (
                                        "You decompose video search queries into 1-4 simpler sub-queries. "
                                        "Return only the sub-queries, one per line, no numbering or explanation. "
                                        "If the query is already simple, return it as-is."
                                    ),
                                },
                                {"role": "user", "content": f"Query: {query}\n\nSub-queries:"},
                            ],
                            "max_tokens": 200,
                            "temperature": 0.2,
                        },
                    )
                    if resp.status_code == 200:
                        text = resp.json()["choices"][0]["message"]["content"]
                        subs = [
                            line.strip().lstrip("-•0123456789. ")
                            for line in text.split("\n")
                            if line.strip() and len(line.strip()) > 3
                        ]
                        if subs:
                            return subs[:4]
        except Exception as exc:
            logger.debug("LLM decomposition failed: %s", exc)

        # Fallback: split on common conjunctions
        import re
        parts = re.split(
            r"\s+(?:and|then|after|before|while|when)\s+",
            query, flags=re.IGNORECASE,
        )
        parts = [p.strip() for p in parts if p.strip()]
        return parts if len(parts) > 1 else [query]

    def _vlm_reflect_rerank(
        self, query: str, candidates: list[dict],
    ) -> list[dict]:
        """Use a VLM to verify each top candidate matches the query.

        Adjusts scores based on VLM verdict. Skipped if no frame_path on the
        candidate or no VLM available.
        """
        if not candidates:
            return candidates

        # Use NVIDIA cloud VLM for reflection (fast + accurate)
        try:
            from backend.providers.nvidia import nvidia
            if not nvidia.available:
                return candidates  # Skip reflection if no VLM
        except ImportError:
            return candidates

        from backend.providers.nvidia import nvidia as _nv

        # Only rerank candidates that have a frame_path
        rerank_count = min(5, len(candidates))  # Top 5 only — VLM calls are expensive
        for i in range(rerank_count):
            c = candidates[i]
            frame = c.get("frame_path")
            if not frame:
                continue

            try:
                prompt = (
                    f"Does this image match the search query: '{query}'? "
                    "Answer with a single word: yes, no, or maybe."
                )
                answer = _nv.caption_image(frame, prompt) or ""
                answer = answer.strip().lower()

                # Boost or penalize the score
                if answer.startswith("yes"):
                    c["score"] = min(1.0, c["score"] * 1.2)
                    c["_reflection"] = "yes"
                elif answer.startswith("no"):
                    c["score"] = c["score"] * 0.5
                    c["_reflection"] = "no"
                else:
                    c["_reflection"] = "maybe"
            except Exception as exc:
                logger.debug("VLM reflection failed for candidate %d: %s", i, exc)

        # Re-sort after reflection adjustments
        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates

    def _visual_search(
        self, query: str, top_k: int, video_id: Optional[str]
    ) -> list[dict]:
        """Search using visual embeddings (SigLIP or NVCLIP).

        Embeds the query text in the visual vector space and searches
        for matching frame embeddings stored as source_type="visual".
        """
        visual_embedding = None

        # Try NVIDIA NVCLIP first
        from backend.providers.nvidia import nvidia
        if nvidia.available:
            visual_embedding = nvidia.embed_text(query)

        # Fall back to local SigLIP
        if visual_embedding is None:
            siglip = None
            try:
                from backend.visual.siglip_embedder import SigLIPEmbedder
                siglip = SigLIPEmbedder()
                if siglip.enabled:
                    visual_embedding = siglip.embed_text(query)
            except Exception as exc:
                logger.debug("Visual query embedding failed: %s", exc)
            finally:
                if siglip is not None:
                    siglip.unload()

        if visual_embedding is None:
            return []

        try:
            return self.store.query(
                embedding=visual_embedding,
                top_k=top_k,
                video_id=video_id,
                source_filter="visual",
            )
        except Exception as exc:
            if not self._is_dimension_mismatch(exc):
                raise

            expected_dim, got_dim = self._parse_dims(str(exc))
            logger.warning(
                "Skipping visual vector search because the active ChromaDB collection "
                "expects %s-dim embeddings but the visual query produced %s-dim.",
                expected_dim,
                got_dim,
            )
            return []

    @staticmethod
    def _merge_multi_vector(
        text_results: list[dict],
        visual_results: list[dict],
        text_weight: float,
        visual_weight: float,
    ) -> list[dict]:
        """Merge results from text and visual searches.

        Segments found in both get a combined score.  Segments found in
        only one modality keep their weighted score.
        """
        total_weight = text_weight + visual_weight
        if total_weight <= 0:
            return text_results

        tw = text_weight / total_weight
        vw = visual_weight / total_weight

        merged: dict[str, dict] = {}

        for r in text_results:
            rid = r["id"]
            merged[rid] = {**r, "score": r["score"] * tw}

        for r in visual_results:
            rid = r["id"]
            if rid in merged:
                merged[rid]["score"] += r["score"] * vw
            else:
                merged[rid] = {**r, "score": r["score"] * vw}

        results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return results

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Stage 4: FTS5 speech search merge
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _merge_speech_results(
        self, candidates: list[dict], query: str, video_id: str | None,
    ) -> list[dict]:
        """Merge FTS5 transcript hits into the candidate pool.

        Speech matches that overlap a vector-search candidate get their
        score boosted. Speech matches with no overlap are injected as
        new candidates (source_type='speech'). This enables queries like
        `"rules"` to surface transcript moments that SigLIP/NVCLIP can't.
        """
        if not self.db:
            return candidates
        try:
            speech_hits = self.db.search_transcripts(query, video_id=video_id, limit=20)
        except Exception:
            return candidates
        if not speech_hits:
            return candidates

        # Index existing candidates by (video_id, time_bucket) for overlap detection
        _BUCKET = 3.0  # seconds — speech hit within this of a vector hit is a match
        by_key: dict[tuple, dict] = {}
        for c in candidates:
            key = (c.get("video_id", ""), round(c.get("start_time", 0) / _BUCKET))
            if key not in by_key or c.get("score", 0) > by_key[key].get("score", 0):
                by_key[key] = c

        injected = 0
        for sh in speech_hits:
            key = (sh["video_id"], round(sh["start_time"] / _BUCKET))
            existing = by_key.get(key)
            if existing:
                # Boost: speech evidence strengthens the vector candidate
                boost = min(0.15, sh["score"] * 0.02)
                existing["score"] = existing.get("score", 0) + boost
                existing["_speech_snippet"] = sh.get("snippet", "")
            else:
                # Inject new candidate from speech-only match.
                # Score must be competitive with visual results (~0.5-0.7
                # range) so it doesn't get buried under top-K truncation.
                candidates.append({
                    "id": f"speech-{sh['video_id']}-{sh['start_time']:.1f}",
                    "video_id": sh["video_id"],
                    "start_time": sh["start_time"],
                    "end_time": sh["end_time"],
                    "text": sh["text"],
                    "source_type": "speech",
                    "score": min(0.85, 0.6 + sh["score"] * 0.05),
                    "frame_path": None,
                    "_speech_snippet": sh.get("snippet", ""),
                })
                injected += 1

        if injected:
            candidates.sort(key=lambda c: c.get("score", 0), reverse=True)
            logger.info("Speech search injected %d new candidates.", injected)

        return candidates

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Stage 5: Per-video diversification
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _diversify(candidates: list[dict], top_k: int) -> list[dict]:
        """Ensure multiple matching videos appear in top-K results.

        Without this, a video with many text-rich segments (e.g. one with
        VLM captions) dominates over a video whose segments match
        only via visual embeddings. This was the cause of lecture_clip
        never appearing in results against cooking_demo.

        Algorithm: round-robin across videos for the first top_k slots,
        ordered by each video's best score. Then fill remaining slots
        in pure score order.
        """
        if len(candidates) <= top_k:
            return candidates

        # Group by video_id, preserving score order within each group
        from collections import defaultdict
        by_video: dict[str, list[dict]] = defaultdict(list)
        for c in candidates:
            by_video[c.get("video_id", "")].append(c)

        if len(by_video) <= 1:
            return candidates  # single video, nothing to diversify

        # Sort videos by their best candidate's score
        video_order = sorted(
            by_video.keys(),
            key=lambda vid: by_video[vid][0].get("score", 0),
            reverse=True,
        )

        # Round-robin: pick 1 from each video in order, repeat until top_k
        diversified: list[dict] = []
        seen_ids: set[str] = set()
        pointers = {vid: 0 for vid in video_order}

        while len(diversified) < top_k:
            added_this_round = False
            for vid in video_order:
                if len(diversified) >= top_k:
                    break
                idx = pointers[vid]
                if idx < len(by_video[vid]):
                    c = by_video[vid][idx]
                    cid = c.get("id", str(id(c)))
                    if cid not in seen_ids:
                        diversified.append(c)
                        seen_ids.add(cid)
                        added_this_round = True
                    pointers[vid] = idx + 1
            if not added_this_round:
                break

        return diversified

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Stage 7: Collapse adjacent frames into moments
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _collapse_moments(
        candidates: list[dict], gap_sec: float = 5.0,
    ) -> list[dict]:
        """Merge adjacent same-video results into single 'moment' entries.

        When three frames at t=11, t=13, t=14 all match, return one result
        spanning [11, 14] with the best score, instead of three separate
        entries that clutter the top-K. Inspired by Moment Search's
        `collapseSearchResults` + `assignProximityGroups` pattern.
        """
        if not candidates:
            return candidates

        from collections import defaultdict
        by_video: dict[str, list[dict]] = defaultdict(list)
        non_video: list[dict] = []
        for c in candidates:
            vid = c.get("video_id")
            if vid:
                by_video[vid].append(c)
            else:
                non_video.append(c)

        collapsed: list[dict] = list(non_video)

        for vid, frames in by_video.items():
            frames.sort(key=lambda f: f.get("start_time", 0))
            buckets: list[list[dict]] = [[frames[0]]]
            for f in frames[1:]:
                prev_end = buckets[-1][-1].get("end_time", buckets[-1][-1].get("start_time", 0))
                cur_start = f.get("start_time", 0)
                if cur_start - prev_end <= gap_sec:
                    buckets[-1].append(f)
                else:
                    buckets.append([f])

            for bucket in buckets:
                best = max(bucket, key=lambda f: f.get("score", 0))
                if len(bucket) > 1:
                    best = dict(best)
                    best["start_time"] = bucket[0].get("start_time", 0)
                    best["end_time"] = bucket[-1].get("end_time", bucket[-1].get("start_time", 0))
                    best["_related_count"] = len(bucket) - 1
                    # Preserve speech snippet from any merged entry
                    for b in bucket:
                        if b.get("_speech_snippet") and not best.get("_speech_snippet"):
                            best["_speech_snippet"] = b["_speech_snippet"]
                            break
                collapsed.append(best)

        collapsed.sort(key=lambda c: c.get("score", 0), reverse=True)
        return collapsed

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Stage 3: Temporal re-ranking
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _temporal_rerank(self, candidates: list[dict]) -> list[dict]:
        """Boost scores when multiple candidates cluster temporally.

        For each candidate, sum the Gaussian-weighted scores of its
        temporal neighbours within ``temporal_window`` seconds.  The
        resulting boost is blended with the original similarity score:

            final = (1 - w) * similarity + w * normalised_boost

        Candidates are then re-sorted by final score.
        """
        n = len(candidates)
        if n <= 1 or self.temporal_weight <= 0:
            return candidates

        # Group by video_id — temporal proximity only makes sense within
        # the same video.
        groups: dict[str, list[int]] = {}
        for i, c in enumerate(candidates):
            vid = c.get("video_id", "")
            groups.setdefault(vid, []).append(i)

        # Compute temporal boost for each candidate
        boosts = [0.0] * n
        sigma = self.temporal_window / 3.0

        for indices in groups.values():
            for i in indices:
                c_i = candidates[i]
                mid_i = (c_i["start_time"] + c_i["end_time"]) / 2.0

                for j in indices:
                    if i == j:
                        continue
                    c_j = candidates[j]
                    mid_j = (c_j["start_time"] + c_j["end_time"]) / 2.0

                    dt = abs(mid_i - mid_j)
                    if dt > self.temporal_window:
                        continue

                    # Gaussian decay weighted by the neighbour's similarity
                    weight = math.exp(-0.5 * (dt / sigma) ** 2)
                    boosts[i] += weight * c_j["score"]

        # Normalise boosts to [0, 1]
        max_boost = max(boosts) if boosts else 1.0
        if max_boost > 0:
            boosts = [b / max_boost for b in boosts]

        # Blend original score with temporal boost
        w = self.temporal_weight
        for i, c in enumerate(candidates):
            original = c["score"]
            c["_original_score"] = original
            c["_temporal_boost"] = round(boosts[i], 4)
            c["score"] = round((1 - w) * original + w * boosts[i], 4)

        # Re-sort descending by final score
        candidates.sort(key=lambda c: c["score"], reverse=True)

        logger.debug(
            "Temporal re-ranking applied (window=%.0fs, weight=%.2f)",
            self.temporal_window,
            self.temporal_weight,
        )
        return candidates

    def _hybrid_rerank(self, candidates: list[dict], query: str) -> list[dict]:
        """Refine ranking with lexical signal and robustness penalties.

        Helps separate semantically similar queries by boosting candidates
        whose text explicitly overlaps with query terms.
        """
        q_tokens = self._tokenise(query)
        if not q_tokens:
            return candidates

        for c in candidates:
            semantic = float(c.get("score", 0.0))
            text = str(c.get("text", ""))
            t_tokens = self._tokenise(text)

            overlap = len(q_tokens & t_tokens)
            lexical = overlap / max(1, len(q_tokens))

            source = str(c.get("source_type", ""))
            source_prior = {
                "fused": 1.00,
                "visual": 0.90,
                "audio": 0.82,
                "event": 0.88,
            }.get(source, 0.85)

            generic_penalty = 0.0
            lowered = text.lower()
            if "no scene description available" in lowered:
                generic_penalty += 0.10
            if "activity: static scene" in lowered:
                generic_penalty += 0.06

            blended = (
                semantic * (1.0 - _LEXICAL_WEIGHT - _SOURCE_PRIOR_WEIGHT)
                + lexical * _LEXICAL_WEIGHT
                + source_prior * _SOURCE_PRIOR_WEIGHT
                - generic_penalty
            )
            c["score"] = round(max(0.0, min(1.0, blended)), 4)
            c["_lexical_overlap"] = overlap

        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Stage 4: Format results
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _format(candidates: list[dict]) -> list[SearchResult]:
        """Convert raw dicts into typed SearchResult dataclasses."""
        results: list[SearchResult] = []

        for c in candidates:
            frame_path = c.get("frame_path")
            video_id = c.get("video_id", "")

            # Build a URL-safe frame reference if a path exists
            frame_url = None
            if frame_path and video_id:
                # Extract just the filename from the stored path
                import os
                fname = os.path.basename(frame_path)
                frame_url = f"/video/{video_id}/frame/{fname}"

            results.append(
                SearchResult(
                    video_id=video_id,
                    segment_id=c.get("id", ""),
                    start_time=c.get("start_time", 0.0),
                    end_time=c.get("end_time", 0.0),
                    text=c.get("text", ""),
                    source_type=c.get("source_type", ""),
                    score=c.get("score", 0.0),
                    event_type=c.get("event_type"),
                    frame_url=frame_url,
                    speech_snippet=c.get("_speech_snippet"),
                    related_count=c.get("_related_count", 0),
                )
            )

        return results

    @staticmethod
    def _is_dimension_mismatch(exc: Exception) -> bool:
        text = str(exc).lower()
        return "expecting embedding with dimension" in text and "got" in text

    @staticmethod
    def _parse_dims(message: str) -> tuple[int | None, int | None]:
        match = re.search(r"dimension of\s+(\d+)\s*,\s*got\s+(\d+)", message)
        if not match:
            return None, None
        return int(match.group(1)), int(match.group(2))

    @staticmethod
    def _tokenise(text: str) -> set[str]:
        tokens = {
            t for t in re.findall(r"[a-zA-Z0-9]+", text.lower())
            if len(t) >= 3 and t not in _STOP_WORDS
        }
        return tokens
