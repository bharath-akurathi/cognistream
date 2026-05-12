#!/usr/bin/env python3
"""Local benchmark utility for CogniStream search quality and pipeline timings.

Usage examples:
  python scripts/benchmark_local.py --video-id <VIDEO_ID>
  python scripts/benchmark_local.py --video-id <VIDEO_ID> --query "person entering room"
  python scripts/benchmark_local.py --video-id <VIDEO_ID> --query-file queries.txt
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any


def _http_json(url: str, method: str = "GET", payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url=url, method=method, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _load_queries(args: argparse.Namespace) -> list[str]:
    if args.query:
        return args.query

    if args.query_file:
        with open(args.query_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    return [
        "person entering room",
        "vehicle movement",
        "crowd activity",
        "suspicious behavior",
        "object left behind",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark CogniStream retrieval diversity and timings")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Backend base URL")
    parser.add_argument("--video-id", required=True, help="Video ID to benchmark")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k for each query")
    parser.add_argument("--query", action="append", help="Query text (can be repeated)")
    parser.add_argument("--query-file", help="Path to newline-delimited queries file")
    parser.add_argument("--report-json", help="Optional output path for benchmark report JSON")
    parser.add_argument("--with-trend", action="store_true", help="Also fetch /benchmark/trend summary")
    args = parser.parse_args()

    queries = _load_queries(args)
    if not queries:
        print("No queries provided.")
        return 2

    all_sets: list[set[str]] = []
    all_scores: list[float] = []
    query_results: list[dict[str, Any]] = []

    print(f"Benchmarking {len(queries)} queries against video {args.video_id} (top_k={args.top_k})")

    for q in queries:
        try:
            data = _http_json(
                f"{args.base_url}/search",
                method="POST",
                payload={
                    "query": q,
                    "video_id": args.video_id,
                    "top_k": args.top_k,
                },
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            print(f"Search failed for query '{q}': HTTP {exc.code} - {body}")
            continue
        except Exception as exc:
            print(f"Search failed for query '{q}': {exc}")
            continue

        results = data.get("results", [])
        ids = {
            (row.get("segment_id") or f"{row.get('video_id')}:{row.get('start_time')}:{row.get('end_time')}")
            for row in results
        }
        ids.discard(None)
        all_sets.append(ids)

        query_scores = [float(row.get("score", 0.0)) for row in results]
        all_scores.extend(query_scores)
        query_results.append(
            {
                "query": q,
                "hit_count": len(results),
                "unique_ids": len(ids),
                "scores": query_scores,
            }
        )

        print(f"- '{q}': {len(results)} hits, unique ids={len(ids)}")

    if not all_sets:
        print("No successful searches were executed.")
        return 1

    pairwise = [
        _jaccard(a, b)
        for a, b in itertools.combinations(all_sets, 2)
    ]
    avg_overlap = statistics.fmean(pairwise) if pairwise else 0.0
    diversity_index = 1.0 - avg_overlap

    print("\nRetrieval summary")
    print(f"- avg_pairwise_overlap_jaccard: {avg_overlap:.4f}")
    print(f"- diversity_index: {diversity_index:.4f}")
    if all_scores:
        print(f"- avg_result_score: {statistics.fmean(all_scores):.4f}")
        print(f"- max_result_score: {max(all_scores):.4f}")

    benchmark_payload: dict[str, Any] | None = None
    trend_payload: dict[str, Any] | None = None

    try:
        bench = _http_json(f"{args.base_url}/video/{args.video_id}/benchmark")
        benchmark_payload = bench
        print("\nPipeline benchmark")
        print(f"- success: {bench.get('success')}")
        print(f"- elapsed_sec: {bench.get('elapsed_sec')}")

        stage_timings = bench.get("stage_timings") or {}
        if stage_timings:
            print("- stage_timings_sec:")
            for key in sorted(stage_timings):
                print(f"  {key}: {stage_timings[key]}")

        quality = bench.get("quality_metrics") or {}
        if quality:
            print("- quality_metrics:")
            for key in sorted(quality):
                print(f"  {key}: {quality[key]}")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print("\nPipeline benchmark")
            print("- No benchmark payload yet. Run processing first.")
        else:
            body = exc.read().decode("utf-8", errors="replace")
            print(f"\nPipeline benchmark fetch failed: HTTP {exc.code} - {body}")
    except Exception as exc:
        print(f"\nPipeline benchmark fetch failed: {exc}")

    if args.with_trend:
        try:
            trend = _http_json(f"{args.base_url}/video/{args.video_id}/benchmark/trend")
            trend_payload = trend
            print("\nBenchmark trend")
            elapsed = trend.get("elapsed_sec") or {}
            print(f"- latest_elapsed_sec: {elapsed.get('latest')}")
            print(f"- oldest_elapsed_sec: {elapsed.get('oldest')}")
            print(f"- delta_latest_minus_oldest: {elapsed.get('delta_latest_minus_oldest')}")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                print("\nBenchmark trend")
                print("- Not enough benchmark runs yet (need at least 2).")
            else:
                body = exc.read().decode("utf-8", errors="replace")
                print(f"\nBenchmark trend fetch failed: HTTP {exc.code} - {body}")
        except Exception as exc:
            print(f"\nBenchmark trend fetch failed: {exc}")

    if args.report_json:
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "base_url": args.base_url,
            "video_id": args.video_id,
            "top_k": args.top_k,
            "query_count": len(queries),
            "queries": query_results,
            "retrieval_summary": {
                "avg_pairwise_overlap_jaccard": round(avg_overlap, 4),
                "diversity_index": round(diversity_index, 4),
                "avg_result_score": round(statistics.fmean(all_scores), 4) if all_scores else None,
                "max_result_score": round(max(all_scores), 4) if all_scores else None,
            },
            "pipeline_benchmark": benchmark_payload,
            "benchmark_trend": trend_payload,
        }
        out_path = os.path.abspath(args.report_json)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
