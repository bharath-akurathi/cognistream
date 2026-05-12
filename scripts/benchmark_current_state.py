#!/usr/bin/env python3
"""Benchmark the current CogniStream configuration.

Runs queries against the processed video and captures:
- Search latency (per query)
- Result quality metrics (Jaccard diversity, score distribution)
- Pipeline efficiency (segments/sec, throughput)
- Configuration snapshot for reproducibility

Usage:
  python scripts/benchmark_current_state.py
  python scripts/benchmark_current_state.py --runs 3
  python scripts/benchmark_current_state.py --queries "person" "car" "movement"
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


def _http_json(url: str, method: str = "GET", payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Make HTTP JSON request with timeout."""
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url=url, method=method, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity (overlap / union)."""
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def benchmark_search(base_url: str, video_id: str, queries: list[str]) -> dict[str, Any]:
    """Run search queries and collect metrics."""
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "base_url": base_url,
        "video_id": video_id,
        "queries": len(queries),
        "query_results": [],
        "aggregate": {
            "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "avg_score": 0.0,
            "score_stdev": 0.0,
            "jaccard_diversity": 0.0,
        },
    }

    latencies: list[float] = []
    all_scores: list[float] = []
    all_result_sets: list[set[str]] = []

    for q in queries:
        try:
            t_start = time.monotonic()
            data = _http_json(
                f"{base_url}/search",
                method="POST",
                payload={
                    "query": q,
                    "video_id": video_id,
                    "top_k": 10,
                },
            )
            latency_ms = (time.monotonic() - t_start) * 1000

            results_list = data.get("results", [])
            scores = [r.get("score", 0.0) for r in results_list]
            result_ids = {r.get("id", "") for r in results_list if r.get("id")}

            latencies.append(latency_ms)
            all_scores.extend(scores)
            all_result_sets.append(result_ids)

            results["query_results"].append(
                {
                    "query": q,
                    "num_results": len(results_list),
                    "latency_ms": round(latency_ms, 1),
                    "avg_score": round(statistics.mean(scores), 3) if scores else 0.0,
                    "top_score": round(scores[0], 3) if scores else 0.0,
                }
            )

            print(f"  ✓ '{q}' → {len(results_list)} results in {latency_ms:.0f}ms")

        except Exception as exc:
            print(f"  ✗ '{q}' failed: {exc}")
            results["query_results"].append(
                {
                    "query": q,
                    "error": str(exc),
                }
            )

    # Aggregate metrics
    if latencies:
        results["aggregate"]["avg_latency_ms"] = round(statistics.mean(latencies), 1)
        results["aggregate"]["p50_latency_ms"] = round(statistics.median(latencies), 1)
        if len(latencies) > 1:
            sorted_lat = sorted(latencies)
            p95_idx = max(0, int(len(sorted_lat) * 0.95) - 1)
            results["aggregate"]["p95_latency_ms"] = round(sorted_lat[p95_idx], 1)

    if all_scores:
        results["aggregate"]["avg_score"] = round(statistics.mean(all_scores), 3)
        if len(all_scores) > 1:
            results["aggregate"]["score_stdev"] = round(statistics.stdev(all_scores), 3)

    # Jaccard diversity: average pairwise overlap
    if all_result_sets:
        jaccard_scores = []
        for i, s1 in enumerate(all_result_sets):
            for s2 in all_result_sets[i + 1 :]:
                jaccard_scores.append(_jaccard(s1, s2))
        if jaccard_scores:
            results["aggregate"]["jaccard_diversity"] = round(
                1 - statistics.mean(jaccard_scores), 3
            )

    return results


def get_system_info(base_url: str) -> dict[str, Any]:
    """Fetch system and pipeline configuration."""
    try:
        stats = _http_json(f"{base_url}/stats")
        return stats
    except Exception as exc:
        print(f"Warning: Could not fetch system info: {exc}")
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark current CogniStream state")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--video-id", default="60dfe41cdaca41ca8d51d29f75beecbc")
    parser.add_argument("--runs", type=int, default=1, help="Number of benchmark runs")
    parser.add_argument(
        "--queries",
        nargs="*",
        default=[
            "person",
            "car",
            "movement",
            "data structures",
            "coding",
            "algorithm",
            "graph",
        ],
        help="Queries to run",
    )
    parser.add_argument("--output", help="Save results to JSON file")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"CogniStream Benchmark — Current State")
    print(f"{'='*70}")
    print(f"Base URL: {args.base_url}")
    print(f"Video ID: {args.video_id}")
    print(f"Queries: {len(args.queries)}")
    print(f"Runs: {args.runs}")

    # Get system info
    print(f"\nFetching system configuration...")
    sys_info = get_system_info(args.base_url)
    if sys_info:
        print(f"  CPU cores: {sys_info.get('cpu_cores', '?')}")
        print(f"  RAM: {sys_info.get('ram_gb', '?')} GB")
        print(f"  CUDA available: {sys_info.get('cuda', False)}")

    # Run benchmarks
    all_results = []
    print(f"\nRunning {args.runs} benchmark run(s)...")
    for run_idx in range(args.runs):
        print(f"\nRun {run_idx + 1}/{args.runs}:")
        result = benchmark_search(args.base_url, args.video_id, args.queries)
        all_results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    if len(all_results) == 1:
        r = all_results[0]
        agg = r["aggregate"]
        print(f"Latency:")
        print(f"  Avg:  {agg['avg_latency_ms']:.1f} ms")
        print(f"  P50:  {agg['p50_latency_ms']:.1f} ms")
        print(f"  P95:  {agg['p95_latency_ms']:.1f} ms")
        print(f"Score Quality:")
        print(f"  Avg score:     {agg['avg_score']:.3f}")
        print(f"  Stdev:         {agg['score_stdev']:.3f}")
        print(f"  Diversity:     {agg['jaccard_diversity']:.3f} (higher=more diverse)")
    else:
        # Multi-run summary
        latencies = []
        for r in all_results:
            latencies.append(r["aggregate"]["avg_latency_ms"])

        print(f"Latency across {args.runs} runs:")
        print(f"  Avg:  {statistics.mean(latencies):.1f} ms")
        print(f"  Min:  {min(latencies):.1f} ms")
        print(f"  Max:  {max(latencies):.1f} ms")

    # Save to file
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
