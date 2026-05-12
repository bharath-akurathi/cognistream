#!/usr/bin/env python3
"""Compare benchmarks across different configurations.

Helps you analyze how different settings impact performance.

Usage:
  python scripts/compare_benchmarks.py reports/baseline-siglip-disabled.json reports/with-siglip.json
  python scripts/compare_benchmarks.py reports/*.json  # Compare all in directory
"""

import json
import statistics
import sys
from pathlib import Path
from typing import Any


def load_benchmark(path: str) -> dict[str, Any]:
    """Load a benchmark JSON file."""
    with open(path) as f:
        data = json.load(f)
        return data[0] if isinstance(data, list) else data


def format_row(label: str, val1: Any, val2: Any = None, delta: bool = False) -> str:
    """Format a comparison row."""
    if val2 is None:
        return f"  {label:.<40} {str(val1):>15}"

    if delta:
        diff = float(val2) - float(val1)
        sign = "+" if diff >= 0 else ""
        pct = (diff / float(val1) * 100) if val1 != 0 else 0
        return f"  {label:.<40} {str(val1):>12} → {str(val2):>12} ({sign}{diff:>6.1f}, {pct:>+5.1f}%)"
    else:
        return f"  {label:.<40} {str(val1):>12}   {str(val2):>12}"


def print_single(name: str, bench: dict[str, Any]) -> None:
    """Print a single benchmark."""
    print(f"\n{'='*80}")
    print(f"  {Path(name).stem}")
    print(f"{'='*80}")

    agg = bench.get("aggregate", {})
    print("\n  LATENCY (milliseconds)")
    print(f"    Avg:          {agg.get('avg_latency_ms', 0):.1f} ms")
    print(f"    P50 (median): {agg.get('p50_latency_ms', 0):.1f} ms")
    print(f"    P95:          {agg.get('p95_latency_ms', 0):.1f} ms")

    print("\n  QUALITY")
    print(f"    Avg score:    {agg.get('avg_score', 0):.3f}")
    print(f"    Stdev:        {agg.get('score_stdev', 0):.3f}")
    print(f"    Diversity:    {agg.get('jaccard_diversity', 0):.3f}")

    print("\n  QUERIES")
    for qr in bench.get("query_results", []):
        error = qr.get("error")
        if error:
            print(f"    ✗ {qr['query']:.<30} {error}")
        else:
            lat = qr.get("latency_ms", 0)
            score = qr.get("avg_score", 0)
            print(f"    ✓ {qr['query']:.<30} {lat:>6.0f}ms, score={score:.3f}")


def print_comparison(name1: str, name2: str, bench1: dict[str, Any], bench2: dict[str, Any]) -> None:
    """Print side-by-side comparison."""
    print(f"\n{'='*80}")
    print(f"  {Path(name1).stem:.<35} vs {Path(name2).stem}")
    print(f"{'='*80}")

    agg1 = bench1.get("aggregate", {})
    agg2 = bench2.get("aggregate", {})

    print("\n  LATENCY")
    print(format_row("Avg", agg1.get("avg_latency_ms", 0), agg2.get("avg_latency_ms", 0), delta=True))
    print(format_row("P50", agg1.get("p50_latency_ms", 0), agg2.get("p50_latency_ms", 0), delta=True))
    print(format_row("P95", agg1.get("p95_latency_ms", 0), agg2.get("p95_latency_ms", 0), delta=True))

    print("\n  QUALITY")
    print(format_row("Avg score", agg1.get("avg_score", 0), agg2.get("avg_score", 0), delta=True))
    print(format_row("Diversity", agg1.get("jaccard_diversity", 0), agg2.get("jaccard_diversity", 0), delta=True))

    print("\n  VERDICT")
    avg_lat_1 = agg1.get("avg_latency_ms", float("inf"))
    avg_lat_2 = agg2.get("avg_latency_ms", float("inf"))
    div_1 = agg1.get("jaccard_diversity", 0)
    div_2 = agg2.get("jaccard_diversity", 0)

    if avg_lat_2 < avg_lat_1 * 0.9:
        print(f"    ✓ {Path(name2).stem:.<35} is faster ({avg_lat_2:.0f}ms vs {avg_lat_1:.0f}ms)")
    elif avg_lat_2 > avg_lat_1 * 1.1:
        print(f"    ✓ {Path(name1).stem:.<35} is faster ({avg_lat_1:.0f}ms vs {avg_lat_2:.0f}ms)")
    else:
        print(f"    ≈ Latency is similar (~{(avg_lat_1 + avg_lat_2) / 2:.0f}ms)")

    if div_2 > div_1 * 1.1:
        print(f"    ✓ {Path(name2).stem:.<35} has better diversity ({div_2:.3f} vs {div_1:.3f})")
    elif div_2 < div_1 * 0.9:
        print(f"    ✓ {Path(name1).stem:.<35} has better diversity ({div_1:.3f} vs {div_2:.3f})")
    else:
        print(f"    ≈ Diversity is similar (~{(div_1 + div_2) / 2:.3f})")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python compare_benchmarks.py <file1.json> [file2.json] ...")
        print("\nExamples:")
        print("  python compare_benchmarks.py reports/baseline.json reports/optimized.json")
        print("  python compare_benchmarks.py reports/*.json")
        return 1

    files = [Path(f) for f in sys.argv[1:]]
    files = [f for f in files if f.exists() and f.suffix == ".json"]

    if not files:
        print("No JSON files found.")
        return 1

    benchmarks = [(str(f), load_benchmark(str(f))) for f in files]

    # Single benchmark: print details
    if len(benchmarks) == 1:
        print_single(benchmarks[0][0], benchmarks[0][1])
        return 0

    # Multiple benchmarks: compare pairwise
    for i, (name1, bench1) in enumerate(benchmarks):
        # Print single details
        print_single(name1, bench1)

        # Compare with next
        if i < len(benchmarks) - 1:
            name2, bench2 = benchmarks[i + 1]
            print_comparison(name1, name2, bench1, bench2)

    return 0


if __name__ == "__main__":
    exit(main())
