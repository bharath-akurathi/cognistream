# CogniStream Benchmark Baseline — April 8, 2026

## Current Configuration (SIGLIP_ENABLED=false)

### Environment
```bash
SIGLIP_ENABLED=false
WHISPER_MODEL_SIZE=large-v3-turbo
OLLAMA_MODEL=moondream
PIPELINE_MODE=auto
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIM=384
```

### Hardware
- **Platform**: macOS (Darwin)
- **CPU cores**: 8
- **RAM**: 8.0 GB
- **CUDA**: Not available
- **Device runtime**: CPU only

### Known Metrics (from last successful run)

#### Ingestion Pipeline
- **Video**: 3 minutes, 10789 frames
- **Shot detection**: 3 segments, ~13 seconds
- **Frame sampling**: 21 keyframes, ~1 second
- **Audio extraction**: FFmpeg, ~1 second
- **Whisper transcription**: 30 audio segments, ~72 seconds (CPU on macOS)
- **SentenceTransformer embedding**: 30 segments, ~12 seconds
- **Knowledge graph**: 89 nodes, 90 edges, <1 second
- **Total time**: ~104 seconds end-to-end

#### Database
- **ChromaDB collection**: cognistream_segments
- **Embedding dimension**: 384 (text only, no SigLIP visual)
- **Segments stored**: 30
- **Storage method**: Persistent local (data/db/chroma/)

---

## Quick Run: Benchmark Checklist

### ✅ Step 1: Start Backend
```bash
cd /Users/akurathi/Desktop/Codes/projects/cognistream
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### ✅ Step 2: Run Baseline Benchmark
```bash
python scripts/benchmark_current_state.py \
  --output reports/baseline-siglip-disabled.json
```

### ✅ Step 3: View Results
```bash
python scripts/compare_benchmarks.py reports/baseline-siglip-disabled.json
```

---

## Planned Benchmarks

### B1: Baseline (Current, SigLIP disabled)
- **Purpose**: Reference point for all comparisons
- **Expected latency**: ~100-200ms/query (CPU embedding)
- **Expected diversity**: 0.5-0.8

### B2: Alternative Whisper Model (Optional)
Change WHISPER_MODEL_SIZE from `large-v3-turbo` to `small`
- **Purpose**: Measure ingestion speed vs. accuracy tradeoff
- **Expected saving**: ~50% faster transcription
- **Drawback**: Lower transcription quality

### B3: With SigLIP Enabled (Future)
Once fixed (dimension alignment):
- **Purpose**: Measure if visual search helps
- **Expected impact**: Better diversity, slower search

### B4: Load Testing (Throughput)
Multiple concurrent queries:
- **Purpose**: Find breaking point
- **Tool**: Apache Bench or custom script

---

## Interpreting Your Benchmarks

### Good Signs ✅
- Latency **< 200ms** per query
- Diversity **> 0.7** (different results per query)
- Consistent latency across runs (stdev < 50%)

### Areas for Optimization 🔍
- Latency **> 500ms**: Check if embedding model is CPU-bound
- Diversity **< 0.5**: May indicate poor embedding or limited results
- High variance: System instability or background processes

---

## Notes

- **Current blocker**: SigLIP dimension mismatch fixed ✅
- **Search working**: Yes, text-only retrieval succeeds
- **Ready for**: Latency/quality benchmarking, configuration testing
- **Next optimization**: Measure impact of Whisper model size

---

Last updated: April 8, 2026 02:29 UTC
