# CogniStream Benchmarking Guide

## Quick Start

1. **Start the server** (in one terminal):
```bash
cd /Users/akurathi/Desktop/Codes/projects/cognistream
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

2. **Run baseline benchmark** (in another terminal):
```bash
python scripts/benchmark_current_state.py --output reports/baseline-siglip-disabled.json
```

3. **View results**:
```bash
cat reports/baseline-siglip-disabled.json | python -m json.tool
```

---

## What Gets Measured

### Latency Metrics
- **Avg latency**: Mean time to process a query (embedding + search)
- **P50 (median)**: Half of queries finish at or before this time
- **P95**: 95% of queries finish before this time

### Quality Metrics
- **Avg score**: Mean similarity score of retrieved results (0-1)
- **Stdev**: Score consistency (lower = more uniform results)
- **Diversity (Jaccard)**: How different are results between queries? (higher = better)

### Pipeline Metrics  
- **Segments processed**: Should match your video (currently 30)
- **Embedding dimension**: Should be 384-dim (text only, SigLIP disabled)

---

## Current Configuration

### Active Settings (`.env`)
```
SIGLIP_ENABLED=false          # Disabled to avoid dimension mismatch
WHISPER_MODEL_SIZE=large-v3-turbo  # High quality transcription (slower)
OLLAMA_MODEL=moondream        # Local VLM (slower than Llama, but works)
```

### Hardware Profile
- **CPU cores**: 8
- **RAM**: 8.0 GB
- **CUDA**: Not available (Mac without GPU support)
- **Device**: Runs on CPU (Whisper, embeddings, all inference)

### Expected Bottlenecks
1. **Whisper transcription**: ~70s for 3 min video (CPU-bound)
2. **SentenceTransformer embedding**: ~12s for 30 segments (CPU-bound)
3. **Search latency**: ~100ms per query (network + embedding)

---

## Benchmark Scenarios

### Scenario A: Baseline (Current)
- SigLIP: Disabled
- Whisper: `large-v3-turbo` (CPU)
- VLM: moondream
- Device: Mac (no CUDA)

**Run**:
```bash
python scripts/benchmark_current_state.py \
  --output reports/baseline-siglip-disabled.json
```

### Scenario B: Faster Whisper (Optional)
Switch to smaller, faster Whisper model:
```bash
# In .env or export:
export WHISPER_MODEL_SIZE=small
```

Then reprocess a video and benchmark.

### Scenario C: Query Performance (No Processing)
If you already have processed videos in the database:
```bash
# Just benchmark search (no ingestion)
python scripts/benchmark_current_state.py \
  --video-id <VIDEO_ID> \
  --queries "your" "custom" "queries"
```

---

## Interpreting Results

### Good Search Results
- **Latency**: < 200ms per query is good for production
- **Diversity**: > 0.7 means queries return different results (good!)
- **Avg score**: 0.4-0.8 is typical for semantic search

### Performance Red Flags
- **Latency**: > 1s suggests chromadb query or embedding is slow
- **All scores ~0.5**: Might indicate poor embedding quality
- **Diversity ~0**: All queries return same results (embedding collapse)

---

## Tracking Across Configurations

After each benchmark run, save the JSON:
```bash
# Baseline
python scripts/benchmark_current_state.py --output reports/baseline-siglip-disabled.json

# After changing config (e.g., enabling SigLIP again)
export SIGLIP_ENABLED=true
rm -rf data/db/chroma  # Clear old embeddings
# ... reprocess video ...
python scripts/benchmark_current_state.py --output reports/with-siglip-enabled.json
```

Then compare:
```bash
python -c "
import json
b1 = json.load(open('reports/baseline-siglip-disabled.json'))
b2 = json.load(open('reports/with-siglip-enabled.json'))
print('Latency delta:', b2[0]['aggregate']['avg_latency_ms'] - b1[0]['aggregate']['avg_latency_ms'], 'ms')
print('Diversity delta:', b2[0]['aggregate']['jaccard_diversity'] - b1[0]['aggregate']['jaccard_diversity'])
"
```

---

## Multi-Run Benchmarks

For stability metrics, run multiple times:
```bash
python scripts/benchmark_current_state.py \
  --runs 5 \
  --output reports/5-run-baseline.json
```

This captures:
- Min/max/avg latency across runs
- Score variance
- System stability (are timings consistent?)

---

## Advanced: Phase Benchmarks

For more complex scenarios, use the existing phase scripts:

```bash
# Video reprocessing (end-to-end pipeline timing)
python scripts/phase0_baseline.py --video-id <ID> --runs 3

# Worker saturation study
python scripts/phase1_worker_saturation.py

# Semantic reuse optimization
python scripts/phase2_reuse_benchmark.py
```

---

## Next Steps

1. ✅ **Run baseline**: `python scripts/benchmark_current_state.py`
2. 📊 **Analyze**: Check latency, diversity, and score distribution
3. 🔧 **Optimize**: Based on bottlenecks, consider:
   - Smaller Whisper model (faster ingestion)
   - Batch query testing (throughput under load)
   - Different embedding models
4. 📈 **Track**: Save each configuration's results for comparison

