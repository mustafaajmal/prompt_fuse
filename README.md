# PromptFuse

**Semantic Prompt Compression and Unification for KV Cache Efficiency**

CSE 232B | Spring 2026 | Aaron Sharif & Mustafa Ajmal

PromptFuse is a middleware layer that sits before [vLLM](https://github.com/vllm-project/vllm) to reduce inference cost through two complementary optimizations:

1. **Segment-level compression** — drops low-importance sentences using perplexity scores from a small proxy LM (Llama-3.2-1B), inspired by LLMLingua but at sentence granularity for grammatical validity.
2. **Semantic unification** — maps paraphrased prompts to shared canonical forms via a fine-tuned bi-encoder and FAISS retrieval, enabling KV cache reuse beyond exact token-level prefix matching.

```
raw_prompt → compressor → unifier → vLLM (prefix cache) → response
```

## Goals

| Metric | Target |
|--------|--------|
| Token reduction | ≥ 30% |
| KV cache hit rate | ≥ 2× over baseline |
| Output quality (ROUGE-L) | ≥ 0.85 |
| Pipeline latency overhead | < 50ms p99 |
| End-to-end speedup | ≥ 1.5× |

## Quick Start

For the complete local CPU workflow (artifacts, diagnostics, and handoff expectations), see `CPU_README.md`.

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[vllm]"       # vLLM serving backend
pip install -e ".[llmlingua]"   # LLMLingua baseline comparison
```

### Generate synthetic paraphrase data

```bash
python scripts/generate_synthetic_paraphrases.py --target-size 500 --clusters 50
```

### Build canonical inventory (warmup)

```bash
python scripts/build_canonical_inventory.py --prompts data/synthetic_paraphrases.json
```

### Fine-tune bi-encoder (optional)

```bash
python scripts/train_bi_encoder.py --data data/synthetic_paraphrases.json
```

Update `configs/default.yaml` with `unifier.fine_tuned_encoder: models/finetuned-minilm`.

### Run benchmark

```bash
# Multi-sentence prompts (compression); use synthetic JSON for unifier-heavy eval
promptfuse-benchmark --prompts data/long_prompts.txt --compare-baselines
promptfuse-benchmark --prompts data/synthetic_paraphrases.json --compare-baselines
```

See [ROADMAP.md](ROADMAP.md) for project status and remaining GPU work.

### Run full CPU pipeline (artifacts + summaries)

```bash
python scripts/run_cpu_pipeline.py
```

### Run CPU readiness checks

```bash
python scripts/run_cpu_readiness_checks.py
```

### Serve middleware

Start vLLM first:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --enable-prefix-caching
```

Then start PromptFuse:

```bash
promptfuse-serve
```

Send requests to `http://localhost:8080/v1/chat/completions` (same OpenAI-compatible API as vLLM).

## Project Structure

```
promptfuse/
├── compressor/          # Segment-level token compressor (proxy LM perplexity)
├── unifier/             # Semantic unifier + FAISS canonical store
├── middleware/          # FastAPI server (vLLM proxy)
├── evaluation/          # ROUGE-L, latency, cache hit metrics
├── pipeline.py          # End-to-end orchestration
└── config.py            # YAML-based configuration

scripts/
├── generate_synthetic_paraphrases.py
├── train_bi_encoder.py
├── build_canonical_inventory.py
├── prepare_datasets.py          # ShareGPT / LMSYS-Chat-1M sampling
├── tau_sweep.py                 # CPU threshold sweep for semantic unifier
├── compressor_diagnostics.py    # Per-segment keep/drop diagnostics
├── aggregate_results.py         # Merge metric artifacts into summary JSON/CSV
└── run_cpu_pipeline.py          # One-command CPU artifact pipeline

configs/default.yaml
data/sample_prompts.txt
tests/
```

## Configuration

Edit `configs/default.yaml`:

- `compressor.compression_ratio` — target token reduction (0.25, 0.40, 0.55)
- `compressor.preserve_patterns` — regex guardrails for instruction segments that should not be dropped
- `unifier.similarity_threshold` — cosine similarity τ for canonical matching
- `serving.vllm_base_url` — vLLM backend URL
- `serving.vllm_metrics_url` — optional metrics endpoint to proxy via `/v1/metrics/vllm-cache`
- `serving.vllm_timeout_s` — upstream timeout for vLLM requests

## Evaluation Baselines

The benchmark includes five modes in a normalized schema:

- **no_compression** — raw prompts, no prefix optimization
- **compression_only** — segment compressor without unification (LLMLingua-style)
- **promptfuse_full** — compression + semantic unification
- **llmlingua_only** — adapter placeholder with graceful unavailable status in CPU-only runs
- **exact_cache_only_stub** — vLLM cache-only placeholder for CPU workflows

## CPU Artifact Outputs

Running `python scripts/run_cpu_pipeline.py` writes artifacts under `results/cpu_final/`:

- `metrics/benchmark_ratio_*.json` — per-ratio normalized baseline artifacts
- `metrics/tau_sweep.json` — threshold sweep diagnostics
- `summary.json` + `summary.csv` — merged report-friendly outputs
- `logs/*.log` — command logs for each pipeline step
- `notes/pipeline_report.json` — run status and failure diagnostics
- `configs/final_cpu.yaml` — config snapshot used for CPU run

## Models

| Role | Model |
|------|-------|
| Target LLM | Llama-3.1-8B-Instruct |
| Proxy LM (compressor) | Llama-3.2-1B |
| Bi-encoder (unifier) | all-MiniLM-L6-v2 (fine-tuned on paraphrase pairs) |

## Datasets

- **ShareGPT** — real user–LLM conversations
- **LMSYS-Chat-1M** — production chat sessions
- **Synthetic paraphrases** — 500 paraphrases across 50 instruction categories

```bash
python scripts/prepare_datasets.py --max-samples 500
```

## Development

```bash
pytest tests/ -v
ruff check promptfuse/
```

## License

MIT
