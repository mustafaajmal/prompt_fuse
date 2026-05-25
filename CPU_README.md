# PromptFuse CPU Workflow

This document explains the current CPU-first workflow for the project, including what is implemented today, how to run it, and what remains GPU-dependent.

## What this workflow is for

Use this when you want to:

- run PromptFuse locally without vLLM/GPU access,
- validate compression + unifier behavior,
- generate reproducible CPU artifacts under `results/cpu_final/`,
- prepare handoff outputs for later GPU experiments.

This workflow is intentionally a proxy/evaluation scaffold and does not replace full GPU serving validation.

## Current CPU architecture

Pipeline:

`raw_prompt -> segment compressor -> semantic unifier -> benchmark/log artifacts`

Key components:

- `promptfuse/compressor/segment_compressor.py`
  - Segment-level compression using proxy LM perplexity.
  - Current default proxy model in config: `openai-community/gpt2`.
- `promptfuse/unifier/semantic_unifier.py`
  - Embedding-based semantic canonicalization using FAISS store.
- `promptfuse/evaluation/benchmark.py`
  - Baseline orchestration and normalized benchmark schema.
- `scripts/tau_sweep.py`
  - Unifier threshold diagnostics.
- `scripts/aggregate_results.py`
  - Merges artifacts into summary JSON/CSV.
- `scripts/run_cpu_pipeline.py`
  - One-command CPU run orchestration.

## Environment setup

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Core CPU commands

### 1) Full CPU pipeline (recommended)

```bash
python scripts/run_cpu_pipeline.py --output-dir results/cpu_final --prompts data/sample_prompts.txt
```

This runs:

1. readiness checks
2. synthetic paraphrase generation (500/50)
3. benchmark ratio sweeps (`0.25`, `0.40`, `0.55`)
4. tau sweep (`0.75`, `0.80`, `0.85`, `0.90`)
5. result aggregation

### 2) Regenerate synthetic data only

```bash
python scripts/generate_synthetic_paraphrases.py --target-size 500 --clusters 50 --output data/synthetic_paraphrases.json
```

### 3) Run tau sweep only

```bash
python scripts/tau_sweep.py --prompts data/synthetic_paraphrases.json --output results/cpu_final/metrics/tau_sweep.json
```

### 4) Export compressor diagnostics

```bash
python scripts/compressor_diagnostics.py --prompts data/sample_prompts.txt --output results/cpu_final/metrics/compressor_diagnostics.json
```

## Output layout

After a successful CPU pipeline run:

- `results/cpu_final/metrics/benchmark_ratio_0_25.json`
- `results/cpu_final/metrics/benchmark_ratio_0_4.json`
- `results/cpu_final/metrics/benchmark_ratio_0_55.json`
- `results/cpu_final/metrics/tau_sweep.json`
- `results/cpu_final/summary.json`
- `results/cpu_final/summary.csv`
- `results/cpu_final/logs/*.log`
- `results/cpu_final/notes/pipeline_report.json`
- `results/cpu_final/notes/status_report.md`
- `results/cpu_final/configs/final_cpu.yaml`

## Interpreting CPU results

- Benchmark ROUGE values are proxy-style (compression transformation quality), not full downstream LLM output fidelity.
- `llmlingua_only` appears as `unavailable` in CPU-only mode unless explicitly wired/runnable.
- `exact_cache_only_stub` is a placeholder for live vLLM cache baseline (GPU phase).
- Tau sweep provides useful unifier behavior signals (hit proxy, canonical count, cluster purity).

## CPU workflow acceptance checks

A healthy run typically satisfies:

- `required_failures: []` in `results/cpu_final/notes/pipeline_report.json`
- all pipeline steps show `"ok": true`
- synthetic dataset contains `500` rows and `50` unique `cluster_id`s
- `summary.json` and `summary.csv` are present and populated

## What remains GPU-dependent

These are intentionally out of scope for this CPU workflow:

- live vLLM serving with prefix cache metrics,
- real KV cache hit/miss comparison vs baseline,
- throughput and TTFT speedup measurements,
- ROUGE-L on actual model outputs from target LLM,
- final exact-cache-only and LLMLingua runtime comparisons on GPU hardware.

## Recommended collaboration split

- CPU owner: maintain this pipeline, artifacts, reproducibility, and diagnostics.
- GPU owner: run vLLM baselines, cache metrics, output-quality evaluation, and throughput benchmarks.
