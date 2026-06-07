# PromptFuse

**Semantic Prompt Compression and Unification for KV Cache Efficiency**

CSE 232B | Spring 2026 | Aaron Sharif & Mustafa Ajmal

PromptFuse is a middleware layer that sits before [vLLM](https://github.com/vllm-project/vllm) to reduce inference cost through two complementary optimizations:

1. **Segment-level compression** — drops low-importance sentences using perplexity scores from a small proxy LM (Llama-3.2-1B), with token-level fallback for short prompts.
2. **Semantic unification** — maps paraphrased prompts to shared canonical forms via a bi-encoder and FAISS retrieval, enabling KV cache reuse beyond exact token-level prefix matching.

```
raw_prompt → compressor → unifier → vLLM (prefix cache) → response
```

## Goals (class demo)

| Metric | Target |
|--------|--------|
| Token reduction | ≥ 30% |
| Unifier hit rate (paraphrases) | high on synthetic clusters |
| Output quality (ROUGE-L) | ≥ 0.85 vs raw |
| Pipeline latency overhead | < 50ms p99 (unifier-only much lower) |
| Prefix diversity reduction | full < raw in A/B experiment |

## Quick Start (WSL2 — recommended)

**Use WSL2**, not Windows PowerShell. See **[WSL.md](WSL.md)** for the full flow.

```bash
cd "/mnt/c/Users/Escalona Cribstafa 5/Documents/dev/prompt_fuse"
./scripts/wsl_setup.sh
# Add HF_TOKEN to .env, then:
./scripts/wsl_prefetch_models.sh
./scripts/start_vllm.sh          # terminal 1
./scripts/start_promptfuse.sh    # terminal 2
```

## Quick Start (Windows native — optional)

```powershell
cd prompt_fuse
.\scripts\setup.ps1
```

This creates `.venv`, installs CUDA PyTorch + dependencies, generates paraphrase data, and warms `data/demo_canonical_inventory`.

**Hugging Face:** Accept licenses for [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) and [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), then:

```powershell
.\.venv\Scripts\huggingface-cli.exe login
```

### Start services (two terminals)

**Terminal 1 — vLLM**

```powershell
.\scripts\start_vllm.ps1
```

**Terminal 2 — PromptFuse**

```powershell
.\scripts\start_promptfuse.ps1
```

### Verify the novel contribution (no vLLM needed)

```powershell
$env:PROMPTFUSE_CONFIG = "configs\demo.yaml"
.\.venv\Scripts\python.exe scripts\run_unifier_eval.py
```

Expect **≥70% subsequent hit rate** on paraphrase clusters (semantic unifier working).

### Full evaluation suite

```powershell
# Pipeline + unifier metrics (vLLM optional)
.\.venv\Scripts\python.exe scripts\run_full_eval.py --no-vllm

# With vLLM for latency / ROUGE
.\.venv\Scripts\python.exe scripts\run_full_eval.py --with-quality
```

Results land in `results/` (`unifier_eval.json`, `compression_eval.json`, `demo_metrics.json`).

### Live in-class demo

```powershell
.\scripts\demo_live.ps1
```

Shows paraphrase A vs B → unified canonical → `/stats` unifier hit rate.

## CPU-only workflow (no GPU)

For local validation without vLLM, see **[CPU_README.md](CPU_README.md)** and **[ROADMAP.md](ROADMAP.md)**.

```bash
python -m venv .venv
source .venv/bin/activate   # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -e ".[dev]"

# One-command CPU artifact pipeline
python scripts/run_cpu_pipeline.py

# Readiness checks
python scripts/run_cpu_readiness_checks.py
```

CPU artifacts land in `results/cpu_final/` (benchmark JSON, τ sweep, summary CSV).

## Linux / WSL2 (demo scripts)

```bash
chmod +x scripts/setup.sh scripts/demo_live.sh
./scripts/setup.sh
./scripts/demo_live.sh
```

## Project Structure

```
promptfuse/
├── compressor/          # Segment + token-level proxy LM compression
├── unifier/             # Bi-encoder + FAISS canonical store
├── middleware/          # FastAPI vLLM proxy
├── evaluation/          # Benchmarks, demo experiment, unifier/compression eval
└── pipeline.py

data/
├── demo_workload.json       # 5 clusters for A/B/C demo
├── complex_workload.json    # 8 RAG/agent-style clusters (stress test)
└── synthetic_paraphrases.json  # paraphrase clusters for unifier eval

scripts/
├── setup.ps1 / setup.sh / wsl_setup.sh
├── start_vllm.ps1 / start_promptfuse.ps1
├── run_full_eval.py / run_unifier_eval.py / run_compression_eval.py
├── run_cpu_pipeline.py / aggregate_results.py / tau_sweep.py
├── warm_demo_inventory.py / demo_live.ps1
└── train_bi_encoder.py
```

## Configuration

`configs/demo.yaml` (recommended for presentation):

- `compressor.compression_ratio: 0.40`
- `compressor.preserve_patterns` — regex guardrails for instruction segments
- `unifier.similarity_threshold: 0.78`
- `unifier.inventory_path: data/demo_canonical_inventory`
- `serving.vllm_base_url` — vLLM backend URL
- `serving.vllm_metrics_url` — optional metrics proxy via `/v1/metrics/vllm-cache`

Override via environment: `PROMPTFUSE_CONFIG=configs/demo.yaml`

## Evaluation modes

| Command | What it proves |
|---------|----------------|
| `run_unifier_eval.py` | Paraphrases map to same canonical (core novelty) |
| `run_compression_eval.py` | ≥30% token reduction at 25/40/55% ratios |
| `promptfuse-demo --no-vllm` | Prefix diversity: full < raw |
| `promptfuse-demo` | End-to-end with vLLM prefix cache |
| `run_quality_eval.py` | ROUGE-L ≥ 0.85 on model outputs |
| `run_cpu_pipeline.py` | CPU artifacts + τ sweep (no GPU) |

## API

- `POST /v1/chat/completions` — OpenAI-compatible (forwards to vLLM after fuse)
- `POST /v1/process` — compress + unify only (debug)
- `GET /stats` — unifier hit rate, inventory size
- `GET /v1/metrics/vllm-cache` — proxy vLLM cache metrics (if configured)

## Models

| Role | Model |
|------|-------|
| Target LLM | Llama-3.1-8B-Instruct |
| Proxy LM (compressor) | Llama-3.2-1B |
| Bi-encoder (unifier) | all-MiniLM-L6-v2 (optional fine-tune) |

## Development

```bash
pytest tests/ -v
ruff check promptfuse/
```

## License

MIT
