# PromptFuse — In-Class Demo Guide

Run on your **RTX 6000 Ada** (Windows native, or WSL2 + CUDA for vLLM).

See [README.md](README.md) for install. This guide is the presentation flow.

## 0. One-time setup

```powershell
cd prompt_fuse
.\scripts\setup.ps1
# Accept Llama 3.1 + 3.2 licenses on huggingface.co, then set HF_TOKEN in .env
```

## 1. Start services

| Terminal | Command |
|----------|---------|
| 1 | `.\scripts\start_vllm.ps1` |
| 2 | `.\scripts\start_promptfuse.ps1` |

vLLM uses `--enable-prefix-caching` so shared prefixes reuse KV blocks.

## 2. Prove the unifier works (30 seconds, no vLLM)

```powershell
.\.venv\Scripts\python.exe scripts\run_unifier_eval.py
```

Look for **subsequent hit rate ≥ 70%** on `data/synthetic_paraphrases.json`. This isolates the **semantic unification** contribution without compression or serving noise.

## 3. Live demo (60 seconds)

```powershell
.\scripts\demo_live.ps1
```

1. Paraphrase A → first canonical registered  
2. Paraphrase B → **unifier hit** → same prefix sent to vLLM  
3. `/stats` shows `unifier_hit_rate` increasing  

## 4. Full A/B/C experiment

```powershell
.\.venv\Scripts\python.exe scripts\run_demo_experiment.py --config configs\demo.yaml
# or: .\.venv\Scripts\promptfuse-demo.exe
```

| Mode | What it shows |
|------|----------------|
| `raw_vllm` | Paraphrases stay different → weak prefix reuse |
| `compress_only` | Fewer tokens, paraphrases still differ |
| `promptfuse_full` | Same canonical prefix → better cache behavior |

**Pipeline-only** (vLLM not up):

```powershell
.\.venv\Scripts\python.exe scripts\run_demo_experiment.py --no-vllm
```

**Complex prompts** (RAG / agent style):

```powershell
.\.venv\Scripts\python.exe scripts\run_demo_experiment.py --no-vllm --workload data\complex_workload.json --output-dir results\complex
```

## 5. Compression + quality

```powershell
# Token reduction sweep (needs GPU for Llama-3.2-1B)
.\.venv\Scripts\python.exe scripts\run_compression_eval.py

# ROUGE-L on model outputs (needs vLLM)
.\.venv\Scripts\python.exe scripts\run_quality_eval.py --limit 8
```

## 6. Everything at once

```powershell
.\.venv\Scripts\python.exe scripts\run_full_eval.py --no-vllm
.\.venv\Scripts\python.exe scripts\run_full_eval.py --with-quality   # after vLLM is up
```

## 7. Slides — copy numbers from

```powershell
Get-Content results\demo_metrics.json | python -m json.tool
Get-Content results\unifier_eval.json | python -m json.tool
Get-Content results\compression_eval.json | python -m json.tool
```

Key fields:

- `unifier_eval.json` → `subsequent_hit_rate`, `cluster_full_hit_rate`
- `demo_metrics.json` → `key_findings.unifier_hit_rate`, `unification_reduces_prefix_diversity`
- `compression_eval.json` → `meets_token_goal`, `by_ratio`

## 8. Talking points

1. **Problem**: vLLM prefix cache needs exact token match; paraphrases miss.  
2. **Compress**: Sentence-level (proxy LM perplexity) → ~40% fewer tokens.  
3. **Unify**: Bi-encoder + FAISS → one canonical per semantic cluster.  
4. **Result**: `unique_final_prompts_per_cluster` drops; unifier hit rate rises; repeat requests faster.

## Optional: fine-tune bi-encoder

```powershell
.\.venv\Scripts\python.exe scripts\train_bi_encoder.py
```

Then set in `configs/demo.yaml`:

```yaml
unifier:
  fine_tuned_encoder: models/finetuned-minilm
  similarity_threshold: 0.85
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| vLLM OOM on 48GB | `--gpu-memory-utilization 0.75` in `start_vllm.ps1` |
| Unifier never hits | Run `warm_demo_inventory.py`; use `configs/demo.yaml` (τ=0.78) |
| Slow first request | Downloads/caches Llama weights; later requests faster |
| `demo_live.ps1` fails | Start both vLLM and PromptFuse first |
| Python not found | Install Python 3.10+; re-run `setup.ps1` |
