# PromptFuse — In-Class Demo Guide

Run everything on your **RTX 6000 Ada** machine (Linux/WSL2 + CUDA).

## 0. One-time setup

```bash
cd prompt_fuse
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# CUDA PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

export HF_TOKEN="hf_..."
huggingface-cli login
```

Request access to `meta-llama/Llama-3.2-1B` and `meta-llama/Llama-3.1-8B-Instruct` on Hugging Face.

## 1. Start services (two terminals)

**Terminal 1 — vLLM**

```bash
source .venv/bin/activate
export HF_TOKEN=...

python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --enable-prefix-caching \
  --port 8000 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096
```

**Terminal 2 — PromptFuse middleware**

```bash
source .venv/bin/activate
export PROMPTFUSE_CONFIG=configs/demo.yaml   # optional; server uses Settings()
promptfuse-serve
# Or: python -m promptfuse.middleware.server
```

Uses `configs/demo.yaml` (τ=0.80, demo inventory path).

## 2. Warm demo inventory

```bash
source .venv/bin/activate
python scripts/warm_demo_inventory.py --config configs/demo.yaml
```

## 3. Live demo (60 seconds)

```bash
chmod +x scripts/demo_live.sh
./scripts/demo_live.sh
```

Shows two paraphrases → unified canonical → stats with unifier hit rate.

## 4. Full A/B/C experiment

Compares three modes on `data/demo_workload.json`:

| Mode | What it shows |
|------|----------------|
| `raw_vllm` | Paraphrases stay different → weak prefix reuse |
| `compress_only` | Fewer tokens, paraphrases still differ |
| `promptfuse_full` | Same canonical prefix → better cache behavior |

```bash
python scripts/run_demo_experiment.py --config configs/demo.yaml
# Or: promptfuse-demo
```

Results: `results/demo_metrics.json` + printed summary table.

**Pipeline-only** (vLLM not up yet):

```bash
python scripts/run_demo_experiment.py --no-vllm --config configs/demo.yaml
```

## 5. Output quality (ROUGE-L)

```bash
python scripts/run_quality_eval.py --config configs/demo.yaml --limit 10
```

Results: `results/quality_eval.json`

## 6. Optional: fine-tune bi-encoder

```bash
python scripts/train_bi_encoder.py --data data/synthetic_paraphrases.json
```

Then in `configs/demo.yaml`:

```yaml
unifier:
  fine_tuned_encoder: models/finetuned-minilm
  similarity_threshold: 0.85
```

## 7. Presentation talking points

1. **Problem**: vLLM prefix cache needs exact token match; paraphrases miss.
2. **Compress**: Sentence-level drop (Llama-3.2-1B perplexity) → ~40% fewer tokens.
3. **Unify**: Bi-encoder + FAISS → one canonical form per semantic cluster.
4. **Result**: `unique_final_prompts_per_cluster` drops (raw → full); unifier hit rate rises; repeat requests faster.

## 8. Slides — copy numbers from

```bash
cat results/demo_metrics.json | python -m json.tool
cat results/quality_eval.json | python -m json.tool
```

Key fields: `key_findings.unifier_hit_rate`, `token_reduction_full`, `unification_reduces_prefix_diversity`.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| vLLM OOM | Lower `--gpu-memory-utilization 0.75` or `--max-model-len 2048` |
| Unifier never hits | Use `configs/demo.yaml` (τ=0.80); run `warm_demo_inventory.py` |
| Slow compressor | First run downloads Llama-3.2-1B; later runs cache weights |
| `demo_live.sh` fails | Start both vLLM and `promptfuse-serve` first |
