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
python scripts/generate_synthetic_paraphrases.py
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
promptfuse-benchmark --prompts data/sample_prompts.txt --compare-baselines
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
└── prepare_datasets.py  # ShareGPT / LMSYS-Chat-1M sampling

configs/default.yaml
data/sample_prompts.txt
tests/
```

## Configuration

Edit `configs/default.yaml`:

- `compressor.compression_ratio` — target token reduction (0.25, 0.40, 0.55)
- `unifier.similarity_threshold` — cosine similarity τ for canonical matching
- `serving.vllm_base_url` — vLLM backend URL

## Evaluation Baselines

The benchmark compares three modes:

- **no_compression** — raw prompts, no prefix optimization
- **compression_only** — segment compressor without unification (LLMLingua-style)
- **promptfuse_full** — compression + semantic unification

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
