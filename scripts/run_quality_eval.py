#!/usr/bin/env python3
"""Compare LLM outputs: raw prompts vs PromptFuse-compressed (ROUGE-L)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from promptfuse.config import Settings
from promptfuse.evaluation.metrics import compute_rouge_l
from promptfuse.evaluation.vllm_client import VLLMClient
from promptfuse.pipeline import PromptFusePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_prompts(path: Path) -> list[str]:
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            if "prompts" in data[0]:
                out: list[str] = []
                for cluster in data:
                    out.extend(cluster["prompts"])
                return out
            return [d.get("prompt", d.get("text", "")) for d in data]
        return data
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="ROUGE-L quality eval via vLLM")
    parser.add_argument("--prompts", type=Path, default=Path("data/sample_prompts.txt"))
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"))
    parser.add_argument("--ratio", type=float, default=0.40)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--output", type=Path, default=Path("results/quality_eval.json"))
    parser.add_argument("--limit", type=int, default=10, help="Max prompts to evaluate")
    args = parser.parse_args()

    settings = Settings(config_path=args.config)
    config = settings.load()

    prompts = load_prompts(args.prompts)[: args.limit]
    if not prompts:
        logger.error("No prompts found in %s", args.prompts)
        sys.exit(1)

    client = VLLMClient(base_url=args.vllm_url, model=config.serving.vllm_model)
    if not client.health_check():
        logger.error("vLLM not reachable at %s — start vLLM first", args.vllm_url)
        sys.exit(1)

    pipeline = PromptFusePipeline(config, lazy_load=False)

    results: list[dict] = []
    rouge_scores: list[float] = []

    for i, prompt in enumerate(prompts):
        logger.info("Evaluating prompt %d/%d", i + 1, len(prompts))

        raw_resp = client.chat(prompt, max_tokens=args.max_tokens, temperature=0.0)
        processed = pipeline.process(prompt, compression_ratio=args.ratio)
        fused_resp = client.chat(processed.final_prompt, max_tokens=args.max_tokens, temperature=0.0)

        rouge_l = compute_rouge_l(raw_resp.content, fused_resp.content)
        rouge_scores.append(rouge_l)

        results.append(
            {
                "prompt": prompt,
                "final_prompt": processed.final_prompt,
                "token_reduction": processed.token_reduction,
                "rouge_l": round(rouge_l, 4),
                "raw_output_preview": raw_resp.content[:300],
                "fused_output_preview": fused_resp.content[:300],
            }
        )

    summary = {
        "num_prompts": len(results),
        "avg_rouge_l": round(sum(rouge_scores) / len(rouge_scores), 4),
        "min_rouge_l": round(min(rouge_scores), 4),
        "max_rouge_l": round(max(rouge_scores), 4),
        "meets_threshold_0.85": sum(1 for s in rouge_scores if s >= 0.85) / len(rouge_scores),
        "compression_ratio": args.ratio,
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Quality Evaluation ===")
    print(f"Prompts evaluated: {summary['num_prompts']}")
    print(f"Avg ROUGE-L:       {summary['avg_rouge_l']:.4f}")
    print(f"≥0.85 threshold:   {summary['meets_threshold_0.85']:.1%} of prompts")
    print(f"Saved to:          {args.output}\n")


if __name__ == "__main__":
    main()
