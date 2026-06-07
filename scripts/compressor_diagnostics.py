#!/usr/bin/env python3
"""Export segment-level compressor diagnostics for debugging and reports."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from promptfuse.compressor.segment_compressor import SegmentCompressor, split_sentences
from promptfuse.config import Settings


@dataclass
class SegmentDiagnostic:
    index: int
    text_preview: str
    token_count: int
    perplexity: float
    protected: bool
    kept: bool


def load_prompts(path: Path) -> list[str]:
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                return [row.get("text", row.get("prompt", "")).strip() for row in data if row.get("text") or row.get("prompt")]
            return [str(item).strip() for item in data if str(item).strip()]
        return [str(item).strip() for item in data.get("prompts", []) if str(item).strip()]
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-segment compressor diagnostics.")
    parser.add_argument("--prompts", type=Path, default=Path("data/sample_prompts.txt"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--ratio", type=float, default=0.40)
    parser.add_argument("--limit", type=int, default=25, help="Max number of prompts to analyze.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cpu_final/metrics/compressor_diagnostics.json"),
    )
    args = parser.parse_args()

    config = Settings(config_path=args.config).load()
    compressor = SegmentCompressor(config.compressor, lazy_load=True)
    prompts = load_prompts(args.prompts)[: args.limit]

    prompt_diagnostics = []
    for prompt_idx, prompt in enumerate(prompts):
        segments = split_sentences(prompt)
        if not segments:
            continue

        scores, _ = compressor._score_segments(segments)
        protected = [compressor._is_protected(seg) for seg in segments]

        result = compressor.compress(prompt, compression_ratio=args.ratio)
        kept_segments = split_sentences(result.compressed)
        kept_counts: dict[str, int] = {}
        for segment in kept_segments:
            kept_counts[segment] = kept_counts.get(segment, 0) + 1

        rows: list[SegmentDiagnostic] = []
        for idx, segment in enumerate(segments):
            currently_kept = kept_counts.get(segment, 0) > 0
            if currently_kept:
                kept_counts[segment] -= 1
            rows.append(
                SegmentDiagnostic(
                    index=idx,
                    text_preview=segment[:120],
                    token_count=scores[idx].token_count,
                    perplexity=scores[idx].perplexity,
                    protected=protected[idx],
                    kept=currently_kept,
                )
            )

        prompt_diagnostics.append(
            {
                "prompt_index": prompt_idx,
                "original_prompt": prompt,
                "compressed_prompt": result.compressed,
                "target_tokens": result.target_tokens,
                "compressed_tokens": result.compressed_tokens,
                "token_reduction": result.token_reduction,
                "segments": [asdict(row) for row in rows],
            }
        )

    payload = {
        "schema_version": "1.0",
        "prompts_path": str(args.prompts),
        "ratio": args.ratio,
        "prompt_count": len(prompt_diagnostics),
        "rows": prompt_diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
