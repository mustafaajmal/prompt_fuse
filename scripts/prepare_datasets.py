#!/usr/bin/env python3
"""Download and prepare evaluation datasets (ShareGPT, LMSYS) with local fallbacks."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Hugging Face dataset IDs to try in order (first success wins)
SHAREGPT_SOURCES = [
    ("RyokoAI/ShareGPT52K", "conversations", "human", "value"),
    ("teknium/OpenHermes-2.5", "conversations", "from", "value"),
    ("HuggingFaceH4/ultrachat_200k", "messages", "user", "content"),
]


def _hf_login_from_env() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
        logger.info("Logged into Hugging Face Hub (token from env)")
    except Exception as exc:
        logger.warning("HF login skipped: %s", exc)


def _extract_first_user_prompt(
    row: dict,
    messages_key: str,
    role_key: str,
    content_key: str,
) -> str | None:
    messages = row.get(messages_key)
    if messages is None:
        return None

    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get(role_key) or msg.get("role")
            if role in ("human", "user", "Human", "User", "from"):
                text = msg.get(content_key) or msg.get("content") or msg.get("value") or ""
                if isinstance(text, str) and text.strip():
                    return text.strip()
    return None


def sample_sharegpt(output: Path, max_samples: int = 1000) -> int:
    from datasets import load_dataset

    prompts: list[str] = []
    last_error: Exception | None = None

    for repo_id, messages_key, role_key, content_key in SHAREGPT_SOURCES:
        try:
            logger.info("Trying ShareGPT-style source: %s", repo_id)
            ds = load_dataset(repo_id, split="train", streaming=True)
            for row in ds:
                text = _extract_first_user_prompt(row, messages_key, role_key, content_key)
                if text and len(text) > 20:
                    prompts.append(text)
                if len(prompts) >= max_samples:
                    break
            if prompts:
                logger.info("Loaded %d prompts from %s", len(prompts), repo_id)
                break
        except Exception as exc:
            last_error = exc
            logger.warning("%s failed: %s", repo_id, exc)

    if not prompts:
        raise RuntimeError(f"All ShareGPT sources failed. Last error: {last_error}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(prompts, f, indent=2)
    logger.info("Wrote %d prompts to %s", len(prompts), output)
    return len(prompts)


def _first_user_message(row: dict) -> str | None:
    for msg in row.get("conversation", []):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


def sample_lmsys(output: Path, max_samples: int = 1000) -> int:
    from datasets import load_dataset

    _hf_login_from_env()
    logger.info("Loading LMSYS-Chat-1M (gated — HF_TOKEN + dataset access required)")

    prompts: list[str] = []
    # Non-streaming slice avoids WSL streaming teardown crash (PyGILState / bad fd).
    scan_cap = min(max_samples * 10, 50_000)
    logger.info("Reading up to %d rows (non-streaming)...", scan_cap)
    ds = load_dataset("lmsys/lmsys-chat-1m", split=f"train[:{scan_cap}]")
    for row in ds:
        text = _first_user_message(row)
        if text:
            prompts.append(text)
        if len(prompts) >= max_samples:
            break

    if len(prompts) < max_samples:
        logger.warning("Only collected %d / %d prompts from slice", len(prompts), max_samples)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(prompts, f, indent=2)
    logger.info("Wrote %d LMSYS prompts to %s", len(prompts), output)
    return len(prompts)


def build_local_eval_bundle(output: Path, max_samples: int = 500) -> int:
    """
    Offline fallback: synthetic paraphrases + complex workload + sample_prompts.
    No Hugging Face download required.
    """
    root = Path(__file__).resolve().parent.parent
    prompts: list[str] = []

    synth = root / "data/synthetic_paraphrases.json"
    if synth.exists():
        with open(synth) as f:
            records = json.load(f)
        prompts.extend(r["text"] for r in records if r.get("text"))

    complex_path = root / "data/complex_workload.json"
    if complex_path.exists():
        with open(complex_path) as f:
            clusters = json.load(f)
        for c in clusters:
            prompts.extend(c.get("prompts", []))
    else:
        # Generate in-memory if file missing
        import sys

        sys.path.insert(0, str(root / "scripts"))
        try:
            from generate_complex_workload import build_workload

            for c in build_workload():
                prompts.extend(c["prompts"])
        except Exception as exc:
            logger.warning("Could not build complex workload: %s", exc)

    sample = root / "data/sample_prompts.txt"
    if sample.exists():
        prompts.extend(line.strip() for line in sample.read_text().splitlines() if line.strip())

    # Dedupe preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for p in prompts:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    prompts = unique[:max_samples]
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(prompts, f, indent=2)
    logger.info("Wrote %d local bundle prompts to %s (no HF download)", len(prompts), output)
    return len(prompts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sharegpt", "lmsys", "local", "all"], default="all")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Skip HF downloads; build data/eval_prompts_local.json only",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.local_only:
        args.dataset = "local"

    if args.dataset in ("local", "all"):
        build_local_eval_bundle(args.output_dir / "eval_prompts_local.json", args.max_samples)

    if args.dataset == "local":
        return

    _hf_login_from_env()

    if args.dataset in ("sharegpt", "all"):
        try:
            sample_sharegpt(args.output_dir / "sharegpt_prompts.json", args.max_samples)
        except Exception as exc:
            logger.warning("ShareGPT download failed: %s", exc)
            logger.info("Use: data/eval_prompts_local.json or run with --local-only")

    if args.dataset in ("lmsys", "all"):
        try:
            sample_lmsys(args.output_dir / "lmsys_prompts.json", args.max_samples)
        except Exception as exc:
            logger.warning("LMSYS download failed: %s", exc)
            logger.info(
                "LMSYS is gated: visit https://huggingface.co/datasets/lmsys/lmsys-chat-1m "
                "→ Agree and share access, set HF_TOKEN in .env, retry."
            )

    logger.info("Done.")


if __name__ == "__main__":
    main()
