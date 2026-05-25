"""Optional LLMLingua baseline (install with pip install -e '.[llmlingua]')."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def llmlingua_available() -> bool:
    try:
        import llmlingua  # noqa: F401

        return True
    except ImportError:
        return False


def compress_with_llmlingua(
    prompt: str,
    *,
    target_ratio: float = 0.40,
    model_name: str = "meta-llama/Llama-3.2-1B",
) -> tuple[str, int, int]:
    """
    Compress using LLMLingua (token-level). Returns (compressed, orig_tokens, comp_tokens).
    """
    if not llmlingua_available():
        raise RuntimeError("llmlingua not installed; pip install -e '.[llmlingua]'")

    from llmlingua import PromptCompressor

    compressor = PromptCompressor(
        model_name=model_name,
        device_map="auto",
        use_llmlingua2=False,
    )
    rate = max(0.05, min(0.95, 1.0 - target_ratio))
    result = compressor.compress_prompt(prompt, rate=rate)
    compressed = result["compressed_prompt"]
    orig = len(compressor.tokenizer.encode(prompt))
    comp = len(compressor.tokenizer.encode(compressed))
    return compressed, orig, comp
