"""Optional baselines (LLMLingua) for comparison experiments."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BaselineCompressionResult:
    original: str
    compressed: str
    method: str


def compress_llmlingua(
    prompt: str,
    *,
    target_ratio: float = 0.40,
    model_name: str = "meta-llama/Llama-3.2-1B",
) -> BaselineCompressionResult:
    """Token-level compression via LLMLingua (optional dependency)."""
    try:
        from llmlingua import PromptCompressor
    except ImportError as exc:
        raise ImportError(
            "LLMLingua not installed. Run: pip install -e '.[llmlingua]'"
        ) from exc

    compressor = PromptCompressor(
        model_name=model_name,
        device_map="auto",
    )
    rate = max(0.05, min(0.95, 1.0 - target_ratio))
    compressed = compressor.compress_prompt(prompt, rate=rate)
    return BaselineCompressionResult(
        original=prompt,
        compressed=compressed,
        method="llmlingua",
    )


def is_llmlingua_available() -> bool:
    try:
        import llmlingua  # noqa: F401

        return True
    except ImportError:
        return False
