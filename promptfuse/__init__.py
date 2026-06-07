"""PromptFuse: semantic prompt compression and unification for KV cache efficiency."""

__version__ = "0.1.0"

from typing import TYPE_CHECKING, Any

__all__ = ["PromptFusePipeline", "__version__"]

if TYPE_CHECKING:
    from promptfuse.pipeline import PromptFusePipeline


def __getattr__(name: str) -> Any:
    if name == "PromptFusePipeline":
        from promptfuse.pipeline import PromptFusePipeline

        return PromptFusePipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
