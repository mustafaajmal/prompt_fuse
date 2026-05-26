"""FastAPI middleware server sitting before vLLM."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from promptfuse.config import PromptFuseConfig, Settings
from promptfuse.pipeline import PromptFusePipeline

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = 512
    temperature: float | None = 0.7
    stream: bool = False
    extra_body: dict[str, Any] = Field(default_factory=dict)


class ProcessRequest(BaseModel):
    prompt: str
    compression_ratio: float | None = None


class ProcessResponse(BaseModel):
    raw_prompt: str
    final_prompt: str
    token_reduction: float
    cache_hit: bool
    compression_ms: float
    unification_ms: float
    total_ms: float
    unifier_similarity: float | None = None
    canonical_id: int | None = None


class PromptFuseServer:
    """Middleware that compresses/unifies prompts before forwarding to vLLM."""

    def __init__(self, config: PromptFuseConfig | None = None, *, config_path: str | Path | None = None):
        settings = Settings(config_path=Path(config_path)) if config_path else Settings()
        self.config = config or settings.load()
        self.pipeline = PromptFusePipeline(self.config, lazy_load=True)
        self.log_path = Path(self.config.serving.log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stats = {"requests": 0, "cache_hits": 0, "total_compression_ms": 0.0}

        self.app = FastAPI(title="PromptFuse", version="0.1.0")
        self._register_routes()
        self._register_lifespan()

    def _register_lifespan(self) -> None:
        @self.app.on_event("startup")
        async def _warm_models() -> None:
            import os

            # Unifier only at startup so /health is fast and we don't fight vLLM for VRAM.
            if self.pipeline.unifier:
                logger.info("Loading bi-encoder for semantic unifier...")
                self.pipeline.unifier._load_encoder()
                logger.info(
                    "Unifier ready (inventory size=%d)",
                    self.pipeline.unifier.store.size,
                )

            preload_compressor = os.environ.get("PROMPTFUSE_PRELOAD_COMPRESSOR", "0") == "1"
            if preload_compressor and self.pipeline.compressor:
                try:
                    logger.info("Preloading proxy LM compressor (PROMPTFUSE_PRELOAD_COMPRESSOR=1)...")
                    self.pipeline.compressor._load_model()
                    logger.info("Compressor preloaded.")
                except Exception as exc:
                    logger.warning(
                        "Compressor not preloaded (%s). Will retry on first request.",
                        exc,
                    )
            elif self.pipeline.compressor:
                logger.info(
                    "Compressor will load on first request (set PROMPTFUSE_PRELOAD_COMPRESSOR=1 to preload)."
                )

    def _register_routes(self) -> None:
        @self.app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok", "service": "promptfuse"}

        @self.app.post("/process", response_model=ProcessResponse)
        @self.app.post("/v1/process", response_model=ProcessResponse)
        async def process_prompt(request: ProcessRequest) -> ProcessResponse:
            """Run compress+unify without calling vLLM (debug / demo)."""
            result = self.pipeline.process(
                request.prompt,
                compression_ratio=request.compression_ratio,
            )
            return ProcessResponse(
                raw_prompt=result.raw_prompt,
                final_prompt=result.final_prompt,
                token_reduction=result.token_reduction,
                cache_hit=result.cache_hit,
                compression_ms=result.compression_ms,
                unification_ms=result.unification_ms,
                total_ms=result.total_ms,
                unifier_similarity=(
                    result.unification.similarity if result.unification else None
                ),
                canonical_id=(
                    result.unification.canonical_id if result.unification else None
                ),
            )

        @self.app.get("/stats")
        async def stats() -> dict[str, Any]:
            n = self._stats["requests"]
            hit_rate = self._stats["cache_hits"] / n if n else 0.0
            avg_ms = self._stats["total_compression_ms"] / n if n else 0.0
            return {
                "requests": n,
                "unifier_cache_hits": self._stats["cache_hits"],
                "unifier_hit_rate": hit_rate,
                "avg_pipeline_ms": avg_ms,
                "canonical_inventory_size": self.pipeline.unifier.store.size
                if self.pipeline.unifier
                else 0,
            }

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
            return await self._handle_chat(request)

        @self.app.post("/v1/completions")
        async def completions(body: dict[str, Any]) -> dict[str, Any]:
            prompt = body.get("prompt", "")
            if isinstance(prompt, list):
                prompt = "\n".join(prompt)
            processed = self.pipeline.process(prompt)
            self._log_event("completion", processed, body)
            body = {**body, "prompt": processed.final_prompt}
            return await self._forward_vllm("/v1/completions", body)

    def _process_user_messages(self, messages: list[ChatMessage]) -> tuple[list[dict[str, str]], Any]:
        """Compress/unify only the last user turn; keep system/assistant text intact."""
        forwarded: list[dict[str, str]] = []
        last_user_idx = None
        for i, msg in enumerate(messages):
            if msg.role == "user":
                last_user_idx = i

        processed = None
        for i, msg in enumerate(messages):
            if i == last_user_idx:
                processed = self.pipeline.process(msg.content)
                forwarded.append({"role": msg.role, "content": processed.final_prompt})
            else:
                forwarded.append({"role": msg.role, "content": msg.content})
        return forwarded, processed

    async def _handle_chat(self, request: ChatCompletionRequest) -> dict[str, Any]:
        forwarded_messages, processed = self._process_user_messages(request.messages)
        if processed:
            self._log_event("chat", processed, request.model_dump())

        payload = {
            "model": request.model or self.config.serving.vllm_model,
            "messages": forwarded_messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": request.stream,
        }
        return await self._forward_vllm("/v1/chat/completions", payload)

    async def _forward_vllm(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.serving.vllm_base_url.rstrip('/')}{path}"
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
        except httpx.HTTPError as exc:
            logger.error("vLLM request failed: %s", exc)
            raise HTTPException(status_code=502, detail=f"vLLM backend error: {exc}") from exc

        vllm_ms = (time.perf_counter() - t0) * 1000
        usage = result.get("usage", {})
        pf_meta = result.setdefault("_promptfuse", {})
        pf_meta["vllm_latency_ms"] = vllm_ms
        pf_meta["vllm_prompt_tokens"] = usage.get("prompt_tokens")
        pf_meta["vllm_completion_tokens"] = usage.get("completion_tokens")
        return result

    def _log_event(self, endpoint: str, processed: Any, request_meta: dict) -> None:
        self._stats["requests"] += 1
        if processed.cache_hit:
            self._stats["cache_hits"] += 1
        self._stats["total_compression_ms"] += processed.total_ms

        event = {
            "timestamp": time.time(),
            "endpoint": endpoint,
            "token_reduction": processed.token_reduction,
            "cache_hit": processed.cache_hit,
            "compression_ms": processed.compression_ms,
            "unification_ms": processed.unification_ms,
            "total_ms": processed.total_ms,
            "original_tokens": processed.compression.original_tokens if processed.compression else None,
            "final_tokens": processed.compression.compressed_tokens if processed.compression else None,
            "final_prompt": processed.final_prompt[:500],
            "unifier_similarity": (
                processed.unification.similarity if processed.unification else None
            ),
            "canonical_id": (
                processed.unification.canonical_id if processed.unification else None
            ),
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def run(self) -> None:
        uvicorn.run(
            self.app,
            host=self.config.serving.host,
            port=self.config.serving.port,
            log_level="info",
        )


def create_app(config_path: str | None = None) -> FastAPI:
    settings = Settings(config_path=Path(config_path)) if config_path else Settings()
    config = settings.load()
    return PromptFuseServer(config, config_path=config_path).app


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    print("PromptFuse: importing (may take 30–60s on /mnt/c)...", flush=True)
    settings = Settings()
    config = settings.load()
    print(
        f"PromptFuse: starting on {config.serving.host}:{config.serving.port} "
        f"→ vLLM {config.serving.vllm_base_url}",
        flush=True,
    )
    server = PromptFuseServer(config, config_path=settings.config_path)
    logger.info(
        "PromptFuse listening on %s:%s → vLLM %s (config: %s)",
        config.serving.host,
        config.serving.port,
        config.serving.vllm_base_url,
        settings.config_path,
    )
    server.run()


if __name__ == "__main__":
    main()
