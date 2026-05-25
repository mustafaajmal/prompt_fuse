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


class PromptFuseServer:
    """Middleware that compresses/unifies prompts before forwarding to vLLM."""

    def __init__(self, config: PromptFuseConfig | None = None):
        self.config = config or Settings().load()
        self.pipeline = PromptFusePipeline(self.config, lazy_load=True)
        self.log_path = Path(self.config.serving.log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stats = {"requests": 0, "cache_hits": 0, "total_compression_ms": 0.0}

        self.app = FastAPI(title="PromptFuse", version="0.1.0")
        self._register_routes()

    def _register_routes(self) -> None:
        @self.app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok", "service": "promptfuse"}

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

    def _extract_prompt(self, messages: list[ChatMessage]) -> str:
        """Concatenate messages into a single prompt string for processing."""
        parts = []
        for msg in messages:
            parts.append(f"{msg.role}: {msg.content}")
        return "\n".join(parts)

    async def _handle_chat(self, request: ChatCompletionRequest) -> dict[str, Any]:
        raw = self._extract_prompt(request.messages)
        processed = self.pipeline.process(raw)
        self._log_event("chat", processed, request.model_dump())

        # Replace user message content with unified prompt for prefix cache alignment
        forwarded_messages = []
        for i, msg in enumerate(request.messages):
            if i == len(request.messages) - 1 and msg.role == "user":
                forwarded_messages.append({"role": msg.role, "content": processed.final_prompt})
            else:
                forwarded_messages.append({"role": msg.role, "content": msg.content})

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
    return PromptFuseServer(config).app


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    server = PromptFuseServer(Settings().load())
    server.run()


if __name__ == "__main__":
    main()
