"""FastAPI middleware server sitting before vLLM."""

from __future__ import annotations

import json
import logging
import time
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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
        async def chat_completions(request: ChatCompletionRequest) -> Any:
            return await self._handle_chat(request)

        @self.app.post("/v1/completions")
        async def completions(body: dict[str, Any]) -> Any:
            prompt = body.get("prompt", "")
            if isinstance(prompt, list):
                prompt = "\n".join(prompt)
            if not isinstance(prompt, str) or not prompt.strip():
                raise HTTPException(status_code=400, detail="Prompt must be a non-empty string.")

            processed = self.pipeline.process(prompt)
            body = {**body, "prompt": processed.final_prompt}
            request_id = str(uuid4())
            return await self._forward_vllm(
                "/v1/completions",
                body,
                endpoint="completion",
                request_id=request_id,
                processed=processed,
                request_meta=body,
            )

        @self.app.get("/v1/metrics/vllm-cache")
        async def vllm_cache_metrics() -> Any:
            if not self.config.serving.vllm_metrics_url:
                raise HTTPException(status_code=404, detail="No vLLM metrics URL configured.")

            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(self.config.serving.vllm_metrics_url)
                    response.raise_for_status()
            except httpx.HTTPError as exc:
                raise HTTPException(status_code=502, detail=f"Failed to fetch vLLM metrics: {exc}") from exc

            with suppress(ValueError):
                return response.json()
            return {"raw_metrics": response.text}

    def _extract_prompt(self, messages: list[ChatMessage]) -> str:
        """Concatenate messages into a single prompt string for processing."""
        parts = []
        for msg in messages:
            parts.append(f"{msg.role}: {msg.content}")
        return "\n".join(parts)

    async def _handle_chat(self, request: ChatCompletionRequest) -> Any:
        if not request.messages:
            raise HTTPException(status_code=400, detail="At least one message is required.")

        raw = self._extract_prompt(request.messages)
        processed = self.pipeline.process(raw)

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
        request_id = str(uuid4())
        return await self._forward_vllm(
            "/v1/chat/completions",
            payload,
            endpoint="chat",
            request_id=request_id,
            processed=processed,
            request_meta=request.model_dump(),
        )

    async def _forward_vllm(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        endpoint: str,
        request_id: str,
        processed: Any,
        request_meta: dict[str, Any],
    ) -> Any:
        url = f"{self.config.serving.vllm_base_url.rstrip('/')}{path}"
        t0 = time.perf_counter()
        timeout = self.config.serving.vllm_timeout_s

        if payload.get("stream"):
            client = httpx.AsyncClient(timeout=timeout)
            request = client.build_request("POST", url, json=payload)
            try:
                response = await client.send(request, stream=True)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                await client.aclose()
                logger.error("vLLM request failed (stream): %s", exc)
                self._log_event(
                    endpoint=endpoint,
                    request_id=request_id,
                    processed=processed,
                    request_meta=request_meta,
                    vllm_ttft_ms=None,
                    vllm_total_ms=(time.perf_counter() - t0) * 1000,
                    stream=True,
                    error=str(exc),
                )
                raise HTTPException(status_code=502, detail=f"vLLM backend error: {exc}") from exc

            state: dict[str, float | int | None] = {"ttft_ms": None, "bytes_streamed": 0}

            async def iterator():
                try:
                    async for chunk in response.aiter_raw():
                        if state["ttft_ms"] is None:
                            state["ttft_ms"] = (time.perf_counter() - t0) * 1000
                        state["bytes_streamed"] = int(state["bytes_streamed"] or 0) + len(chunk)
                        yield chunk
                finally:
                    total_ms = (time.perf_counter() - t0) * 1000
                    with suppress(Exception):
                        await response.aclose()
                    with suppress(Exception):
                        await client.aclose()
                    self._log_event(
                        endpoint=endpoint,
                        request_id=request_id,
                        processed=processed,
                        request_meta=request_meta,
                        vllm_ttft_ms=float(state["ttft_ms"]) if state["ttft_ms"] is not None else None,
                        vllm_total_ms=total_ms,
                        stream=True,
                        bytes_streamed=int(state["bytes_streamed"] or 0),
                    )

            content_type = response.headers.get("content-type", "text/event-stream")
            return StreamingResponse(iterator(), media_type=content_type)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
        except httpx.HTTPError as exc:
            logger.error("vLLM request failed: %s", exc)
            self._log_event(
                endpoint=endpoint,
                request_id=request_id,
                processed=processed,
                request_meta=request_meta,
                vllm_ttft_ms=None,
                vllm_total_ms=(time.perf_counter() - t0) * 1000,
                stream=False,
                error=str(exc),
            )
            raise HTTPException(status_code=502, detail=f"vLLM backend error: {exc}") from exc

        total_ms = (time.perf_counter() - t0) * 1000
        result.setdefault("_promptfuse", {}).update(
            {
                "request_id": request_id,
                "vllm_latency_ms": total_ms,
                "pipeline_ms": processed.total_ms,
                "cache_hit": processed.cache_hit,
                "token_reduction": processed.token_reduction,
            }
        )
        self._log_event(
            endpoint=endpoint,
            request_id=request_id,
            processed=processed,
            request_meta=request_meta,
            vllm_ttft_ms=total_ms,  # TTFT is unavailable on non-streamed responses.
            vllm_total_ms=total_ms,
            stream=False,
        )
        return result

    def _log_event(
        self,
        *,
        endpoint: str,
        request_id: str,
        processed: Any,
        request_meta: dict,
        vllm_ttft_ms: float | None,
        vllm_total_ms: float,
        stream: bool,
        error: str | None = None,
        bytes_streamed: int | None = None,
    ) -> None:
        self._stats["requests"] += 1
        if processed.cache_hit:
            self._stats["cache_hits"] += 1
        self._stats["total_compression_ms"] += processed.total_ms

        original_tokens = processed.compression.original_tokens if processed.compression else None
        final_tokens = processed.compression.compressed_tokens if processed.compression else None
        event = {
            "request_id": request_id,
            "timestamp": time.time(),
            "endpoint": endpoint,
            "stream": stream,
            "token_reduction": processed.token_reduction,
            "cache_hit": processed.cache_hit,
            "compression_ms": processed.compression_ms,
            "unification_ms": processed.unification_ms,
            "pipeline_total_ms": processed.total_ms,
            "vllm_ttft_ms": vllm_ttft_ms,
            "vllm_total_ms": vllm_total_ms,
            "original_tokens": original_tokens,
            "final_tokens": final_tokens,
            "token_delta": (original_tokens - final_tokens)
            if original_tokens is not None and final_tokens is not None
            else None,
            "canonical_id": processed.unification.canonical_id if processed.unification else None,
            "similarity": processed.unification.similarity if processed.unification else None,
            "error": error,
            "bytes_streamed": bytes_streamed,
            "request_meta": {
                "model": request_meta.get("model"),
                "max_tokens": request_meta.get("max_tokens"),
                "temperature": request_meta.get("temperature"),
            },
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
    server = PromptFuseServer()
    server.run()


if __name__ == "__main__":
    main()
