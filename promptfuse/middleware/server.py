"""FastAPI middleware server sitting before vLLM."""

from __future__ import annotations

import asyncio
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


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Thread-safe stats counter
# ---------------------------------------------------------------------------

class _Stats:
    """Atomic-ish stats accumulator safe under asyncio + thread-pool."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.requests: int = 0
        self.cache_hits: int = 0
        self.total_compression_ms: float = 0.0

    async def record(self, *, cache_hit: bool, pipeline_ms: float) -> None:
        async with self._lock:
            self.requests += 1
            if cache_hit:
                self.cache_hits += 1
            self.total_compression_ms += pipeline_ms

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            n = self.requests
            return {
                "requests": n,
                "unifier_cache_hits": self.cache_hits,
                "unifier_hit_rate": self.cache_hits / n if n else 0.0,
                "avg_pipeline_ms": self.total_compression_ms / n if n else 0.0,
            }


# ---------------------------------------------------------------------------
# Buffered JSONL log writer
# ---------------------------------------------------------------------------

class _LogWriter:
    """Writes JSONL events with periodic flushing instead of per-event I/O."""

    def __init__(self, path: Path, flush_interval: float = 2.0) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._buffer: list[str] = []
        self._lock = asyncio.Lock()
        self._flush_interval = flush_interval
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._flush_loop())

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(self._flush_interval)
            await self.flush()

    async def write(self, event: dict[str, Any]) -> None:
        async with self._lock:
            self._buffer.append(json.dumps(event, default=str))

    async def flush(self) -> None:
        async with self._lock:
            if not self._buffer:
                return
            batch = self._buffer.copy()
            self._buffer.clear()
        with open(self._path, "a") as f:
            f.write("\n".join(batch) + "\n")

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
        await self.flush()


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class PromptFuseServer:
    """Middleware that compresses/unifies prompts before forwarding to vLLM.

    Key design choice: only the **system prompt** (and optionally the last
    user message) are compressed and unified.  Earlier assistant/user turns
    are forwarded verbatim because:

    1.  The system prompt is the shared prefix across requests — it's where
        KV cache reuse actually happens.
    2.  Compressing conversation history changes meaning in context-dependent
        ways the compressor can't account for.
    3.  The unifier can only match canonical forms for instructions, not
        arbitrary dialogue.
    """

    _COMPRESSIBLE_ROLES = frozenset({"system"})

    def __init__(self, config: PromptFuseConfig | None = None, *, config_path: str | Path | None = None):
        settings = Settings(config_path=Path(config_path)) if config_path else Settings()
        self.config = config or settings.load()
        self.pipeline = PromptFusePipeline(self.config, lazy_load=True)
        self._stats = _Stats()
        self._log = _LogWriter(Path(self.config.serving.log_path))

        self.app = FastAPI(title="PromptFuse", version="0.1.0")
        self._register_routes()
        self._register_lifecycle()

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
            snap = await self._stats.snapshot()
            snap["canonical_inventory_size"] = (
                self.pipeline.unifier.store.size if self.pipeline.unifier else 0
            )
            return snap

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
                raise HTTPException(
                    status_code=404, detail="No vLLM metrics URL configured."
                )
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(self.config.serving.vllm_metrics_url)
                    response.raise_for_status()
            except httpx.HTTPError as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to fetch vLLM metrics: {exc}",
                ) from exc

            with suppress(ValueError):
                return response.json()
            return {"raw_metrics": response.text}

    def _register_lifecycle(self) -> None:
        @self.app.on_event("startup")
        async def on_startup() -> None:
            import os

            self._log.start()

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

        @self.app.on_event("shutdown")
        async def on_shutdown() -> None:
            await self._log.close()

    def _process_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[list[dict[str, str]], Any]:
        """Compress/unify only compressible roles; pass others through."""
        forwarded: list[dict[str, str]] = []
        pipeline_result = None

        for msg in messages:
            if msg.role in self._COMPRESSIBLE_ROLES and msg.content.strip():
                processed = self.pipeline.process(msg.content)
                forwarded.append({"role": msg.role, "content": processed.final_prompt})
                if pipeline_result is None:
                    pipeline_result = processed
            else:
                forwarded.append({"role": msg.role, "content": msg.content})

        if pipeline_result is None:
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].role == "user" and messages[i].content.strip():
                    processed = self.pipeline.process(messages[i].content)
                    forwarded[i] = {
                        "role": messages[i].role,
                        "content": processed.final_prompt,
                    }
                    pipeline_result = processed
                    break

        if pipeline_result is None:
            raw = messages[-1].content if messages else ""
            pipeline_result = self.pipeline.noop_result(raw)

        return forwarded, pipeline_result

    async def _handle_chat(self, request: ChatCompletionRequest) -> Any:
        if not request.messages:
            raise HTTPException(status_code=400, detail="At least one message is required.")

        forwarded_messages, processed = self._process_messages(request.messages)

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
            return await self._forward_stream(
                url, payload, timeout=timeout, t0=t0,
                endpoint=endpoint, request_id=request_id,
                processed=processed, request_meta=request_meta,
            )

        return await self._forward_unary(
            url, payload, timeout=timeout, t0=t0,
            endpoint=endpoint, request_id=request_id,
            processed=processed, request_meta=request_meta,
        )

    async def _forward_unary(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        timeout: float,
        t0: float,
        endpoint: str,
        request_id: str,
        processed: Any,
        request_meta: dict[str, Any],
    ) -> Any:
        error: str | None = None
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
        except httpx.HTTPError as exc:
            error = str(exc)
            logger.error("vLLM request failed: %s", exc)
            total_ms = (time.perf_counter() - t0) * 1000
            await self._record_and_log(
                endpoint=endpoint, request_id=request_id,
                processed=processed, request_meta=request_meta,
                vllm_ttft_ms=None, vllm_total_ms=total_ms,
                stream=False, error=error,
            )
            raise HTTPException(status_code=502, detail=f"vLLM backend error: {exc}") from exc

        total_ms = (time.perf_counter() - t0) * 1000
        usage = result.get("usage", {})
        result.setdefault("_promptfuse", {}).update({
            "request_id": request_id,
            "vllm_latency_ms": total_ms,
            "pipeline_ms": processed.total_ms,
            "cache_hit": processed.cache_hit,
            "token_reduction": processed.token_reduction,
            "vllm_prompt_tokens": usage.get("prompt_tokens"),
            "vllm_completion_tokens": usage.get("completion_tokens"),
        })
        await self._record_and_log(
            endpoint=endpoint, request_id=request_id,
            processed=processed, request_meta=request_meta,
            vllm_ttft_ms=total_ms, vllm_total_ms=total_ms,
            stream=False,
        )
        return result

    async def _forward_stream(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        timeout: float,
        t0: float,
        endpoint: str,
        request_id: str,
        processed: Any,
        request_meta: dict[str, Any],
    ) -> StreamingResponse:
        client = httpx.AsyncClient(timeout=timeout)
        req = client.build_request("POST", url, json=payload)
        try:
            response = await client.send(req, stream=True)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            await client.aclose()
            logger.error("vLLM request failed (stream): %s", exc)
            total_ms = (time.perf_counter() - t0) * 1000
            await self._record_and_log(
                endpoint=endpoint, request_id=request_id,
                processed=processed, request_meta=request_meta,
                vllm_ttft_ms=None, vllm_total_ms=total_ms,
                stream=True, error=str(exc),
            )
            raise HTTPException(status_code=502, detail=f"vLLM backend error: {exc}") from exc

        state: dict[str, Any] = {"ttft_ms": None, "bytes_streamed": 0}

        async def iterator():
            try:
                async for chunk in response.aiter_raw():
                    if state["ttft_ms"] is None:
                        state["ttft_ms"] = (time.perf_counter() - t0) * 1000
                    state["bytes_streamed"] += len(chunk)
                    yield chunk
            finally:
                total_ms = (time.perf_counter() - t0) * 1000
                with suppress(Exception):
                    await response.aclose()
                with suppress(Exception):
                    await client.aclose()
                with suppress(Exception):
                    await self._record_and_log(
                        endpoint=endpoint, request_id=request_id,
                        processed=processed, request_meta=request_meta,
                        vllm_ttft_ms=state["ttft_ms"],
                        vllm_total_ms=total_ms,
                        stream=True,
                        bytes_streamed=state["bytes_streamed"],
                    )

        content_type = response.headers.get("content-type", "text/event-stream")
        return StreamingResponse(iterator(), media_type=content_type)

    async def _record_and_log(
        self,
        *,
        endpoint: str,
        request_id: str,
        processed: Any,
        request_meta: dict[str, Any],
        vllm_ttft_ms: float | None,
        vllm_total_ms: float,
        stream: bool,
        error: str | None = None,
        bytes_streamed: int | None = None,
    ) -> None:
        await self._stats.record(
            cache_hit=processed.cache_hit,
            pipeline_ms=processed.total_ms,
        )

        original_tokens = (
            processed.compression.original_tokens if processed.compression else None
        )
        final_tokens = (
            processed.compression.compressed_tokens if processed.compression else None
        )

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
            "final_prompt": processed.final_prompt[:500],
            "canonical_id": (
                processed.unification.canonical_id if processed.unification else None
            ),
            "similarity": (
                processed.unification.similarity if processed.unification else None
            ),
            "error": error,
            "bytes_streamed": bytes_streamed,
            "request_meta": {
                "model": request_meta.get("model"),
                "max_tokens": request_meta.get("max_tokens"),
                "temperature": request_meta.get("temperature"),
            },
        }
        await self._log.write(event)

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
