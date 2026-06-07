"""CPU-safe integration tests for PromptFuse middleware server."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import promptfuse.middleware.server as server_module
from promptfuse.config import PromptFuseConfig
from promptfuse.middleware.server import PromptFuseServer


class _DummyCompression:
    original_tokens = 10
    compressed_tokens = 6


class _DummyUnification:
    similarity = 0.91
    canonical_id = 7


class _DummyProcessed:
    final_prompt = "canonicalized prompt"
    token_reduction = 0.4
    cache_hit = True
    compression_ms = 3.5
    unification_ms = 1.2
    total_ms = 5.4
    compression = _DummyCompression()
    unification = _DummyUnification()


class _DummyPipeline:
    def __init__(self, *_args, **_kwargs):
        self.unifier = None

    def process(self, _raw_prompt: str, _compression_ratio: float | None = None) -> _DummyProcessed:
        return _DummyProcessed()


def _build_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> PromptFuseServer:
    monkeypatch.setattr(server_module, "PromptFusePipeline", _DummyPipeline)
    config = PromptFuseConfig()
    config.serving.log_path = str(tmp_path / "promptfuse.jsonl")
    config.serving.vllm_base_url = "http://example-vllm"
    config.serving.vllm_timeout_s = 10.0
    return PromptFuseServer(config)


def _read_event(log_path: Path) -> dict:
    with open(log_path) as f:
        lines = [line.strip() for line in f if line.strip()]
    assert lines
    return json.loads(lines[-1])


def test_request_validation_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    server = _build_server(monkeypatch, tmp_path)
    client = TestClient(server.app)

    completion_resp = client.post("/v1/completions", json={"prompt": ""})
    assert completion_resp.status_code == 400

    chat_resp = client.post("/v1/chat/completions", json={"messages": []})
    assert chat_resp.status_code == 400


@pytest.mark.asyncio
async def test_non_stream_forwarding_and_log_schema(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    server = _build_server(monkeypatch, tmp_path)
    log_path = Path(server.config.serving.log_path)

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"id": "cmpl-test"}

    class _FakeAsyncClient:
        def __init__(self, *_, **__):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *_args, **_kwargs):
            return _FakeResponse()

    monkeypatch.setattr(server_module.httpx, "AsyncClient", _FakeAsyncClient)

    result = await server._forward_vllm(
        "/v1/chat/completions",
        {"stream": False},
        endpoint="chat",
        request_id="req-123",
        processed=_DummyProcessed(),
        request_meta={"model": "test-model", "max_tokens": 32, "temperature": 0.1},
    )

    assert result["id"] == "cmpl-test"
    assert "_promptfuse" in result
    event = _read_event(log_path)
    for field in (
        "request_id",
        "endpoint",
        "stream",
        "pipeline_total_ms",
        "vllm_ttft_ms",
        "vllm_total_ms",
        "token_delta",
        "canonical_id",
        "similarity",
        "request_meta",
    ):
        assert field in event


@pytest.mark.asyncio
async def test_stream_passthrough_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    server = _build_server(monkeypatch, tmp_path)
    log_path = Path(server.config.serving.log_path)
    streamed_chunks = [b"data: hello\n\n", b"data: [DONE]\n\n"]

    class _FakeStreamResponse:
        headers = {"content-type": "text/event-stream"}

        def raise_for_status(self):
            return None

        async def aiter_raw(self):
            for chunk in streamed_chunks:
                yield chunk

        async def aclose(self):
            return None

    class _FakeAsyncClient:
        def __init__(self, *_, **__):
            pass

        def build_request(self, *_args, **_kwargs):
            return object()

        async def send(self, *_args, **_kwargs):
            return _FakeStreamResponse()

        async def aclose(self):
            return None

    monkeypatch.setattr(server_module.httpx, "AsyncClient", _FakeAsyncClient)

    response = await server._forward_vllm(
        "/v1/chat/completions",
        {"stream": True},
        endpoint="chat",
        request_id="req-stream",
        processed=_DummyProcessed(),
        request_meta={"model": "test-model", "max_tokens": 32, "temperature": 0.1},
    )
    body = b"".join([chunk async for chunk in response.body_iterator])
    assert body == b"".join(streamed_chunks)

    event = _read_event(log_path)
    assert event["stream"] is True
    assert event["bytes_streamed"] == len(body)
