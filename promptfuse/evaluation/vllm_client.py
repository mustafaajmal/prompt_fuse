"""HTTP client for vLLM OpenAI-compatible API."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class VLLMResponse:
    content: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    raw: dict[str, Any]


class VLLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        timeout_s: float = 300.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(f"{self.base_url}/health")
                return r.status_code == 200
        except httpx.HTTPError:
            return False

    def chat(
        self,
        user_prompt: str,
        *,
        max_tokens: int = 64,
        temperature: float = 0.0,
        system_prompt: str | None = None,
    ) -> VLLMResponse:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        t0 = time.perf_counter()
        with httpx.Client(timeout=self.timeout_s) as client:
            response = client.post(f"{self.base_url}/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
        latency_ms = (time.perf_counter() - t0) * 1000

        choice = data["choices"][0]
        content = choice["message"]["content"]
        usage = data.get("usage", {})

        return VLLMResponse(
            content=content,
            latency_ms=latency_ms,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
            model=data.get("model", self.model),
            raw=data,
        )
