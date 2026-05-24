"""Configuration loading for PromptFuse."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class CompressorConfig(BaseModel):
    proxy_model: str = "meta-llama/Llama-3.2-1B"
    compression_ratio: float = 0.40
    device: str = "auto"
    max_length: int = 4096


class UnifierConfig(BaseModel):
    encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    fine_tuned_encoder: str | None = None
    similarity_threshold: float = 0.85
    inventory_path: str = "data/canonical_inventory"
    embedding_dim: int = 384
    faiss_nprobe: int = 16


class ServingConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    vllm_base_url: str = "http://localhost:8000"
    vllm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    enable_compression: bool = True
    enable_unification: bool = True
    log_path: str = "logs/promptfuse.jsonl"


class EvaluationConfig(BaseModel):
    target_llm: str = "meta-llama/Llama-3.1-8B-Instruct"
    compression_ratios: list[float] = Field(default_factory=lambda: [0.25, 0.40, 0.55])
    rouge_l_threshold: float = 0.85
    latency_p99_ms: float = 50.0
    cache_hit_multiplier: float = 2.0
    speedup_target: float = 1.5


class PromptFuseConfig(BaseModel):
    compressor: CompressorConfig = Field(default_factory=CompressorConfig)
    unifier: UnifierConfig = Field(default_factory=UnifierConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PromptFuseConfig:
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        return cls.model_validate(data)


class Settings(BaseSettings):
    config_path: Path = Path("configs/default.yaml")

    def load(self) -> PromptFuseConfig:
        if self.config_path.exists():
            return PromptFuseConfig.from_yaml(self.config_path)
        return PromptFuseConfig()
