"""Demo experiment: raw vLLM vs compression-only vs full PromptFuse."""

from __future__ import annotations

import json
import logging
import shutil
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from promptfuse.config import PromptFuseConfig, Settings
from promptfuse.evaluation.vllm_client import VLLMClient
from promptfuse.pipeline import PromptFusePipeline

logger = logging.getLogger(__name__)

DemoMode = Literal["raw_vllm", "compress_only", "promptfuse_full", "pipeline_only"]

MODE_LABELS = {
    "raw_vllm": "Raw + vLLM prefix cache",
    "compress_only": "Compress only + vLLM",
    "promptfuse_full": "Full PromptFuse + vLLM",
    "pipeline_only": "Pipeline only (no vLLM)",
}


@dataclass
class DemoCluster:
    cluster: str
    prompts: list[str]


@dataclass
class RequestRecord:
    mode: str
    cluster: str
    paraphrase_index: int
    run_index: int
    raw_prompt: str
    final_prompt: str
    unifier_hit: bool
    unifier_similarity: float | None
    token_reduction: float
    original_tokens: int | None
    final_tokens: int | None
    pipeline_ms: float
    vllm_latency_ms: float | None
    vllm_prompt_tokens: int | None
    vllm_completion_tokens: int | None
    output_preview: str | None = None


@dataclass
class ModeSummary:
    mode: str
    label: str
    total_requests: int
    avg_token_reduction: float
    unifier_hit_rate: float
    avg_pipeline_ms: float
    avg_vllm_latency_ms: float | None
    avg_vllm_latency_run2_ms: float | None
    vllm_speedup_run2_vs_run1: float | None
    unique_final_prompts_per_cluster: float | None = None


def load_demo_workload(path: Path) -> list[DemoCluster]:
    with open(path) as f:
        data = json.load(f)
    return [DemoCluster(cluster=c["cluster"], prompts=c["prompts"]) for c in data]


def _reset_inventory(inventory_path: Path) -> None:
    if inventory_path.exists():
        shutil.rmtree(inventory_path)
    inventory_path.mkdir(parents=True, exist_ok=True)


def _build_pipeline(
    config: PromptFuseConfig,
    mode: DemoMode,
    *,
    lazy_load: bool = False,
) -> PromptFusePipeline:
    if mode == "raw_vllm":
        return PromptFusePipeline(
            config,
            lazy_load=lazy_load,
            enable_compression=False,
            enable_unification=False,
        )
    if mode == "compress_only":
        return PromptFusePipeline(
            config,
            lazy_load=lazy_load,
            enable_compression=True,
            enable_unification=False,
        )
    return PromptFusePipeline(
        config,
        lazy_load=lazy_load,
        enable_compression=True,
        enable_unification=True,
    )


def run_mode(
    mode: DemoMode,
    workload: list[DemoCluster],
    config: PromptFuseConfig,
    *,
    compression_ratio: float = 0.40,
    vllm_client: VLLMClient | None = None,
    max_tokens: int = 64,
    repeats: int = 2,
    reset_inventory: bool = True,
    warm_inventory: bool = False,
) -> list[RequestRecord]:
    """Run all clusters for one experiment mode."""
    records: list[RequestRecord] = []

    if mode == "promptfuse_full" and config.unifier:
        inv_path = Path(config.unifier.inventory_path)
        if reset_inventory and not warm_inventory:
            _reset_inventory(inv_path)
        elif warm_inventory and not inv_path.exists():
            logger.warning("warm_inventory requested but %s missing — run warm_demo_inventory.py", inv_path)

    pipeline = _build_pipeline(config, mode, lazy_load=False)

    for cluster in workload:
        for p_idx, raw_prompt in enumerate(cluster.prompts):
            for run_idx in range(1, repeats + 1):
                processed = pipeline.process(raw_prompt, compression_ratio=compression_ratio)

                unifier_sim = None
                if processed.unification:
                    unifier_sim = processed.unification.similarity

                record = RequestRecord(
                    mode=mode,
                    cluster=cluster.cluster,
                    paraphrase_index=p_idx,
                    run_index=run_idx,
                    raw_prompt=raw_prompt,
                    final_prompt=processed.final_prompt,
                    unifier_hit=processed.cache_hit,
                    unifier_similarity=unifier_sim,
                    token_reduction=processed.token_reduction,
                    original_tokens=(
                        processed.compression.original_tokens if processed.compression else None
                    ),
                    final_tokens=(
                        processed.compression.compressed_tokens
                        if processed.compression
                        else (
                            len(processed.final_prompt.split())
                            if processed.final_prompt
                            else None
                        )
                    ),
                    pipeline_ms=processed.total_ms,
                    vllm_latency_ms=None,
                    vllm_prompt_tokens=None,
                    vllm_completion_tokens=None,
                )

                if vllm_client is not None and mode != "pipeline_only":
                    try:
                        resp = vllm_client.chat(
                            processed.final_prompt if mode != "raw_vllm" else raw_prompt,
                            max_tokens=max_tokens,
                            temperature=0.0,
                        )
                        record.vllm_latency_ms = resp.latency_ms
                        record.vllm_prompt_tokens = resp.prompt_tokens
                        record.vllm_completion_tokens = resp.completion_tokens
                        record.output_preview = resp.content[:200]
                    except Exception as exc:
                        logger.error("vLLM request failed: %s", exc)
                        record.output_preview = f"ERROR: {exc}"

                records.append(record)
                logger.info(
                    "[%s] %s p%d run%d pipeline=%.1fms vllm=%s hit=%s",
                    mode,
                    cluster.cluster,
                    p_idx,
                    run_idx,
                    record.pipeline_ms,
                    f"{record.vllm_latency_ms:.1f}ms" if record.vllm_latency_ms else "n/a",
                    record.unifier_hit,
                )

    if mode == "promptfuse_full" and pipeline.unifier:
        pipeline.unifier.save_inventory()

    return records


def summarize_mode(records: list[RequestRecord]) -> ModeSummary:
    mode = records[0].mode if records else "unknown"
    reductions = [r.token_reduction for r in records if r.token_reduction > 0]
    pipeline_lat = [r.pipeline_ms for r in records]
    vllm_lat = [r.vllm_latency_ms for r in records if r.vllm_latency_ms is not None]
    hits = sum(1 for r in records if r.unifier_hit)

    run1 = [r.vllm_latency_ms for r in records if r.run_index == 1 and r.vllm_latency_ms]
    run2 = [r.vllm_latency_ms for r in records if r.run_index == 2 and r.vllm_latency_ms]
    speedup = None
    if run1 and run2 and statistics.mean(run1) > 0:
        speedup = statistics.mean(run1) / statistics.mean(run2)

    clusters: dict[str, set[str]] = {}
    for r in records:
        clusters.setdefault(r.cluster, set()).add(r.final_prompt)
    unique_per_cluster = (
        statistics.mean([len(v) for v in clusters.values()]) if clusters else None
    )

    return ModeSummary(
        mode=mode,
        label=MODE_LABELS.get(mode, mode),
        total_requests=len(records),
        avg_token_reduction=statistics.mean(reductions) if reductions else 0.0,
        unifier_hit_rate=hits / len(records) if records else 0.0,
        avg_pipeline_ms=statistics.mean(pipeline_lat) if pipeline_lat else 0.0,
        avg_vllm_latency_ms=statistics.mean(vllm_lat) if vllm_lat else None,
        avg_vllm_latency_run2_ms=statistics.mean(run2) if run2 else None,
        vllm_speedup_run2_vs_run1=speedup,
        unique_final_prompts_per_cluster=unique_per_cluster,
    )


def run_full_experiment(
    workload_path: Path,
    *,
    config_path: Path | None = None,
    modes: list[DemoMode] | None = None,
    compression_ratio: float = 0.40,
    vllm_base_url: str = "http://localhost:8000",
    vllm_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    max_tokens: int = 64,
    use_vllm: bool = True,
    output_dir: Path = Path("results"),
    warm_inventory: bool = False,
) -> dict[str, Any]:
    settings = Settings(config_path=config_path) if config_path else Settings()
    config = settings.load()

    workload = load_demo_workload(workload_path)
    if modes is None:
        modes = (
            ["raw_vllm", "compress_only", "promptfuse_full"]
            if use_vllm
            else ["pipeline_only", "compress_only", "promptfuse_full"]
        )

    vllm_client = None
    if use_vllm:
        vllm_client = VLLMClient(base_url=vllm_base_url, model=vllm_model)
        if not vllm_client.health_check():
            logger.warning(
                "vLLM not reachable at %s — falling back to pipeline-only metrics",
                vllm_base_url,
            )
            vllm_client = None
            modes = [m for m in modes if m not in ("raw_vllm", "compress_only", "promptfuse_full")]
            modes.extend(["compress_only", "promptfuse_full"])
            if "pipeline_only" not in modes:
                modes.insert(0, "pipeline_only")

    all_records: list[RequestRecord] = []
    summaries: list[ModeSummary] = []

    t0 = time.perf_counter()
    for mode in modes:
        logger.info("=== Running mode: %s ===", mode)
        effective_vllm = vllm_client if mode != "pipeline_only" else None
        records = run_mode(
            mode,  # type: ignore[arg-type]
            workload,
            config,
            compression_ratio=compression_ratio,
            vllm_client=effective_vllm,
            max_tokens=max_tokens,
            reset_inventory=not warm_inventory,
            warm_inventory=warm_inventory,
        )
        all_records.extend(records)
        summaries.append(summarize_mode(records))

    elapsed = time.perf_counter() - t0

    full = next((s for s in summaries if s.mode == "promptfuse_full"), None)
    compress = next((s for s in summaries if s.mode == "compress_only"), None)
    raw = next((s for s in summaries if s.mode == "raw_vllm"), None)

    comparison: dict[str, Any] = {
        "experiment": "promptfuse_demo",
        "workload": str(workload_path),
        "compression_ratio": compression_ratio,
        "wall_time_s": round(elapsed, 2),
        "use_vllm": vllm_client is not None,
        "summaries": [asdict(s) for s in summaries],
        "records": [asdict(r) for r in all_records],
    }

    if full and compress:
        comparison["key_findings"] = {
            "token_reduction_full": round(full.avg_token_reduction, 4),
            "unifier_hit_rate": round(full.unifier_hit_rate, 4),
            "unique_prompts_raw": raw.unique_final_prompts_per_cluster if raw else None,
            "unique_prompts_full": full.unique_final_prompts_per_cluster,
            "unification_reduces_prefix_diversity": (
                raw
                and full.unique_final_prompts_per_cluster is not None
                and raw.unique_final_prompts_per_cluster is not None
                and full.unique_final_prompts_per_cluster
                < raw.unique_final_prompts_per_cluster
            ),
            "vllm_speedup_full_run2": full.vllm_speedup_run2_vs_run1,
            "vllm_speedup_compress_run2": compress.vllm_speedup_run2_vs_run1,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "demo_metrics.json"
    with open(out_file, "w") as f:
        json.dump(comparison, f, indent=2)

    comparison["_output_file"] = str(out_file)
    return comparison


def print_summary_table(comparison: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("PromptFuse Demo Experiment Results")
    print("=" * 72)
    print(f"{'Mode':<22} {'Tok↓':>8} {'Unifier%':>10} {'Pipe ms':>10} {'vLLM ms':>10} {'Run2↓':>8}")
    print("-" * 72)
    for s in comparison.get("summaries", []):
        print(
            f"{s['label'][:22]:<22} "
            f"{s['avg_token_reduction']:>7.1%} "
            f"{s['unifier_hit_rate']:>9.1%} "
            f"{s['avg_pipeline_ms']:>9.1f} "
            f"{(s['avg_vllm_latency_ms'] or 0):>9.1f} "
            f"{(s['vllm_speedup_run2_vs_run1'] or 0):>7.2f}x"
        )
    print("-" * 72)
    kf = comparison.get("key_findings", {})
    if kf:
        print("\nKey findings:")
        print(f"  • Unifier hit rate: {kf.get('unifier_hit_rate', 0):.1%}")
        print(f"  • Token reduction (full): {kf.get('token_reduction_full', 0):.1%}")
        if kf.get("unification_reduces_prefix_diversity"):
            print(
                f"  • Prefix diversity: raw={kf.get('unique_prompts_raw'):.1f} "
                f"→ full={kf.get('unique_prompts_full'):.1f} unique prompts/cluster"
            )
        if kf.get("vllm_speedup_full_run2"):
            print(f"  • vLLM run-2 speedup (full): {kf['vllm_speedup_full_run2']:.2f}x")
    print("=" * 72 + "\n")
