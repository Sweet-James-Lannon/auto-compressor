"""Centralized runtime settings with validation and effective-value reporting."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional

from utils import env_bool, env_int, get_effective_cpu_count

logger = logging.getLogger(__name__)

DEFAULT_PARALLEL_MAX_WORKERS = 8


@dataclass(frozen=True)
class SplitRuntimeSettings:
    split_minimize_parts: bool
    split_enable_binary_fallback: bool
    split_adaptive_max_attempts: int
    split_ultra_jpegq: int
    split_ultra_gap_pct: float


@dataclass(frozen=True)
class DeliveryRuntimeSettings:
    delivery_split_skip_inflation_pct: float


def _parse_positive_int(raw: Optional[str], *, name: str) -> Optional[int]:
    if raw is None:
        return None
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning("[settings] Invalid %s=%s; using default", name, raw)
        return None
    if value <= 0:
        logger.warning("[settings] Non-positive %s=%s; using default", name, raw)
        return None
    return value


def _auto_async_workers(effective_cpu: int) -> int:
    return max(1, min(3, effective_cpu // 2))


@lru_cache(maxsize=1)
def get_split_runtime_settings() -> SplitRuntimeSettings:
    split_minimize_parts = env_bool("SPLIT_MINIMIZE_PARTS", False)
    split_enable_binary_fallback = env_bool("SPLIT_ENABLE_BINARY_FALLBACK", False)

    raw_attempts = os.environ.get("SPLIT_ADAPTIVE_MAX_ATTEMPTS")
    if raw_attempts is None:
        split_adaptive_max_attempts = 3
    else:
        try:
            split_adaptive_max_attempts = max(1, min(10, int(raw_attempts)))
        except ValueError:
            split_adaptive_max_attempts = 3
            logger.warning("[settings] Invalid SPLIT_ADAPTIVE_MAX_ATTEMPTS=%s; using 3", raw_attempts)

    raw_jpegq = os.environ.get("SPLIT_ULTRA_JPEGQ")
    if raw_jpegq is None:
        split_ultra_jpegq = 50
    else:
        try:
            split_ultra_jpegq = max(20, min(90, int(raw_jpegq)))
        except ValueError:
            split_ultra_jpegq = 50
            logger.warning("[settings] Invalid SPLIT_ULTRA_JPEGQ=%s; using 50", raw_jpegq)

    raw_gap = os.environ.get("SPLIT_ULTRA_GAP_PCT")
    if raw_gap is None:
        split_ultra_gap_pct = 0.12
    else:
        try:
            split_ultra_gap_pct = max(0.0, min(0.5, float(raw_gap)))
        except ValueError:
            split_ultra_gap_pct = 0.12
            logger.warning("[settings] Invalid SPLIT_ULTRA_GAP_PCT=%s; using 0.12", raw_gap)

    return SplitRuntimeSettings(
        split_minimize_parts=split_minimize_parts,
        split_enable_binary_fallback=split_enable_binary_fallback,
        split_adaptive_max_attempts=split_adaptive_max_attempts,
        split_ultra_jpegq=split_ultra_jpegq,
        split_ultra_gap_pct=split_ultra_gap_pct,
    )


@lru_cache(maxsize=1)
def get_delivery_runtime_settings() -> DeliveryRuntimeSettings:
    raw = os.environ.get("DELIVERY_SPLIT_SKIP_INFLATION_PCT")
    if raw is None:
        pct = 0.70
    else:
        try:
            pct = float(raw)
        except ValueError:
            pct = 0.70
            logger.warning("[settings] Invalid DELIVERY_SPLIT_SKIP_INFLATION_PCT=%s; using 0.70", raw)
    pct = max(0.0, pct)
    return DeliveryRuntimeSettings(delivery_split_skip_inflation_pct=pct)


def resolve_parallel_compute_plan(effective_cpu: Optional[int] = None) -> Dict[str, Any]:
    """Build a CPU-safe parallel compute plan with explicit cap chain details."""
    if effective_cpu is None:
        effective_cpu = get_effective_cpu_count()
    effective_cpu = max(1, int(effective_cpu))

    env_parallel_raw = os.environ.get("PARALLEL_MAX_WORKERS")
    env_parallel_workers = _parse_positive_int(env_parallel_raw, name="PARALLEL_MAX_WORKERS")

    env_async_raw = os.environ.get("ASYNC_WORKERS")
    env_async_workers = _parse_positive_int(env_async_raw, name="ASYNC_WORKERS")

    async_workers = min(effective_cpu, env_async_workers or _auto_async_workers(effective_cpu))
    gs_threads = max(1, env_int("GS_NUM_RENDERING_THREADS", 1))

    requested_parallel_workers = (
        env_parallel_workers
        if env_parallel_workers is not None
        else min(effective_cpu, DEFAULT_PARALLEL_MAX_WORKERS)
    )

    total_parallel_budget = max(1, effective_cpu // gs_threads)
    per_job_parallel_budget = max(1, total_parallel_budget // max(1, async_workers))
    parallel_budget_enforce = env_bool("PARALLEL_BUDGET_ENFORCE", True)

    capped_reasons: list[str] = []

    cpu_thread_capped_workers = min(requested_parallel_workers, total_parallel_budget)
    if cpu_thread_capped_workers < requested_parallel_workers:
        capped_reasons.append("cpu_thread_budget")

    effective_parallel_workers = cpu_thread_capped_workers
    if parallel_budget_enforce:
        fairness_capped_workers = min(cpu_thread_capped_workers, per_job_parallel_budget)
        if fairness_capped_workers < cpu_thread_capped_workers:
            capped_reasons.append("per_job_fairness")
        effective_parallel_workers = fairness_capped_workers

    effective_parallel_workers = max(1, effective_parallel_workers)

    return {
        "effective_cpu": effective_cpu,
        "async_workers": async_workers,
        "gs_threads_per_worker": gs_threads,
        "env_parallel_workers": env_parallel_workers,
        "requested_parallel_workers": requested_parallel_workers,
        "configured_parallel_workers": requested_parallel_workers,
        "total_parallel_budget": total_parallel_budget,
        "per_job_parallel_budget": per_job_parallel_budget,
        "parallel_budget_enforce": parallel_budget_enforce,
        "effective_parallel_workers": effective_parallel_workers,
        "capped_by_budget": bool(capped_reasons),
        "capped_reasons": capped_reasons,
        "cap_chain": {
            "requested": requested_parallel_workers,
            "cpu_thread_budget_cap": total_parallel_budget,
            "after_cpu_thread_cap": cpu_thread_capped_workers,
            "per_job_budget_cap": per_job_parallel_budget,
            "effective": effective_parallel_workers,
        },
    }
