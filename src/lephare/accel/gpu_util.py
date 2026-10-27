"""Shared GPU batch sizing helpers for accelerated χ²."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


DEFAULT_MEM_FRACTION = 0.8

# Last object-chunk sizes / OOM retries, for benchmarks (``gpu_util.STATS``).
STATS: dict = {}


def chunk_mem_fraction(default: float | None = None) -> float:
    """Fraction of free VRAM an object chunk may use (env ``LEPHARE_ACCEL_CHI2_MEM_FRACTION``)."""
    env = os.environ.get("LEPHARE_ACCEL_CHI2_MEM_FRACTION")
    if env:
        return float(env)
    return DEFAULT_MEM_FRACTION if default is None else float(default)


def chunk_max_objects(default: int | None = None) -> int | None:
    """Object-chunk cap (env ``LEPHARE_ACCEL_CHI2_MAX_CHUNK``; ``0`` = uncapped)."""
    env = os.environ.get("LEPHARE_ACCEL_CHI2_MAX_CHUNK")
    if env is not None and env.strip() != "":
        n = int(env)
        return None if n <= 0 else n
    return default


def suggest_obj_chunk(
    per_obj_bytes: int,
    mem_fraction: float | None = None,
    min_chunk: int = 256,
    max_chunk: int | None = None,
    reserve_bytes: int = 0,
) -> int:
    """Largest object chunk whose ``per_obj_bytes`` device footprint fits the free VRAM.

    ``mem_fraction`` / ``max_chunk`` default to the env knobs above (0.8, uncapped).
    """
    frac = chunk_mem_fraction(mem_fraction)
    cap = chunk_max_objects(max_chunk)
    usable = int(_gpu_free_bytes() * frac) - int(reserve_bytes)
    n = max(min_chunk, usable // max(1, int(per_obj_bytes)))
    if cap is not None:
        n = min(n, cap)
    return int(n)


def suggest_chi2_chunk_size(
    n_templates: int,
    dtype=np.float64,
    mem_fraction: float | None = None,
    min_chunk: int = 256,
    max_chunk: int | None = None,
) -> int:
    """Largest object chunk that fits ``mem_fraction`` of usable GPU memory for chi2.

    Budget assumes resident mag-lib + fused-tile temporaries. Primary buffer is
    ``chi2[N, N_t]`` (~2–3× for intermediates / double-buffer headroom). Override
    with env ``LEPHARE_ACCEL_CHI2_CHUNK`` for forced sizing; ``mem_fraction`` and
    ``max_chunk`` default to ``LEPHARE_ACCEL_CHI2_MEM_FRACTION`` (0.8) and
    ``LEPHARE_ACCEL_CHI2_MAX_CHUNK`` (uncapped).
    """
    forced = os.environ.get("LEPHARE_ACCEL_CHI2_CHUNK")
    if forced:
        return max(min_chunk, int(forced))

    nbytes = 8 if np.dtype(dtype) == np.float64 else 4
    # chi2 + fused intermediates (photometry/PDF/exp) + double-buffer slack
    per_obj = int(n_templates * nbytes * 3.0)
    if per_obj <= 0:
        return min_chunk
    return suggest_obj_chunk(per_obj, mem_fraction, min_chunk, max_chunk)


def is_gpu_oom(exc: BaseException) -> bool:
    """CuPy / CUDA / XLA out-of-memory errors."""
    name = type(exc).__name__
    if name in ("OutOfMemoryError", "MemoryError"):
        return True
    msg = str(exc)
    return "out of memory" in msg.lower() or "RESOURCE_EXHAUSTED" in msg


def free_gpu_pools() -> None:
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def run_with_oom_backoff(fn, n_obj: int, chunk: int, min_chunk: int = 64, tag: str = "chunk") -> int:
    """Call ``fn(start, end)`` over ``[0, n_obj)`` in ``chunk``-row slices.

    On a GPU out-of-memory error the pools are freed and the same slice is retried at
    half the size (down to ``min_chunk``). Returns the last chunk size used; records it
    and the retry count in ``STATS[tag]``.
    """
    chunk = max(1, min(int(chunk), max(int(n_obj), 1)))
    retries = 0
    start = 0
    while start < n_obj:
        end = min(start + chunk, n_obj)
        try:
            fn(start, end)
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is an OOM
            if not is_gpu_oom(exc) or chunk <= min_chunk:
                raise
            free_gpu_pools()
            chunk = max(min_chunk, chunk // 2)
            retries += 1
            continue
        start = end
    STATS[tag] = {"chunk": int(chunk), "oom_retries": int(retries), "n_obj": int(n_obj)}
    return chunk


def suggest_cpu_obj_chunk(n_obj: int, default: int = 32) -> int:
    """JAX CPU object batch. A 4k-wide tile is a skinny ``K=9`` GEMM (~2 threads).

    Override with ``LEPHARE_ACCEL_CPU_OBJ_CHUNK``.
    """
    n_obj = max(1, int(n_obj))
    forced = os.environ.get("LEPHARE_ACCEL_CPU_OBJ_CHUNK")
    if forced:
        return max(1, min(int(forced), n_obj))
    return max(1, min(int(default), n_obj))


def suggest_template_tile_size(
    n_obj: int,
    n_templates: int,
    dtype=np.float64,
    mem_fraction: float = 0.55,
    min_tile: int = 2048,
    max_tile: int | None = 65536,
    host_memory: bool = False,
) -> int:
    """Largest template tile ``T`` so ``chi2[n_obj, T]`` fits in VRAM.

    Used when ``chi2[N, N_t]`` does not fit but ``N`` is modest. Override with
    ``LEPHARE_ACCEL_TEMPLATE_TILE``. Cap keeps GEMMs in a sweet spot on ~8 GB GPUs.
    CPU (``host_memory=True``) uses RAM and allows larger tiles.
    """
    forced = os.environ.get("LEPHARE_ACCEL_TEMPLATE_TILE")
    if forced:
        return max(min_tile, int(forced))

    nbytes = 8 if np.dtype(dtype) == np.float64 else 4
    free_bytes = _host_free_bytes() if host_memory else _gpu_free_bytes()
    usable = int(free_bytes * mem_fraction)
    if host_memory and max_tile is None:
        max_tile = 262_144
    # chi2[N,T] + dm-ish intermediates (~2×) + photometry/pdf slack
    per_t = max(1, int(n_obj) * nbytes * 3)
    t = max(min_tile, usable // per_t)
    t = min(t, int(n_templates))
    if max_tile is not None:
        t = min(t, int(max_tile))
    # Prefer multiples of 1024 for XLA padding friendliness
    if t >= 1024:
        t = (t // 1024) * 1024
    return max(min_tile, int(t))


def choose_tiling_axis(
    n_obj: int,
    n_templates: int,
    dtype=np.float64,
    mem_fraction: float | None = None,
) -> tuple[str, int]:
    """Pick ``(\"object\", chunk)`` or ``(\"template\", T)`` for FP64 χ².

    Prefer template tiling when a full ``chi2[N, Nt]`` does not fit but a
    wide template tile for all objects does. Large-``N`` catalogs keep object
    tiling so ``chi2[N, T]`` never exceeds VRAM.
    """
    forced_axis = os.environ.get("LEPHARE_ACCEL_TILE_AXIS", "").strip().lower()
    nbytes = 8 if np.dtype(dtype) == np.float64 else 4
    mem_fraction = chunk_mem_fraction(mem_fraction)
    free_bytes = _gpu_free_bytes()
    budget = int(free_bytes * mem_fraction)
    full_bytes = int(n_obj) * int(n_templates) * nbytes * 3

    if forced_axis in ("object", "obj"):
        return "object", suggest_chi2_chunk_size(n_templates, dtype, mem_fraction)
    if forced_axis in ("template", "tmpl", "sed"):
        t = suggest_template_tile_size(n_obj, n_templates, dtype, mem_fraction)
        return "template", t

    if full_bytes <= budget:
        # Whole χ² fits — single object tile of all objects
        return "object", max(int(n_obj), 1)

    t = suggest_template_tile_size(n_obj, n_templates, dtype, mem_fraction)
    tile_bytes = int(n_obj) * int(t) * nbytes * 3
    # Template path needs all objects resident; bail to object tiles if not
    if tile_bytes <= budget and t >= 2048 and n_obj <= 50_000:
        return "template", t

    return "object", suggest_chi2_chunk_size(n_templates, dtype, mem_fraction)


def _gpu_free_bytes() -> int:
    """Usable bytes for a new chi2 chunk (accounts for runtime already resident)."""
    try:
        import cupy as cp

        free, total = cp.cuda.runtime.memGetInfo()
        pool_spare = 0
        try:
            pool = cp.get_default_memory_pool()
            pool_spare = max(0, int(pool.total_bytes()) - int(pool.used_bytes()))
        except Exception:
            pool_spare = 0
        # Runtime "free" excludes both live JAX buffers and unused CuPy pool
        # blocks. Count unused pool as reusable; do not pretend half the card
        # is free when another runtime (JAX) already holds it.
        return int(max(free + pool_spare, 256 * 1024 * 1024))
    except Exception:
        pass
    try:
        import subprocess

        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip().splitlines()[0]
        free_mib, total_mib = [float(x.strip()) for x in out.split(",")]
        free_b = int(free_mib * 1024 * 1024)
        total_b = int(total_mib * 1024 * 1024)
        return max(free_b, int(total_b * 0.50))
    except Exception:
        return 4 * 1024**3


def _host_free_bytes() -> int:
    """Host RAM budget for CPU template tiles."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 16 * 1024**3
