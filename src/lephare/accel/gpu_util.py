"""Shared GPU batch sizing helpers for accelerated χ²."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def suggest_chi2_chunk_size(
    n_templates: int,
    dtype=np.float64,
    mem_fraction: float = 0.55,
    min_chunk: int = 256,
    max_chunk: int | None = 100_000,
) -> int:
    """Largest object chunk that fits ~mem_fraction of usable GPU memory for chi2.

    Budget assumes resident mag-lib + fused-tile temporaries. Primary buffer is
    ``chi2[N, N_t]`` (~2–3× for intermediates / double-buffer headroom). Override
    with env ``LEPHARE_ACCEL_CHI2_CHUNK`` for forced sizing.
    """
    forced = os.environ.get("LEPHARE_ACCEL_CHI2_CHUNK")
    if forced:
        return max(min_chunk, int(forced))

    nbytes = 8 if np.dtype(dtype) == np.float64 else 4
    free_bytes = _gpu_free_bytes()
    # chi2 + fused intermediates (photometry/PDF/exp) + double-buffer slack
    per_obj = int(n_templates * nbytes * 3.0)
    if per_obj <= 0:
        return min_chunk
    usable = int(free_bytes * mem_fraction)
    n = max(min_chunk, usable // per_obj)
    if max_chunk is not None:
        n = min(n, max_chunk)
    return int(n)


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
    mem_fraction: float = 0.55,
) -> tuple[str, int]:
    """Pick ``(\"object\", chunk)`` or ``(\"template\", T)`` for FP64 χ².

    Prefer template tiling when a full ``chi2[N, Nt]`` does not fit but a
    wide template tile for all objects does. Large-``N`` catalogs keep object
    tiling so ``chi2[N, T]`` never exceeds VRAM.
    """
    forced_axis = os.environ.get("LEPHARE_ACCEL_TILE_AXIS", "").strip().lower()
    nbytes = 8 if np.dtype(dtype) == np.float64 else 4
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
