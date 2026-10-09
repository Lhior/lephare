"""Backend registry for optional LePHARE accelerators."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .kernels_ref import process_batch as process_numpy
from .maglib import MagLibTensors

ProcessFn = Callable[..., tuple[np.ndarray, np.ndarray]]

BACKENDS = ("lephare", "jax_hybrid", "cuda_hybrid", "jax_full", "cuda_full")


def jax_available() -> bool:
    try:
        import jax  # noqa: F401

        return True
    except ImportError:
        return False


def cuda_available() -> bool:
    try:
        from .cuda_backend import cuda_available as _ca

        return _ca()
    except Exception:
        return False


def available_backends() -> list[str]:
    out = ["lephare"]
    if jax_available():
        out.extend(["jax_hybrid", "jax_full"])
    if cuda_available():
        out.extend(["cuda_hybrid", "cuda_full"])
    return out


def get_backend(name: str) -> ProcessFn:
    name = str(name)
    if name in ("numpy", "ref"):
        return process_numpy
    if name in ("jax_hybrid", "jax_full"):
        if not jax_available():
            raise ImportError(
                f"backend={name} requires JAX. pip install 'lephare[jax]'"
            )
        from .jax_backend import process_batch_jax

        return process_batch_jax
    if name in ("cuda_hybrid", "cuda_full"):
        if not cuda_available():
            raise ImportError(
                f"backend={name} requires CuPy+GPU. pip install 'lephare[cuda]'"
            )
        from .cuda_backend import process_batch_cuda

        return process_batch_cuda
    if name == "lephare":
        raise ValueError("Use classic lp.process for backend='lephare'")
    raise ValueError(f"Unknown backend {name!r}; choose from {BACKENDS}")


def process_with_backend(
    backend: str,
    mags: np.ndarray,
    magerrs: np.ndarray,
    tensors: MagLibTensors,
    context=None,
    offsets=None,
    dtype=np.float64,
) -> tuple:
    """Run estimate; accel backends may return ``(pdfs, zgrid, zmode, zmean[, ancil])``."""
    fn = get_backend(backend)
    return fn(mags, magerrs, tensors, context=context, offsets=offsets, dtype=dtype)
