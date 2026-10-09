"""Optional accelerated LePHARE backends (NumPy reference, JAX, CUDA)."""

from .maglib import MagLibTensors, export_maglib_from_photoz, load_maglib, save_maglib
from .kernels_ref import process_batch, prepare_photometry
from .dispatch import get_backend, available_backends, BACKENDS

__all__ = [
    "MagLibTensors",
    "export_maglib_from_photoz",
    "load_maglib",
    "save_maglib",
    "process_batch",
    "prepare_photometry",
    "get_backend",
    "available_backends",
    "BACKENDS",
]
