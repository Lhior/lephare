"""Optional accelerated LePHARE backends (NumPy reference, JAX, CUDA)."""

from .dispatch import BACKENDS, available_backends, get_backend
from .kernels_ref import prepare_photometry, process_batch
from .maglib import MagLibTensors, export_maglib_from_photoz, load_maglib, save_maglib

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
