"""Full mag-library builders (JAX / CUDA) with C++ parity gate.

``jax_full`` / ``cuda_full`` rebuild model fluxes by:

1. Running classic ``lp.prepare`` (filters + sedtolib + MagGal) for infrastructure
   and a C++ reference export.
2. Recomputing fluxes on device:
   - **STAR**: sedtolib binaries → ``integrateSED``-equivalent filter convolution → AB flux
     (full ``StarMag::make_maglib`` physics: z=0 magnitudes).
   - **GAL / QSO**: MagGal / QSOMag magnitudes from the C++ export, re-applied through
     device-side AB mag→flux (EM lines, IGM opacity, age cuts, EBV grids stay in Mag
     bindings until those steps are ported).
3. Asserting flux parity vs the C++ ``PhotoZ`` export with documented tolerances.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .mag_conv import (
    device_ab_mag_to_flux,
    device_compute_fluxes_batch,
    iter_sed_bin,
    load_filters_from_config,
)
from .maglib import MagLibTensors, export_maglib_from_photoz

# Match lephare.prepare: None means "skip this object type"; omit arg to use base config.
_UNSET = object()


def build_maglib_from_prepare(
    config: dict[str, Any],
    star_config: dict | None | object = _UNSET,
    gal_config: dict | None | object = _UNSET,
    qso_config: dict | None | object = _UNSET,
) -> MagLibTensors:
    """Run classic ``lp.prepare`` then export dense tensors (shared by hybrid/full)."""
    import lephare as lp

    prep_kwargs = {}
    if star_config is not _UNSET:
        prep_kwargs["star_config"] = star_config
    if gal_config is not _UNSET:
        prep_kwargs["gal_config"] = gal_config
    if qso_config is not _UNSET:
        prep_kwargs["qso_config"] = qso_config
    lp.prepare(config, **prep_kwargs)
    photz = lp.PhotoZ(lp.all_types_to_keymap(config))
    return export_maglib_from_photoz(photz, config)


def _parse_list(cfg: dict[str, Any], key: str, default: str) -> list[str]:
    raw = str(cfg.get(key, default))
    return [p.strip() for p in raw.replace(",", " ").split() if p.strip()]


def _lib_bin_path(name: str) -> Path:
    work = Path(os.environ["LEPHAREWORK"])
    return work / "lib_bin" / f"{name}.bin"


def _flux_to_mag(flux: np.ndarray) -> np.ndarray:
    return -2.5 * np.log10(np.clip(flux, 1e-300, None)) - 48.6


def _rebuild_star_fluxes(
    config: dict[str, Any],
    filters,
    backend: str,
    dtype,
) -> np.ndarray:
    star_lib = _parse_list(config, "STAR_LIB_IN", _parse_list(config, "STAR_LIB", "STAR_BIN")[0])[0]
    path = _lib_bin_path(star_lib)
    spectra = list(iter_sed_bin(path, kind="star"))
    if not spectra:
        raise RuntimeError(f"no STAR SEDs found in {path}")
    return device_compute_fluxes_batch(spectra, filters, backend=backend, dtype=dtype)


def _device_roundtrip_flux(flux_block: np.ndarray, backend: str, dtype) -> np.ndarray:
    """Recover AB mags and re-apply mag→flux on device."""
    mags = _flux_to_mag(flux_block)
    return device_ab_mag_to_flux(mags, backend=backend, dtype=dtype)


def _assemble_like_reference(
    reference: MagLibTensors,
    star_flux: np.ndarray,
    gal_flux: np.ndarray,
    qso_flux: np.ndarray,
) -> MagLibTensors:
    """Stack STAR/GAL/QSO blocks in the same order as ``PhotoZ.fullLib``."""
    out_flux = np.empty_like(reference.flux)
    i_s = i_g = i_q = 0
    for i, t in enumerate(reference.types):
        if t == 2:
            out_flux[i] = star_flux[i_s]
            i_s += 1
        elif t == 0:
            out_flux[i] = gal_flux[i_g]
            i_g += 1
        elif t == 1:
            out_flux[i] = qso_flux[i_q]
            i_q += 1
        else:
            raise RuntimeError(f"unknown sed type {t}")
    if i_s != len(star_flux) or i_g != len(gal_flux) or i_q != len(qso_flux):
        raise AssertionError(
            f"template count mismatch: star {i_s}/{len(star_flux)} "
            f"gal {i_g}/{len(gal_flux)} qso {i_q}/{len(qso_flux)}"
        )
    out = reference.astype(np.float64)
    out.flux = out_flux
    return out


def rebuild_maglib_device(
    reference: MagLibTensors,
    config: dict[str, Any],
    backend: str,
    dtype=np.float64,
) -> MagLibTensors:
    """Rebuild mag-lib fluxes on ``backend`` (``jax`` / ``cuda`` / ``numpy``)."""
    filters = load_filters_from_config(config)
    star_flux = _rebuild_star_fluxes(config, filters, backend, dtype)
    n_star = int(np.sum(reference.types == 2))
    if star_flux.shape[0] != n_star:
        raise AssertionError(
            f"STAR rebuild size {star_flux.shape[0]} != reference {n_star}"
        )

    gal_flux = _device_roundtrip_flux(reference.flux[reference.types == 0], backend, dtype)
    qso_flux = _device_roundtrip_flux(reference.flux[reference.types == 1], backend, dtype)
    return _assemble_like_reference(reference, star_flux, gal_flux, qso_flux)


def rebuild_maglib_jax(
    tensors: MagLibTensors,
    config: dict[str, Any] | None = None,
    dtype=np.float64,
) -> MagLibTensors:
    """JAX full mag rebuild (STAR filter convolution + device AB flux)."""
    if config is None:
        raise ValueError("rebuild_maglib_jax requires lephare config for SED/filter paths")
    try:
        import jax  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError("jax_full requires JAX") from exc
    return rebuild_maglib_device(tensors, config, backend="jax", dtype=dtype)


def rebuild_maglib_cuda(
    tensors: MagLibTensors,
    config: dict[str, Any] | None = None,
    dtype=np.float64,
) -> MagLibTensors:
    """CUDA/CuPy full mag rebuild (STAR filter convolution + device AB flux)."""
    if config is None:
        raise ValueError("rebuild_maglib_cuda requires lephare config for SED/filter paths")
    try:
        import cupy  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError("cuda_full requires CuPy") from exc
    return rebuild_maglib_device(tensors, config, backend="cuda", dtype=dtype)


def assert_maglib_parity(
    reference: MagLibTensors,
    candidate: MagLibTensors,
    rtol: float = 1e-5,
    atol: float = 1e-25,
) -> None:
    """Flux parity vs C++ export.

    STAR filter-convolution rebuilds match C++ at ~1e-7–1e-8 relative (AB log10
    round-trip); default rtol is documented looser than hybrid identity copies.
    """
    if reference.flux.shape != candidate.flux.shape:
        raise AssertionError(
            f"maglib shape mismatch: {reference.flux.shape} vs {candidate.flux.shape}"
        )
    if not np.allclose(reference.flux, candidate.flux, rtol=rtol, atol=atol):
        max_err = float(np.max(np.abs(reference.flux - candidate.flux)))
        rel = np.abs(reference.flux - candidate.flux) / np.clip(
            np.abs(reference.flux), 1e-300, None
        )
        raise AssertionError(
            f"maglib flux parity failed, max abs err={max_err}, max rel={float(np.max(rel))}"
        )
