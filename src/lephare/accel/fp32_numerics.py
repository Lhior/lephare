"""Shared FP32 amplitude scaling so χ² stays finite (CUDA and JAX).

LePHARE AB fluxes are ~1e-29; squaring them underflows float32. A blind
``×1e28`` can overflow bright templates. Scale so max |flux| ~ 1e4 (χ²
invariant). Cap at 1e28.
"""

from __future__ import annotations

import numpy as np

FP32_FLUX_SCALE_CAP = 1.0e28
FP32_TARGET_ABS = 1.0e4
FP32_CLIP_ABS = 1.0e6
FP32_DCHI_MAX = 80.0


def fp32_amp_scale(flux: np.ndarray) -> float:
    """χ²-invariant scale so native FP32 squares stay finite.

    Uses max |flux| when the library is compact. If a handful of templates are
    1e40 brighter than the median (Diffsky/Cardinal maglibs), scale from the
    median so typical AB fluxes (~1e-29…1e-40) do not underflow when squared.
    """
    absf = np.abs(np.asarray(flux, dtype=np.float64).ravel())
    absf = absf[np.isfinite(absf) & (absf > 0.0)]
    if absf.size == 0:
        return 1.0
    typical = float(np.median(absf))
    mmax = float(np.max(absf))
    if not np.isfinite(typical) or typical <= 0.0:
        return 1.0
    ref = mmax if (mmax / typical) < 1.0e6 else typical
    return float(min(FP32_FLUX_SCALE_CAP, FP32_TARGET_ABS / ref))


def fp32_apply_scale(flux: np.ndarray, scale: float) -> np.ndarray:
    """Apply ``scale`` and clip so squares stay inside float32."""
    out = np.asarray(flux, dtype=np.float64) * float(scale)
    return np.clip(out, -FP32_CLIP_ABS, FP32_CLIP_ABS)


def mag_abs_dm_bounds(
    mag0: np.ndarray,
    z_lib: np.ndarray,
    types: np.ndarray,
    prior: np.ndarray,
    funz0: float,
    dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """MAG_ABS window as ``dm_min < dm < dm_max`` (same reject as C++ ``log10``)."""
    mag0 = np.asarray(mag0, dtype=np.float64)
    z_lib = np.asarray(z_lib, dtype=np.float64)
    types = np.asarray(types, dtype=np.int32)
    prior = np.asarray(prior, dtype=np.float64)
    mag0_eff = np.where(z_lib < 1e-10, mag0 - float(funz0), mag0)
    huge = np.float64(1.0e30 if np.dtype(dtype) == np.float32 else 1.0e300)
    dm_min = np.zeros(mag0.shape[0], dtype=np.float64)
    dm_max = np.full(mag0.shape[0], huge, dtype=np.float64)
    gal_on = bool(prior[0] < 0 and prior[1] < 0)
    qso_on = bool(prior[2] < 0 and prior[3] < 0)
    if gal_on:
        gal = types == 0
        dm_min[gal] = np.power(10.0, (mag0_eff[gal] - prior[1]) / 2.5)
        dm_max[gal] = np.power(10.0, (mag0_eff[gal] - prior[0]) / 2.5)
    if qso_on:
        qso = types == 1
        dm_min[qso] = np.power(10.0, (mag0_eff[qso] - prior[3]) / 2.5)
        dm_max[qso] = np.power(10.0, (mag0_eff[qso] - prior[2]) / 2.5)
    if np.dtype(dtype) == np.float32:
        dm_min = np.clip(dm_min, 0.0, 1.0e30)
        dm_max = np.clip(dm_max, 0.0, 1.0e30)
    return dm_min.astype(dtype, copy=False), dm_max.astype(dtype, copy=False)
