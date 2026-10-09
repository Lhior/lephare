"""RAIL-facing best-fit ancil from final χ² (matches onesource fit/interp/uncertaintiesMin)."""

from __future__ import annotations

import numpy as np

from .kernels_ref import HIGH_CHI2
from .maglib import MagLibTensors

INVALID_Z = -99.9
INVALID_INDEX = -99


def quadratic_extremum(
    x1: float, x2: float, x3: float, y1: float, y2: float, y3: float
) -> tuple[float, float]:
    """Port of ``PDF::quadratic_extremum`` (C++)."""
    x23 = x2 - x3
    x31 = x3 - x1
    x12 = x1 - x2
    delta = x23 * x1 * x1 + x31 * x2 * x2 + x12 * x3 * x3
    a = y1 * x23 + y2 * x31 + y3 * x12
    if a == 0.0 or delta == 0.0:
        raise ValueError("The three points are aligned")
    b = y1 * (x2 * x2 - x3 * x3) - y2 * (x1 * x1 - x3 * x3) + y3 * (x1 * x1 - x2 * x2)
    c = y1 * x2 * x3 * x23 + y2 * x1 * x3 * x31 + y3 * x1 * x2 * x12
    ba = b / a
    xmin = 0.5 * ba
    ymin = -0.25 * ba * b / delta + c / delta
    return float(xmin), float(ymin)


def improve_extremum_chi2(xaxis: np.ndarray, chi2: np.ndarray) -> tuple[float, float]:
    """Port of ``PDF::improve_extremum(true)`` for a 1-D χ² curve on ``xaxis``."""
    xaxis = np.asarray(xaxis, dtype=np.float64)
    chi2 = np.asarray(chi2, dtype=np.float64)
    if xaxis.size == 0:
        return INVALID_Z, HIGH_CHI2
    id_ = int(np.argmin(chi2))
    if id_ == 0 or id_ == xaxis.size - 1:
        return float(xaxis[id_]), float(chi2[id_])
    x = xaxis[id_ - 1 : id_ + 2]
    y = chi2[id_ - 1 : id_ + 2]
    if np.any(y >= HIGH_CHI2):
        return float(xaxis[id_]), float(chi2[id_])
    if abs(x[1] - x[0]) > 0.5 or abs(x[2] - x[1]) > 0.5:
        return float(x[1]), float(y[1])
    try:
        return quadratic_extremum(
            float(x[0]),
            float(x[1]),
            float(x[2]),
            float(y[0]),
            float(y[1]),
            float(y[2]),
        )
    except ValueError:
        return float(xaxis[id_]), float(chi2[id_])


def _type_argmins(chi2: np.ndarray, types: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-object argmin / min χ² for types 0,1,2. ``chi2`` is ``[N, Nt]`` or ``[Nt]``."""
    chi2 = np.asarray(chi2, dtype=np.float64)
    single = chi2.ndim == 1
    if single:
        chi2 = chi2[None, :]
    n_obj = chi2.shape[0]
    types = np.asarray(types, dtype=np.int32)
    indmin = np.full((n_obj, 3), INVALID_INDEX, dtype=np.int64)
    chimin = np.full((n_obj, 3), HIGH_CHI2, dtype=np.float64)
    for t in range(3):
        mask = types == t
        if not np.any(mask):
            continue
        vals = np.where(mask[None, :], chi2, HIGH_CHI2)
        idx = np.argmin(vals, axis=1)
        cmin = vals[np.arange(n_obj), idx]
        valid = cmin < HIGH_CHI2
        indmin[valid, t] = idx[valid]
        chimin[valid, t] = cmin[valid]
    if single:
        return indmin[0], chimin[0]
    return indmin, chimin


def min_chi2_per_zbin(
    chi2: np.ndarray,
    types: np.ndarray,
    z_lib: np.ndarray,
    gridz: np.ndarray,
    type_code: int,
) -> np.ndarray:
    """MIN_ZG / MIN_ZQ style: min χ² in each ``gridz`` bin for one object type."""
    chi2 = np.asarray(chi2, dtype=np.float64)
    single = chi2.ndim == 1
    if single:
        chi2 = chi2[None, :]
    n_obj = chi2.shape[0]
    gridz = np.asarray(gridz, dtype=np.float64)
    n_z = int(gridz.size)
    out = np.full((n_obj, n_z), HIGH_CHI2, dtype=np.float64)
    mask = np.asarray(types, dtype=np.int32) == int(type_code)
    idx = np.flatnonzero(mask)
    if idx.size == 0 or n_z == 0:
        return out[0] if single else out
    zg = np.asarray(z_lib, dtype=np.float64)[idx]
    zmin = float(gridz[0])
    inv = 1.0 / (gridz[1] - gridz[0]) if n_z > 1 else 1.0
    bin_idx = np.clip(np.rint((zg - zmin) * inv).astype(np.int64), 0, n_z - 1)
    sub = chi2[:, idx]
    for b in range(n_z):
        cols = bin_idx == b
        if np.any(cols):
            out[:, b] = np.min(sub[:, cols], axis=1)
    return out[0] if single else out


def empty_ancil(n_obj: int) -> dict[str, np.ndarray]:
    """Allocate default RAIL output_keys arrays."""
    return {
        "Z_BEST": np.full(n_obj, INVALID_Z, dtype=np.float64),
        "CHI_BEST": np.full(n_obj, HIGH_CHI2, dtype=np.float64),
        "ZQ_BEST": np.full(n_obj, INVALID_Z, dtype=np.float64),
        "CHI_QSO": np.full(n_obj, HIGH_CHI2, dtype=np.float64),
        "MOD_STAR": np.full(n_obj, float(INVALID_INDEX), dtype=np.float64),
        "CHI_STAR": np.full(n_obj, HIGH_CHI2, dtype=np.float64),
    }


def bestfit_ancil_from_chi2(
    chi2: np.ndarray,
    tensors: MagLibTensors,
    z_interp: bool | None = None,
) -> dict[str, np.ndarray]:
    """Build default RAIL ``output_keys`` from final per-object χ².

    Follows ``onesource::fit`` winners, optional ``interp`` via MIN_ZG/MIN_ZQ
    ``improve_extremum``, then ``uncertaintiesMin`` (``zgmin[0]`` / ``zqmin[0]``).
    """
    chi2 = np.asarray(chi2, dtype=np.float64)
    single = chi2.ndim == 1
    if single:
        chi2 = chi2[None, :]
    n_obj = chi2.shape[0]
    if z_interp is None:
        z_interp = bool(getattr(tensors, "z_interp", True))

    indmin, chimin = _type_argmins(chi2, tensors.types)
    z_lib = np.asarray(tensors.z, dtype=np.float64)
    nummod = getattr(tensors, "nummod", None)
    if nummod is None:
        nummod = np.full(z_lib.shape[0], INVALID_INDEX, dtype=np.int32)
    else:
        nummod = np.asarray(nummod, dtype=np.int32)

    z_best = np.full(n_obj, INVALID_Z, dtype=np.float64)
    zq_best = np.full(n_obj, INVALID_Z, dtype=np.float64)
    mod_star = np.full(n_obj, float(INVALID_INDEX), dtype=np.float64)

    for i in range(n_obj):
        if indmin[i, 0] >= 0:
            z_best[i] = z_lib[indmin[i, 0]]
        if indmin[i, 1] >= 0:
            zq_best[i] = z_lib[indmin[i, 1]]
        if indmin[i, 2] >= 0:
            mod_star[i] = float(nummod[indmin[i, 2]])

    if z_interp:
        min_zg = min_chi2_per_zbin(chi2, tensors.types, z_lib, tensors.gridz, 0)
        min_zq = min_chi2_per_zbin(chi2, tensors.types, z_lib, tensors.gridz, 1)
        gridz = np.asarray(tensors.gridz, dtype=np.float64)
        for i in range(n_obj):
            if indmin[i, 0] >= 0 and np.min(min_zg[i]) < HIGH_CHI2:
                z_best[i], _ = improve_extremum_chi2(gridz, min_zg[i])
            if indmin[i, 1] >= 0 and np.min(min_zq[i]) < HIGH_CHI2:
                zq_best[i], _ = improve_extremum_chi2(gridz, min_zq[i])

    out = {
        "Z_BEST": z_best,
        "CHI_BEST": chimin[:, 0].copy(),
        "ZQ_BEST": zq_best,
        "CHI_QSO": chimin[:, 1].copy(),
        "MOD_STAR": mod_star,
        "CHI_STAR": chimin[:, 2].copy(),
    }
    if single:
        return {k: np.asarray(v[0]) for k, v in out.items()}
    return out


def bestfit_ancil_from_type_mins(
    indmin: np.ndarray,
    chimin: np.ndarray,
    tensors: MagLibTensors,
    min_zg: np.ndarray | None = None,
    min_zq: np.ndarray | None = None,
    z_interp: bool | None = None,
) -> dict[str, np.ndarray]:
    """Build RAIL ``output_keys`` from per-type winners (+ optional MIN_ZG/ZQ).

    Same results as :func:`bestfit_ancil_from_chi2` when ``min_zg`` / ``min_zq``
    are the exact per-bin minima accumulated while streaming templates.
    """
    indmin = np.asarray(indmin, dtype=np.int64)
    chimin = np.asarray(chimin, dtype=np.float64)
    single = indmin.ndim == 1
    if single:
        indmin = indmin[None, :]
        chimin = chimin[None, :]
        if min_zg is not None:
            min_zg = np.asarray(min_zg, dtype=np.float64)[None, :]
        if min_zq is not None:
            min_zq = np.asarray(min_zq, dtype=np.float64)[None, :]
    n_obj = indmin.shape[0]
    if z_interp is None:
        z_interp = bool(getattr(tensors, "z_interp", True))

    z_lib = np.asarray(tensors.z, dtype=np.float64)
    nummod = getattr(tensors, "nummod", None)
    if nummod is None:
        nummod = np.full(z_lib.shape[0], INVALID_INDEX, dtype=np.int32)
    else:
        nummod = np.asarray(nummod, dtype=np.int32)

    z_best = np.full(n_obj, INVALID_Z, dtype=np.float64)
    zq_best = np.full(n_obj, INVALID_Z, dtype=np.float64)
    mod_star = np.full(n_obj, float(INVALID_INDEX), dtype=np.float64)

    for i in range(n_obj):
        if indmin[i, 0] >= 0:
            z_best[i] = z_lib[indmin[i, 0]]
        if indmin[i, 1] >= 0:
            zq_best[i] = z_lib[indmin[i, 1]]
        if indmin[i, 2] >= 0:
            mod_star[i] = float(nummod[indmin[i, 2]])

    if z_interp:
        gridz = np.asarray(tensors.gridz, dtype=np.float64)
        if min_zg is not None:
            min_zg = np.asarray(min_zg, dtype=np.float64)
            for i in range(n_obj):
                if indmin[i, 0] >= 0 and np.min(min_zg[i]) < HIGH_CHI2:
                    z_best[i], _ = improve_extremum_chi2(gridz, min_zg[i])
        if min_zq is not None:
            min_zq = np.asarray(min_zq, dtype=np.float64)
            for i in range(n_obj):
                if indmin[i, 1] >= 0 and np.min(min_zq[i]) < HIGH_CHI2:
                    zq_best[i], _ = improve_extremum_chi2(gridz, min_zq[i])

    out = {
        "Z_BEST": z_best,
        "CHI_BEST": chimin[:, 0].copy(),
        "ZQ_BEST": zq_best,
        "CHI_QSO": chimin[:, 1].copy(),
        "MOD_STAR": mod_star,
        "CHI_STAR": chimin[:, 2].copy(),
    }
    if single:
        return {k: np.asarray(v[0]) for k, v in out.items()}
    return out


def merge_ancil(
    dest: dict[str, np.ndarray],
    src: dict[str, np.ndarray],
    indices: np.ndarray,
) -> None:
    """Scatter ``src`` rows into ``dest`` at ``indices``."""
    indices = np.asarray(indices, dtype=np.int64)
    for k, arr in src.items():
        dest[k][indices] = np.asarray(arr, dtype=np.float64)
