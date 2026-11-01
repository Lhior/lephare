"""Absolute magnitudes for the accelerated backends (LePhare 1.0 ports).

Library level (``photoz_lib.cpp``): ``bestFilter`` / ``minimizekcolor`` pick the
observed band used for each rest-frame band per redshift, ``maxkcolor`` the
k-colour spread used as the error. Per object (``onesource.cpp``):
``interp_lib`` interpolates model mags / k-corrections of the best GAL template
to ``Z_BEST`` and ``absmag`` combines them with the observed photometry.
"""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from typing import Any

import numpy as np

from .kernels_ref import prepare_photometry
from .maglib import MagLibTensors
from .physical import (
    cosmology_from_config,
    dist_mod,
    keyword_doubles,
    keyword_ints,
)

HIGH_MAG = 1000.0
_LIB_CACHE: dict = {}


def _flux2mag(flux):
    with np.errstate(divide="ignore", invalid="ignore"):
        return -2.5 * np.log10(flux) - 48.6


def indexz(red: np.ndarray, gridz: np.ndarray) -> np.ndarray:
    """``indexz``: nearest ``gridz`` index, ties go to the upper neighbour."""
    red = np.asarray(red, dtype=np.float64)
    up = np.clip(np.searchsorted(gridz, red, side="left"), 1, gridz.size - 1)
    low = up - 1
    out = np.where(np.abs(gridz[up] - red) <= np.abs(red - gridz[low]), up, low)
    out = np.where(red <= gridz[0], 0, out)
    return np.where(red >= gridz[-1], gridz.size - 1, out)


def _context_bit(context: int, k: int) -> bool:
    return bool((int(context) >> k) & 1)


def _kcolor_extremes(indz, kcolor, n_z):
    """Per-(z bin, column) min / max of ``kcolor`` [G, C] (1e4 / -1e4 when empty)."""
    n_c = kcolor.shape[1]
    lo = np.full((n_z, n_c), 1.0e4)
    hi = np.full((n_z, n_c), -1.0e4)
    for c in range(n_c):
        np.minimum.at(lo[:, c], indz, kcolor[:, c])
        np.maximum.at(hi[:, c], indz, kcolor[:, c])
    return lo, hi


def library_filters(tensors: MagLibTensors, config: Mapping[str, Any]):
    """``bestFilter`` + ``maxkcolor`` → ``(best_flt [N_z, N_b] int, maxkcol [N_z, N_b])``."""
    n_b = int(tensors.n_bands)
    gridz = np.asarray(tensors.gridz, dtype=np.float64)
    n_z = gridz.size
    method = keyword_ints(config, "MABS_METHOD", "0")[0]
    bapp = [b - 1 for b in keyword_ints(config, "MABS_REF", "1")]
    bapp_op = [b - 1 for b in keyword_ints(config, "MABS_FILT", "1")]
    nbin = len(bapp_op)
    zbmin = keyword_doubles(config, "MABS_ZBIN", "0", nbin + 1)[:-1]
    zbmax = keyword_doubles(config, "MABS_ZBIN", "6", nbin + 1)[1:]
    magabscont = keyword_ints(config, "MABS_CONTEXT", "0")
    # identity, not address: a freed library's address can be reused by a new one
    key = (id(tensors), method, tuple(bapp), tuple(bapp_op), tuple(zbmin), tuple(zbmax), tuple(magabscont))
    hit = _LIB_CACHE.get(key)
    if hit is not None and hit[0]() is tensors:
        return hit[1]

    gal = np.flatnonzero(np.asarray(tensors.types) == 0)
    indz = indexz(np.asarray(tensors.z)[gal], gridz)
    mag = np.asarray(tensors.mag, dtype=np.float64)
    m0 = mag[np.asarray(tensors.index_z0)[gal]]
    kcorr = np.asarray(tensors.kcorr, dtype=np.float64)[gal]

    best = np.full((n_z, n_b), -1, dtype=np.int64)
    if method == 0:
        best[:] = np.arange(n_b)
    elif method == 2:
        best[:] = bapp[:n_b] if len(bapp) == n_b else bapp[0]
    elif method == 4:
        for i in range(n_z):
            j = 0
            while not (zbmin[j] <= gridz[i] < zbmax[j]) and j < len(zbmax) - 1:
                j += 1
            best[i] = bapp_op[j]
    if method == 1:
        if magabscont[0] > 0:
            if len(magabscont) == n_b:
                possi = [k for k in range(n_b) if _context_bit(magabscont[k], k)]
            else:
                possi = [k for k in range(n_b) if _context_bit(magabscont[0], k)]
        else:
            possi = list(range(n_b))
        if possi:
            for r in range(n_b):
                kc = m0[:, [r]] - m0[:, possi] - kcorr[:, possi]
                lo, hi = _kcolor_extremes(indz, kc, n_z)
                # first possible filter with the smallest spread (strict < scan)
                best[:, r] = np.asarray(possi)[np.argmin(hi - lo, axis=1)]

    # maxkcolor
    bf = best[indz]  # [G, N_b]
    ok = bf >= 0
    bfc = np.where(ok, bf, 0)
    rows = np.arange(gal.size)[:, None]
    kc = m0 - m0[rows, bfc] - kcorr[rows, bfc]
    lo = np.full((n_z, n_b), 1.0e4)
    hi = np.full((n_z, n_b), -1.0e4)
    for k in range(n_b):
        sel = ok[:, k]
        np.minimum.at(lo[:, k], indz[sel], kc[sel, k])
        np.maximum.at(hi[:, k], indz[sel], kc[sel, k])
    out = (best, hi - lo)
    _LIB_CACHE.clear()
    _LIB_CACHE[key] = (weakref.ref(tensors), out)
    return out


def _library_neighbours(tensors: MagLibTensors):
    """Per-template targets of the index walks in ``interp_lib``."""
    red = np.asarray(tensors.z, dtype=np.float64)
    n_t = red.size
    idx = np.arange(n_t)
    # runs of identical redshift (emission-line duplicates) → next different z
    new_z = np.ones(n_t, dtype=bool)
    new_z[1:] = red[1:] != red[:-1]
    z_start = np.maximum.accumulate(np.where(new_z, idx, 0))
    z_end = np.minimum.accumulate(np.where(np.r_[new_z[1:], True], idx, n_t - 1)[::-1])[::-1]
    # runs of the same model (SED::is_same_model: nummod, ebv, age)
    nm = np.asarray(tensors.nummod)
    ebv = np.asarray(tensors.ebv)
    age = np.asarray(tensors.age)
    new_m = np.ones(n_t, dtype=bool)
    new_m[1:] = (nm[1:] != nm[:-1]) | (ebv[1:] != ebv[:-1]) | (age[1:] != age[:-1])
    m_start = np.maximum.accumulate(np.where(new_m, idx, 0))
    last_z0 = np.maximum.accumulate(np.where(red <= 1e-20, idx, -1))
    walk0 = np.where(last_z0 >= m_start, last_z0, np.maximum(m_start - 1, 0))
    return z_start, z_end, walk0


def interp_lib(valid, consiz, dmmin, indmin_gal, tensors: MagLibTensors):
    """``onesource::interp_lib`` → ``(magm, magm0, kap)`` each ``[N_obj, N_b]``."""
    n_t = tensors.flux.shape[0]
    red = np.asarray(tensors.z, dtype=np.float64)
    mag = np.asarray(tensors.mag, dtype=np.float64)
    kcorr = np.asarray(tensors.kcorr, dtype=np.float64)
    flux = np.asarray(tensors.flux, dtype=np.float64)
    z_start, z_end, walk0 = _library_neighbours(tensors)

    ia = np.where(valid, np.asarray(indmin_gal, dtype=np.int64), 0)
    up = consiz - red[ia] >= 0  # copysign(1, +0.0) == +1
    nxt = np.where(up, z_end[ia] + 1, z_start[ia] - 1)
    deca = np.where((nxt >= 0) & (nxt < n_t), nxt - ia, 0)
    ib = ia + deca
    deca = np.where(red[ib] < 1e-10, 0, deca)
    ib = ia + deca

    both = (flux[ia] > 0) & (flux[ib] > 0)
    magma = np.where(both, _flux2mag(np.where(both, flux[ia], 1.0)), mag[ia])
    magmb = np.where(both, _flux2mag(np.where(both, flux[ib], 1.0)), mag[ib])
    same = (
        (deca != 0)
        & (np.asarray(tensors.nummod)[ia] == np.asarray(tensors.nummod)[ib])
        & (np.asarray(tensors.ebv)[ia] == np.asarray(tensors.ebv)[ib])
        & (np.asarray(tensors.age)[ia] == np.asarray(tensors.age)[ib])
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(same, np.abs((consiz - red[ia]) / (red[ia] - red[ib])), 0.0)
    magm = np.empty_like(magma)
    kap = np.empty_like(magma)
    for k in range(magma.shape[1]):
        a = np.where(mag[ib, k] > 90, 0.0, a)
        magm[:, k] = np.where(same, a * magmb[:, k] + (1.0 - a) * magma[:, k], mag[ia, k])
        a = np.where(kcorr[ib, k] > 90, 0.0, a)
        kap[:, k] = np.where(same, a * kcorr[ib, k] + (1.0 - a) * kcorr[ia, k], kcorr[ia, k])
    magm0 = mag[walk0[ia]]
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = 2.5 * np.log10(dmmin)
    magm = magm - shift[:, None]
    magm0 = magm0 - shift[:, None]
    inv = ~valid
    for arr in (magm, magm0, kap):
        arr[inv] = -999.0
    return magm, magm0, kap


def absolute_magnitudes(
    mags,
    magerrs,
    context,
    offsets,
    valid,
    consiz,
    dmmin,
    indmin_gal,
    tensors: MagLibTensors,
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """``interp_lib`` + ``absmag``.

    Returns ``MAG_ABS()``, ``EMAG_ABS()``, ``K_COR()``, ``MABS_FILT()`` and ``MAG_MOD()``.
    """
    best, maxkcol = library_filters(tensors, config)
    magm, magm0, kap = interp_lib(valid, consiz, dmmin, indmin_gal, tensors)

    # convertMag before offsets (msab), adapt_mag after offsets (mab, sab)
    ab0, sab0, *_ = prepare_photometry(mags, magerrs, context, tensors)
    ab, *_ = prepare_photometry(mags, magerrs, context, tensors, offsets=offsets)
    pos = ab0 > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        msab = np.where(pos & (sab0 >= 0), 1.086 * sab0 / ab0, np.where(pos, -1.0, HIGH_MAG))
    mab = np.where(ab > 0, _flux2mag(np.where(ab > 0, ab, 1.0)), HIGH_MAG)
    sab = sab0
    if offsets is not None:
        corr = np.power(10.0, 0.4 * np.asarray(offsets, dtype=np.float64))
        sab = np.where((ab0 > 0) | (sab0 > 0), sab0 * corr[None, :], sab0)

    gridz = np.asarray(tensors.gridz, dtype=np.float64)
    iz = np.minimum(np.searchsorted(gridz, consiz, side="left"), gridz.size - 1)
    dmod = dist_mod(consiz, cosmology_from_config(config))
    n_obj, n_b = magm.shape
    rows = np.arange(n_obj)
    mabs = np.empty((n_obj, n_b))
    emabs = np.empty((n_obj, n_b))
    absfilt = np.empty((n_obj, n_b))
    for k in range(n_b):
        fobs = best[iz, k]
        fc = np.where(fobs >= 0, fobs, 0)
        ms = msab[rows, fc]
        use = (fobs >= 0) & (ms <= 0.3) & (ms >= 0)
        val = mab[rows, fc] + magm0[:, k] - magm0[rows, fc] - kap[rows, fc] - dmod
        err = np.sqrt(maxkcol[iz, k] ** 2 + sab[rows, fc] ** 2)
        mabs[:, k] = np.where(use, val, magm0[:, k])
        emabs[:, k] = np.where(use, err, -1.0)
        absfilt[:, k] = np.where(use, fobs, -1)
    return {
        "MAG_ABS()": mabs,
        "EMAG_ABS()": emabs,
        "K_COR()": kap,
        "MABS_FILT()": absfilt,
        "MAG_MOD()": magm,
    }
