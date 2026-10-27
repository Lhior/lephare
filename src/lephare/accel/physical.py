"""Galaxy physical properties for the accelerated backends (LePhare 1.0 ports).

After the χ² fit, LePhare's ``PhotoZ::run_photoz`` derives per-source galaxy
properties from the GAL templates:

* ``onesource::generatePDF`` marginalizes ``exp(-χ²/2)`` over every GAL template
  into MASS / SFR / SSFR / LDUST / AGE (and COL1 / COL2 / MREF) histograms;
  ``uncertaintiesBay`` turns them into ``*_MED`` / ``*_INF`` / ``*_SUP``.
* ``onesource::interp`` rescales the best-fit scaling to the interpolated
  ``Z_BEST`` and ``compute_best_fit_physical_quantities`` gives ``*_BEST``.
* ``interp_lib`` + ``absmag`` give ``MAG_ABS()`` & co (see :mod:`.absmag`).

The histograms need every GAL template's χ² and scaling per object, so they are
accumulated on device (JAX or CUDA; NumPy reference) from the final fit state
the backends export in ``ancil`` (``_INDMIN_GAL`` and ``_BUSNORMA`` after
RM_DISCREPANT). Everything else uses only the best-fit template and runs on the
host.
"""

from __future__ import annotations

import contextvars
import os
import weakref
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .bestfit import INVALID_INDEX, INVALID_Z
from .fp32_numerics import FP32_DCHI_MAX, mag_abs_dm_bounds
from .kernels_ref import HIGH_CHI2, prepare_photometry
from .maglib import MagLibTensors

INVALID_PHYS = -999.0
LSOL = 3.826e33
_CKMS = 2.99792458e5

# onesource(pos, gridz) PDF grids: (min, max, size); hist layout follows this order.
PHYS_PDF_GRIDS: dict[str, tuple[float, float, int]] = {
    "MASS": (3.0, 13.0, 201),
    "SFR": (-10.0, 5.0, 151),
    "SSFR": (-25.0, -5.0, 201),
    "LDUST": (6.0, 14.0, 161),
    "AGE": (6.0, 10.5, 91),
    "COL1": (-10.0, 10.0, 201),
    "COL2": (-10.0, 10.0, 201),
    "MREF": (-30.0, -5.0, 250),
}
_PDF_NAMES = tuple(PHYS_PDF_GRIDS)
_PDF_OFFSET = dict(zip(_PDF_NAMES, np.cumsum([0] + [g[2] for g in PHYS_PDF_GRIDS.values()])))
N_HIST = int(sum(g[2] for g in PHYS_PDF_GRIDS.values()))

BEST_KEYS = (
    "MOD_BEST",
    "SCALE_BEST",
    "MASS_BEST",
    "SFR_BEST",
    "SSFR_BEST",
    "LDUST_BEST",
    "AGE_BEST",
    "EBV_BEST",
    "EXTLAW_BEST",
    "LUM_NUV_BEST",
    "LUM_R_BEST",
    "LUM_K_BEST",
)
# key -> (pdf name, position in uncertaintiesBay vector: 0=median, 1/2=68% bounds)
PDF_KEYS = {
    f"{name}_{suffix}": (name, pos)
    for name in _PDF_NAMES
    for pos, suffix in enumerate(("MED", "INF", "SUP"))
}
ABSMAG_KEYS = ("MAG_ABS()", "EMAG_ABS()", "K_COR()", "MABS_FILT()", "MAG_MOD()")
GALAXY_PROPERTY_KEYS = BEST_KEYS + tuple(PDF_KEYS) + ABSMAG_KEYS


# ---------------------------------------------------------------- config parsing


def split_keyword(config: Mapping[str, Any], key: str, default: str, n_items: int) -> list[str]:
    """``keyword::split_string``: comma list, a lone value fills ``n_items``, bad counts → default."""
    raw = config.get(key)
    value = str(default) if raw is None or str(raw).strip() == "" else str(raw)
    elems = [e.replace("\t", "").replace(" ", "") for e in value.split(",")]
    if n_items > 1 and len(elems) == 1:
        elems = elems * n_items
    if n_items > 0 and len(elems) != n_items:
        elems = [str(default)] * n_items
    return elems


def keyword_ints(config, key, default, n_items=-1) -> list[int]:
    return [int(float(v)) for v in split_keyword(config, key, default, n_items)]


def keyword_doubles(config, key, default, n_items=-1) -> list[float]:
    return [float(v) for v in split_keyword(config, key, default, n_items)]


def keyword_bool(config, key, default) -> bool:
    return split_keyword(config, key, default, 1)[0][:1].upper() in ("Y", "T", "1")


def cosmology_from_config(config: Mapping[str, Any]) -> tuple[float, float, float]:
    vals = keyword_doubles(config, "COSMOLOGY", "70,0.3,0.7", 3)
    return float(vals[0]), float(vals[1]), float(vals[2])


def dist_mod(z, cosmology: tuple[float, float, float]) -> np.ndarray:
    """``cosmo::distMod`` (50-step midpoint integral for Λ ≠ 0)."""
    h0, om0, l0 = cosmology
    z = np.asarray(z, dtype=np.float64)
    zz = np.where(z < 1e-10, 1.0, z)
    if l0 == 0:
        if om0 > 0:
            dmet = om0 * zz - (om0 - 2) * (1 - np.sqrt(1 + om0 * zz))
            dmet = 2 * _CKMS / (h0 * om0 * om0 * (1 + zz)) * dmet
        else:
            dmet = _CKMS * zz * (1.0 + zz / 2.0) / (h0 * (1 + zz))
    elif om0 < 1:
        total = np.zeros_like(zz)
        dz = zz / 50.0
        for i in range(50):
            zi = (float(i) + 0.5) * dz
            ez = np.sqrt(om0 * (1.0 + zi) ** 3 + (1 - om0 - l0) * (1.0 + zi) ** 2 + l0)
            total = total + dz / ez
        dmet = _CKMS / h0 * total
    else:
        raise ValueError(f"Cosmology not supported by LePhare: {cosmology}")
    return np.where(z < 1e-10, 0.0, 5.0 * np.log10((1 + zz) * dmet) + 25)


# ------------------------------------------------------------- PDF primitives


def pdf_index(values, name: str, xp=np):
    """``PDF::index``: nearest bin (round half away), clamp outside, NaN → last bin."""
    lo, hi, n = PHYS_PDF_GRIDS[name]
    inv = 1.0 / ((hi - lo) / (n - 1))
    with np.errstate(invalid="ignore"):  # ±inf / NaN land in the clamped branches
        x = (values - lo) * inv
        fl = xp.floor(x)
        near = fl + ((x - fl) >= 0.5)
        inside = (values >= lo) & (values <= hi)
        out = xp.where(inside, near, xp.where(values < lo, 0.0, float(n - 1)))
    return out.astype(xp.int32)


def _grid(name: str) -> np.ndarray:
    lo, hi, n = PHYS_PDF_GRIDS[name]
    step = (hi - lo) / (n - 1)
    return lo + np.arange(n, dtype=np.float64) * step


def _interp_x(level, c1, x1, c2, x2, xp=np):
    """``PDF::linear_interp(level, c1, x1, c2, x2)`` (x from cumulant level)."""
    slope_ok = c2 > c1
    denom = xp.where(slope_ok, c2 - c1, 1.0)
    return xp.where(slope_ok, x1 + (level - c1) * (x2 - x1) / denom, x1)


def _normalized_cumulant(pdf, x, xp=np):
    """``PDF::normalization`` then trapezoid cumulant / last; returns (cum, defined)."""
    width = xp.diff(x)
    area = xp.sum((pdf[:, :-1] + pdf[:, 1:]) / 2.0 * width, axis=1)
    pdf = xp.where(area[:, None] > 0, pdf / xp.where(area > 0, area, 1.0)[:, None], pdf)
    cum = xp.zeros_like(pdf)
    cum[:, 1:] = xp.cumsum(width * (pdf[:, 1:] + pdf[:, :-1]) / 2.0, axis=1)
    last = cum[:, -1]
    defined = last > 0
    cum = xp.where(defined[:, None], cum / xp.where(defined, last, 1.0)[:, None], cum)
    return cum, defined


def _upper_bound(cum, level, xp=np):
    """Row-wise ``std::upper_bound`` index for monotone rows."""
    return xp.sum(cum <= level[:, None], axis=1)


def _take(a, idx, xp=np):
    return xp.take_along_axis(a, idx[:, None], axis=1)[:, 0]


def pdf_median_interval(pdf, name: str, xp=np):
    """``levelCumu2x(0.5)`` and ``credible_interval(68, median)`` for each row.

    Runs on ``xp`` (NumPy or CuPy) arrays; returns arrays on the same device.
    """
    x = xp.asarray(_grid(name))
    lo, hi, n = PHYS_PDF_GRIDS[name]
    n_obj = pdf.shape[0]
    cum, defined = _normalized_cumulant(xp.asarray(pdf, dtype=xp.float64), x, xp)

    def _take_(a, idx):
        return _take(a, idx, xp)

    half = xp.full(n_obj, 0.5)
    ind = _upper_bound(cum, half, xp)
    ind_c = xp.clip(ind, 1, n - 1)
    med = _interp_x(half, _take_(cum, ind_c - 1), x[ind_c - 1], _take_(cum, ind_c), x[ind_c], xp)
    med = xp.where(ind > 0, med, x[0])
    med = xp.where(defined, med, INVALID_Z)

    # credible_interval(level=68 as float, val=med)
    level = float(np.float32(np.float32(68.0) / 100.0))
    inside = (med >= lo) & (med < hi) & defined
    step = (hi - lo) / (n - 1)
    idx_lo = xp.clip(((xp.where(inside, med, lo) - lo) / step).astype(xp.int64), 0, n - 2)
    c_lo = _take_(cum, idx_lo)
    c_hi = _take_(cum, idx_lo + 1)
    cum_val = c_lo + (c_hi - c_lo) / (x[idx_lo + 1] - x[idx_lo]) * (med - x[idx_lo])
    left = cum_val
    right = 1.0 - left
    both = (right >= level / 2.0) & (left >= level / 2.0)
    upper = xp.where(both, left + level / 2.0, xp.where(right <= level / 2.0, 1.0, level))
    lower = xp.where(both, left - level / 2.0, xp.where(right <= level / 2.0, 1.0 - level, 0.0))

    def _bound(target):
        ind = xp.clip(_upper_bound(cum, target, xp), 1, n)
        c_prev = _take_(cum, ind - 1)
        ind_hi = xp.minimum(ind, n - 1)
        val = _interp_x(target, c_prev, x[ind - 1], _take_(cum, ind_hi), x[ind_hi], xp)
        return xp.where(c_prev < target, val, x[ind - 1])

    inf = xp.where(inside, _bound(lower), med)
    sup = xp.where(inside, _bound(upper), med)
    return med, inf, sup


# ------------------------------------------------------- per-template table


@dataclass
class GalPhysTable:
    """Per-GAL-template inputs of ``generatePDF`` (``SEDlight`` columns)."""

    gal_idx: np.ndarray  # [G] library rows with type GAL
    lgmass: np.ndarray  # [G] log10(mass) (NaN/-inf if undefined)
    lgsfr: np.ndarray  # [G]
    ltir: np.ndarray  # [G]
    mref0: np.ndarray  # [G] rest-frame M_REF magnitude (colRF[2])
    ssfr_bin: np.ndarray  # [G] int32 absolute hist index
    age_bin: np.ndarray  # [G]
    col1_bin: np.ndarray  # [G]
    col2_bin: np.ndarray  # [G]
    col_analysis: bool


_TABLE_CACHE: dict = {}


def _rf_color_bands(config: Mapping[str, Any], n_b: int) -> tuple[list[int], int, bool]:
    flt = keyword_ints(config, "RF_COLORS", "-1", 4)
    if len(flt) > 1:
        flt = [f - 1 for f in flt]
    ref = keyword_ints(config, "M_REF", "0")[0]
    if ref > 0:
        ref -= 1
    ok = all(0 <= f < n_b for f in flt) and ref >= 0
    return flt, ref, ok


def gal_phys_table(tensors: MagLibTensors, config: Mapping[str, Any]) -> GalPhysTable:
    flt, ref, col_ok = _rf_color_bands(config, int(tensors.n_bands))
    # identity, not address: a freed library's address can be reused by a new one
    key = (id(tensors), tuple(flt), ref)
    cached = _TABLE_CACHE.get(key)
    if cached is not None and cached[0]() is tensors:
        return cached[1]
    gal = np.flatnonzero(np.asarray(tensors.types) == 0)
    mass = np.asarray(tensors.mass, dtype=np.float64)[gal]
    sfr = np.asarray(tensors.sfr, dtype=np.float64)[gal]
    age = np.asarray(tensors.age, dtype=np.float64)[gal]
    with np.errstate(divide="ignore", invalid="ignore"):
        ssfr = np.where(mass > 0, sfr / mass, -999.0)  # GalSED::readMagBin
        lgmass = np.log10(mass)
        lgsfr = np.log10(sfr)
        lgssfr = np.log10(ssfr)
        lgage = np.log10(age)
    if col_ok:
        mag_z0 = np.asarray(tensors.mag, dtype=np.float64)[np.asarray(tensors.index_z0)[gal]]
        col1 = mag_z0[:, flt[0]] - mag_z0[:, flt[1]]
        col2 = mag_z0[:, flt[2]] - mag_z0[:, flt[3]]
        mref0 = mag_z0[:, ref]
    else:
        col1 = col2 = mref0 = np.full(gal.size, -9999.0)  # INVALID_MAG

    def _abs_bin(v, name):
        return (pdf_index(v, name) + _PDF_OFFSET[name]).astype(np.int32)

    table = GalPhysTable(
        gal_idx=gal,
        lgmass=lgmass,
        lgsfr=lgsfr,
        ltir=np.asarray(tensors.ltir, dtype=np.float64)[gal],
        mref0=mref0,
        ssfr_bin=_abs_bin(lgssfr, "SSFR"),
        age_bin=_abs_bin(lgage, "AGE"),
        col1_bin=_abs_bin(col1, "COL1"),
        col2_bin=_abs_bin(col2, "COL2"),
        col_analysis=bool(col_ok),
    )
    _TABLE_CACHE.clear()
    _TABLE_CACHE[key] = (weakref.ref(tensors), table)
    return table


# ------------------------------------------------------- histogram accumulation


def _dchi_cap(dtype) -> float:
    return FP32_DCHI_MAX if np.dtype(dtype) == np.float32 else 700.0


def _gal_priors(tensors: MagLibTensors, gal: np.ndarray, dtype):
    dm_min, dm_max = mag_abs_dm_bounds(
        tensors.mag0, tensors.z, tensors.types, tensors.prior_lib, tensors.funz0, dtype
    )
    gal_on = bool(tensors.prior_lib[0] < 0 and tensors.prior_lib[1] < 0)
    return gal_on, dm_min[gal], dm_max[gal]


def _accumulate_np(hist, prob, lgdm, tab: GalPhysTable, cols):
    """Scatter ``generatePDF`` weights for one [N, T] block into ``hist`` [N, N_HIST]."""
    lgmass = tab.lgmass[cols][None, :]
    mass_ok = (prob > 0) & (lgdm > -100) & (lgmass > -100)
    ltir = tab.ltir[cols][None, :]
    entries = [
        (mass_ok, pdf_index(lgdm + lgmass, "MASS") + _PDF_OFFSET["MASS"]),
        (mass_ok, pdf_index(tab.lgsfr[cols][None, :] + lgdm, "SFR") + _PDF_OFFSET["SFR"]),
        (mass_ok, np.broadcast_to(tab.ssfr_bin[cols][None, :], prob.shape)),
        (mass_ok, np.broadcast_to(tab.age_bin[cols][None, :], prob.shape)),
        (mass_ok & (ltir >= 0), pdf_index(ltir + lgdm, "LDUST") + _PDF_OFFSET["LDUST"]),
    ]
    if tab.col_analysis:
        any_ok = prob > 0
        entries += [
            (any_ok, np.broadcast_to(tab.col1_bin[cols][None, :], prob.shape)),
            (any_ok, np.broadcast_to(tab.col2_bin[cols][None, :], prob.shape)),
            (any_ok, pdf_index(tab.mref0[cols][None, :] - 2.5 * lgdm, "MREF") + _PDF_OFFSET["MREF"]),
        ]
    flat_rows = np.arange(prob.shape[0])[:, None] * N_HIST
    for ok, idx in entries:
        np.add.at(hist.reshape(-1), (flat_rows + idx)[ok], prob[ok])


def phys_hist_numpy(ab, sab, busnorma, chimin_gal, tensors, table, dtype=np.float64):
    """Float64 NumPy reference for the ``generatePDF`` physical histograms."""
    n_obj = ab.shape[0]
    hist = np.zeros((n_obj, N_HIST), dtype=np.float64)
    gal = table.gal_idx
    gal_on, dm_min, dm_max = _gal_priors(tensors, gal, np.float64)
    inv = np.where(busnorma > 0, busnorma / np.where(sab != 0, sab, 1.0), 0.0)
    inv2 = inv * inv
    abinv = ab * inv2
    base = np.sum((ab * inv) ** 2, axis=1)
    flux = np.asarray(tensors.flux, dtype=np.float64)
    cap = _dchi_cap(dtype)
    tile, chunk = 4096, 512
    for o0 in range(0, n_obj, chunk):
        rows = slice(o0, min(o0 + chunk, n_obj))
        for s in range(0, gal.size, tile):
            cols = np.arange(s, min(s + tile, gal.size))
            f = flux[gal[cols]]
            avmago = abinv[rows] @ f.T
            avmagt = inv2[rows] @ (f * f).T
            pos = avmagt > 0
            safe = np.where(pos, avmagt, 1.0)
            dm = np.where(pos, avmago / safe, -999.0)
            chi2 = np.where(pos, base[rows, None] - avmago * avmago / safe, base[rows, None])
            if gal_on:
                reject = (dm > 0) & ((dm <= dm_min[cols]) | (dm >= dm_max[cols]))
                chi2 = np.where(reject, HIGH_CHI2, chi2)
            dchi = np.maximum(chi2 - chimin_gal[rows, None], 0.0)
            ok = (chi2 < HIGH_CHI2) & (dchi <= cap)
            prob = np.where(ok, np.exp(-0.5 * dchi), 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                lgdm = np.log10(dm)
            _accumulate_np(hist[rows], prob, lgdm, table, cols)
    return hist


def _phys_hist_jax(mags, magerrs, context, offsets, busnorma, chimin_gal, tensors, table, dtype):
    from .jax_backend import _cache_fns, _require_jax, ensure_gpu_backend

    jax, jnp = _require_jax()
    platform = ensure_gpu_backend()
    fns = _cache_fns(tensors, dtype, jax, jnp, platform)
    n_obj, n_b = mags.shape
    has_c = context is not None
    has_o = offsets is not None
    ab, sab, _bus, _busul, _nbul = fns["prep"](
        jnp.asarray(mags, dtype=dtype),
        jnp.asarray(magerrs, dtype=dtype),
        jnp.asarray(context if has_c else np.zeros(n_obj), dtype=jnp.int64),
        jnp.asarray(offsets if has_o else np.zeros(n_b), dtype=dtype),
        has_c,
        has_o,
    )
    bus = jnp.asarray(busnorma, dtype=dtype)
    inv = jnp.where(bus > 0, bus / sab, 0.0)
    inv2 = inv * inv
    abinv = ab * inv2
    base = jnp.sum((ab * inv) ** 2, axis=1)

    gal_on, dm_min, dm_max = _gal_priors(tensors, table.gal_idx, dtype)
    # FP32 runs keep the per-template math and the histogram in float32 (FP64 is 1:64 on
    # consumer GPUs); FP64 runs are unchanged
    acc = jnp.float64 if np.dtype(dtype) == np.float64 else jnp.float32
    n_gal = int(table.gal_idx.size)
    tile = 2048
    n_pad = max(tile, -(-n_gal // tile) * tile)
    pad = n_pad - n_gal

    def _p(a, fill):
        a = np.asarray(a)
        return jnp.asarray(np.concatenate([a, np.full(pad, fill, dtype=a.dtype)]))

    lib = (
        jnp.concatenate([fns["flux_gal_bt"], jnp.zeros((n_b, pad), dtype=fns["jd"])], axis=1),
        _p(np.ones(n_gal, dtype=bool), False),
        _p(dm_min, 0),
        _p(dm_max, 0),
        *(_p(a, np.nan).astype(acc) for a in (table.lgmass, table.lgsfr, table.ltir, table.mref0)),
        *(_p(b, 0) for b in (table.ssfr_bin, table.age_bin, table.col1_bin, table.col2_bin)),
    )
    cap = _dchi_cap(dtype)
    col = table.col_analysis

    def _tile(lib, inv2, abinv, base, cmin, hist, s):
        flux_bt, valid, dmn, dmx, lgmass, lgsfr, ltir, mref0, *statics = lib
        f = jax.lax.dynamic_slice_in_dim(flux_bt, s, tile, axis=1)
        sl = lambda a: jax.lax.dynamic_slice_in_dim(a, s, tile)  # noqa: E731
        avmago = jnp.matmul(abinv, f, precision=jax.lax.Precision.HIGHEST)  # no TF32
        avmagt = jnp.matmul(inv2, f * f, precision=jax.lax.Precision.HIGHEST)
        pos = avmagt > 0
        safe = jnp.where(pos, avmagt, 1.0)
        dm = jnp.where(pos, avmago / safe, -999.0)
        chi2 = jnp.where(pos, base[:, None] - avmago * avmago / safe, base[:, None])
        if gal_on:
            reject = (dm > 0) & ((dm <= sl(dmn)[None, :]) | (dm >= sl(dmx)[None, :]))
            chi2 = jnp.where(reject, HIGH_CHI2, chi2)
        dchi = jnp.maximum(chi2 - cmin[:, None], 0.0).astype(acc)
        ok = (chi2 < HIGH_CHI2) & (dchi <= cap) & sl(valid)[None, :]
        prob = jnp.where(ok, jnp.exp(-0.5 * dchi), 0.0)
        lgdm = jnp.log10(dm.astype(acc))
        lm = sl(lgmass)[None, :]
        mass_ok = ok & (lgdm > -100) & (lm > -100)
        lt = sl(ltir)[None, :]
        shape = prob.shape
        ss, ag, c1, c2 = (jnp.broadcast_to(sl(b)[None, :], shape) for b in statics)
        idx = [
            pdf_index(lgdm + lm, "MASS", jnp) + _PDF_OFFSET["MASS"],
            pdf_index(sl(lgsfr)[None, :] + lgdm, "SFR", jnp) + _PDF_OFFSET["SFR"],
            ss,
            ag,
            pdf_index(lt + lgdm, "LDUST", jnp) + _PDF_OFFSET["LDUST"],
        ]
        w = [mass_ok, mass_ok, mass_ok, mass_ok, mass_ok & (lt >= 0)]
        if col:
            mref = pdf_index(sl(mref0)[None, :] - 2.5 * lgdm, "MREF", jnp) + _PDF_OFFSET["MREF"]
            idx += [c1, c2, mref]
            w += [ok, ok, ok]
        idx = jnp.stack(idx, axis=-1)
        wt = jnp.where(jnp.stack(w, axis=-1), prob[..., None], 0.0)
        rows = jnp.arange(shape[0])[:, None, None] * N_HIST
        flat = (rows + idx).reshape(-1)
        return hist.reshape(-1).at[flat].add(wt.reshape(-1)).reshape(hist.shape)

    tile_fn = jax.jit(_tile)
    chunk = 512
    out = np.zeros((n_obj, N_HIST), dtype=np.float64)
    cmin_all = jnp.asarray(chimin_gal, dtype=dtype)
    for o0 in range(0, n_obj, chunk):
        o1 = min(o0 + chunk, n_obj)
        sel = slice(o0, o1)
        npad = chunk - (o1 - o0)

        def _rows(a, fill):
            return jnp.concatenate([a[sel], jnp.full((npad,) + a.shape[1:], fill, dtype=a.dtype)])

        args = (_rows(inv2, 0), _rows(abinv, 0), _rows(base, 0), _rows(cmin_all, HIGH_CHI2))
        hist = jnp.zeros((chunk, N_HIST), dtype=acc)
        for s in range(0, n_pad, tile):
            hist = tile_fn(lib, *args, hist, s)
        out[o0:o1] = np.asarray(jax.device_get(hist))[: o1 - o0]
    return out


_CUDA_KERNELS: dict = {}


def _phys_cuda_src(real: str, acc: str = "double", ncopy: int = 1) -> str:
    """``generatePDF`` histogram kernel. ``real`` is the per-template math type (chi2,
    weights, bin indices); ``acc`` is the shared-memory histogram accumulator type.
    FP64 uses double/double; FP32 runs its math in float (1:64 FP64 on consumer GPUs).
    ``ncopy`` private shared histograms (one per warp group) cut atomic contention:
    a posterior lands in a few hot bins, and shared atomics on them serialize."""

    def lit(v) -> str:
        return f"{float(v)!r}f" if real == "float" else f"{float(v)!r}"

    def grid_args(name):
        lo, hi, n = PHYS_PDF_GRIDS[name]
        return f"{lit(lo)}, {lit(hi)}, {n}"

    off = {k: int(v) for k, v in _PDF_OFFSET.items()}
    return f"""
#define HIGH_CHI2 ({HIGH_CHI2:.1f})
#ifndef USE_PRIORS
#define USE_PRIORS 0
#endif

__device__ __forceinline__ int pdf_index({real} v, {real} lo, {real} hi, int n) {{
    if (v >= lo && v <= hi) {{
        {real} x = (v - lo) * ({lit(1)} / ((hi - lo) / ({real})(n - 1)));
        {real} f = floor(x);
        return (int)f + ((x - f) >= {lit(0.5)} ? 1 : 0);
    }}
    if (v < lo) return 0;
    return n - 1;
}}

extern "C" __global__
void phys_hist_kernel(
    const {real}* __restrict__ flux_bt,
    const {real}* __restrict__ invsabSq,
    const {real}* __restrict__ abinvsabSq,
    const {real}* __restrict__ base,
    const {real}* __restrict__ dm_min,
    const {real}* __restrict__ dm_max,
    const {real}* __restrict__ chimin,
    const {real}* __restrict__ lgmass,
    const {real}* __restrict__ lgsfr,
    const {real}* __restrict__ ltir,
    const {real}* __restrict__ mref0,
    const int* __restrict__ ssfr_bin,
    const int* __restrict__ age_bin,
    const int* __restrict__ col1_bin,
    const int* __restrict__ col2_bin,
    {acc}* __restrict__ hist,
    int N_obj, int N_g, int N_b, {real} dchi_max, int col_analysis
) {{
    int o = (int)blockIdx.x;
    if (o >= N_obj) return;
    extern __shared__ unsigned char s_raw[];
    {acc}* s_all = ({acc}*)s_raw;
    for (int b = (int)threadIdx.x; b < {ncopy} * {N_HIST}; b += (int)blockDim.x) s_all[b] = 0;
    {acc}* s_hist = s_all + (((int)threadIdx.x / 32) % {ncopy}) * {N_HIST};
    {real} r_inv2[16];
    {real} r_abinv[16];
    for (int k = 0; k < 16; ++k) {{
        if (k < N_b) {{
            r_inv2[k] = invsabSq[o * N_b + k];
            r_abinv[k] = abinvsabSq[o * N_b + k];
        }}
    }}
    {real} r_base = base[o];
    {real} cmin = chimin[o];
    __syncthreads();

    for (int i = (int)threadIdx.x; i < N_g; i += (int)blockDim.x) {{
        {real} avmago = 0;
        {real} avmagt = 0;
        for (int k = 0; k < N_b; ++k) {{
            {real} f = flux_bt[k * N_g + i];
            avmago += f * r_abinv[k];
            avmagt += f * f * r_inv2[k];
        }}
        {real} chi2 = r_base;
        if (avmagt > 0) chi2 = r_base - (avmago * avmago) / avmagt;
        {real} dm = (avmagt > 0) ? (avmago / avmagt) : ({real})(-999.0);
#if USE_PRIORS
        if (dm > 0 && (dm <= dm_min[i] || dm >= dm_max[i])) chi2 = HIGH_CHI2;
#endif
        if (chi2 >= HIGH_CHI2) continue;
        {real} dchi = chi2 - cmin;
        if (dchi < 0) dchi = 0;
        if (dchi > dchi_max) continue;
        {real} prob = exp({lit(-0.5)} * dchi);
        {real} lgdm = log10(dm);
        {real} lm = lgmass[i];
        if (lgdm > {lit(-100)} && lm > {lit(-100)}) {{
            atomicAdd(&s_hist[{off['MASS']} + pdf_index(lgdm + lm, {grid_args('MASS')})], prob);
            atomicAdd(&s_hist[{off['SFR']} + pdf_index(lgsfr[i] + lgdm, {grid_args('SFR')})], prob);
            atomicAdd(&s_hist[ssfr_bin[i]], prob);
            atomicAdd(&s_hist[age_bin[i]], prob);
            if (ltir[i] >= 0)
                atomicAdd(&s_hist[{off['LDUST']} + pdf_index(ltir[i] + lgdm, {grid_args('LDUST')})], prob);
        }}
        if (col_analysis) {{
            atomicAdd(&s_hist[col1_bin[i]], prob);
            atomicAdd(&s_hist[col2_bin[i]], prob);
            atomicAdd(&s_hist[{off['MREF']} + pdf_index(mref0[i] - {lit(2.5)} * lgdm, {grid_args('MREF')})], prob);
        }}
    }}
    __syncthreads();
    for (int b = (int)threadIdx.x; b < {N_HIST}; b += (int)blockDim.x) {{
        {acc} tot = 0;
        for (int c = 0; c < {ncopy}; ++c) tot += s_all[c * {N_HIST} + b];
        hist[(long long)o * {N_HIST} + b] = tot;
    }}
}}
"""


def _phys_cuda_chunks(mags, magerrs, context, offsets, busnorma, chimin_gal, tensors, table, dtype, consume):
    """Run the CUDA ``generatePDF`` histogram over VRAM-sized object chunks and call
    ``consume(o0, o1, d_hist)`` with each device histogram ``[o1 - o0, N_HIST]``."""
    from .cuda_backend import prepare_photometry_cuda_device
    from .cuda_template import _device_lib, _require_cupy
    from .gpu_util import run_with_oom_backoff, suggest_obj_chunk

    cp = _require_cupy()
    dtype = np.dtype(dtype).type
    real = "double" if dtype == np.float64 else "float"
    acc = "double" if dtype == np.float64 else "float"
    acc_dtype = np.float64 if acc == "double" else np.float32
    # FP32: one private float histogram per warp (8 x 1457 x 4 B fits the 48 KB default);
    # FP64 keeps a single double histogram (its runtime is FP64-bound, not atomic-bound).
    ncopy = 1 if acc == "double" else 8
    lib = _device_lib(cp, tensors, dtype=dtype)
    gal_on, dm_min, dm_max = _gal_priors(tensors, table.gal_idx, dtype)
    key = (real, acc, ncopy, gal_on)
    if key not in _CUDA_KERNELS:
        mod = cp.RawModule(
            code=_phys_cuda_src(real, acc, ncopy),
            options=("-std=c++11", f"-DUSE_PRIORS={int(gal_on)}"),
        )
        _CUDA_KERNELS[key] = mod.get_function("phys_hist_kernel")
    kernel = _CUDA_KERNELS[key]

    n_obj, n_b = mags.shape
    n_g = int(table.gal_idx.size)
    if not (n_obj and n_g):
        return False
    per_template = tuple(
        cp.asarray(a, dtype=dtype)
        for a in (dm_min, dm_max, table.lgmass, table.lgsfr, table.ltir, table.mref0)
    ) + tuple(
        cp.asarray(a, dtype=np.int32)
        for a in (table.ssfr_bin, table.age_bin, table.col1_bin, table.col2_bin)
    )
    ctx = None if context is None else np.asarray(context)
    per_obj_ctx = ctx is not None and ctx.ndim > 0 and ctx.shape[0] == n_obj
    nbytes = np.dtype(dtype).itemsize

    def run(o0, o1):
        c = ctx[o0:o1] if per_obj_ctx else ctx
        d_ab, d_sab, _bus, _busul, _nbul = prepare_photometry_cuda_device(
            mags[o0:o1], magerrs[o0:o1], tensors, context=c, offsets=offsets, dtype=dtype
        )
        if dtype == np.float32:
            scale = cp.asarray(lib["fp32_scale"], dtype=cp.float32)
            d_ab = d_ab * scale
            d_sab = d_sab * scale
        d_bus = cp.asarray(busnorma[o0:o1], dtype=dtype)
        zero = cp.asarray(0, dtype=dtype)
        d_inv = cp.where(d_bus > 0, d_bus / d_sab, zero)
        d_inv2 = cp.ascontiguousarray(d_inv * d_inv)
        d_abinv = cp.ascontiguousarray(d_ab * d_inv2)
        d_base = cp.ascontiguousarray(cp.sum((d_ab * d_inv) ** 2, axis=1))
        del d_ab, d_sab, d_bus, d_inv, _bus, _busul, _nbul
        d_cmin = cp.asarray(chimin_gal[o0:o1], dtype=dtype)
        d_hist = cp.zeros((o1 - o0, N_HIST), dtype=acc_dtype)
        dmn, dmx, *tab = per_template
        kernel(
            (o1 - o0,),
            (256,),
            (
                lib["flux_gal"],
                d_inv2,
                d_abinv,
                d_base,
                dmn,
                dmx,
                d_cmin,
                *tab,
                d_hist,
                np.int32(o1 - o0),
                np.int32(n_g),
                np.int32(n_b),
                dtype(_dchi_cap(dtype)),
                np.int32(table.col_analysis),
            ),
            shared_mem=ncopy * N_HIST * np.dtype(acc_dtype).itemsize,
        )
        consume(o0, o1, d_hist)

    # Histogram + float64 cumulant temporaries of the statistics (~4 N_HIST doubles)
    # dominate; photometry prep is ~12 band vectors.
    per_obj = N_HIST * (np.dtype(acc_dtype).itemsize + 4 * 8) + 12 * n_b * nbytes
    chunk = suggest_obj_chunk(per_obj, min_chunk=256)
    run_with_oom_backoff(run, n_obj, chunk, min_chunk=256, tag="phys_hist")
    return True


def _phys_hist_cuda(mags, magerrs, context, offsets, busnorma, chimin_gal, tensors, table, dtype):
    from .cuda_template import _require_cupy

    cp = _require_cupy()
    out = np.zeros((mags.shape[0], N_HIST), dtype=np.float64)

    def consume(o0, o1, d_hist):
        out[o0:o1] = cp.asnumpy(d_hist).astype(np.float64)

    _phys_cuda_chunks(mags, magerrs, context, offsets, busnorma, chimin_gal, tensors, table, dtype, consume)
    return out


def _phys_props_cuda(keys, mags, magerrs, context, offsets, busnorma, chimin_gal, tensors, table, dtype):
    """:func:`pdf_properties` of the CUDA histograms, reduced on the device per chunk
    (the ``[N, N_HIST]`` histogram never reaches the host)."""
    from .cuda_template import _require_cupy

    cp = _require_cupy()
    n_obj = mags.shape[0]
    out: dict[str, np.ndarray] = {}

    def consume(o0, o1, d_hist):
        props = pdf_properties(d_hist.astype(cp.float64), keys, xp=cp)
        for k, v in props.items():
            if k not in out:
                out[k] = np.empty(n_obj, dtype=np.float64)
            out[k][o0:o1] = cp.asnumpy(v)

    ran = _phys_cuda_chunks(
        mags, magerrs, context, offsets, busnorma, chimin_gal, tensors, table, dtype, consume
    )
    if not ran:
        return pdf_properties(np.zeros((n_obj, N_HIST)), keys)
    return out


# ------------------------------------------- generatePDF fused into the CUDA fit

_FUSED_PHYS = contextvars.ContextVar("rail_lephare_fused_phys", default=None)
_FUSED_KERNELS: dict = {}
_FUSED_TABLES: dict = {}


class fused_galaxy_properties:
    """Context in which the CUDA fused fit also accumulates the ``generatePDF``
    histograms in its second pass (one GAL chi2 sweep instead of two) and returns the
    PDF statistics (``MASS_MED`` ...) as ``_PHYS:<key>`` ancil rows flagged by
    ``_PHYSDONE``. Disable with ``LEPHARE_ACCEL_FUSED_PHYS=0``."""

    def __init__(self, keys, tensors: MagLibTensors, config: Mapping[str, Any]):
        self.req = None
        pdf_keys = [k for k in keys if k in PDF_KEYS]
        if not pdf_keys or os.environ.get("LEPHARE_ACCEL_FUSED_PHYS", "1") == "0":
            return
        try:
            check_supported(tensors, config)
        except (ValueError, NotImplementedError):
            return  # galaxy_properties raises the same error with context
        self.req = dict(keys=pdf_keys, tensors=tensors, table=gal_phys_table(tensors, config))

    def __enter__(self):
        self._tok = _FUSED_PHYS.set(self.req)
        return self

    def __exit__(self, *exc):
        _FUSED_PHYS.reset(self._tok)


def fused_request():
    return _FUSED_PHYS.get()


def fused_extra_obj_bytes(dtype) -> int:
    """Device bytes per object the fused histograms + statistics add to a fit chunk."""
    acc = 8 if np.dtype(dtype) == np.float64 else 4
    return N_HIST * (acc + 4 * 8)


def _fused_src(real: str, acc: str, ncopy: int) -> str:
    """``fuse_pass2_kernel`` (BAY_ZG p(z) over GAL templates) that also scatters the
    ``phys_hist_kernel`` weights: same chi2, same ``prob``; the p(z) uses the fit's
    MAG_ABS bounds and the histograms their own (as the two separate kernels do)."""

    def lit(v) -> str:
        return f"{float(v)!r}f" if real == "float" else f"{float(v)!r}"

    def grid_args(name):
        lo, hi, n = PHYS_PDF_GRIDS[name]
        return f"{lit(lo)}, {lit(hi)}, {n}"

    off = {k: int(v) for k, v in _PDF_OFFSET.items()}
    exp = "expf" if real == "float" else "exp"
    return _phys_cuda_src(real, acc, ncopy) + f"""
extern "C" __global__
void pass2_phys_kernel(
    const {real}* __restrict__ flux_bt,
    const {real}* __restrict__ invsabSq,
    const {real}* __restrict__ abinvsabSq,
    const {real}* __restrict__ base,
    const {real}* __restrict__ dm_min,
    const {real}* __restrict__ dm_max,
    const int* __restrict__ z_bin,
    const {real}* __restrict__ chimin_gal,
    {real}* __restrict__ pdf,
    int N_obj, int N_t, int N_b, int N_z,
    const {real}* __restrict__ hdm_min,
    const {real}* __restrict__ hdm_max,
    const {real}* __restrict__ lgmass,
    const {real}* __restrict__ lgsfr,
    const {real}* __restrict__ ltir,
    const {real}* __restrict__ mref0,
    const int* __restrict__ ssfr_bin,
    const int* __restrict__ age_bin,
    const int* __restrict__ col1_bin,
    const int* __restrict__ col2_bin,
    {acc}* __restrict__ hist,
    {real} dchi_max, int col_analysis
) {{
    int o = (int)blockIdx.x;
    if (o >= N_obj) return;
    const int nwarps = (int)blockDim.x / 32;
    const int warp = (int)threadIdx.x / 32;
    extern __shared__ unsigned char raw[];
    {real} *s_pdf = ({real} *)raw;
    size_t pdf_bytes = (((size_t)nwarps * N_z * sizeof({real}) + 15) / 16) * 16;
    {acc} *s_all = ({acc} *)(raw + pdf_bytes);
    {acc} *s_hist = s_all + (warp % {ncopy}) * {N_HIST};
#ifdef N_B
    const int nb = N_B;
#else
    const int nb = N_b;
#endif
    {real} r_inv2[16];
    {real} r_abinv[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) {{
        if (k < nb) {{
            r_inv2[k] = invsabSq[o * nb + k];
            r_abinv[k] = abinvsabSq[o * nb + k];
        }}
    }}
    {real} r_base = base[o];
    {real} cmin = chimin_gal[o];
    for (int b = (int)threadIdx.x; b < nwarps * N_z; b += (int)blockDim.x) s_pdf[b] = 0;
    for (int b = (int)threadIdx.x; b < {ncopy} * {N_HIST}; b += (int)blockDim.x) s_all[b] = 0;
    __syncthreads();

    for (int i = (int)threadIdx.x; i < N_t; i += (int)blockDim.x) {{
        {real} avmago = 0;
        {real} avmagt = 0;
#ifdef N_B
#pragma unroll
        for (int k = 0; k < N_B; ++k) {{
#else
        for (int k = 0; k < nb; ++k) {{
#endif
            {real} f = flux_bt[k * N_t + i];
            avmago += f * r_abinv[k];
            avmagt += f * f * r_inv2[k];
        }}
        {real} chi2 = r_base;
        if (avmagt > 0) chi2 = r_base - (avmago * avmago) / avmagt;
        {real} dm = (avmagt > 0) ? (avmago / avmagt) : ({real})(-999.0);
        {real} chi2_pdf = chi2;
        {real} chi2_h = chi2;
#if USE_PRIORS
        if (dm > 0 && (dm <= dm_min[i] || dm >= dm_max[i])) chi2_pdf = HIGH_CHI2;
        if (dm > 0 && (dm <= hdm_min[i] || dm >= hdm_max[i])) chi2_h = HIGH_CHI2;
#endif
        if (chi2_pdf < HIGH_CHI2) {{
            {real} dchi = chi2_pdf - cmin;
            if (dchi < 0) dchi = 0;
            if (dchi <= dchi_max) {{
                int zb = z_bin[i];
                if (zb >= 0 && zb < N_z)
                    atomicAdd(&s_pdf[warp * N_z + zb], {exp}(({real})(-0.5) * dchi));
            }}
        }}
        if (chi2_h >= HIGH_CHI2) continue;
        {real} dchi = chi2_h - cmin;
        if (dchi < 0) dchi = 0;
        if (dchi > dchi_max) continue;
        {real} prob = exp({lit(-0.5)} * dchi);
        {real} lgdm = log10(dm);
        {real} lm = lgmass[i];
        if (lgdm > {lit(-100)} && lm > {lit(-100)}) {{
            atomicAdd(&s_hist[{off['MASS']} + pdf_index(lgdm + lm, {grid_args('MASS')})], prob);
            atomicAdd(&s_hist[{off['SFR']} + pdf_index(lgsfr[i] + lgdm, {grid_args('SFR')})], prob);
            atomicAdd(&s_hist[ssfr_bin[i]], prob);
            atomicAdd(&s_hist[age_bin[i]], prob);
            if (ltir[i] >= 0)
                atomicAdd(&s_hist[{off['LDUST']} + pdf_index(ltir[i] + lgdm, {grid_args('LDUST')})], prob);
        }}
        if (col_analysis) {{
            atomicAdd(&s_hist[col1_bin[i]], prob);
            atomicAdd(&s_hist[col2_bin[i]], prob);
            atomicAdd(&s_hist[{off['MREF']} + pdf_index(mref0[i] - {lit(2.5)} * lgdm, {grid_args('MREF')})], prob);
        }}
    }}
    __syncthreads();
    for (int b = (int)threadIdx.x; b < N_z; b += (int)blockDim.x) {{
        {real} a = 0;
        for (int w = 0; w < nwarps; ++w) a += s_pdf[w * N_z + b];
        pdf[o * N_z + b] = a;
    }}
    for (int b = (int)threadIdx.x; b < {N_HIST}; b += (int)blockDim.x) {{
        {acc} tot = 0;
        for (int c = 0; c < {ncopy}; ++c) tot += s_all[c * {N_HIST} + b];
        hist[(long long)o * {N_HIST} + b] = tot;
    }}
}}
"""


def fused_pass2_phys(cp, req, lib, d_inv2, d_abinv, d_base, d_cmin_gal, d_pdf, n_b, n_z, dtype,
                     use_priors: bool, threads: int, unroll: int):
    """Launch :func:`_fused_src` for every object row; fills ``d_pdf`` like the fit's
    second pass and returns the requested PDF statistics as host arrays."""
    from .cuda_template import _set_max_dynamic_smem

    dtype = np.dtype(dtype).type
    real = "double" if dtype == np.float64 else "float"
    acc_dtype = np.float64 if dtype == np.float64 else np.float32
    ncopy = int(os.environ.get("LEPHARE_ACCEL_PHYS_NCOPY", "1" if dtype == np.float64 else "4"))
    key = (real, ncopy, bool(use_priors), int(unroll))
    if key not in _FUSED_KERNELS:
        opts = ["-std=c++11", f"-DUSE_PRIORS={int(use_priors)}"]
        if unroll:
            opts.append(f"-DN_B={int(unroll)}")
        mod = cp.RawModule(code=_fused_src(real, real, ncopy), options=tuple(opts))
        _FUSED_KERNELS[key] = mod.get_function("pass2_phys_kernel")
    kernel = _FUSED_KERNELS[key]

    tensors, table = req["tensors"], req["table"]
    tkey = (id(table), np.dtype(dtype).str)
    if tkey not in _FUSED_TABLES:
        _on, hdm_min, hdm_max = _gal_priors(tensors, table.gal_idx, dtype)
        _FUSED_TABLES.clear()
        _FUSED_TABLES[tkey] = (table,) + tuple(
            cp.asarray(a, dtype=dtype)
            for a in (hdm_min, hdm_max, table.lgmass, table.lgsfr, table.ltir, table.mref0)
        ) + tuple(
            cp.asarray(a, dtype=np.int32)
            for a in (table.ssfr_bin, table.age_bin, table.col1_bin, table.col2_bin)
        )
    per_template = _FUSED_TABLES[tkey][1:]

    n_obj = int(d_base.shape[0])
    n_gal = int(lib["n_gal"])
    nbytes = np.dtype(dtype).itemsize
    nwarps = threads // 32
    smem = ((nwarps * n_z * nbytes + 15) // 16) * 16 + ncopy * N_HIST * np.dtype(acc_dtype).itemsize
    _set_max_dynamic_smem(kernel, smem)
    d_hist = cp.empty((n_obj, N_HIST), dtype=acc_dtype)
    kernel(
        (n_obj,),
        (threads,),
        (
            lib["flux_gal"], d_inv2, d_abinv, d_base,
            lib["dm_min_gal"], lib["dm_max_gal"], lib["z_bin_gal"], d_cmin_gal, d_pdf,
            np.int32(n_obj), np.int32(n_gal), np.int32(n_b), np.int32(n_z),
            *per_template, d_hist,
            dtype(_dchi_cap(dtype)), np.int32(table.col_analysis),
        ),
        shared_mem=smem,
    )
    props = pdf_properties(d_hist.astype(cp.float64), req["keys"], xp=cp)
    del d_hist
    return {k: cp.asnumpy(v) for k, v in props.items()}


def phys_hist(
    backend: str,
    mags,
    magerrs,
    context,
    offsets,
    busnorma,
    chimin_gal,
    tensors,
    table,
    dtype=np.float64,
):
    """``generatePDF`` physical histograms ``[N_obj, N_HIST]`` on ``backend``'s device."""
    if backend.startswith("jax"):
        return _phys_hist_jax(mags, magerrs, context, offsets, busnorma, chimin_gal, tensors, table, dtype)
    if backend.startswith("cuda"):
        return _phys_hist_cuda(mags, magerrs, context, offsets, busnorma, chimin_gal, tensors, table, dtype)
    ab, sab, *_ = prepare_photometry(mags, magerrs, context, tensors, offsets=offsets)
    return phys_hist_numpy(ab, sab, busnorma, chimin_gal, tensors, table, dtype)


# ------------------------------------------------------- best-fit quantities


def best_fit_scaling(ab, sab, busnorma, flux_rows) -> np.ndarray:
    """``onesource::fit`` scaling ``dm`` for one template per object (C++ band order)."""
    avmago = np.zeros(ab.shape[0])
    avmagt = np.zeros(ab.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        invsab = np.where(busnorma > 0, busnorma * 1.0 / sab, 0.0)
    for k in range(ab.shape[1]):
        inv2 = invsab[:, k] * invsab[:, k]
        avmago += flux_rows[:, k] * (ab[:, k] * inv2)
        avmagt += flux_rows[:, k] * flux_rows[:, k] * inv2
    return np.where(avmagt > 0, avmago / np.where(avmagt > 0, avmagt, 1.0), -999.0)


def best_fit_state(ab, sab, busnorma, indmin_gal, z_best, tensors, config):
    """Return ``(valid, consiz, dmmin)`` after ``onesource::interp`` (Z_METHOD=BEST)."""
    idx = np.asarray(indmin_gal, dtype=np.int64)
    valid = idx >= 0
    safe = np.where(valid, idx, 0)
    flux_rows = np.asarray(tensors.flux, dtype=np.float64)[safe]
    dm = np.where(valid, best_fit_scaling(ab, sab, busnorma, flux_rows), 0.0)
    z_lib = np.asarray(tensors.z, dtype=np.float64)[safe]
    consiz = np.where(valid, np.asarray(z_best, dtype=np.float64), INVALID_Z)
    # Z_BEST from the backends is interpolated iff tensors.z_interp (onesource::interp).
    if tensors.z_interp:
        cosmo = cosmology_from_config(config)
        rescale = np.power(10.0, 0.4 * (dist_mod(consiz, cosmo) - dist_mod(z_lib, cosmo)))
        dm = np.where(valid, dm * rescale, dm)
    else:
        consiz = np.where(valid, z_lib, INVALID_Z)
    return valid, consiz, dm


def best_fit_properties(valid, dmmin, indmin_gal, tensors) -> dict[str, np.ndarray]:
    """``compute_best_fit_physical_quantities`` + ``MOD_BEST`` / ``SCALE_BEST``."""
    n_obj = valid.size
    idx = np.where(valid, np.asarray(indmin_gal, dtype=np.int64), 0)
    out = {k: np.full(n_obj, INVALID_PHYS) for k in BEST_KEYS}
    out["MOD_BEST"] = np.where(valid, np.asarray(tensors.nummod)[idx], INVALID_INDEX).astype(np.float64)
    out["SCALE_BEST"] = np.where(valid, dmmin, 0.0)
    ok = valid & (dmmin > 0)
    dm = np.where(ok, dmmin, 1.0)
    mass = np.asarray(tensors.mass, dtype=np.float64)[idx]
    sfr = np.asarray(tensors.sfr, dtype=np.float64)[idx]
    age = np.asarray(tensors.age, dtype=np.float64)[idx]
    has_mass = ok & (mass > 0)
    lum = {"LUM_NUV_BEST": (tensors.luv, 400.0, 2300.0), "LUM_R_BEST": (tensors.lopt, 1000.0, 6000.0),
           "LUM_K_BEST": (tensors.lnir, 2000.0, 22000.0)}
    with np.errstate(divide="ignore", invalid="ignore"):
        out["MASS_BEST"] = np.where(has_mass, np.log10(mass * dm), INVALID_PHYS)
        out["SFR_BEST"] = np.where(has_mass, np.log10(sfr * dm), INVALID_PHYS)
        out["SSFR_BEST"] = np.where(has_mass, np.log10(sfr / np.where(mass > 0, mass, 1.0)), INVALID_PHYS)
        ltir = np.asarray(tensors.ltir, dtype=np.float64)[idx]
        out["LDUST_BEST"] = np.where(has_mass, ltir + np.log10(dm), INVALID_PHYS)
        out["AGE_BEST"] = np.where(ok & (age != INVALID_PHYS), np.log10(age), INVALID_PHYS)
        for key, (arr, width, lam) in lum.items():
            l0 = np.asarray(arr, dtype=np.float64)[idx]
            out[key] = np.where(ok, l0 + np.log10(3.0e18 * width / lam**2 * (dm / LSOL)), INVALID_PHYS)
    out["EBV_BEST"] = np.where(ok, np.asarray(tensors.ebv, dtype=np.float64)[idx], INVALID_PHYS)
    out["EXTLAW_BEST"] = np.where(ok, np.asarray(tensors.extlaw, dtype=np.float64)[idx] + 1, INVALID_PHYS)
    return out


def pdf_properties(hist, keys, xp=np) -> dict:
    out = {}
    names = sorted({PDF_KEYS[k][0] for k in keys if k in PDF_KEYS}, key=_PDF_NAMES.index)
    for name in names:
        o = int(_PDF_OFFSET[name])
        stats = pdf_median_interval(hist[:, o : o + PHYS_PDF_GRIDS[name][2]], name, xp)
        for suffix, pos in (("MED", 0), ("INF", 1), ("SUP", 2)):
            out[f"{name}_{suffix}"] = stats[pos]
    return {k: v for k, v in out.items() if k in keys}


# ---------------------------------------------------------------- entry point


def check_supported(tensors: MagLibTensors, config: Mapping[str, Any]) -> None:
    if not tensors.has_phys():
        raise ValueError(
            "Galaxy properties need per-template SED metadata (mass, sfr, luminosities, "
            "k-corrections) in the maglib; re-run the accelerated inform stage with this "
            "version of rail_lephare to rebuild maglib_tensors.npz."
        )
    if split_keyword(config, "Z_METHOD", "BEST", 1)[0][:1].upper() == "M":
        raise NotImplementedError("Galaxy properties with Z_METHOD=MED are not supported on accel backends")
    if keyword_bool(config, "ZFIX", "NO"):
        raise NotImplementedError("Galaxy properties with ZFIX=YES are not supported on accel backends")


def galaxy_properties(
    keys,
    mags: np.ndarray,
    magerrs: np.ndarray,
    tensors: MagLibTensors,
    fit: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    context=None,
    offsets=None,
    backend: str = "numpy",
    dtype=np.float64,
) -> dict[str, np.ndarray]:
    """Galaxy-property ``output_keys`` matching ``lp.process`` output columns.

    ``fit`` is the backend ``ancil`` with ``_INDMIN_GAL``, ``_BUSNORMA``,
    ``Z_BEST`` and ``CHI_BEST``. Returns only keys in :data:`GALAXY_PROPERTY_KEYS`.
    """
    keys = [k for k in keys if k in GALAXY_PROPERTY_KEYS]
    if not keys:
        return {}
    check_supported(tensors, config)
    mags = np.asarray(mags, dtype=np.float64)
    magerrs = np.asarray(magerrs, dtype=np.float64)
    ab, sab, *_ = prepare_photometry(mags, magerrs, context, tensors, offsets=offsets)
    busnorma = np.asarray(fit["_BUSNORMA"], dtype=np.float64)
    indmin_gal = np.asarray(fit["_INDMIN_GAL"]).astype(np.int64)
    valid, consiz, dmmin = best_fit_state(ab, sab, busnorma, indmin_gal, fit["Z_BEST"], tensors, config)

    out: dict[str, np.ndarray] = {}
    if any(k in BEST_KEYS for k in keys):
        out.update(best_fit_properties(valid, dmmin, indmin_gal, tensors))
    if any(k in PDF_KEYS for k in keys):
        table = gal_phys_table(tensors, config)
        chimin_gal = np.asarray(fit["CHI_BEST"], dtype=np.float64)
        pdf_keys = [k for k in keys if k in PDF_KEYS]
        done = np.asarray(fit.get("_PHYSDONE", np.zeros(len(busnorma))), dtype=np.float64) > 0.5
        if backend.startswith("cuda") and all(f"_PHYS:{k}" in fit for k in pdf_keys) and done.any():
            # histograms already accumulated in the fused fit; redo only the rows whose
            # final fit changed afterwards (RM_DISCREPANT band removal)
            for k in pdf_keys:
                out[k] = np.asarray(fit[f"_PHYS:{k}"], dtype=np.float64).copy()
            rows = np.flatnonzero(~done)
            if rows.size:
                ctx = None if context is None else np.asarray(context)
                ctx = ctx[rows] if ctx is not None and ctx.ndim > 0 and ctx.shape[0] == len(done) else ctx
                redo = _phys_props_cuda(
                    pdf_keys, mags[rows], magerrs[rows], ctx, offsets, busnorma[rows],
                    chimin_gal[rows], tensors, table, dtype,
                )
                for k in pdf_keys:
                    out[k][rows] = redo[k]
        elif backend.startswith("cuda"):
            out.update(
                _phys_props_cuda(
                    keys, mags, magerrs, context, offsets, busnorma, chimin_gal, tensors, table, dtype
                )
            )
        else:
            hist = phys_hist(
                backend, mags, magerrs, context, offsets, busnorma, chimin_gal, tensors, table,
                dtype=dtype,
            )
            out.update(pdf_properties(hist, keys))
    if any(k in ABSMAG_KEYS for k in keys):
        from .absmag import absolute_magnitudes

        out.update(
            absolute_magnitudes(
                mags, magerrs, context, offsets, valid, consiz, dmmin, indmin_gal, tensors, config
            )
        )
    return {k: out[k] for k in keys}
