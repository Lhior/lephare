"""JAX GPU backend for hybrid / full accelerated LePHARE estimation.

Pallas/Triton ``B=6`` kernels are not included: fused XLA still
materializes a large ``chi2`` temporary (OOM at n=1500 / Nt=210k on an 8 GB
card) and was slower than split GEMM+reduce. Dense production speed is the
CUDA fused path in ``cuda_template``.
"""

from __future__ import annotations

import os

import numpy as np

from .fp32_numerics import FP32_DCHI_MAX, fp32_amp_scale, fp32_apply_scale, mag_abs_dm_bounds
from .gpu_util import (
    choose_tiling_axis,
    suggest_chi2_chunk_size,
    suggest_cpu_obj_chunk,
    suggest_template_tile_size,
)
from .kernels_ref import (
    HIGH_CHI2,
    pdf_zmode_zmean,
    prepare_bay_zg_index,
    prepare_photometry,
    rm_discrepant,
)
from .maglib import MagLibTensors


def _require_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "backend requires JAX. Install with: pip install 'lephare[jax]'"
        ) from exc
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    flag = os.environ.get("JAX_ENABLE_X64", "1").strip().lower()
    jax.config.update("jax_enable_x64", flag not in ("0", "false", "no"))
    return jax, jnp


def ensure_gpu_backend() -> str:
    """Select CUDA as the default JAX device when present."""
    jax, _ = _require_jax()
    plats = os.environ.get("JAX_PLATFORMS", "")
    first = plats.split(",")[0].strip().lower() if plats else ""
    if first == "cpu":
        return "cpu"
    try:
        gpus = jax.devices("gpu")
        if gpus:
            jax.config.update("jax_default_device", gpus[0])
            return "gpu"
    except RuntimeError:
        pass
    return str(jax.default_backend())


def _prepare_photometry_arrays(
    mags,
    magerrs,
    context,
    offsets,
    err_scale,
    err_factor,
    gbcont,
    contforb,
    jnp,
    flux_scale=1.0,
):
    """Vectorized photometry prep (JAX-traceable)."""
    ab = jnp.where(
        jnp.isfinite(mags) & (mags < 90),
        jnp.power(10.0, -0.4 * (mags + 48.6)),
        -9999.0,
    )
    sab = jnp.where(ab > 0, ab * magerrs / 1.086, -9999.0)

    n_b = mags.shape[-1]
    sab_list = []
    for k in range(n_b):
        scale = err_scale[k] if err_scale.size > 1 else err_scale[0]
        fac = err_factor[k] if err_factor.size > 1 else err_factor[0]
        good = (sab[:, k] >= 0) & (ab[:, k] != 0.0)
        sab_mag = jnp.where(good, 1.086 * sab[:, k] / ab[:, k], 0.0)
        sab_mag = jnp.sqrt(sab_mag * sab_mag + scale * scale)
        sk = jnp.where(good, jnp.abs(ab[:, k]) * sab_mag / 1.086 * fac, sab[:, k])
        sab_list.append(sk)
    sab = jnp.stack(sab_list, axis=1)

    if offsets is not None:
        corr = jnp.power(10.0, 0.4 * offsets)
        use = (ab > 0) | (sab > 0)
        ab = jnp.where(use, ab * corr[None, :], ab)

    if context is None:
        context = jnp.zeros(mags.shape[0], dtype=jnp.int64)
        for k in range(n_b):
            mask = (mags[:, k] > 0) & jnp.isfinite(mags[:, k])
            mask = mask & (magerrs[:, k] > 0) & jnp.isfinite(magerrs[:, k])
            context = context + mask.astype(jnp.int64) * (1 << k)
    else:
        context = context.astype(jnp.int64)

    cont = jnp.where(gbcont >= 0, jnp.full_like(context, gbcont), context)
    busnorma_cols = []
    busul_cols = []
    for k in range(n_b):
        bit = jnp.int64(1 << k)
        used = jnp.where(cont <= 0, jnp.int64(1), ((cont & bit) != 0).astype(jnp.int64))
        if contforb > 0 and (contforb & (1 << k)):
            used = jnp.zeros_like(used)
        used = jnp.where((sab[:, k] < 0) & (ab[:, k] < 0), jnp.int64(0), used)
        zero_err = sab[:, k] == 0
        used = jnp.where(zero_err, jnp.int64(0), used)
        ab = ab.at[:, k].set(jnp.where(zero_err, -99.9, ab[:, k]))
        sab = sab.at[:, k].set(jnp.where(zero_err, -99.9, sab[:, k]))
        bn = jnp.where(sab[:, k] < 0, jnp.int64(0), used)
        busul_k = ((used == 1) & (sab[:, k] < 0)).astype(ab.dtype)
        busnorma_cols.append(bn.astype(ab.dtype))
        busul_cols.append(busul_k)
    busnorma = jnp.stack(busnorma_cols, axis=1)
    busul = jnp.stack(busul_cols, axis=1)
    nbul = jnp.sum(busul, axis=1).astype(jnp.int32)
    if ab.dtype == jnp.float32:
        scale = jnp.asarray(flux_scale, dtype=ab.dtype)
        ab = ab * scale
        sab = sab * scale
    return ab, sab, busnorma, busul, nbul


def _build_fit_fns(tensors: MagLibTensors, dtype, jax, jnp):
    jd = jnp.float64 if dtype == np.float64 else jnp.float32
    flux_np = np.asarray(tensors.flux, dtype=np.float64)
    amp_scale = 1.0
    if dtype == np.float32:
        amp_scale = fp32_amp_scale(flux_np)
        flux_np = fp32_apply_scale(flux_np, amp_scale)
    flux = jnp.asarray(flux_np, dtype=jd)  # [Nt, Nb]
    flux_bt = flux.T  # [Nb, Nt]
    flux_sq_bt = flux_bt * flux_bt
    types = jnp.asarray(tensors.types)
    mag0 = jnp.asarray(tensors.mag0, dtype=jd)
    z_lib = jnp.asarray(tensors.z, dtype=jd)
    prior = jnp.asarray(tensors.prior_lib, dtype=jd)
    funz0 = jnp.asarray(tensors.funz0, dtype=jd)
    err_scale = jnp.asarray(tensors.err_scale, dtype=jd)
    err_factor = jnp.asarray(tensors.err_factor, dtype=jd)
    gbcont = int(tensors.gbcont)
    contforb = int(tensors.contforb)
    threshold = float(tensors.rm_discrepant)
    gal_on = bool(tensors.prior_lib[0] < 0 and tensors.prior_lib[1] < 0)
    qso_on = bool(tensors.prior_lib[2] < 0 and tensors.prior_lib[3] < 0)
    priors_on = gal_on or qso_on
    zgrid = jnp.asarray(tensors.gridz, dtype=jd)

    index = prepare_bay_zg_index(tensors)
    gal_idx = jnp.asarray(index.gal_idx)
    bin_cols = tuple(jnp.asarray(c) for c in index.bin_cols)
    dz = jnp.asarray(index.dz, dtype=jd)
    n_z = int(index.n_z)
    n_b = int(tensors.flux.shape[1])
    n_t = int(tensors.flux.shape[0])

    z_np = np.asarray(tensors.z, dtype=np.float64)
    gridz_np = np.asarray(tensors.gridz, dtype=np.float64)
    if n_z <= 1:
        z_bin_full_np = np.zeros(n_t, dtype=np.int64)
    else:
        inv = 1.0 / (gridz_np[1] - gridz_np[0])
        z_bin_full_np = np.clip(
            np.rint((z_np - gridz_np[0]) * inv).astype(np.int64), 0, n_z - 1
        )
    z_bin_full = jnp.asarray(z_bin_full_np)

    # Packed GAL columns for pass-2 (STAR/QSO unused in BAY_ZG)
    flux_gal_bt = flux_bt[:, gal_idx] if gal_idx.size else flux_bt[:, :0]
    flux_gal_sq_bt = flux_gal_bt * flux_gal_bt
    mag0_gal = mag0[gal_idx] if gal_idx.size else mag0[:0]
    z_gal = z_lib[gal_idx] if gal_idx.size else z_lib[:0]
    z_bin_gal = z_bin_full[gal_idx] if gal_idx.size else z_bin_full[:0]
    types_gal = types[gal_idx] if gal_idx.size else types[:0]
    dm_min_np, dm_max_np = mag_abs_dm_bounds(
        np.asarray(tensors.mag0),
        np.asarray(tensors.z),
        np.asarray(tensors.types),
        np.asarray(tensors.prior_lib),
        float(tensors.funz0),
        dtype,
    )
    dm_min = jnp.asarray(dm_min_np)
    dm_max = jnp.asarray(dm_max_np)
    dm_min_gal = dm_min[gal_idx] if gal_idx.size else dm_min[:0]
    dm_max_gal = dm_max[gal_idx] if gal_idx.size else dm_max[:0]
    dchi_cap = jnp.asarray(FP32_DCHI_MAX if dtype == np.float32 else 700.0, dtype=jd)

    def _apply_priors(chi2, dm, types_t, dm_min_t, dm_max_t):
        if not priors_on:
            return chi2
        dm_ok = dm > 0
        reject = dm_ok & ((dm <= dm_min_t[None, :]) | (dm >= dm_max_t[None, :]))
        if gal_on:
            chi2 = jnp.where(reject & (types_t[None, :] == 0), HIGH_CHI2, chi2)
        if qso_on:
            chi2 = jnp.where(reject & (types_t[None, :] == 1), HIGH_CHI2, chi2)
        return chi2

    def fit_one(ab, sab, busnorma):
        invsab = jnp.where(busnorma > 0, busnorma / sab, 0.0)
        invsabSq = invsab * invsab
        s2n = ab * invsab
        abinvsabSq = ab * invsabSq
        avmago = flux @ abinvsabSq
        avmagt = jnp.sum(flux * flux * invsabSq, axis=-1)
        dm = jnp.where(avmagt > 0, avmago / avmagt, -999.0)
        inter = s2n[None, :] - dm[:, None] * flux * invsab[None, :]
        chi2 = jnp.sum(inter * inter, axis=-1)
        chi2 = _apply_priors(chi2[None, :], dm[None, :], types, dm_min, dm_max)[0]
        return chi2, dm

    def fit_batch_chi2(ab, sab, busnorma):
        """Object-batched χ² via GEMMs without materializing [N,T,B]."""
        invsab = jnp.where(busnorma > 0, busnorma / sab, 0.0)
        invsabSq = invsab * invsab
        abinvsabSq = ab * invsabSq
        s2n = ab * invsab
        avmago = abinvsabSq @ flux_bt
        avmagt = invsabSq @ flux_sq_bt
        dm = jnp.where(avmagt > 0, avmago / avmagt, -999.0)
        base = jnp.sum(s2n * s2n, axis=-1)
        chi2 = jnp.where(
            avmagt > 0,
            base[:, None] - (avmago * avmago) / avmagt,
            base[:, None],
        )
        return _apply_priors(chi2, dm, types, dm_min, dm_max)

    def chi2_template_tile(ab, sab, busnorma, flux_t, flux_sq_t, types_t, dm_min_t, dm_max_t):
        """χ² vs template tile; ``flux_*`` are ``[Nb, T]``."""
        invsab = jnp.where(busnorma > 0, busnorma / sab, 0.0)
        invsabSq = invsab * invsab
        abinvsabSq = ab * invsabSq
        s2n = ab * invsab
        avmago = abinvsabSq @ flux_t
        avmagt = invsabSq @ flux_sq_t
        dm = jnp.where(avmagt > 0, avmago / avmagt, -999.0)
        base = jnp.sum(s2n * s2n, axis=-1)
        chi2 = jnp.where(
            avmagt > 0,
            base[:, None] - (avmago * avmago) / avmagt,
            base[:, None],
        )
        chi2 = _apply_priors(chi2, dm, types_t, dm_min_t, dm_max_t)
        return jnp.where(types_t[None, :] < 0, HIGH_CHI2, chi2)

    def pdf_from_chi2(chi2):
        n_obj = chi2.shape[0]
        if n_z == 0:
            return jnp.zeros((n_obj, 0), dtype=jd)
        if gal_idx.size == 0:
            return jnp.zeros((n_obj, n_z), dtype=jd)
        chi2g = chi2[:, gal_idx]
        chimin0 = jnp.min(chi2g, axis=1, keepdims=True)
        ok = chi2g < HIGH_CHI2
        dchi = jnp.maximum(chi2g - chimin0, 0)
        ok = ok & (dchi <= dchi_cap)
        prob = jnp.where(ok, jnp.exp(-0.5 * dchi), 0.0)
        cols = []
        for bcols in bin_cols:
            if bcols.size:
                cols.append(jnp.sum(prob[:, bcols], axis=1))
            else:
                cols.append(jnp.zeros(n_obj, dtype=jd))
        pdf = jnp.stack(cols, axis=1)
        if n_z > 1 and dz.size > 0:
            somme = jnp.sum((pdf[:, :-1] + pdf[:, 1:]) * 0.5 * dz[None, :], axis=1)
            pdf = jnp.where(
                somme[:, None] > 0,
                pdf / jnp.where(somme[:, None] > 0, somme[:, None], 1.0),
                pdf,
            )
        return pdf

    def normalize_pdf(pdf):
        if n_z > 1 and dz.size > 0:
            somme = jnp.sum((pdf[:, :-1] + pdf[:, 1:]) * 0.5 * dz[None, :], axis=1)
            pdf = jnp.where(
                somme[:, None] > 0,
                pdf / jnp.where(somme[:, None] > 0, somme[:, None], 1.0),
                pdf,
            )
        return pdf

    def improved_batch(chi2):
        mins = []
        for t in range(3):
            mask = types == t
            vals = jnp.where(mask[None, :], chi2, HIGH_CHI2)
            mins.append(jnp.min(vals, axis=1))
        return jnp.min(jnp.stack(mins, axis=0), axis=0)

    def zancil_from_pdf(pdf):
        """Device zmode (argmax) + zmean; ties use first max bin."""
        idx = jnp.argmax(pdf, axis=1)
        zmode = zgrid[idx]
        denom = jnp.sum(pdf, axis=1)
        zmean = jnp.where(denom > 0, jnp.sum(pdf * zgrid[None, :], axis=1) / denom, jnp.nan)
        return zmode, zmean

    def estimate_tile(mags, magerrs, context, offsets_arr, has_context, has_offsets):
        """Fused easy path: mags → PDF/zmode/zmean + easy mask (no χ²/photometry D2H)."""
        ctx = context if has_context else None
        off = offsets_arr if has_offsets else None
        ab, sab, busnorma, busul, nbul = _prepare_photometry_arrays(
            mags, magerrs, ctx, off, err_scale, err_factor, gbcont, contforb, jnp,
            flux_scale=amp_scale,
        )
        chi2 = fit_batch_chi2(ab, sab, busnorma)
        improved = improved_batch(chi2)
        pdf = pdf_from_chi2(chi2)
        zmode, zmean = zancil_from_pdf(pdf)
        easy = (nbul == 0) & (improved <= threshold)
        return pdf, zmode, zmean, easy, nbul

    def prepare_photometry_fn(mags, magerrs, context, offsets_arr, has_context, has_offsets):
        ctx = context if has_context else None
        off = offsets_arr if has_offsets else None
        return _prepare_photometry_arrays(
            mags, magerrs, ctx, off, err_scale, err_factor, gbcont, contforb, jnp,
            flux_scale=amp_scale,
        )

    def pass1_update(chi2, types_t, z_bin_t, offset, chimin, indmin, min_zg, min_zq):
        """Accumulate per-type mins and MIN_ZG / MIN_ZQ from one template tile."""
        n_obj = chi2.shape[0]
        rows = jnp.arange(n_obj)

        def _upd_type(t, chimin, indmin):
            mask = types_t == t
            vals = jnp.where(mask[None, :], chi2, HIGH_CHI2)
            loc = jnp.argmin(vals, axis=1)
            cmin = vals[rows, loc]
            better = cmin < chimin[:, t]
            chimin = chimin.at[:, t].set(jnp.where(better, cmin, chimin[:, t]))
            indmin = indmin.at[:, t].set(
                jnp.where(better, loc.astype(indmin.dtype) + offset, indmin[:, t])
            )
            return chimin, indmin

        for t in range(3):
            chimin, indmin = _upd_type(t, chimin, indmin)

        # O(N*T) scatter-min into z bins (not O(Nz*N*T) masked reductions)
        def _scatter_min_rows(vals):
            return jax.vmap(
                lambda row: jnp.full(n_z, HIGH_CHI2, dtype=jd).at[z_bin_t].min(row)
            )(vals)

        gal_vals = jnp.where(types_t[None, :] == 0, chi2, HIGH_CHI2)
        qso_vals = jnp.where(types_t[None, :] == 1, chi2, HIGH_CHI2)
        min_zg = jnp.minimum(min_zg, _scatter_min_rows(gal_vals))
        min_zq = jnp.minimum(min_zq, _scatter_min_rows(qso_vals))
        return chimin, indmin, min_zg, min_zq

    def pass2_accumulate(chi2, types_t, z_bin_t, chimin_gal, pdf):
        """Scatter-add BAY_ZG weights using global GAL ``chimin``."""
        ok = (types_t[None, :] == 0) & (chi2 < HIGH_CHI2)
        dchi = jnp.maximum(chi2 - chimin_gal[:, None], 0)
        ok = ok & (dchi <= dchi_cap)
        prob = jnp.where(ok, jnp.exp(-0.5 * dchi), 0.0)
        n_obj = chi2.shape[0]
        flat = (jnp.arange(n_obj)[:, None] * n_z + z_bin_t[None, :]).reshape(-1)
        flat_pdf = pdf.reshape(-1)
        flat_pdf = flat_pdf.at[flat].add(prob.reshape(-1))
        return flat_pdf.reshape(n_obj, n_z)

    def fused_pass1_tile(
        ab,
        sab,
        busnorma,
        flux_t,
        flux_sq_t,
        types_t,
        dm_min_t,
        dm_max_t,
        z_bin_t,
        offset,
        chimin,
        indmin,
        min_zg,
        min_zq,
    ):
        chi2 = chi2_template_tile(
            ab, sab, busnorma, flux_t, flux_sq_t, types_t, dm_min_t, dm_max_t
        )
        return pass1_update(
            chi2, types_t, z_bin_t, offset, chimin, indmin, min_zg, min_zq
        )

    def fused_pass2_tile(
        ab, sab, busnorma, flux_t, flux_sq_t, types_t, dm_min_t, dm_max_t, z_bin_t, chimin_gal, pdf
    ):
        chi2 = chi2_template_tile(
            ab, sab, busnorma, flux_t, flux_sq_t, types_t, dm_min_t, dm_max_t
        )
        return pass2_accumulate(chi2, types_t, z_bin_t, chimin_gal, pdf)

    fit_one_jit = jax.jit(fit_one)
    fit_batch_jit = jax.jit(jax.vmap(fit_one, in_axes=(0, 0, 0)))
    fit_batch_chi2_jit = jax.jit(fit_batch_chi2)
    pdf_jit = jax.jit(pdf_from_chi2)
    improved_jit = jax.jit(improved_batch)
    estimate_tile_jit = jax.jit(estimate_tile, static_argnames=("has_context", "has_offsets"))
    prep_jit = jax.jit(prepare_photometry_fn, static_argnames=("has_context", "has_offsets"))
    chi2_tile_jit = jax.jit(chi2_template_tile)
    pass1_jit = jax.jit(pass1_update)
    pass2_jit = jax.jit(pass2_accumulate)
    fused_p1_jit = jax.jit(fused_pass1_tile)
    fused_p2_jit = jax.jit(fused_pass2_tile)
    normalize_jit = jax.jit(normalize_pdf)
    return {
        "fit_one": fit_one_jit,
        "fit_batch": fit_batch_jit,
        "fit_batch_chi2": fit_batch_chi2_jit,
        "pdf": pdf_jit,
        "improved": improved_jit,
        "estimate_tile": estimate_tile_jit,
        "prep": prep_jit,
        "chi2_tile": chi2_tile_jit,
        "pass1": pass1_jit,
        "pass2": pass2_jit,
        "fused_pass1_tile": fused_p1_jit,
        "fused_pass2_tile": fused_p2_jit,
        "normalize_pdf": normalize_jit,
        "flux": flux,
        "flux_bt": flux_bt,
        "flux_sq_bt": flux_sq_bt,
        "types": types,
        "mag0": mag0,
        "z_lib": z_lib,
        "z_bin_full": z_bin_full,
        "dm_min": dm_min,
        "dm_max": dm_max,
        "flux_gal_bt": flux_gal_bt,
        "flux_gal_sq_bt": flux_gal_sq_bt,
        "types_gal": types_gal,
        "mag0_gal": mag0_gal,
        "z_gal": z_gal,
        "z_bin_gal": z_bin_gal,
        "dm_min_gal": dm_min_gal,
        "dm_max_gal": dm_max_gal,
        "n_b": n_b,
        "n_t": n_t,
        "n_gal": int(gal_idx.size),
        "zgrid": zgrid,
        "n_z": n_z,
        "jd": jd,
        "threshold": threshold,
        "jax": jax,
        "jnp": jnp,
    }


_FIT_CACHE: dict = {}
_TILE_PAD_CACHE: dict = {}


def clear_accel_caches() -> None:
    """Drop device maglib copies so a later dtype/backend can fit in VRAM."""
    _FIT_CACHE.clear()
    _TILE_PAD_CACHE.clear()
    try:
        import jax

        jax.clear_caches()
    except Exception:
        pass


def _cache_fns(tensors: MagLibTensors, dtype, jax, jnp, platform: str):
    key = (tensors.flux.__array_interface__["data"][0], tensors.flux.shape, dtype, platform)
    if key not in _FIT_CACHE:
        clear_accel_caches()
        _FIT_CACHE[key] = _build_fit_fns(tensors, dtype, jax, jnp)
    return _FIT_CACHE[key]


def _pad_axis1(arr, pad, fill, jnp):
    if pad <= 0:
        return arr
    if arr.ndim == 1:
        return jnp.concatenate([arr, jnp.full((pad,), fill, dtype=arr.dtype)])
    zeros = jnp.zeros((arr.shape[0], pad), dtype=arr.dtype)
    return jnp.concatenate([arr, zeros], axis=1)


def _padded_libs(fns, tile_t: int):
    """Device-pad full maglib and GAL-only pack to a multiple of ``tile_t``."""
    jnp = fns["jnp"]
    probe = jnp.zeros(())
    dev = probe.device if not callable(getattr(probe, "device", None)) else probe.device()
    key = (id(fns), int(tile_t), str(dev))
    cached = _TILE_PAD_CACHE.get(key)
    if cached is not None:
        return cached
    tile_t = int(tile_t)
    n_t = int(fns["n_t"])
    n_gal = int(fns["n_gal"])
    n_pad = ((n_t + tile_t - 1) // tile_t) * tile_t
    pad = n_pad - n_t
    n_pad_g = ((max(n_gal, 1) + tile_t - 1) // tile_t) * tile_t if n_gal else 0
    pad_g = n_pad_g - n_gal if n_gal else 0

    full = {
        "flux_bt": _pad_axis1(fns["flux_bt"], pad, 0, jnp),
        "flux_sq_bt": _pad_axis1(fns["flux_sq_bt"], pad, 0, jnp),
        "types": _pad_axis1(fns["types"], pad, -1, jnp),
        "mag0": _pad_axis1(fns["mag0"], pad, 0, jnp),
        "z": _pad_axis1(fns["z_lib"], pad, 0, jnp),
        "z_bin": _pad_axis1(fns["z_bin_full"], pad, 0, jnp),
        "dm_min": _pad_axis1(fns["dm_min"], pad, 0, jnp),
        "dm_max": _pad_axis1(fns["dm_max"], pad, 0, jnp),
        "n_pad": n_pad,
        "n_tiles": n_pad // tile_t,
    }
    gal = {
        "flux_bt": _pad_axis1(fns["flux_gal_bt"], pad_g, 0, jnp),
        "flux_sq_bt": _pad_axis1(fns["flux_gal_sq_bt"], pad_g, 0, jnp),
        "types": _pad_axis1(fns["types_gal"], pad_g, -1, jnp),
        "mag0": _pad_axis1(fns["mag0_gal"], pad_g, 0, jnp),
        "z": _pad_axis1(fns["z_gal"], pad_g, 0, jnp),
        "z_bin": _pad_axis1(fns["z_bin_gal"], pad_g, 0, jnp),
        "dm_min": _pad_axis1(fns["dm_min_gal"], pad_g, 0, jnp),
        "dm_max": _pad_axis1(fns["dm_max_gal"], pad_g, 0, jnp),
        "n_pad": n_pad_g,
        "n_tiles": n_pad_g // tile_t if n_pad_g else 0,
    }
    out = {"full": full, "gal": gal, "tile_t": tile_t}
    _TILE_PAD_CACHE[key] = out
    return out


def _type_mins(chi2: np.ndarray, types: np.ndarray) -> float:
    vals = []
    for t in range(3):
        mask = types == t
        if np.any(mask):
            vals.append(float(np.min(chi2[mask])))
        else:
            vals.append(HIGH_CHI2)
    return float(np.min(vals))


def prepare_photometry_jax(
    mags: np.ndarray,
    magerrs: np.ndarray,
    tensors: MagLibTensors,
    context: np.ndarray | None = None,
    offsets: np.ndarray | None = None,
    dtype=np.float64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Device photometry prep; returns host NumPy arrays (parity helper)."""
    jax, jnp = _require_jax()
    platform = ensure_gpu_backend()
    if dtype == np.float64:
        jax.config.update("jax_enable_x64", True)
    fns = _cache_fns(tensors, dtype, jax, jnp, platform)
    has_c = context is not None
    has_o = offsets is not None
    m = jnp.asarray(mags, dtype=dtype)
    e = jnp.asarray(magerrs, dtype=dtype)
    c = jnp.asarray(context, dtype=jnp.int64) if has_c else jnp.zeros(mags.shape[0], dtype=jnp.int64)
    o = jnp.asarray(offsets, dtype=dtype) if has_o else jnp.zeros(mags.shape[1], dtype=dtype)
    ab, sab, bus, busul, nbul = fns["prep"](m, e, c, o, has_c, has_o)
    return (
        np.asarray(jax.device_get(ab)),
        np.asarray(jax.device_get(sab)),
        np.asarray(jax.device_get(bus)),
        np.asarray(jax.device_get(busul)),
        np.asarray(jax.device_get(nbul)),
    )


def chi2_batch_jax(
    ab: np.ndarray,
    sab: np.ndarray,
    busnorma: np.ndarray,
    tensors: MagLibTensors,
    dtype=np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched χ² for ``[N_obj, N_b]`` photometry → ``chi2[N_obj, N_t]``."""
    jax, jnp = _require_jax()
    platform = ensure_gpu_backend()
    if dtype == np.float64:
        jax.config.update("jax_enable_x64", True)
    fns = _cache_fns(tensors, dtype, jax, jnp, platform)
    chi2, dm = fns["fit_batch"](
        jnp.asarray(ab, dtype=dtype),
        jnp.asarray(sab, dtype=dtype),
        jnp.asarray(busnorma, dtype=dtype),
    )
    return np.asarray(jax.device_get(chi2)), np.asarray(jax.device_get(dm))


def pdf_batch_jax(
    chi2: np.ndarray,
    tensors: MagLibTensors,
    dtype=np.float64,
) -> np.ndarray:
    """Device BAY_ZG PDF; ``chi2`` may be host or device array."""
    jax, jnp = _require_jax()
    platform = ensure_gpu_backend()
    if dtype == np.float64:
        jax.config.update("jax_enable_x64", True)
    fns = _cache_fns(tensors, dtype, jax, jnp, platform)
    return np.asarray(jax.device_get(fns["pdf"](jnp.asarray(chi2, dtype=dtype))), dtype=np.float64)


def _finish_hard_rows(
    hard_acc,
    mags,
    magerrs,
    tensors,
    ctx_full,
    has_c,
    has_o,
    offsets,
    fns,
    ancil,
    pdfs,
    dtype,
    jax,
    jnp,
):
    """Host RM_DISCREPANT for hard/UL; fill pdfs + ancil for those rows."""
    from .bestfit import bestfit_ancil_from_chi2, merge_ancil

    if not hard_acc:
        return
    fit_batch = fns["fit_batch"]
    pdf_fn = fns["pdf"]
    types_np = tensors.types
    threshold = tensors.rm_discrepant
    n_b = mags.shape[1]
    idxs = np.asarray([h[0] for h in hard_acc], dtype=np.int64)
    ctx = ctx_full[idxs] if has_c else None
    ab, sab, busnorma, busul, nbul = prepare_photometry(
        mags[idxs],
        magerrs[idxs],
        ctx,
        tensors,
        offsets=offsets if has_o else None,
    )
    ul_rows = np.flatnonzero(nbul > 0)
    hard_rows = np.flatnonzero(nbul == 0)
    chi2_by_local: dict[int, np.ndarray] = {}
    bus_by_local: dict[int, np.ndarray] = {}

    for row in ul_rows:
        i = int(idxs[row])
        c2, _, _, bus = rm_discrepant(
            ab[row], sab[row], busnorma[row], busul[row], int(nbul[row]), tensors
        )
        pdfs[i] = np.asarray(
            jax.device_get(pdf_fn(jnp.asarray(c2[None, :], dtype=dtype)))
        )[0]
        chi2_by_local[int(row)] = np.asarray(c2, dtype=np.float64)
        bus_by_local[int(row)] = bus

    if hard_rows.size:
        hard_chunk = 512
        for hs in range(0, hard_rows.size, hard_chunk):
            he = min(hs + hard_chunk, hard_rows.size)
            rows = hard_rows[hs:he]
            c2_all, _ = fit_batch(
                jnp.asarray(ab[rows], dtype=dtype),
                jnp.asarray(sab[rows], dtype=dtype),
                jnp.asarray(busnorma[rows], dtype=dtype),
            )
            c2_all = np.asarray(jax.device_get(c2_all))
            buses = [busnorma[r].copy() for r in rows]
            chi2_final = list(c2_all)
            improved = [_type_mins(c2_all[t], types_np) for t in range(len(rows))]

            for _pass in range(2):
                active = [t for t, imp in enumerate(improved) if imp > threshold]
                if not active:
                    break
                var_ab = []
                var_sab = []
                var_bus = []
                owners = []
                for t in active:
                    row = int(rows[t])
                    bus = buses[t]
                    for k in range(n_b):
                        if bus[k] == 1:
                            b2 = bus.copy()
                            b2[k] = 0
                            var_ab.append(ab[row])
                            var_sab.append(sab[row])
                            var_bus.append(b2)
                            owners.append((t, k))
                if not var_bus:
                    break
                c2_vars, _ = fit_batch(
                    jnp.asarray(np.stack(var_ab), dtype=dtype),
                    jnp.asarray(np.stack(var_sab), dtype=dtype),
                    jnp.asarray(np.stack(var_bus), dtype=dtype),
                )
                c2_vars = np.asarray(jax.device_get(c2_vars))
                best = {t: (improved[t], -1, None) for t in active}
                for v, (t, k) in enumerate(owners):
                    newmin = _type_mins(c2_vars[v], types_np)
                    if newmin < best[t][0]:
                        best[t] = (newmin, k, c2_vars[v])
                for t, (newmin, k, c2_best) in best.items():
                    if k >= 0:
                        buses[t][k] = 0
                        improved[t] = newmin
                        chi2_final[t] = c2_best

            c2_stack = np.stack(chi2_final, axis=0)
            pdf_hard = np.asarray(
                jax.device_get(pdf_fn(jnp.asarray(c2_stack, dtype=dtype)))
            )
            for t, row in enumerate(rows):
                i = int(idxs[row])
                pdfs[i] = pdf_hard[t]
                chi2_by_local[int(row)] = np.asarray(c2_stack[t], dtype=np.float64)
                bus_by_local[int(row)] = buses[t]

    ordered = np.asarray(sorted(chi2_by_local.keys()), dtype=np.int64)
    if ordered.size:
        c2_stack = np.stack([chi2_by_local[int(r)] for r in ordered], axis=0)
        bus_stack = np.stack([bus_by_local[int(r)] for r in ordered], axis=0)
        merge_ancil(
            ancil,
            bestfit_ancil_from_chi2(c2_stack, tensors, busnorma=bus_stack),
            idxs[ordered],
        )


def _process_template_tiles(
    mags,
    magerrs,
    tensors,
    context,
    offsets,
    dtype,
    return_ancil,
    tile_t: int,
    jax,
    jnp,
    fns,
):
    """Two-pass FP64 template tiling: global min then BAY_ZG scatter-add.

    Maglib is device-padded once. Pass 2 streams GAL templates only.
    Set ``LEPHARE_ACCEL_FUSED_CHI2=1`` to jit χ²+reduce together (often slower
    on consumer GPUs; default is split GEMM then reduce).
    """
    from .bestfit import bestfit_ancil_from_type_mins, empty_ancil, merge_ancil

    n_obj, n_b = mags.shape
    n_z = int(fns["n_z"])
    threshold = float(fns["threshold"])
    jd = fns["jd"]
    tile_t = int(tile_t)
    use_fused = os.environ.get("LEPHARE_ACCEL_FUSED_CHI2", "0") == "1"

    has_c = context is not None
    has_o = offsets is not None
    ctx_full = (
        np.asarray(context, dtype=np.int64)
        if has_c
        else np.zeros(n_obj, dtype=np.int64)
    )
    off_arr = (
        np.asarray(offsets, dtype=dtype)
        if has_o
        else np.zeros(n_b, dtype=dtype)
    )

    libs = _padded_libs(fns, tile_t)
    full, gal = libs["full"], libs["gal"]

    ab, sab, busnorma, busul, nbul = fns["prep"](
        jnp.asarray(mags, dtype=dtype),
        jnp.asarray(magerrs, dtype=dtype),
        jnp.asarray(ctx_full, dtype=jnp.int64),
        jnp.asarray(off_arr, dtype=dtype),
        has_c,
        has_o,
    )
    nbul_np = np.asarray(jax.device_get(nbul))
    busnorma_np = np.asarray(jax.device_get(busnorma), dtype=np.float64)

    chimin = jnp.full((n_obj, 3), HIGH_CHI2, dtype=jd)
    indmin = jnp.full((n_obj, 3), -99, dtype=jnp.int64)
    min_zg = jnp.full((n_obj, n_z), HIGH_CHI2, dtype=jd)
    min_zq = jnp.full((n_obj, n_z), HIGH_CHI2, dtype=jd)

    chi2_tile = fns["chi2_tile"]
    pass1 = fns["pass1"]
    pass2 = fns["pass2"]
    fused_p1 = fns["fused_pass1_tile"]
    fused_p2 = fns["fused_pass2_tile"]

    for ti in range(full["n_tiles"]):
        s = ti * tile_t
        e = s + tile_t
        ft = full["flux_bt"][:, s:e]
        fsq = full["flux_sq_bt"][:, s:e]
        ty = full["types"][s:e]
        dmin = full["dm_min"][s:e]
        dmax = full["dm_max"][s:e]
        zb = full["z_bin"][s:e]
        if use_fused:
            chimin, indmin, min_zg, min_zq = fused_p1(
                ab, sab, busnorma, ft, fsq, ty, dmin, dmax, zb, s, chimin, indmin, min_zg, min_zq
            )
        else:
            c2 = chi2_tile(ab, sab, busnorma, ft, fsq, ty, dmin, dmax)
            chimin, indmin, min_zg, min_zq = pass1(
                c2, ty, zb, s, chimin, indmin, min_zg, min_zq
            )

    chimin_gal = chimin[:, 0]
    pdf = jnp.zeros((n_obj, n_z), dtype=jd)
    for ti in range(gal["n_tiles"]):
        s = ti * tile_t
        e = s + tile_t
        ft = gal["flux_bt"][:, s:e]
        fsq = gal["flux_sq_bt"][:, s:e]
        ty = gal["types"][s:e]
        dmin = gal["dm_min"][s:e]
        dmax = gal["dm_max"][s:e]
        zb = gal["z_bin"][s:e]
        if use_fused:
            pdf = fused_p2(ab, sab, busnorma, ft, fsq, ty, dmin, dmax, zb, chimin_gal, pdf)
        else:
            c2 = chi2_tile(ab, sab, busnorma, ft, fsq, ty, dmin, dmax)
            pdf = pass2(c2, ty, zb, chimin_gal, pdf)

    pdf = fns["normalize_pdf"](pdf)
    pdfs = np.array(jax.device_get(pdf), dtype=np.float64, copy=True)
    chimin_np = np.array(jax.device_get(chimin), dtype=np.float64, copy=True)
    indmin_np = np.array(jax.device_get(indmin), dtype=np.int64, copy=True)
    min_zg_np = np.array(jax.device_get(min_zg), dtype=np.float64, copy=True)
    min_zq_np = np.array(jax.device_get(min_zq), dtype=np.float64, copy=True)

    improved = np.min(chimin_np, axis=1)
    easy_mask = (nbul_np == 0) & (improved <= threshold)
    if dtype == np.float32:
        easy_mask = np.ones(n_obj, dtype=bool)
    hard_acc = [(int(i), int(nbul_np[i])) for i in np.flatnonzero(~easy_mask)]

    ancil = empty_ancil(n_obj, n_b)
    # Easy ancil from pass-1 winners (no second full χ²)
    easy_idxs = np.flatnonzero(easy_mask)
    if easy_idxs.size:
        merge_ancil(
            ancil,
            bestfit_ancil_from_type_mins(
                indmin_np[easy_idxs],
                chimin_np[easy_idxs],
                tensors,
                min_zg=min_zg_np[easy_idxs],
                min_zq=min_zq_np[easy_idxs],
                busnorma=busnorma_np[easy_idxs],
            ),
            easy_idxs,
        )

    _finish_hard_rows(
        hard_acc,
        mags,
        magerrs,
        tensors,
        ctx_full,
        has_c,
        has_o,
        offsets,
        fns,
        ancil,
        pdfs,
        dtype,
        jax,
        jnp,
    )

    zgrid = tensors.gridz.astype(np.float64, copy=False)
    zmode, zmean = pdf_zmode_zmean(pdfs, zgrid)
    if not return_ancil:
        return pdfs, zgrid
    return pdfs, zgrid, zmode, zmean, ancil


def _process_object_tiles(
    mags,
    magerrs,
    tensors,
    context,
    offsets,
    dtype,
    return_ancil,
    chunk: int,
    jax,
    jnp,
    fns,
):
    """Existing object-tile fused easy path (large-N / small-Nt fallback)."""
    from .bestfit import bestfit_ancil_from_chi2, empty_ancil, merge_ancil

    estimate_tile = fns["estimate_tile"]
    fit_batch = fns["fit_batch"]
    n_obj, n_b = mags.shape
    n_z = len(tensors.gridz)
    pdfs = np.zeros((n_obj, n_z), dtype=np.float64)
    easy_mask = np.ones(n_obj, dtype=bool)
    ancil = empty_ancil(n_obj, n_b)

    has_c = context is not None
    has_o = offsets is not None
    ctx_full = (
        np.asarray(context, dtype=np.int64)
        if has_c
        else np.zeros(n_obj, dtype=np.int64)
    )
    off_arr = (
        np.asarray(offsets, dtype=dtype)
        if has_o
        else np.zeros(n_b, dtype=dtype)
    )

    def _consume_easy(start, end, dev, hard_acc):
        pdf, _zm, _zmn, easy, nbul = jax.device_get(dev)
        easy = np.asarray(easy).astype(bool)
        nbul_np = np.asarray(nbul)
        pdfs[start:end] = np.asarray(pdf)
        easy_mask[start:end] = easy
        for j in np.flatnonzero(~easy):
            hard_acc.append((start + int(j), int(nbul_np[j])))

    def _fill_easy_ancil():
        """Fold ancil via object-chunk χ² only for winners needed by bestfit.

        Still one extra χ² pass on the object axis (fits in VRAM by construction).
        """
        easy_idxs = np.flatnonzero(easy_mask)
        if easy_idxs.size == 0:
            return
        easy_chunk = max(256, min(chunk, 2048))
        for es in range(0, easy_idxs.size, easy_chunk):
            ee = min(es + easy_chunk, easy_idxs.size)
            idxs = easy_idxs[es:ee]
            ctx = ctx_full[idxs] if has_c else None
            ab, sab, busnorma, _busul, _nbul = prepare_photometry(
                mags[idxs],
                magerrs[idxs],
                ctx,
                tensors,
                offsets=offsets if has_o else None,
            )
            c2, _ = fit_batch(
                jnp.asarray(ab, dtype=dtype),
                jnp.asarray(sab, dtype=dtype),
                jnp.asarray(busnorma, dtype=dtype),
            )
            c2 = np.asarray(jax.device_get(c2), dtype=np.float64)
            merge_ancil(ancil, bestfit_ancil_from_chi2(c2, tensors, busnorma=busnorma), idxs)

    def _launch(start, end):
        return estimate_tile(
            jnp.asarray(mags[start:end], dtype=dtype),
            jnp.asarray(magerrs[start:end], dtype=dtype),
            jnp.asarray(ctx_full[start:end], dtype=jnp.int64),
            jnp.asarray(off_arr, dtype=dtype),
            has_c,
            has_o,
        )

    for _attempt in range(8):
        try:
            end0 = min(chunk, n_obj)
            probe = _launch(0, end0)
            jax.block_until_ready(probe)
            del probe
            break
        except Exception as exc:
            msg = str(exc).lower()
            if "resource_exhausted" not in msg and "out of memory" not in msg:
                raise
            chunk = max(256, chunk // 2)
            if _attempt == 7:
                raise

    hard_acc: list = []
    pending = None
    for start in range(0, n_obj, chunk):
        end = min(start + chunk, n_obj)
        dev = _launch(start, end)
        if pending is not None:
            _consume_easy(*pending, hard_acc)
        pending = (start, end, dev)
    if pending is not None:
        _consume_easy(*pending, hard_acc)
    if dtype == np.float32:
        hard_acc = []
        easy_mask[:] = True

    _finish_hard_rows(
        hard_acc,
        mags,
        magerrs,
        tensors,
        ctx_full,
        has_c,
        has_o,
        offsets,
        fns,
        ancil,
        pdfs,
        dtype,
        jax,
        jnp,
    )
    _fill_easy_ancil()

    zgrid = tensors.gridz.astype(np.float64, copy=False)
    zmode, zmean = pdf_zmode_zmean(pdfs, zgrid)
    if not return_ancil:
        return pdfs, zgrid
    return pdfs, zgrid, zmode, zmean, ancil


def _process_cpu_object_chunks(
    mags,
    magerrs,
    tensors,
    context,
    offsets,
    dtype,
    return_ancil,
    obj_chunk: int,
    tile_t: int,
    jax,
    jnp,
    fns,
):
    """CPU estimate: small object batches + large template tiles.

    A full 4k-object GEMM is ``[N, 9] @ [9, T]`` and Eigen typically sits on
    ~2 threads. Chunking ``N`` lets the template axis stay wide.
    """
    from .bestfit import empty_ancil, merge_ancil

    n_obj = int(mags.shape[0])
    obj_chunk = max(1, int(obj_chunk))
    n_z = int(len(tensors.gridz))
    pdfs = np.zeros((n_obj, n_z), dtype=np.float64)
    zmode = np.zeros(n_obj, dtype=np.float64)
    zmean = np.zeros(n_obj, dtype=np.float64)
    ancil = empty_ancil(n_obj, int(mags.shape[1])) if return_ancil else None
    zgrid = None
    for start in range(0, n_obj, obj_chunk):
        end = min(start + obj_chunk, n_obj)
        ctx = context[start:end] if context is not None else None
        out = _process_template_tiles(
            mags[start:end],
            magerrs[start:end],
            tensors,
            ctx,
            offsets,
            dtype,
            return_ancil,
            tile_t,
            jax,
            jnp,
            fns,
        )
        if return_ancil:
            pdf, zgrid, zm, zmn, an = out
            zmode[start:end] = zm
            zmean[start:end] = zmn
            merge_ancil(ancil, an, np.arange(start, end))
        else:
            pdf, zgrid = out
        pdfs[start:end] = pdf
    if not return_ancil:
        return pdfs, zgrid
    return pdfs, zgrid, zmode, zmean, ancil


def process_batch_jax(
    mags: np.ndarray,
    magerrs: np.ndarray,
    tensors: MagLibTensors,
    context: np.ndarray | None = None,
    offsets: np.ndarray | None = None,
    dtype=np.float64,
    return_ancil: bool = True,
) -> tuple:
    """Device-native estimate: mags→χ²→PDF; host RM for hard/UL only.

    Chooses template-axis tiling for large ``N_t`` / modest ``N``, else object
    tiles. FP64 BAY_ZG / ancil math is unchanged.

    Returns
    -------
    pdfs, zgrid[, zmode, zmean, ancil]
    """
    jax, jnp = _require_jax()
    platform = ensure_gpu_backend()
    if dtype == np.float64:
        jax.config.update("jax_enable_x64", True)

    fns = _cache_fns(tensors, dtype, jax, jnp, platform)
    n_obj = int(mags.shape[0])
    n_t = int(tensors.flux.shape[0])

    if platform == "cpu":
        obj_chunk = suggest_cpu_obj_chunk(n_obj)
        tile_t = suggest_template_tile_size(
            obj_chunk,
            n_t,
            dtype=dtype,
            host_memory=True,
            max_tile=262_144,
        )
        return _process_cpu_object_chunks(
            mags,
            magerrs,
            tensors,
            context,
            offsets,
            dtype,
            return_ancil,
            obj_chunk,
            tile_t,
            jax,
            jnp,
            fns,
        )

    axis, size = choose_tiling_axis(n_obj, n_t, dtype=dtype)

    def _run(axis_name: str, tile_or_chunk: int, obj_chunk: int | None = None):
        if axis_name == "template":
            if obj_chunk is not None and obj_chunk < n_obj:
                return _process_cpu_object_chunks(
                    mags,
                    magerrs,
                    tensors,
                    context,
                    offsets,
                    dtype,
                    return_ancil,
                    obj_chunk,
                    tile_or_chunk,
                    jax,
                    jnp,
                    fns,
                )
            return _process_template_tiles(
                mags,
                magerrs,
                tensors,
                context,
                offsets,
                dtype,
                return_ancil,
                tile_or_chunk,
                jax,
                jnp,
                fns,
            )
        return _process_object_tiles(
            mags,
            magerrs,
            tensors,
            context,
            offsets,
            dtype,
            return_ancil,
            tile_or_chunk if tile_or_chunk > 0 else suggest_chi2_chunk_size(n_t, dtype=dtype),
            jax,
            jnp,
            fns,
        )

    try:
        return _run(axis, size)
    except Exception as exc:
        msg = str(exc)
        if "RESOURCE_EXHAUSTED" not in msg and "out of memory" not in msg.lower():
            raise
        print("JAX estimate OOM; retrying with object chunks", flush=True)
        clear_accel_caches()
        fns = _cache_fns(tensors, dtype, jax, jnp, platform)
        tile_t = min(4096, suggest_template_tile_size(min(256, n_obj), n_t, dtype=dtype))
        obj_chunk = min(256, n_obj)
        return _run("template", tile_t, obj_chunk=obj_chunk)
