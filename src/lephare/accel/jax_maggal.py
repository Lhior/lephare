"""JAX GPU MagGal: rest-frame SEDs → dense ``MagLibTensors`` without C++ MagGal.

Host: extinction, IGM merge-resample, emission lines (match ``SED.cpp``).
Device: ``vmap`` over redshift of filter integrals + AB mag→flux.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .mag_conv import (
    FilterCurve,
    SedSpectrum,
    device_ab_mag_to_flux,
    device_compute_fluxes_batch,
    iter_sed_bin,
    load_filters_from_config,
)
from .mag_physics import (
    ExtLaw,
    HIGH_MAG,
    INVALID_MAG,
    LYA_SPLIT,
    OpaTable,
    apply_ext_lines,
    apply_extinction_cpp,
    apply_opa_cpp,
    compute_fluxes_cpp,
    compute_magnitudes_cpp,
    generate_em_emp_sfr,
    generate_em_emp_uv,
    generate_em_spectra,
    load_extinction_law_file,
    load_opa_tables,
    mag_plus_lines,
    opa_index,
    parse_csv_floats,
    parse_csv_ints,
    redshift_spectrum,
    rescale_em_lines,
    warning_integrate_sed,
)
from .maglib import MagLibTensors, _parse_doubles

C_LIGHT_A_S = 2.99792458e18


@dataclass
class MagGalSetup:
    filters: list[FilterCurve]
    ext_laws: list[ExtLaw]
    mw_ext: ExtLaw
    ebv: np.ndarray
    modext: np.ndarray  # [nextlaw, 2]
    frac_em: np.ndarray
    emlines: str
    add_dust: bool
    gridz: np.ndarray
    grid_t: np.ndarray
    grid_dm: np.ndarray
    opa_all: list[OpaTable]
    mabs_ref: int
    add_emlines: tuple[int, int]


def _lib_bin_path(name: str) -> Path:
    return Path(os.environ["LEPHAREWORK"]) / "lib_bin" / f"{name}.bin"


def _split_list(cfg: dict[str, Any], key: str, default: str) -> list[str]:
    raw = str(cfg.get(key, default))
    return [p.strip() for p in raw.replace(",", " ").split() if p.strip()]


def maggal_setup_from_config(config: dict[str, Any]) -> MagGalSetup:
    import lephare as lp

    filters = load_filters_from_config(config)
    lepharedir = os.environ.get("LEPHAREDIR", getattr(lp, "LEPHAREDIR", ""))
    ext_names = _split_list(config, "EXTINC_LAW", "calzetti.dat")
    ext_laws = [
        load_extinction_law_file(Path(lepharedir) / "ext" / name, i)
        for i, name in enumerate(ext_names)
    ]
    mw_ext = load_extinction_law_file(Path(lepharedir) / "ext" / "MW_seaton.dat", len(ext_laws))
    ebv = np.asarray(parse_csv_floats(str(config.get("EB_V", "0"))), dtype=np.float64)
    nextlaw = len(ext_laws)
    mod = parse_csv_ints(str(config.get("MOD_EXTINC", "0,0")))
    if len(mod) < nextlaw * 2:
        mod = (mod + [0] * (nextlaw * 2))[: nextlaw * 2]
    modext = np.asarray(mod, dtype=np.int32).reshape(nextlaw, 2)
    emlines = str(config.get("EM_LINES", "NO")).split()[0].strip()
    if emlines[:1].upper() in ("Y",):
        emlines = "EMP_UV"
    if emlines[:2].upper() == "NO":
        frac_em = np.array([1.0], dtype=np.float64)
    else:
        frac_em = np.asarray(
            parse_csv_floats(str(config.get("EM_DISPERSION", "1"))), dtype=np.float64
        )
        if frac_em.size == 0:
            frac_em = np.array([1.0], dtype=np.float64)
    zstep = parse_csv_floats(str(config.get("Z_STEP", "0.04,0,6")))
    dz, zmin, zmax = zstep[0], zstep[1], zstep[2]
    gridz = np.asarray(lp.zgrid(dz, zmin, zmax), dtype=np.float64)
    cosmo_vals = _parse_doubles(config, "COSMOLOGY", "70,0.3,0.7")
    lcdm = lp.cosmo(cosmo_vals[0], cosmo_vals[1], cosmo_vals[2])
    grid_t = np.asarray(lcdm.time(gridz), dtype=np.float64)
    grid_dm = np.asarray(lcdm.distMod(gridz), dtype=np.float64)
    opa_all = load_opa_tables(lepharedir)
    mabs_ref = int(float(str(config.get("MAG_REF", "3")).split(",")[0])) - 1
    mabs_ref = max(mabs_ref, 0)
    em_mod = parse_csv_ints(str(config.get("ADD_EMLINES", "-9999,-9999")))
    if len(em_mod) < 2:
        em_mod = [-9999, -9999]
    add_dust = str(config.get("ADD_DUSTEM", "NO")).strip().upper() in ("YES", "Y", "TRUE", "1")
    return MagGalSetup(
        filters=filters,
        ext_laws=ext_laws,
        mw_ext=mw_ext,
        ebv=ebv,
        modext=modext,
        frac_em=frac_em,
        emlines=emlines,
        add_dust=add_dust,
        gridz=gridz,
        grid_t=grid_t,
        grid_dm=grid_dm,
        opa_all=opa_all,
        mabs_ref=mabs_ref,
        add_emlines=(int(em_mod[0]), int(em_mod[1])),
    )


def _ext_applies(nummod: int, i_law: int, ebv: float, modext: np.ndarray) -> bool:
    if ebv < 1e-10:
        return i_law == 0
    return nummod >= int(modext[i_law, 0]) and nummod <= int(modext[i_law, 1])


def _stack_opa(opa_all: list[OpaTable]) -> tuple[np.ndarray, np.ndarray]:
    nmax = max(int(t.lamb.size) for t in opa_all)
    ntab = len(opa_all)
    ol = np.zeros((ntab, nmax), dtype=np.float64)
    ot = np.ones((ntab, nmax), dtype=np.float64)
    for i, t in enumerate(opa_all):
        n = int(t.lamb.size)
        ol[i, :n] = t.lamb
        ot[i, :n] = t.trans
        if n < nmax:
            ol[i, n:] = t.lamb[-1]
            ot[i, n:] = t.trans[-1]
    return ol, ot


def _pad_spectra(lambs: Sequence[np.ndarray], fluxes: Sequence[np.ndarray]):
    n = len(lambs)
    nmax = max(int(a.size) for a in lambs)
    pad_l = np.zeros((n, nmax), dtype=np.float64)
    pad_f = np.zeros((n, nmax), dtype=np.float64)
    for i, (l, f) in enumerate(zip(lambs, fluxes)):
        pad_l[i, : l.size] = l
        pad_f[i, : f.size] = f
        if l.size < nmax:
            # jnp.interp requires strictly increasing xp; repeating λ poisons sparse SEDs.
            pad_l[i, l.size :] = l[-1] + np.arange(1, nmax - l.size + 1, dtype=np.float64) * 0.01
            pad_f[i, l.size :] = 0.0
    return pad_l, pad_f


def _filter_pack(filters: list[FilterCurve], dtype):
    import jax.numpy as jnp

    nf = max(int(f.lamb.size) for f in filters)
    n_b = len(filters)
    fl = np.zeros((n_b, nf), dtype=np.float64)
    ft = np.zeros((n_b, nf), dtype=np.float64)
    lmin = np.empty(n_b, dtype=np.float64)
    lmax = np.empty(n_b, dtype=np.float64)
    fcorr = np.empty(n_b, dtype=np.float64)
    for i, flt in enumerate(filters):
        n = int(flt.lamb.size)
        fl[i, :n] = flt.lamb
        ft[i, :n] = flt.trans
        if n < nf:
            fl[i, n:] = flt.lamb[-1] + np.arange(1, nf - n + 1, dtype=np.float64) * 0.01
            ft[i, n:] = 0.0
        lmin[i] = flt.lmin
        lmax[i] = flt.lmax
        fcorr[i] = flt.fcorr
    return (
        jnp.asarray(fl, dtype=dtype),
        jnp.asarray(ft, dtype=dtype),
        jnp.asarray(lmin, dtype=dtype),
        jnp.asarray(lmax, dtype=dtype),
        jnp.asarray(fcorr, dtype=dtype),
    )


def _jax_union_mags_fn(filters: list[FilterCurve], dtype=np.float64, *, as_fnu: bool = False):
    """Jitted kernel: rest-frame combos × z → AB mags or line ``fmel/arean`` fluxes."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    fl, ft, lmin, lmax, fcorr = _filter_pack(filters, dtype)

    def mag_one_filter(sed_l, sed_f, dm, fl_i, ft_i, lmin_i, lmax_i, fcorr_i):
        invalid = (sed_l[0] >= lmin_i) | (sed_l[-1] <= lmax_i)
        grid = jnp.sort(jnp.concatenate([sed_l, fl_i]))
        sed_i = jnp.interp(grid, sed_l, sed_f, left=0.0, right=0.0)
        tr_i = jnp.interp(grid, fl_i, ft_i, left=0.0, right=0.0)
        tr_i = jnp.where((grid >= lmin_i) & (grid <= lmax_i), tr_i, 0.0)
        dl = grid[1:] - grid[:-1]
        lmean = 0.5 * (grid[1:] + grid[:-1])
        tmean = 0.5 * (tr_i[1:] + tr_i[:-1])
        fmean = 0.5 * (sed_i[1:] + sed_i[:-1])
        conv = C_LIGHT_A_S / (lmean * lmean)
        arean = jnp.sum(tmean * conv * dl)
        fmel = jnp.sum(fmean * tmean * dl)
        ratio = fmel / arean
        if as_fnu:
            # ``SED::compute_fluxes`` (no fcorr) after ``rescale(10**(-0.4*distMod))``.
            val = ratio * jnp.power(10.0, -0.4 * dm)
            val = jnp.where((fmel > 0) & (arean > 0), val, jnp.asarray(0.0, dtype=dtype))
            return jnp.where(invalid, jnp.asarray(0.0, dtype=dtype), val)
        mag = -2.5 * jnp.log10(ratio * fcorr_i) - 48.6 + dm
        mag = jnp.where((fmel > 0) & (arean > 0), mag, jnp.asarray(HIGH_MAG, dtype=dtype))
        return jnp.where(invalid, jnp.asarray(INVALID_MAG, dtype=dtype), mag)

    def mags_sed(sed_l, sed_f, dm):
        return jax.vmap(mag_one_filter, in_axes=(None, None, None, 0, 0, 0, 0, 0))(
            sed_l, sed_f, dm, fl, ft, lmin, lmax, fcorr
        )

    fill = 0.0 if as_fnu else HIGH_MAG

    def combo_all_z(sed_l, sed_f, zs, dms, opa_l, opa_t, opa_idx, age_ok):
        def at_z(z, dm, idx, ok):
            transm = jnp.interp(sed_l, opa_l[idx], opa_t[idx], left=1.0, right=1.0)
            flux = jnp.where(sed_l < LYA_SPLIT, sed_f * transm, sed_f)
            lo_obs = sed_l[0] * (1.0 + z)
            hi_obs = sed_l[-1] * (1.0 + z)
            need_blue = lo_obs > jnp.min(lmin)
            need_red = hi_obs < jnp.max(lmax)
            lamb = sed_l * (1.0 + z)
            flux = flux / (1.0 + z)
            # Keep λ strictly increasing: never duplicate endpoints (breaks jnp.interp).
            eps = jnp.asarray(1.0e-8, dtype=dtype)
            lo = jnp.where(need_blue, jnp.asarray(0.0, dtype=dtype), lamb[0] - eps)
            hi = jnp.where(
                need_red,
                jnp.asarray(1.0e8 * (1.0 + z), dtype=dtype),
                lamb[-1] + eps,
            )
            lamb = jnp.concatenate([lo[None], lamb, hi[None]])
            flux = jnp.concatenate(
                [
                    jnp.where(need_blue, jnp.asarray(0.0, dtype=dtype), flux[0:1]),
                    flux,
                    jnp.where(need_red, jnp.asarray(0.0, dtype=dtype), flux[-1:]),
                ]
            )
            vals = mags_sed(lamb, flux, dm)
            return jnp.where(ok, vals, jnp.asarray(fill, dtype=dtype))

        return jax.vmap(at_z)(zs, dms, opa_idx, age_ok)

    return jax.jit(jax.vmap(combo_all_z, in_axes=(0, 0, None, None, None, None, None, 0)))


_UNION_MAGS_CACHE: dict = {}


def jax_continuum_mags_grid(
    rest_l: np.ndarray,
    rest_f: np.ndarray,
    age_ok: np.ndarray,
    setup: MagGalSetup,
    dtype=np.float64,
    *,
    as_fnu: bool = False,
) -> np.ndarray:
    """``(N_combo, N_z, N_band)`` AB mags on GPU: IGM, redshift, union-grid integrals.

    One compiled kernel over all combos and redshifts (not a Python z loop).
    """
    import jax
    import jax.numpy as jnp

    key = (int(rest_l.shape[1]), len(setup.filters), bool(as_fnu))
    fn = _UNION_MAGS_CACHE.get(key)
    if fn is None:
        fn = _jax_union_mags_fn(setup.filters, dtype=dtype, as_fnu=as_fnu)
        _UNION_MAGS_CACHE[key] = fn
    opa_l, opa_t = _stack_opa(setup.opa_all)
    opa_idx = np.asarray([opa_index(float(z)) for z in setup.gridz], dtype=np.int32)
    n_c, nwave = int(rest_l.shape[0]), int(rest_l.shape[1])
    n_z = int(setup.gridz.size)
    n_b = len(setup.filters)
    # Union-grid intermediates ~ nwave per (combo, z, band). Chunk both axes.
    budget = 250_000
    z_chunk = int(max(1, min(n_z, budget // max(nwave * n_b, 1))))
    c_chunk = int(max(1, min(n_c, budget // max(nwave * z_chunk * n_b, 1))))
    jl_ol = jnp.asarray(opa_l, dtype=dtype)
    jl_ot = jnp.asarray(opa_t, dtype=dtype)
    c_parts = []
    for i0 in range(0, n_c, c_chunk):
        sl = slice(i0, min(i0 + c_chunk, n_c))
        jl_l = jnp.asarray(rest_l[sl], dtype=dtype)
        jl_f = jnp.asarray(rest_f[sl], dtype=dtype)
        z_parts = []
        for k0 in range(0, n_z, z_chunk):
            zs = slice(k0, min(k0 + z_chunk, n_z))
            z_parts.append(
                np.asarray(
                    jax.device_get(
                        fn(
                            jl_l,
                            jl_f,
                            jnp.asarray(setup.gridz[zs], dtype=dtype),
                            jnp.asarray(setup.grid_dm[zs], dtype=dtype),
                            jl_ol,
                            jl_ot,
                            jnp.asarray(opa_idx[zs], dtype=np.int32),
                            jnp.asarray(age_ok[sl, zs], dtype=bool),
                        )
                    ),
                    dtype=np.float64,
                )
            )
        c_parts.append(np.concatenate(z_parts, axis=1))
        del jl_l, jl_f, z_parts
    out = np.concatenate(c_parts, axis=0)
    del c_parts
    return out


def _em_rest_frame(spec: SedSpectrum, ebv: float, frac: float, setup: MagGalSetup):
    em = setup.emlines
    if em.startswith("EMP_SFR"):
        ll, lv = generate_em_emp_sfr(spec.sfr, spec.luv, spec.lopt)
    else:
        ll, lv = generate_em_emp_uv(spec.luv, spec.lopt)
    ll, lv = apply_ext_lines(ll, lv, ebv, setup.mw_ext)
    lv = rescale_em_lines(lv, frac)
    return generate_em_spectra(ll, lv, nstep=40)


def _add_em_lines_gpu(mags_cz: np.ndarray, recs, age_ok: np.ndarray, setup: MagGalSetup) -> np.ndarray:
    """Rest-frame line SEDs once per combo; IGM / z / filter integrals on GPU."""
    line_l, line_f = [], []
    for rec in recs:
        spec, ebv, frac = rec[0], rec[1], rec[2]
        el, ev = _em_rest_frame(spec, float(ebv), float(frac), setup)
        if el.size == 0:
            line_l.append(np.array([1.0, 2.0], dtype=np.float64))
            line_f.append(np.zeros(2, dtype=np.float64))
        else:
            line_l.append(el)
            line_f.append(ev)
    pad_l, pad_f = _pad_spectra(line_l, line_f)
    fnu = jax_continuum_mags_grid(pad_l, pad_f, age_ok, setup, as_fnu=True)
    n_b = int(mags_cz.shape[-1])
    out = mag_plus_lines(mags_cz.reshape(-1, n_b), fnu.reshape(-1, n_b)).reshape(mags_cz.shape)
    out[~age_ok] = HIGH_MAG
    return out


def jax_compute_magnitudes_batch(
    lambs: Sequence[np.ndarray],
    fluxes: Sequence[np.ndarray],
    dist_mods: np.ndarray,
    filters: list[FilterCurve],
    dtype=np.float64,
) -> np.ndarray:
    """GPU batch of already-redshifted spectra (union-grid ``integrateSED``)."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    pad_l, pad_f = _pad_spectra(lambs, fluxes)
    fl, ft, lmin, lmax, fcorr = _filter_pack(filters, dtype)
    jl = jnp.asarray(pad_l, dtype=dtype)
    jf = jnp.asarray(pad_f, dtype=dtype)
    dms = jnp.asarray(dist_mods, dtype=dtype)

    def mag_one_filter(sed_l, sed_f, dm, fl_i, ft_i, lmin_i, lmax_i, fcorr_i):
        invalid = (sed_l[0] >= lmin_i) | (sed_l[-1] <= lmax_i)
        grid = jnp.sort(jnp.concatenate([sed_l, fl_i]))
        sed_i = jnp.interp(grid, sed_l, sed_f, left=0.0, right=0.0)
        tr_i = jnp.interp(grid, fl_i, ft_i, left=0.0, right=0.0)
        tr_i = jnp.where((grid >= lmin_i) & (grid <= lmax_i), tr_i, 0.0)
        dl = grid[1:] - grid[:-1]
        lmean = 0.5 * (grid[1:] + grid[:-1])
        tmean = 0.5 * (tr_i[1:] + tr_i[:-1])
        fmean = 0.5 * (sed_i[1:] + sed_i[:-1])
        conv = C_LIGHT_A_S / (lmean * lmean)
        arean = jnp.sum(tmean * conv * dl)
        fmel = jnp.sum(fmean * tmean * dl)
        mag = -2.5 * jnp.log10(fmel / arean * fcorr_i) - 48.6 + dm
        mag = jnp.where((fmel > 0) & (arean > 0), mag, jnp.asarray(HIGH_MAG, dtype=dtype))
        return jnp.where(invalid, jnp.asarray(INVALID_MAG, dtype=dtype), mag)

    def one_sed(sed_l, sed_f, dm):
        return jax.vmap(mag_one_filter, in_axes=(None, None, None, 0, 0, 0, 0, 0))(
            sed_l, sed_f, dm, fl, ft, lmin, lmax, fcorr
        )

    out = jax.jit(jax.vmap(one_sed))(jl, jf, dms)
    return np.asarray(jax.device_get(out), dtype=np.float64)


def _continuum_at_z(
    lamb: np.ndarray,
    flux: np.ndarray,
    z: float,
    dist_mod: float,
    setup: MagGalSetup,
    integrator: str,
) -> np.ndarray:
    lamb, flux = warning_integrate_sed(lamb, flux, z, setup.filters)
    lamb, flux = apply_opa_cpp(lamb, flux, z, setup.opa_all)
    lamb, flux = redshift_spectrum(lamb, flux, z)
    if integrator == "jax":
        return jax_compute_magnitudes_batch([lamb], [flux], np.array([dist_mod]), setup.filters)[0]
    return compute_magnitudes_cpp(lamb, flux, setup.filters, dist_mod=dist_mod)


def _em_line_fluxes(
    spec: SedSpectrum,
    z: float,
    dist_mod: float,
    ebv: float,
    frac_em: float,
    setup: MagGalSetup,
) -> np.ndarray:
    em = setup.emlines
    n_b = len(setup.filters)
    if em[:2].upper() == "NO" or em[:1] not in ("E", "P", "e", "p"):
        return np.zeros(n_b, dtype=np.float64)
    if em.startswith("EMP_SFR"):
        ll, lv = generate_em_emp_sfr(spec.sfr, spec.luv, spec.lopt)
    else:
        ll, lv = generate_em_emp_uv(spec.luv, spec.lopt)
    ll, lv = apply_ext_lines(ll, lv, ebv, setup.mw_ext)
    lv = rescale_em_lines(lv, frac_em)
    el, ev = generate_em_spectra(ll, lv, nstep=40)
    if el.size == 0:
        return np.zeros(n_b, dtype=np.float64)
    el, ev = apply_opa_cpp(el, ev, z, setup.opa_all)
    el, ev = redshift_spectrum(el, ev, z)
    ev = ev * (10.0 ** (-0.4 * dist_mod))
    return compute_fluxes_cpp(el, ev, setup.filters)


def maggal_one_gal(
    spec: SedSpectrum,
    setup: MagGalSetup,
    *,
    integrator: str = "cpp",
    em_lines: bool | None = None,
) -> dict[str, np.ndarray]:
    """Expand one rest-frame GAL SED over ext × EBV × fracEm × z (C++ order)."""
    use_em = setup.emlines[:1] in ("E", "P", "e", "p") if em_lines is None else em_lines
    n_z = int(setup.gridz.size)
    n_b = len(setup.filters)
    rest_l, rest_f = [], []
    meta = []
    for i in range(len(setup.ext_laws)):
        for ebv in setup.ebv:
            if not _ext_applies(spec.nummod, i, float(ebv), setup.modext):
                continue
            lamb_e, flux_e = apply_extinction_cpp(spec.lamb, spec.flux, float(ebv), setup.ext_laws[i])
            for frac in setup.frac_em:
                rest_l.append(lamb_e)
                rest_f.append(flux_e)
                meta.append((float(ebv), float(frac), int(setup.ext_laws[i].numext)))
    if not rest_l:
        empty = np.zeros((0, n_b), dtype=np.float64)
        return {
            "mag": empty,
            "flux": empty,
            "z": np.zeros(0),
            "ebv": np.zeros(0),
            "age": np.zeros(0),
            "nummod": np.zeros(0, dtype=np.int32),
            "mag0": np.zeros(0),
            "frac_em": np.zeros(0),
            "extlaw": np.zeros(0, dtype=np.int32),
        }
    age_ok = np.broadcast_to(setup.grid_t > spec.age, (len(rest_l), n_z)).copy()
    if integrator == "jax":
        pad_l, pad_f = _pad_spectra(rest_l, rest_f)
        mags_cz = np.array(jax_continuum_mags_grid(pad_l, pad_f, age_ok, setup), copy=True)
    else:
        mags_cz = np.full((len(rest_l), n_z, n_b), HIGH_MAG, dtype=np.float64)
        for c, (lamb_e, flux_e) in enumerate(zip(rest_l, rest_f)):
            for k, z in enumerate(setup.gridz):
                if not age_ok[c, k]:
                    continue
                lamb, flux = warning_integrate_sed(lamb_e, flux_e, float(z), setup.filters)
                lamb, flux = apply_opa_cpp(lamb, flux, float(z), setup.opa_all)
                lamb, flux = redshift_spectrum(lamb, flux, float(z))
                mags_cz[c, k] = compute_magnitudes_cpp(
                    lamb, flux, setup.filters, dist_mod=float(setup.grid_dm[k])
                )
    if use_em:
        mags_cz = _add_em_lines_gpu(mags_cz, [(spec, m[0], m[1], m[2]) for m in meta], age_ok, setup)

    rows_mag, rows_z, rows_ebv, rows_age, rows_num, rows_ext, rows_frac = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    mag0_of: list[float] = []
    leftover_m0 = HIGH_MAG
    mabs = min(setup.mabs_ref, n_b - 1)
    for c, (ebv, frac, extid) in enumerate(meta):
        mag_z0 = None
        for k, z in enumerate(setup.gridz):
            if not age_ok[c, k]:
                continue
            row = np.array(mags_cz[c, k], dtype=np.float64, copy=True)
            if mag_z0 is None and float(z) < 1e-5:
                mag_z0 = row.copy()
            rows_mag.append(row)
            rows_z.append(float(z))
            rows_ebv.append(ebv)
            rows_age.append(float(spec.age))
            rows_num.append(int(spec.nummod))
            rows_ext.append(extid)
            rows_frac.append(frac)
        mag0_of.append(mag_z0[mabs] if mag_z0 is not None else leftover_m0)
        if mag_z0 is not None:
            leftover_m0 = mag_z0[mabs]
    mag = np.vstack(rows_mag) if rows_mag else np.zeros((0, n_b))
    mag0 = []
    ic = 0
    for c, (ebv, frac, extid) in enumerate(meta):
        nkeep = int(np.sum(age_ok[c]))
        mag0.extend([mag0_of[c]] * nkeep)
        ic += nkeep
    mag0 = np.asarray(mag0, dtype=np.float64)
    backend = "jax" if integrator == "jax" else "numpy"
    return {
        "mag": mag,
        "flux": device_ab_mag_to_flux(mag, backend=backend),
        "z": np.asarray(rows_z, dtype=np.float64),
        "ebv": np.asarray(rows_ebv, dtype=np.float64),
        "age": np.asarray(rows_age, dtype=np.float64),
        "nummod": np.asarray(rows_num, dtype=np.int32),
        "mag0": mag0,
        "frac_em": np.asarray(rows_frac, dtype=np.float64),
        "extlaw": np.asarray(rows_ext, dtype=np.int32),
    }


def _concat_maggal_blocks(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    chunks = [c for c in chunks if c["mag"].shape[0]]
    if not chunks:
        n_b = chunks[0]["mag"].shape[1] if chunks else 0
        return chunks[0] if False else {
            "mag": np.zeros((0, 1)),
            "flux": np.zeros((0, 1)),
            "z": np.zeros(0),
            "ebv": np.zeros(0),
            "age": np.zeros(0),
            "nummod": np.zeros(0, dtype=np.int32),
            "mag0": np.zeros(0),
            "frac_em": np.zeros(0),
            "extlaw": np.zeros(0, dtype=np.int32),
        }
    keys = chunks[0].keys()
    out = {}
    for k in keys:
        out[k] = np.concatenate([c[k] for c in chunks], axis=0)
    return out


def maggal_gal_library(
    spectra: Sequence[SedSpectrum],
    setup: MagGalSetup,
    *,
    integrator: str = "jax",
    em_lines: bool | None = None,
) -> dict[str, np.ndarray]:
    """All GAL rest-frame SEDs in one GPU ``(N_combo, N_z)`` launch when ``integrator='jax'``."""
    use_em = setup.emlines[:1] in ("E", "P", "e", "p") if em_lines is None else em_lines
    if integrator != "jax":
        return _concat_maggal_blocks(
            [maggal_one_gal(s, setup, integrator=integrator, em_lines=em_lines) for s in spectra]
        )
    n_z = int(setup.gridz.size)
    n_b = len(setup.filters)
    rest_l, rest_f, recs = [], [], []
    for spec in spectra:
        for i in range(len(setup.ext_laws)):
            for ebv in setup.ebv:
                if not _ext_applies(spec.nummod, i, float(ebv), setup.modext):
                    continue
                lamb_e, flux_e = apply_extinction_cpp(
                    spec.lamb, spec.flux, float(ebv), setup.ext_laws[i]
                )
                for frac in setup.frac_em:
                    rest_l.append(lamb_e)
                    rest_f.append(flux_e)
                    recs.append((spec, float(ebv), float(frac), int(setup.ext_laws[i].numext)))
    if not rest_l:
        return maggal_one_gal(spectra[0], setup, integrator="jax", em_lines=em_lines)
    pad_l, pad_f = _pad_spectra(rest_l, rest_f)
    age_ok = np.zeros((len(recs), n_z), dtype=bool)
    for c, (spec, *_rest) in enumerate(recs):
        age_ok[c] = setup.grid_t > spec.age
    mags_cz = np.array(jax_continuum_mags_grid(pad_l, pad_f, age_ok, setup), copy=True)
    if use_em:
        mags_cz = _add_em_lines_gpu(mags_cz, recs, age_ok, setup)
    rows_mag, rows_z, rows_ebv, rows_age, rows_num, rows_ext, rows_frac = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    mag0_of = {}
    leftover_m0 = HIGH_MAG
    mabs = min(setup.mabs_ref, n_b - 1)
    for c, (spec, ebv, frac, extid) in enumerate(recs):
        mag_z0 = None
        for k, z in enumerate(setup.gridz):
            if not age_ok[c, k]:
                continue
            row = np.array(mags_cz[c, k], dtype=np.float64, copy=True)
            if mag_z0 is None and float(z) < 1e-5:
                mag_z0 = row.copy()
            rows_mag.append(row)
            rows_z.append(float(z))
            rows_ebv.append(ebv)
            rows_age.append(float(spec.age))
            rows_num.append(int(spec.nummod))
            rows_ext.append(extid)
            rows_frac.append(frac)
        mag0_of[c] = mag_z0[mabs] if mag_z0 is not None else leftover_m0
        if mag_z0 is not None:
            leftover_m0 = mag_z0[mabs]
    mag = np.vstack(rows_mag)
    mag0 = []
    for c, _rec in enumerate(recs):
        nkeep = int(np.sum(age_ok[c]))
        mag0.extend([mag0_of[c]] * nkeep)
    return {
        "mag": mag,
        "flux": device_ab_mag_to_flux(mag, backend="jax"),
        "z": np.asarray(rows_z, dtype=np.float64),
        "ebv": np.asarray(rows_ebv, dtype=np.float64),
        "age": np.asarray(rows_age, dtype=np.float64),
        "nummod": np.asarray(rows_num, dtype=np.int32),
        "mag0": np.asarray(mag0, dtype=np.float64),
        "frac_em": np.asarray(rows_frac, dtype=np.float64),
        "extlaw": np.asarray(rows_ext, dtype=np.int32),
    }


def maggal_qso_library(
    spectra: Sequence[SedSpectrum],
    setup: MagGalSetup,
    *,
    integrator: str = "jax",
) -> dict[str, np.ndarray]:
    if integrator != "jax":
        return _concat_maggal_blocks(
            [maggal_one_qso(s, setup, integrator=integrator) for s in spectra]
        )
    n_z = int(setup.gridz.size)
    rest_l, rest_f, recs = [], [], []
    for spec in spectra:
        for i in range(len(setup.ext_laws)):
            for ebv in setup.ebv:
                if not _ext_applies(spec.nummod, i, float(ebv), setup.modext):
                    continue
                lamb_e, flux_e = apply_extinction_cpp(
                    spec.lamb, spec.flux, float(ebv), setup.ext_laws[i]
                )
                rest_l.append(lamb_e)
                rest_f.append(flux_e)
                recs.append((spec, float(ebv)))
    if not rest_l:
        return maggal_one_qso(spectra[0], setup, integrator="jax")
    pad_l, pad_f = _pad_spectra(rest_l, rest_f)
    age_ok = np.ones((len(recs), n_z), dtype=bool)
    mags_cz = np.array(jax_continuum_mags_grid(pad_l, pad_f, age_ok, setup), copy=True)
    rows_mag, rows_z, rows_ebv, rows_num = [], [], [], []
    mag0_of = {}
    mabs = min(setup.mabs_ref, len(setup.filters) - 1)
    for c, (spec, ebv) in enumerate(recs):
        mag_z0 = None
        for k, z in enumerate(setup.gridz):
            row = np.array(mags_cz[c, k], dtype=np.float64, copy=True)
            if mag_z0 is None and float(z) < 1e-5:
                mag_z0 = row.copy()
            rows_mag.append(row)
            rows_z.append(float(z))
            rows_ebv.append(ebv)
            rows_num.append(int(spec.nummod))
        mag0_of[c] = mag_z0 if mag_z0 is not None else rows_mag[-1]
    mag = np.vstack(rows_mag)
    mag0 = np.concatenate(
        [np.full(n_z, mag0_of[c][mabs], dtype=np.float64) for c in range(len(recs))]
    )
    return {
        "mag": mag,
        "flux": device_ab_mag_to_flux(mag, backend="jax"),
        "z": np.asarray(rows_z, dtype=np.float64),
        "ebv": np.asarray(rows_ebv, dtype=np.float64),
        "age": np.zeros(mag.shape[0], dtype=np.float64),
        "nummod": np.asarray(rows_num, dtype=np.int32),
        "mag0": mag0,
    }


def maggal_one_qso(
    spec: SedSpectrum,
    setup: MagGalSetup,
    *,
    integrator: str = "cpp",
) -> dict[str, np.ndarray]:
    """QSO collapse-3: ext × EBV × z (no age cut, no emission lines)."""
    n_z = int(setup.gridz.size)
    n_b = len(setup.filters)
    rest_l, rest_f, meta = [], [], []
    for i in range(len(setup.ext_laws)):
        for ebv in setup.ebv:
            if not _ext_applies(spec.nummod, i, float(ebv), setup.modext):
                continue
            lamb_e, flux_e = apply_extinction_cpp(spec.lamb, spec.flux, float(ebv), setup.ext_laws[i])
            rest_l.append(lamb_e)
            rest_f.append(flux_e)
            meta.append(float(ebv))
    if not rest_l:
        empty = np.zeros((0, n_b), dtype=np.float64)
        return {
            "mag": empty,
            "flux": empty,
            "z": np.zeros(0),
            "ebv": np.zeros(0),
            "age": np.zeros(0),
            "nummod": np.zeros(0, dtype=np.int32),
            "mag0": np.zeros(0),
        }
    age_ok = np.ones((len(rest_l), n_z), dtype=bool)
    if integrator == "jax":
        pad_l, pad_f = _pad_spectra(rest_l, rest_f)
        mags_cz = np.array(jax_continuum_mags_grid(pad_l, pad_f, age_ok, setup), copy=True)
    else:
        mags_cz = np.full((len(rest_l), n_z, n_b), HIGH_MAG, dtype=np.float64)
        for c, (lamb_e, flux_e) in enumerate(zip(rest_l, rest_f)):
            for k, z in enumerate(setup.gridz):
                lamb, flux = warning_integrate_sed(lamb_e, flux_e, float(z), setup.filters)
                lamb, flux = apply_opa_cpp(lamb, flux, float(z), setup.opa_all)
                lamb, flux = redshift_spectrum(lamb, flux, float(z))
                mags_cz[c, k] = compute_magnitudes_cpp(
                    lamb, flux, setup.filters, dist_mod=float(setup.grid_dm[k])
                )
    rows_mag, rows_z, rows_ebv, rows_num = [], [], [], []
    mag0_ref = None
    mabs = min(setup.mabs_ref, n_b - 1)
    for c, ebv in enumerate(meta):
        mag_z0 = None
        for k, z in enumerate(setup.gridz):
            row = np.array(mags_cz[c, k], dtype=np.float64, copy=True)
            if mag_z0 is None and float(z) < 1e-5:
                mag_z0 = row.copy()
            rows_mag.append(row)
            rows_z.append(float(z))
            rows_ebv.append(ebv)
            rows_num.append(int(spec.nummod))
        if mag_z0 is not None:
            mag0_ref = mag_z0
    mag = np.vstack(rows_mag)
    if mag0_ref is None:
        mag0_ref = mag[0]
    mag0 = np.full(mag.shape[0], mag0_ref[mabs])
    return {
        "mag": mag,
        "flux": device_ab_mag_to_flux(mag, backend="jax" if integrator == "jax" else "numpy"),
        "z": np.asarray(rows_z, dtype=np.float64),
        "ebv": np.asarray(rows_ebv, dtype=np.float64),
        "age": np.zeros(mag.shape[0], dtype=np.float64),
        "nummod": np.asarray(rows_num, dtype=np.int32),
        "mag0": mag0,
    }


def _sidecar(config: dict[str, Any], n_bands: int, gridz: np.ndarray) -> dict[str, Any]:
    import lephare as lp

    mag_abs = _parse_doubles(config, "MAG_ABS", "-24,-5")
    mag_abs_agn = _parse_doubles(
        config,
        "MAG_ABS_AGN" if "MAG_ABS_AGN" in config else "MAG_ABS_QSO",
        "-30,-10",
    )
    if len(mag_abs) < 2:
        mag_abs = [-24.0, -5.0]
    if len(mag_abs_agn) < 2:
        mag_abs_agn = [-30.0, -10.0]
    gb0, gf0 = mag_abs[0], mag_abs[1]
    if gb0 > gf0:
        gb0, gf0 = gf0, gb0
    qb0, qf0 = mag_abs_agn[0], mag_abs_agn[1]
    if qb0 > qf0:
        qb0, qf0 = qf0, qb0
    prior_lib = np.array([gb0, gf0, qb0, qf0], dtype=np.float64)
    mabs_ref = int(float(str(config.get("MAG_REF", "3")).split(",")[0])) - 1
    mabs_ref = max(mabs_ref, 0)
    cosmo_vals = _parse_doubles(config, "COSMOLOGY", "70,0.3,0.7")
    lcdm = lp.cosmo(cosmo_vals[0], cosmo_vals[1], cosmo_vals[2])
    funz0 = float(lcdm.distMod(gridz[1] / 20.0)) if len(gridz) > 1 else 0.0
    err_scale = np.asarray(_parse_doubles(config, "ERR_SCALE", "0.02", n_bands), dtype=np.float64)
    if err_scale.size == 1:
        err_scale = np.full(n_bands, err_scale[0], dtype=np.float64)
    err_factor = np.asarray(_parse_doubles(config, "ERR_FACTOR", "1.5", n_bands), dtype=np.float64)
    if err_factor.size == 1:
        err_factor = np.full(n_bands, err_factor[0], dtype=np.float64)
    z_interp = str(config.get("Z_INTERP", "YES")).strip().upper() in ("YES", "Y", "TRUE", "1")
    return dict(
        prior_lib=prior_lib,
        mabs_ref=mabs_ref,
        funz0=funz0,
        err_scale=err_scale[:n_bands],
        err_factor=err_factor[:n_bands],
        gbcont=int(float(str(config.get("GLB_CONTEXT", "0")).split(",")[0])),
        contforb=int(float(str(config.get("FORB_CONTEXT", "-1")).split(",")[0])),
        rm_discrepant=float(str(config.get("RM_DISCREPANT_BD", "1e9")).split(",")[0]),
        z_interp=z_interp,
    )


def _merge_string_config(base: dict[str, Any], override) -> dict[str, Any]:
    from lephare.prepare import _DEFAULT

    out = dict(base)
    if override is None or override is _DEFAULT:
        return out
    out.update(override)
    return out


def ensure_filt_and_sedtolib(
    config: dict[str, Any],
    star_config=None,
    gal_config=None,
    qso_config=None,
) -> None:
    """Filter + sedtolib only (no MagGal). Same overrides as ``lp.prepare``."""
    import lephare as lp

    from lephare.prepare import _DEFAULT, overwrite_config

    if star_config is None:
        star_config = _DEFAULT
    if gal_config is None:
        gal_config = _DEFAULT
    if qso_config is None:
        qso_config = _DEFAULT
    cfg = lp.all_types_to_keymap(config)
    object_types = dict(STAR=star_config, GAL=gal_config, QSO=qso_config)
    filter_lib = lp.FilterSvc.from_keymap(cfg)
    filter_output = os.path.join(os.environ["LEPHAREWORK"], "filt", cfg["FILTER_FILE"].value)
    os.makedirs(os.path.dirname(filter_output), exist_ok=True)
    lp.write_output_filter(filter_output + ".dat", filter_output + ".doc", filter_lib)
    lp.write_yaml_config(cfg, f"{filter_output}_config.yaml")
    os.makedirs(os.path.join(os.environ["LEPHAREWORK"], "lib_bin"), exist_ok=True)

    for object_type, override in object_types.items():
        if override is None:
            continue
        if override is _DEFAULT:
            updated = dict(cfg)
        else:
            updated = overwrite_config(dict(cfg), lp.all_types_to_keymap(override))
        lib_name = updated[f"{object_type}_LIB"].value
        bin_path = _lib_bin_path(lib_name)
        if bin_path.is_file():
            continue
        sedlib = lp.Sedtolib(config_keymap=updated)
        list_loc = updated[f"{object_type}_SED"].value
        if list_loc.find("sed/") >= 0:
            list_loc = os.path.join(lp.LEPHAREDIR, list_loc)
        sedlib.run(typ=object_type, **{f"{object_type.lower()}_sed": list_loc})


def build_maglib_jax_from_lib_bin(
    config: dict[str, Any],
    *,
    integrator: str = "jax",
    star_config=None,
    gal_config=None,
    qso_config=None,
) -> MagLibTensors:
    """Build ``MagLibTensors`` from ``lib_bin`` on the JAX MagGal path."""
    from lephare.prepare import _DEFAULT

    if star_config is None:
        star_config = _DEFAULT
    if gal_config is None:
        gal_config = _DEFAULT
    if qso_config is None:
        qso_config = _DEFAULT
    ensure_filt_and_sedtolib(config, star_config, gal_config, qso_config)
    gal_cfg = _merge_string_config(config, gal_config)
    qso_cfg = _merge_string_config(config, qso_config)
    setup = maggal_setup_from_config(gal_cfg)
    qsetup = maggal_setup_from_config(qso_cfg)
    n_b = len(setup.filters)
    blocks_flux, blocks_z, blocks_t, blocks_m0, blocks_nm, blocks_age, blocks_ebv = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    zphot = _split_list(config, "ZPHOTLIB", "STAR_LIB_OUT,GAL_LIB_OUT,QSO_LIB_OUT")
    # ZPHOTLIB names mag libraries; map back to lib_bin via *_LIB_IN / *_LIB
    kind_order = []
    for name in zphot:
        up = name.upper()
        if "STAR" in up:
            kind_order.append("star")
        elif "QSO" in up or "AGN" in up:
            kind_order.append("qso")
        else:
            kind_order.append("gal")

    def _append(block: dict[str, np.ndarray], typ: int) -> None:
        if block["flux"].shape[0] == 0:
            return
        blocks_flux.append(block["flux"])
        blocks_z.append(block["z"])
        blocks_t.append(np.full(block["z"].shape[0], typ, dtype=np.int32))
        blocks_m0.append(block["mag0"])
        blocks_nm.append(block["nummod"])
        blocks_age.append(block["age"])
        blocks_ebv.append(block["ebv"])

    for kind in kind_order:
        t_kind = time.perf_counter()
        if kind == "star":
            name = _split_list(config, "STAR_LIB_IN", _split_list(config, "STAR_LIB", "STAR_BIN")[0])[0]
            spectra = list(iter_sed_bin(_lib_bin_path(name), kind="star"))
            if not spectra:
                continue
            flux = device_compute_fluxes_batch(
                spectra, setup.filters, backend="jax" if integrator == "jax" else "numpy"
            )
            n = flux.shape[0]
            mag = -2.5 * np.log10(np.clip(flux, 1e-300, None)) - 48.6
            mabs = min(setup.mabs_ref, n_b - 1)
            _append(
                {
                    "flux": flux,
                    "z": np.zeros(n, dtype=np.float64),
                    "mag0": mag[:, mabs],
                    "nummod": np.asarray([s.nummod for s in spectra], dtype=np.int32),
                    "age": np.zeros(n, dtype=np.float64),
                    "ebv": np.zeros(n, dtype=np.float64),
                },
                2,
            )
            print(
                f"[jax MagGal] STAR n_sed={n} {time.perf_counter() - t_kind:.2f}s",
                flush=True,
            )
            if integrator == "jax":
                import jax
                jax.clear_caches()
        elif kind == "gal":
            name = _split_list(config, "GAL_LIB_IN", _split_list(config, "GAL_LIB", "GAL_BIN")[0])[0]
            gal_specs = list(iter_sed_bin(_lib_bin_path(name), kind="gal"))
            _append(maggal_gal_library(gal_specs, setup, integrator=integrator), 0)
            print(
                f"[jax MagGal] GAL n_sed={len(gal_specs)} "
                f"{time.perf_counter() - t_kind:.2f}s",
                flush=True,
            )
            if integrator == "jax":
                import jax
                jax.clear_caches()
        else:
            name = _split_list(qso_cfg, "QSO_LIB_IN", _split_list(qso_cfg, "QSO_LIB", "QSO_BIN")[0])[0]
            qso_specs = list(iter_sed_bin(_lib_bin_path(name), kind="qso"))
            _append(maggal_qso_library(qso_specs, qsetup, integrator=integrator), 1)
            print(
                f"[jax MagGal] QSO n_sed={len(qso_specs)} "
                f"{time.perf_counter() - t_kind:.2f}s",
                flush=True,
            )

    flux = np.vstack(blocks_flux) if blocks_flux else np.zeros((0, n_b))
    z = np.concatenate(blocks_z) if blocks_z else np.zeros(0)
    types = np.concatenate(blocks_t) if blocks_t else np.zeros(0, dtype=np.int32)
    mag0 = np.concatenate(blocks_m0) if blocks_m0 else np.zeros(0)
    nummod = np.concatenate(blocks_nm) if blocks_nm else np.zeros(0, dtype=np.int32)
    age = np.concatenate(blocks_age) if blocks_age else np.zeros(0)
    ebv = np.concatenate(blocks_ebv) if blocks_ebv else np.zeros(0)
    if int(np.sum(types == 0)) <= 0:
        raise RuntimeError("JAX MagGal produced no GAL templates")
    if np.unique(z).size <= 1:
        raise RuntimeError("JAX MagGal has no redshift diversity")
    meta = _sidecar(config, n_b, setup.gridz)
    return MagLibTensors(
        flux=np.asarray(flux, dtype=np.float64),
        z=np.asarray(z, dtype=np.float64),
        types=types.astype(np.int32),
        mag0=np.asarray(mag0, dtype=np.float64),
        gridz=setup.gridz,
        prior_lib=meta["prior_lib"],
        funz0=meta["funz0"],
        err_scale=meta["err_scale"],
        err_factor=meta["err_factor"],
        gbcont=meta["gbcont"],
        contforb=meta["contforb"],
        rm_discrepant=meta["rm_discrepant"],
        mabs_ref=meta["mabs_ref"],
        n_bands=n_b,
        nummod=nummod.astype(np.int32),
        age=np.asarray(age, dtype=np.float64),
        ebv=np.asarray(ebv, dtype=np.float64),
        z_interp=meta["z_interp"],
    )
