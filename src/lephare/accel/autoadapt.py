"""C++ AUTO_ADAPT offset loop (NumPy reference)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .kernels_ref import HIGH_CHI2, prepare_photometry, rm_discrepant
from .maglib import MagLibTensors

HIGH_MAG = 1000.0
AB_ZP = 48.6
INVALID_INDEX = -99


def _parse_doubles(raw: str, default: str) -> list[float]:
    text = str(raw if raw is not None else default)
    parts = [p.strip() for p in text.replace(",", " ").split() if p.strip()]
    return [float(p) for p in parts]


def _parse_int(raw, default: int = 1) -> int:
    parts = str(raw if raw is not None else default).replace(",", " ").split()
    return int(float(parts[0]))


def flux2mag(flux: np.ndarray) -> np.ndarray:
    flux = np.asarray(flux, dtype=np.float64)
    out = np.full(flux.shape, HIGH_MAG, dtype=np.float64)
    ok = flux > 0
    out[ok] = -2.5 * np.log10(flux[ok]) - AB_ZP
    return out


def indexz(red: np.ndarray, gridz: np.ndarray) -> np.ndarray:
    """Closest ``gridz`` index (matches ``lephare`` ``indexz``)."""
    gridz = np.asarray(gridz, dtype=np.float64)
    red = np.asarray(red, dtype=np.float64)
    scalar = red.ndim == 0
    red = np.atleast_1d(red)
    n = int(gridz.size)
    idx = np.empty(red.shape, dtype=np.int32)
    lo_m = red <= gridz[0]
    hi_m = red >= gridz[-1]
    mid = ~lo_m & ~hi_m
    idx[lo_m] = 0
    idx[hi_m] = n - 1
    if np.any(mid):
        r = red[mid]
        up = np.searchsorted(gridz, r, side="left")
        up = np.clip(up, 0, n - 1)
        low = np.clip(up - 1, 0, n - 1)
        choose_up = np.abs(gridz[up] - r) <= np.abs(r - gridz[low])
        idx[mid] = np.where(choose_up, up, low).astype(np.int32)
    return idx[0] if scalar else idx


def apply_sysshift_or_none(config: dict[str, Any], n_bands: int) -> np.ndarray | None:
    shifts = _parse_doubles(config.get("APPLY_SYSSHIFT", "0."), "0.")
    if len(shifts) == n_bands:
        return np.asarray(shifts, dtype=np.float64)
    return None


def select_adapt_sources(
    mags: np.ndarray,
    zs: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Boolean mask matching ``PhotoZ::read_autoadapt_sources`` cuts."""
    mags = np.asarray(mags, dtype=np.float64)
    zs = np.asarray(zs, dtype=np.float64)
    n_b = mags.shape[1]
    fl_auto = _parse_int(config.get("ADAPT_BAND", "1"), 1) - 1
    fl_auto = int(np.clip(fl_auto, 0, n_b - 1))
    lim = _parse_doubles(config.get("ADAPT_LIM", "15,35"), "15,35")
    zbin = _parse_doubles(config.get("ADAPT_ZBIN", "0.001,6"), "0.001,6")
    tmin, tmax = lim[0], lim[1] if len(lim) > 1 else 35.0
    zmin, zmax = zbin[0], zbin[1] if len(zbin) > 1 else 6.0
    mag_sel = np.where(mags[:, fl_auto] > 0, mags[:, fl_auto], HIGH_MAG)
    return (zs > zmin) & (zs < zmax) & (mag_sel > tmin) & (mag_sel < tmax)


def adapt_mag(ab_ori: np.ndarray, sab_ori: np.ndarray, a0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """C++ ``onesource::adapt_mag``: flux and flux error both scaled by ``10**(0.4 a0)``."""
    a0 = np.asarray(a0, dtype=np.float64)
    scale = np.power(10.0, 0.4 * a0)[None, :]
    ab_ori = np.asarray(ab_ori, dtype=np.float64)
    sab_ori = np.asarray(sab_ori, dtype=np.float64)
    use = (ab_ori > 0) | (sab_ori > 0)
    return np.where(use, ab_ori * scale, ab_ori), np.where(use, sab_ori * scale, sab_ori)


def _is_same_model(tensors: MagLibTensors, i: int, j: int) -> bool:
    nummod = tensors.nummod
    if nummod is None:
        return i == j
    if int(nummod[i]) != int(nummod[j]):
        return False
    age = tensors.age
    ebv = tensors.ebv
    if age is not None and float(age[i]) != float(age[j]):
        return False
    if ebv is not None and float(ebv[i]) != float(ebv[j]):
        return False
    return True


def interp_lib_magm(
    tensors: MagLibTensors,
    indmin_gal: np.ndarray,
    dm_gal: np.ndarray,
    zs: np.ndarray,
) -> np.ndarray:
    """GAL ``interp_lib`` predicted magnitudes ``magm[N_obj, N_b]``."""
    flux = np.asarray(tensors.flux, dtype=np.float64)
    z_lib = np.asarray(tensors.z, dtype=np.float64)
    n_obj = int(indmin_gal.shape[0])
    n_b = int(flux.shape[1])
    n_t = int(flux.shape[0])
    mag_lib = flux2mag(flux)
    magm = np.full((n_obj, n_b), -999.0, dtype=np.float64)
    for o in range(n_obj):
        i0 = int(indmin_gal[o])
        dm = float(dm_gal[o])
        if i0 < 0 or dm <= 0:
            continue
        consiz = float(zs[o])
        sens = int(np.copysign(1.0, consiz - z_lib[i0]))
        if sens == 0:
            sens = 1
        deca = sens
        nxt = i0 + deca
        if 0 <= nxt <= n_t - 1:
            while z_lib[i0] == z_lib[i0 + deca]:
                deca += sens
                if not (0 <= i0 + deca < n_t):
                    deca = 0
                    break
        else:
            deca = 0
        if deca != 0 and z_lib[i0 + deca] < 1e-10:
            deca = 0
        seda = mag_lib[i0].copy()
        if deca != 0 and _is_same_model(tensors, i0, i0 + deca):
            sedb = mag_lib[i0 + deca]
            a = abs((consiz - z_lib[i0]) / (z_lib[i0] - z_lib[i0 + deca]))
            row = np.empty(n_b, dtype=np.float64)
            for k in range(n_b):
                if sedb[k] > 90:
                    a = 0.0
                row[k] = a * sedb[k] + (1.0 - a) * seda[k]
            magm[o] = row
        else:
            magm[o] = seda
        magm[o] = magm[o] - 2.5 * np.log10(dm)
    return magm


def build_zslice_csr(tensors: MagLibTensors) -> tuple[np.ndarray, np.ndarray]:
    """CSR of template indices per gridz bin (``validLib`` / ``indexes_in_vec``)."""
    gridz = np.asarray(tensors.gridz, dtype=np.float64)
    z_lib = np.asarray(tensors.z, dtype=np.float64)
    n_z = int(gridz.size)
    zb = indexz(z_lib, gridz)
    match = np.abs(z_lib - gridz[zb]) < 1e-10
    t_ok = np.flatnonzero(match)
    bins = zb[t_ok]
    order = np.argsort(bins, kind="mergesort")
    idx_s = t_ok[order].astype(np.int32)
    counts = np.bincount(bins[order], minlength=n_z)
    offsets = np.zeros(n_z + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts, dtype=np.int32)
    return offsets, idx_s


def zfix_type_mins(
    ab: np.ndarray,
    sab: np.ndarray,
    busnorma: np.ndarray,
    busul: np.ndarray,
    nbul: np.ndarray,
    tensors: MagLibTensors,
    zs: np.ndarray,
    csr_off: np.ndarray | None = None,
    csr_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Z-fixed ``fit``: GAL/QSO/STAR ``indmin`` and GAL ``dm``."""
    n_obj = int(ab.shape[0])
    gridz = np.asarray(tensors.gridz, dtype=np.float64)
    zb = indexz(np.asarray(zs, dtype=np.float64), gridz)
    if csr_off is None:
        csr_off, csr_idx = build_zslice_csr(tensors)
    indmin = np.full((n_obj, 3), INVALID_INDEX, dtype=np.int64)
    dm_gal = np.full(n_obj, -999.0, dtype=np.float64)
    types = np.asarray(tensors.types, dtype=np.int32)
    flux = np.asarray(tensors.flux, dtype=np.float64)
    mag0 = np.asarray(tensors.mag0, dtype=np.float64)
    z_lib = np.asarray(tensors.z, dtype=np.float64)
    prior = np.asarray(tensors.prior_lib, dtype=np.float64)
    funz0 = float(tensors.funz0)
    mabs_gal = bool(prior[0] < 0 and prior[1] < 0)
    mabs_agn = bool(prior[2] < 0 and prior[3] < 0)

    invsab = np.where(busnorma > 0, busnorma / sab, 0.0)
    invsab_sq = invsab * invsab
    s2n = ab * invsab
    abinvsab_sq = ab * invsab_sq

    ul_obj = np.flatnonzero(np.asarray(nbul, dtype=np.int32) > 0)
    ul_set = set(int(i) for i in ul_obj)

    for b in np.unique(zb):
        b = int(b)
        objs = np.flatnonzero(zb == b)
        sl = csr_idx[int(csr_off[b]) : int(csr_off[b + 1])]
        if sl.size == 0 or objs.size == 0:
            continue
        need_host = [int(o) for o in objs if o in ul_set]
        gpu_like = np.array([int(o) for o in objs if o not in ul_set], dtype=np.int64)
        if gpu_like.size:
            f = flux[sl]
            avmago = f @ abinvsab_sq[gpu_like].T
            avmagt = (f * f) @ invsab_sq[gpu_like].T
            dm = np.where(avmagt > 0, avmago / avmagt, -999.0)
            chi2 = np.sum(
                (s2n[gpu_like][None, :, :] - dm[:, :, None] * f[:, None, :] * invsab[gpu_like][None, :, :])
                ** 2,
                axis=2,
            )
            loc_types = types[sl]
            loc_mag0 = mag0[sl]
            loc_z = z_lib[sl]
            if mabs_gal or mabs_agn:
                abs_mag = loc_mag0[:, None] - 2.5 * np.log10(np.clip(dm, 1e-300, None))
                abs_mag = np.where(loc_z[:, None] < 1e-10, abs_mag - funz0, abs_mag)
                if mabs_gal:
                    rej = (loc_types[:, None] == 0) & ((abs_mag <= prior[0]) | (abs_mag >= prior[1]))
                    chi2 = np.where(rej, HIGH_CHI2, chi2)
                if mabs_agn:
                    rej = (loc_types[:, None] == 1) & ((abs_mag <= prior[2]) | (abs_mag >= prior[3]))
                    chi2 = np.where(rej, HIGH_CHI2, chi2)
            for t in range(3):
                m = loc_types == t
                if not np.any(m):
                    continue
                sub = chi2[m]
                valid = sub < HIGH_CHI2
                # argmin among valid; if none, leave INVALID
                filled = np.where(valid, sub, np.inf)
                j_loc = np.argmin(filled, axis=0)
                ok = np.any(valid, axis=0)
                t_idx = np.flatnonzero(m)
                chosen = sl[t_idx[j_loc]]
                for k, o in enumerate(gpu_like):
                    if not ok[k]:
                        continue
                    indmin[int(o), t] = int(chosen[k])
                    if t == 0:
                        dm_gal[int(o)] = float(dm[t_idx[j_loc[k]], k])
        for o in need_host:
            sub = MagLibTensors(
                flux=tensors.flux[sl],
                z=tensors.z[sl],
                types=types[sl],
                mag0=tensors.mag0[sl],
                gridz=tensors.gridz,
                prior_lib=tensors.prior_lib,
                funz0=tensors.funz0,
                err_scale=tensors.err_scale,
                err_factor=tensors.err_factor,
                gbcont=tensors.gbcont,
                contforb=tensors.contforb,
                rm_discrepant=tensors.rm_discrepant,
                mabs_ref=tensors.mabs_ref,
                n_bands=tensors.n_bands,
                nummod=None if tensors.nummod is None else tensors.nummod[sl],
                age=None if tensors.age is None else tensors.age[sl],
                ebv=None if tensors.ebv is None else tensors.ebv[sl],
                z_interp=tensors.z_interp,
            )
            chi2, dm, _, _ = rm_discrepant(ab[o], sab[o], busnorma[o], busul[o], int(nbul[o]), sub)
            for t in range(3):
                m = sub.types == t
                if not np.any(m):
                    continue
                loc = np.where(m)[0]
                chi = chi2[loc]
                valid = chi < HIGH_CHI2
                if not np.any(valid):
                    continue
                j = loc[np.argmin(np.where(valid, chi, HIGH_CHI2))]
                indmin[o, t] = int(sl[j])
                if t == 0:
                    dm_gal[o] = float(dm[j])
    return indmin, dm_gal


def median_offsets(
    mab_ori: np.ndarray,
    magm: np.ndarray,
    indmin_gal: np.ndarray,
    busnorma: np.ndarray,
) -> np.ndarray:
    """Per-band median of ``mab_ori - magm`` with C++ cuts."""
    n_b = mab_ori.shape[1]
    a0 = np.zeros(n_b, dtype=np.float64)
    for k in range(n_b):
        ok = (mab_ori[:, k] > 0) & (indmin_gal > 0) & (busnorma[:, k] == 1)
        inter = mab_ori[:, k] - magm[:, k]
        ok = ok & (np.abs(inter) < 1.0)
        diff = inter[ok]
        if diff.size > 10:
            ordered = np.sort(diff)[::-1]
            a0[k] = float(ordered[int(diff.size / 2.0)])
        else:
            a0[k] = 0.0
    return a0


def _converged(a0: np.ndarray, a0pre: np.ndarray) -> bool:
    n_b = a0.size
    conv = float(np.sum(np.abs(a0 - a0pre))) / n_b < 0.02
    if np.any(np.abs(a0 - a0pre) > 0.03):
        conv = False
    return bool(conv)


ZFixFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]


def calculate_offsets(
    mags: np.ndarray,
    magerrs: np.ndarray,
    zs: np.ndarray,
    tensors: MagLibTensors,
    config: dict[str, Any],
    context: np.ndarray | None = None,
    verbose: bool = False,
    zfix_fn: ZFixFn | None = None,
) -> np.ndarray:
    """AUTO_ADAPT offsets (C++ ``run_autoadapt`` / ``compute_offsets``)."""
    n_b = int(mags.shape[1])
    auto = str(config.get("AUTO_ADAPT", "NO")).strip().upper() in ("YES", "Y", "TRUE", "1")
    sys_a0 = apply_sysshift_or_none(config, n_b)
    if sys_a0 is not None:
        if auto:
            print("Auto adapt is turned off because systematic shifts are applied manually")
        return sys_a0
    if not auto:
        return np.zeros(n_b, dtype=np.float64)

    mask = select_adapt_sources(mags, zs, config)
    if verbose:
        print(f"\n Number of sources read for auto adapt {int(mask.sum())}")
    if not np.any(mask):
        return np.zeros(n_b, dtype=np.float64)

    mags_s = np.asarray(mags, dtype=np.float64)[mask]
    magerrs_s = np.asarray(magerrs, dtype=np.float64)[mask]
    zs_s = np.asarray(zs, dtype=np.float64)[mask]
    ctx = None if context is None else np.asarray(context)[mask]

    ab_ori, sab_ori, busnorma, busul, nbul = prepare_photometry(mags_s, magerrs_s, ctx, tensors, offsets=None)
    mab_ori = flux2mag(ab_ori)
    csr_off, csr_idx = build_zslice_csr(tensors)

    def _default_fit(ab, sab_, bus, busul_, nbul_, zs_):
        return zfix_type_mins(ab, sab_, bus, busul_, nbul_, tensors, zs_, csr_off, csr_idx)

    fit = zfix_fn or _default_fit

    a0 = np.zeros(n_b, dtype=np.float64)
    converge = False
    iteration = 0
    while (not converge) and iteration < 10:
        ab, sab = adapt_mag(ab_ori, sab_ori, a0)
        indmin, dm_gal = fit(ab, sab, busnorma, busul, nbul, zs_s)
        magm = interp_lib_magm(tensors, indmin[:, 0], dm_gal, zs_s)
        a0pre = a0.copy()
        a0 = median_offsets(mab_ori, magm, indmin[:, 0], busnorma)
        converge = _converged(a0, a0pre)
        if iteration == 10:
            converge = True
        iteration += 1
        print(f" Done with iteration {iteration} and converge {int(converge)}")
        if verbose:
            print("Offsets:  " + ",".join(str(x) for x in a0))
    if verbose:
        print("Done with auto-adapt")
    return a0
