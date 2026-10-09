"""C++-faithful MagGal continuum / IGM / emission-line helpers (host FP64).

Algorithms follow ``lephare/src/lib/SED.cpp`` (``resample``, ``applyExt``,
``applyOpa``, ``integrateSED``, ``generateEmEmpUV``, ``generateEmSpectra``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mag_conv import C_LIGHT_A_S, FilterCurve, ab_mag_to_flux

HIGH_MAG = 1000.0
INVALID_MAG = -9999.0
LYA_SPLIT = 1215.67
PI = np.pi

# EMP_UV rest-frame air wavelengths and H-beta-normalized ratios (GalSED::generateEmEmpUV).
LB_LINE = np.array(
    [
        1215.67, 1335.708, 1663.48, 1908.73, 2140.896, 2325.726, 2799.116,
        3726.04, 3728.8, 3869.063, 3888.856, 3968.771, 4026.191, 4068.601,
        4076.349, 4101.734, 4363.21, 4340.472, 4471.48, 4712.249, 4861.352,
        4958.911, 5006.843, 5199.079, 5754.594, 5875.62, 6300.304, 6312.064,
        6548.041, 6562.787, 6583.461, 6678.151, 6716.44, 6730.812, 7065.19,
        7135.803, 7319.458, 7330.196, 7751.057, 9068.611, 9530.586, 10286.731,
        10320.491, 10336.407, 10827.334, 18750.976, 12818.072, 10938.174,
        10049.368, 9545.969, 9229.667, 9014.909, 8862.782, 8750.473, 8665.018,
        8598.392, 8545.383, 40511.742, 26251.517, 21655.267, 19445.581,
        18174.237, 17362.143, 16806.509, 16407.21,
    ],
    dtype=np.float64,
)
Z0_LINE_EMP_UV = np.array(
    [
        22.2, 0, 0, 0, 0, 0, 0, 0.81, 0.81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0.21, 0.59, 0, 0, 0, 0, 0, 0, 2.90, 0.0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0,
    ],
    dtype=np.float64,
)
Z0_LINE_EMP_SFR = np.array(
    [
        22.2, 0, 0, 0, 0, 0, 0, 1.425, 1.425, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1.36, 4.081, 0, 0, 0, 0, 0, 0, 2.85, 0.86, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0,
    ],
    dtype=np.float64,
)


@dataclass
class ExtLaw:
    lamb: np.ndarray
    kval: np.ndarray
    lmin: float
    lmax: float
    numext: int


@dataclass
class OpaTable:
    lamb: np.ndarray
    trans: np.ndarray
    lmin: float
    lmax: float


def _sort_ori(lamb: np.ndarray, val: np.ndarray, ori: np.ndarray):
    order = np.argsort(lamb, kind="mergesort")
    return lamb[order].copy(), val[order].copy(), ori[order].copy()


def resample(
    lamb: np.ndarray,
    val: np.ndarray,
    ori: np.ndarray,
    origine: int,
    lmin: float = 0.0,
    lmax: float = 1.0e20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``SED::resample``: interpolate origin ``origine`` onto the merged grid."""
    mask = (lamb >= lmin) & (lamb <= lmax)
    lamb = lamb[mask].copy()
    val = val[mask].copy()
    ori = ori[mask].copy()
    n = int(lamb.size)
    prev_l = -999.0
    prev_v = -999.0
    for i in range(n):
        if ori[i] == origine:
            prev_l = lamb[i]
            prev_v = val[i]
            continue
        if prev_l > 0:
            next_l = -999.0
            next_v = -999.0
            for k in range(i, n):
                if ori[k] == origine:
                    next_l = lamb[k]
                    next_v = val[k]
                    break
            if prev_l > 0 and next_l > prev_l:
                slope = (next_v - prev_v) / (next_l - prev_l)
                val[i] = prev_v + (lamb[i] - prev_l) * slope
                ori[i] = origine
            else:
                val[i] = prev_v
                ori[i] = -99
        else:
            val[i] = -99.0
            ori[i] = -99
    return lamb, val, ori


def apply_extinction_cpp(
    sed_lamb: np.ndarray,
    sed_flux: np.ndarray,
    ebv: float,
    law: ExtLaw,
) -> tuple[np.ndarray, np.ndarray]:
    """``SED::applyExt`` (ori=1 SED, ori=2 extinction)."""
    if ebv <= 1e-20:
        return np.asarray(sed_lamb, dtype=np.float64), np.asarray(sed_flux, dtype=np.float64)
    sl = np.asarray(sed_lamb, dtype=np.float64)
    sv = np.asarray(sed_flux, dtype=np.float64)
    so = np.ones(sl.size, dtype=np.int32)
    el = np.asarray(law.lamb, dtype=np.float64)
    ev = np.asarray(law.kval, dtype=np.float64)
    eo = np.full(el.size, 2, dtype=np.int32)
    lamb = np.concatenate([sl, el])
    val = np.concatenate([sv, ev])
    ori = np.concatenate([so, eo])
    lamb, val, ori = _sort_ori(lamb, val, ori)
    _, new_ext, ext_ori = resample(lamb, val, ori, origine=2, lmin=0.0, lmax=1e20)
    keep_l = []
    keep_v = []
    for k in range(lamb.size):
        if ori[k] != 1:
            continue
        lam = lamb[k]
        v = val[k]
        if law.lmin < lam < law.lmax:
            kval = 0.0 if ext_ori[k] < 0 else new_ext[k]
            v = val[k] * (10.0 ** (-0.4 * ebv * kval))
        keep_l.append(lam)
        keep_v.append(v)
    return np.asarray(keep_l, dtype=np.float64), np.asarray(keep_v, dtype=np.float64)


def opa_index(z: float) -> int:
    """``lround(red / 0.100000000000001)`` clamped to 80."""
    ind = int(round(float(z) / 0.100000000000001))
    if ind < 0:
        ind = 0
    if ind > 80:
        ind = 80
    return ind


def apply_opa_cpp(
    sed_lamb: np.ndarray,
    sed_flux: np.ndarray,
    z: float,
    opa_all: list[OpaTable],
) -> tuple[np.ndarray, np.ndarray]:
    """``SED::applyOpa``."""
    sl = np.asarray(sed_lamb, dtype=np.float64)
    sv = np.asarray(sed_flux, dtype=np.float64)
    ind = opa_index(z)
    opa = opa_all[ind]
    below = sl < LYA_SPLIT
    lamb_b, flux_b = sl[below], sv[below]
    lamb_e, flux_e = sl[~below], sv[~below]
    ol = np.asarray(opa.lamb, dtype=np.float64)
    ov = np.asarray(opa.trans, dtype=np.float64)
    oo = np.full(ol.size, 3, dtype=np.int32)
    so = np.ones(lamb_b.size, dtype=np.int32)
    lamb = np.concatenate([lamb_b, ol])
    val = np.concatenate([flux_b, ov])
    ori = np.concatenate([so, oo])
    lamb, val, ori = _sort_ori(lamb, val, ori)
    _, new_opa, opa_ori = resample(lamb, val, ori, origine=3, lmin=0.0, lmax=1e20)
    keep_l = []
    keep_v = []
    for k in range(lamb.size):
        if ori[k] != 1:
            continue
        lam = lamb[k]
        v = val[k]
        if opa.lmin < lam < opa.lmax:
            t = 1.0 if opa_ori[k] < 0 else new_opa[k]
            v = val[k] * t
        keep_l.append(lam)
        keep_v.append(v)
    out_l = np.concatenate([np.asarray(keep_l, dtype=np.float64), lamb_e])
    out_v = np.concatenate([np.asarray(keep_v, dtype=np.float64), flux_e])
    return out_l, out_v


def warning_integrate_sed(
    sed_lamb: np.ndarray,
    sed_flux: np.ndarray,
    z: float,
    filters: list[FilterCurve],
) -> tuple[np.ndarray, np.ndarray]:
    """``SED::warning_integrateSED`` (rest-frame, uses ``lambda*(1+z)``)."""
    lamb = np.asarray(sed_lamb, dtype=np.float64).copy()
    flux = np.asarray(sed_flux, dtype=np.float64).copy()
    one_pz = 1.0 + float(z)
    for flt in filters:
        if lamb[0] * one_pz > flt.lmin:
            lamb = np.concatenate([[0.0], lamb])
            flux = np.concatenate([[0.0], flux])
        if lamb[-1] * one_pz < flt.lmax:
            lamb = np.concatenate([lamb, [1.0e8 * one_pz]])
            flux = np.concatenate([flux, [0.0]])
    return lamb, flux


def redshift_spectrum(sed_lamb: np.ndarray, sed_flux: np.ndarray, z: float):
    one_pz = 1.0 + float(z)
    return sed_lamb * one_pz, sed_flux / one_pz


def trapzd(sed_lamb: np.ndarray, sed_flux: np.ndarray) -> float:
    if sed_lamb.size < 2:
        return 0.0
    return float(np.sum(0.5 * (sed_flux[1:] + sed_flux[:-1]) * (sed_lamb[1:] - sed_lamb[:-1])))


def integrate_sed_cpp(
    sed_lamb: np.ndarray,
    sed_flux: np.ndarray,
    flt: FilterCurve,
) -> tuple[float, float] | None:
    """Return ``(fmel, arean)`` matching ``SED::integrateSED`` indices 3 and 1."""
    sl = np.asarray(sed_lamb, dtype=np.float64)
    sv = np.asarray(sed_flux, dtype=np.float64)
    if sl.size < 2 or sl[0] >= flt.lmin or sl[-1] <= flt.lmax:
        return None
    lamb_all = [0.0]
    val_all = [0.0]
    ori_all = [1]
    first_before = True
    for i in range(sl.size):
        lam, v = sl[i], sv[i]
        if lam < flt.lmin:
            if first_before:
                lamb_all[0] = lam
                val_all[0] = v
            else:
                lamb_all[0] = lam
                val_all[0] = v
            continue
        if flt.lmin <= lam <= flt.lmax:
            lamb_all.append(lam)
            val_all.append(v)
            ori_all.append(1)
            continue
        if lam > flt.lmax:
            lamb_all.append(lam)
            val_all.append(v)
            ori_all.append(1)
            break
    fl = np.asarray(flt.lamb, dtype=np.float64)
    ft = np.asarray(flt.trans, dtype=np.float64)
    lamb = np.concatenate([np.asarray(lamb_all, dtype=np.float64), fl])
    val = np.concatenate([np.asarray(val_all, dtype=np.float64), ft])
    ori = np.concatenate(
        [np.asarray(ori_all, dtype=np.int32), np.zeros(fl.size, dtype=np.int32)]
    )
    lamb, val, ori = _sort_ori(lamb, val, ori)
    if ori[0] == 0:
        lamb = np.concatenate([[lamb[0] - 1.0], lamb])
        val = np.concatenate([[0.0], val])
        ori = np.concatenate([[1], ori])
    if ori[-1] == 0:
        lamb = np.concatenate([lamb, [lamb[-1] + 1.0]])
        val = np.concatenate([val, [0.0]])
        ori = np.concatenate([ori, [1]])
    tl, tv, to = resample(lamb, val.copy(), ori.copy(), origine=0, lmin=0.0, lmax=1e50)
    sl2, sv2, so2 = resample(lamb, val.copy(), ori.copy(), origine=1, lmin=0.0, lmax=1e50)
    fmel = 0.0
    arean = 0.0
    for k in range(sl2.size - 1):
        vs0 = 0.0 if so2[k] < 0 else sv2[k]
        vs1 = 0.0 if so2[k + 1] < 0 else sv2[k + 1]
        vt0 = 0.0 if to[k] < 0 else tv[k]
        vt1 = 0.0 if to[k + 1] < 0 else tv[k + 1]
        lmean = 0.5 * (sl2[k] + sl2[k + 1])
        tmean = 0.5 * (vt0 + vt1)
        fmean = 0.5 * (vs0 + vs1)
        dlbd = sl2[k + 1] - sl2[k]
        conv = C_LIGHT_A_S / (lmean * lmean)
        arean += tmean * conv * dlbd
        fmel += fmean * tmean * dlbd
    return fmel, arean


def compute_magnitudes_cpp(
    sed_lamb: np.ndarray,
    sed_flux: np.ndarray,
    filters: list[FilterCurve],
    dist_mod: float = 0.0,
) -> np.ndarray:
    mags = np.empty(len(filters), dtype=np.float64)
    for i, flt in enumerate(filters):
        integ = integrate_sed_cpp(sed_lamb, sed_flux, flt)
        if integ is None:
            mags[i] = INVALID_MAG
            continue
        fmel, arean = integ
        if fmel > 0.0 and arean > 0.0:
            mags[i] = -2.5 * np.log10(fmel / arean * flt.fcorr) - 48.6 + dist_mod
        else:
            mags[i] = HIGH_MAG
    return mags


def compute_fluxes_cpp(
    sed_lamb: np.ndarray,
    sed_flux: np.ndarray,
    filters: list[FilterCurve],
) -> np.ndarray:
    """``SED::compute_fluxes``: ``fmel/arean`` (no fcorr, no distMod)."""
    out = np.zeros(len(filters), dtype=np.float64)
    for i, flt in enumerate(filters):
        integ = integrate_sed_cpp(sed_lamb, sed_flux, flt)
        if integ is None:
            out[i] = -9999.0
            continue
        fmel, arean = integ
        out[i] = 0.0 if arean == 0 else fmel / arean
    return out


def mag_plus_lines(mags: np.ndarray, fl_em: np.ndarray) -> np.ndarray:
    """``GalSED::sumEmLines``."""
    flux = ab_mag_to_flux(mags) + np.asarray(fl_em, dtype=np.float64)
    out = np.empty_like(mags, dtype=np.float64)
    ok = flux > 0
    out[ok] = -2.5 * np.log10(flux[ok]) - 48.6
    out[~ok] = 999.9
    return out


def generate_em_emp_uv(luv: float, lopt: float) -> tuple[np.ndarray, np.ndarray]:
    mnuv = -2.5 * luv + 51.59
    nuvr = -2.5 * (luv - lopt)
    if nuvr < 4:
        scale = 10.0 ** (-0.4 * mnuv - 6.224)
    else:
        scale = 0.0
    scale = scale / 2.85
    val = scale * Z0_LINE_EMP_UV
    order = np.argsort(LB_LINE, kind="mergesort")
    return LB_LINE[order].copy(), val[order].copy()


def generate_em_emp_sfr(sfr: float, luv: float, lopt: float) -> tuple[np.ndarray, np.ndarray]:
    """``GalSED::generateEmEmpSFR`` Kennicutt H-alpha scaling."""
    nuvr = -2.5 * (luv - lopt)
    if nuvr < 4 and sfr > 0:
        scale = 10.0 ** (
            np.log10(sfr) + 41.27 - np.log10(4 * PI * 100 * (3.08568**2)) - 36
        )
    else:
        scale = 0.0
    scale = scale / 2.85
    val = scale * Z0_LINE_EMP_SFR
    order = np.argsort(LB_LINE, kind="mergesort")
    return LB_LINE[order].copy(), val[order].copy()


def apply_ext_lines(
    line_lamb: np.ndarray,
    line_val: np.ndarray,
    ebv: float,
    law: ExtLaw,
) -> tuple[np.ndarray, np.ndarray]:
    """``SED::applyExtLines`` with ebvFac=1 and Hayes Lya escape."""
    if ebv <= 1e-20:
        return np.asarray(line_lamb, dtype=np.float64), np.asarray(line_val, dtype=np.float64)
    sl = np.asarray(line_lamb, dtype=np.float64)
    sv = np.asarray(line_val, dtype=np.float64)
    so = np.full(sl.size, 5, dtype=np.int32)
    el = np.asarray(law.lamb, dtype=np.float64)
    ev = np.asarray(law.kval, dtype=np.float64)
    eo = np.full(el.size, 2, dtype=np.int32)
    lamb = np.concatenate([sl, el])
    val = np.concatenate([sv, ev])
    ori = np.concatenate([so, eo])
    lamb, val, ori = _sort_ori(lamb, val, ori)
    _, new_ext, ext_ori = resample(lamb, val, ori, origine=2, lmin=0.0, lmax=1e20)
    keep_l = []
    keep_v = []
    line_i = 0
    c_lya, k_lya = 0.445, 13.8
    for k in range(lamb.size):
        if ori[k] != 5:
            continue
        lam = lamb[k]
        v = val[k]
        if law.lmin < lam < law.lmax:
            kval = 0.0 if ext_ori[k] < 0 else new_ext[k]
            v = val[k] * (10.0 ** (-0.4 * ebv * kval))
            if 1215.0 < lam < 1216.0:
                v = v * (c_lya * (10.0 ** (-0.4 * ebv * k_lya)))
        line_i += 1
        keep_l.append(lam)
        keep_v.append(v)
    return np.asarray(keep_l, dtype=np.float64), np.asarray(keep_v, dtype=np.float64)


def rescale_em_lines(line_val: np.ndarray, frac_em: float) -> np.ndarray:
    return np.asarray(line_val, dtype=np.float64) * float(frac_em)


def generate_em_spectra(
    line_lamb: np.ndarray,
    line_val: np.ndarray,
    nstep: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """``GalSED::generateEmSpectra``."""
    nbsigma = 6.0
    sigma = 100.0 * line_lamb / C_LIGHT_A_S / 2.355 * 1.0e13
    pts_l = []
    pts_v = []
    for j in range(line_lamb.size):
        if line_val[j] <= 0:
            continue
        step = nbsigma * 2.0 * sigma[j] / float(nstep)
        for i in range(nstep):
            lb = line_lamb[j] - nbsigma * sigma[j] + float(i) * step
            pts_l.append(lb)
            pts_v.append(0.0)
    if not pts_l:
        return np.zeros(0), np.zeros(0)
    lamb = np.asarray(pts_l, dtype=np.float64)
    flux = np.zeros_like(lamb)
    order = np.argsort(lamb, kind="mergesort")
    lamb, flux = lamb[order], flux[order]
    for j in range(line_lamb.size):
        if line_val[j] <= 0:
            continue
        sig = sigma[j]
        line_fl = (line_val[j] / (sig * np.sqrt(2.0 * PI))) * np.exp(
            -((lamb - line_lamb[j]) ** 2) / (2.0 * sig * sig)
        )
        flux = flux + line_fl
    insert = []
    for k in range(1, lamb.size - 1):
        if (lamb[k] + 1.0) < lamb[k + 1]:
            insert.append(lamb[k] + 0.1)
        if (lamb[k - 1] + 1.0) < lamb[k]:
            insert.append(lamb[k] - 0.1)
    insert.extend([0.0, 1.0e7])
    lamb = np.concatenate([lamb, np.asarray(insert, dtype=np.float64)])
    flux = np.concatenate([flux, np.zeros(len(insert), dtype=np.float64)])
    order = np.argsort(lamb, kind="mergesort")
    return lamb[order], flux[order]


def load_extinction_law_file(path: str | Path, numext: int) -> ExtLaw:
    rows = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            rows.append((float(parts[0]), float(parts[1])))
    arr = np.asarray(rows, dtype=np.float64)
    order = np.argsort(arr[:, 0], kind="mergesort")
    arr = arr[order]
    return ExtLaw(
        lamb=arr[:, 0],
        kval=arr[:, 1],
        lmin=float(arr[0, 0]),
        lmax=float(arr[-1, 0]),
        numext=int(numext),
    )


def load_opa_tables(lepharedir: str | Path) -> list[OpaTable]:
    root = Path(lepharedir)
    list_path = root / "opa" / "OPACITY.dat"
    tables: list[OpaTable] = []
    with list_path.open() as fh:
        for i, line in enumerate(fh):
            if i >= 81:
                break
            parts = line.split()
            if len(parts) < 2:
                continue
            name = parts[1]
            rows = []
            with (root / "opa" / name).open() as ofh:
                for lit in ofh:
                    lit = lit.strip()
                    if not lit or lit.startswith("#"):
                        continue
                    p = lit.split()
                    if len(p) < 2:
                        continue
                    rows.append((float(p[0]), float(p[1])))
            arr = np.asarray(rows, dtype=np.float64)
            order = np.argsort(arr[:, 0], kind="mergesort")
            arr = arr[order]
            lamb, trans = arr[:, 0], arr[:, 1]
            if lamb[-1] > 1215.66:
                lamb = lamb[:-1]
                trans = trans[:-1]
            lamb = np.concatenate([lamb, [1215.66, 1215.67]])
            trans = np.concatenate([trans, [trans[-1], 1.0]])
            tables.append(
                OpaTable(
                    lamb=lamb,
                    trans=trans,
                    lmin=float(lamb[0]),
                    lmax=1215.67,
                )
            )
    if len(tables) != 81:
        raise RuntimeError(f"expected 81 opacity tables, got {len(tables)}")
    return tables


def parse_csv_floats(raw: str) -> list[float]:
    return [float(p) for p in str(raw).replace(",", " ").split() if p.strip()]


def parse_csv_ints(raw: str) -> list[int]:
    return [int(float(p)) for p in str(raw).replace(",", " ").split() if p.strip()]
