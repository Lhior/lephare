"""C++-faithful MagGal continuum / IGM / emission-line helpers (host FP64).

Algorithms follow ``lephare/src/lib/SED.cpp`` (``resample``, ``applyExt``,
``applyOpa``, ``integrateSED``, ``generateEmEmpUV``, ``generateEmSpectra``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import _lephare
from .mag_conv import C_LIGHT_A_S, FilterCurve, ab_mag_to_flux

HIGH_MAG = 1000.0
INVALID_MAG = -9999.0
LYA_SPLIT = 1215.67
PI = np.pi

# EMP_UV rest-frame air wavelengths and H-beta-normalized ratios used by GalSED::generateEmEmpUV
# and generateEmEmpSFR, taken from the compiled library so they follow emission_lines.h.
LB_LINE = np.asarray(_lephare._emission_lines, dtype=np.float64)
EMPIRICAL_RATIO = np.asarray(_lephare._empirical_ratio, dtype=np.float64)
Z0_LINE_EMP_UV = EMPIRICAL_RATIO
Z0_LINE_EMP_SFR = EMPIRICAL_RATIO


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


def fast_interpolate(x, y, z, d: float) -> np.ndarray:
    """``fast_interpolate`` (lephare ``globals.cpp``): linear interpolation of ``(x, y)``
    at sorted ``z``, ``d`` outside ``[x[0], x[-1]]``."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    out = np.full(z.size, float(d))
    if x.size < 2 or z.size == 0:
        return out
    inside = (z >= x[0]) & (z <= x[-1])
    # C++ walks i while x[i+1] < z: the segment is the first one with x[i+1] >= z
    i = np.clip(np.searchsorted(x, z, side="left") - 1, 0, x.size - 2)
    t = (z - x[i]) / (x[i + 1] - x[i])
    out[inside] = (y[i] + t * (y[i + 1] - y[i]))[inside]
    return out


def apply_extinction_cpp(
    sed_lamb: np.ndarray,
    sed_flux: np.ndarray,
    ebv: float,
    law: ExtLaw,
) -> tuple[np.ndarray, np.ndarray]:
    """``SED::apply_extinction`` (v1.0.0): ``flux *= 10^(-0.4 ebv k(lambda))`` with
    ``k = fast_interpolate(law, lambda, 0)``."""
    sl = np.asarray(sed_lamb, dtype=np.float64)
    sv = np.asarray(sed_flux, dtype=np.float64)
    if abs(ebv) <= 1e-20 or sl.size == 0:
        return sl, sv
    k = fast_interpolate(law.lamb, law.kval, sl, 0.0)
    return sl.copy(), sv * np.power(10.0, -0.4 * ebv * k)


def opa_index(z: float) -> int:
    """``lround(red / 0.100000000000001)`` clamped to 80."""
    ind = int(round(float(z) / 0.100000000000001))
    if ind < 0:
        ind = 0
    if ind > 80:
        ind = 80
    return ind


def opa_transmission(lamb: np.ndarray, opa: OpaTable) -> np.ndarray:
    """IGM multiplier of ``SED::applyOpa`` (v1.0.0) at rest-frame ``lamb``:
    ``fast_interpolate(opa, lamb, 1)`` below 1215.67, 1 elsewhere."""
    lam = np.asarray(lamb, dtype=np.float64)
    out = np.ones(lam.size, dtype=np.float64)
    below = lam < LYA_SPLIT
    if below.any():
        out[below] = fast_interpolate(opa.lamb, opa.trans, lam[below], 1.0)
    return out


def apply_opa_cpp(
    sed_lamb: np.ndarray,
    sed_flux: np.ndarray,
    z: float,
    opa_all: list[OpaTable],
) -> tuple[np.ndarray, np.ndarray]:
    """``SED::applyOpa`` (lephare v1.0.0): multiply fluxes below Lyman-alpha by the
    interpolated transmission of the opacity table for ``z``; wavelengths unchanged."""
    sl = np.asarray(sed_lamb, dtype=np.float64)
    sv = np.asarray(sed_flux, dtype=np.float64)
    return sl.copy(), sv * opa_transmission(sl, opa_all[opa_index(z)])


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


def _interp_linear_point(x: np.ndarray, y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """``interp_linear_point`` (``oneElLambda.cpp``) at each ``q``: 0 outside ``x``."""
    out = np.zeros(q.size)
    inside = (q >= x[0]) & (q <= x[-1])
    idx = np.clip(np.searchsorted(x, q, side="left"), 1, x.size - 1)
    t = (q - x[idx - 1]) / (x[idx] - x[idx - 1])
    val = y[idx - 1] + t * (y[idx] - y[idx - 1])
    val = np.where(q == x[0], y[0], np.where(q == x[-1], y[-1], val))
    out[inside] = val[inside]
    return out


def integrate_sed_cpp(
    sed_lamb: np.ndarray,
    sed_flux: np.ndarray,
    flt: FilterCurve,
) -> tuple[float, float] | None:
    """``(r3, r1)`` of ``SED::integrateSED`` (lephare v1.0.0); None if not integrable."""
    sl = np.asarray(sed_lamb, dtype=np.float64)
    sv = np.asarray(sed_flux, dtype=np.float64)
    fl = np.asarray(flt.lamb, dtype=np.float64)
    ft = np.asarray(flt.trans, dtype=np.float64)
    if sl.size < 2 or sl[0] > fl[0] or sl[-1] < fl[-1]:
        return None
    lo, hi = max(sl[0], fl[0]), min(sl[-1], fl[-1])
    if lo >= hi:
        return 0.0, 0.0
    # restricted_resampling: sorted unique union inside [lo, hi], both curves interpolated
    x = np.unique(np.concatenate([sl[(sl >= lo) & (sl <= hi)], fl[(fl >= lo) & (fl <= hi)]]))
    fs = _interp_linear_point(sl, sv, x)
    tr = _interp_linear_point(fl, ft, x)
    r3 = r5 = 0.0
    for i in range(x.size - 1):
        lmean = (x[i] + x[i + 1]) / 2
        tmean = (tr[i] + tr[i + 1]) / 2
        fmean = (fs[i] + fs[i + 1]) / 2
        tmp = tmean * (x[i + 1] - x[i])
        r3 += tmp * fmean
        r5 += tmp / lmean**2
    return r3, r5 * C_LIGHT_A_S


def compute_magnitudes_cpp(
    sed_lamb: np.ndarray,
    sed_flux: np.ndarray,
    filters: list[FilterCurve],
    dist_mod: float = 0.0,
) -> np.ndarray:
    """``SED::compute_magnitudes``."""
    mags = np.empty(len(filters), dtype=np.float64)
    for i, flt in enumerate(filters):
        integ = integrate_sed_cpp(sed_lamb, sed_flux, flt)
        if integ is None:
            mags[i] = INVALID_MAG
            continue
        r3, r1 = integ
        mags[i] = -2.5 * np.log10(r3 / r1 * flt.fcorr) - 48.6 + dist_mod if r3 > 0.0 else HIGH_MAG
    return mags


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
        scale = 10.0 ** (np.log10(sfr) + 41.27 - np.log10(4 * PI * 100 * (3.08568**2)) - 36)
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
    """``SED::apply_extinction_to_lines`` (v1.0.0, ebvFac = 1, no z dependence) with the
    Hayes+11 Lyman-alpha escape fraction."""
    sl = np.asarray(line_lamb, dtype=np.float64)
    sv = np.asarray(line_val, dtype=np.float64)
    if ebv <= 1e-20 or sl.size == 0:
        return sl, sv
    k = fast_interpolate(law.lamb, law.kval, sl, 0.0)
    f = 1.0
    out = sv * np.power(10.0, -0.4 * ebv / f * k)
    lya = (sl > 1215.0) & (sl < 1216.0)
    out = np.where(lya, out * (0.445 * np.power(10.0, -0.4 * ebv / f * 13.8)), out)
    return sl.copy(), out


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
