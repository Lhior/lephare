"""Float64 NumPy reference for LePHARE χ², RM_DISCREPANT, and BAY_ZG PDF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .maglib import MagLibTensors

HIGH_CHI2 = 1.0e9


@dataclass(frozen=True)
class BayZgIndex:
    """Precomputed BAY_ZG binning shared by host and device PDF kernels."""

    gal_mask: np.ndarray  # [N_t] bool
    gal_idx: np.ndarray  # [N_gal] int64 template indices with type==GAL
    bin_idx: np.ndarray  # [N_gal] int64 gridz bin for each GAL template
    dz: np.ndarray  # [N_z-1] trapezoid widths (empty if N_z < 2)
    n_z: int
    gridz: np.ndarray  # [N_z]
    bin_cols: tuple  # len N_z; each is int64 indices into gal axis landing in that bin


def prepare_bay_zg_index(tensors: MagLibTensors) -> BayZgIndex:
    """Static GAL mask and nearest-gridz indices for BAY_ZG accumulation."""
    gridz = np.asarray(tensors.gridz, dtype=np.float64)
    gal_mask = np.asarray(tensors.types, dtype=np.int32) == 0
    gal_idx = np.flatnonzero(gal_mask).astype(np.int64)
    n_z = int(gridz.size)
    if gal_idx.size == 0 or n_z == 0:
        return BayZgIndex(
            gal_mask=gal_mask,
            gal_idx=gal_idx,
            bin_idx=np.zeros(0, dtype=np.int64),
            dz=np.diff(gridz) if n_z > 1 else np.zeros(0, dtype=np.float64),
            n_z=n_z,
            gridz=gridz,
            bin_cols=tuple(np.zeros(0, dtype=np.int64) for _ in range(n_z)),
        )
    zg = np.asarray(tensors.z, dtype=np.float64)[gal_idx]
    zmin = float(gridz[0])
    inv = 1.0 / (gridz[1] - gridz[0]) if n_z > 1 else 1.0
    bin_idx = np.clip(np.rint((zg - zmin) * inv).astype(np.int64), 0, n_z - 1)
    dz = np.diff(gridz) if n_z > 1 else np.zeros(0, dtype=np.float64)
    bin_cols = tuple(np.flatnonzero(bin_idx == b).astype(np.int64) for b in range(n_z))
    return BayZgIndex(
        gal_mask=gal_mask,
        gal_idx=gal_idx,
        bin_idx=bin_idx,
        dz=dz,
        n_z=n_z,
        gridz=gridz,
        bin_cols=bin_cols,
    )


def mag2flux(mag: np.ndarray) -> np.ndarray:
    return np.where(
        np.isfinite(mag) & (mag < 90),
        10.0 ** (-0.4 * (mag + 48.6)),
        -9999.0,
    )


def prepare_photometry(
    mags: np.ndarray,
    magerrs: np.ndarray,
    context: np.ndarray | None,
    tensors: MagLibTensors,
    offsets: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert RAIL magnitudes to flux arrays matching ``PhotoZ.prep_data``.

    Parameters
    ----------
    mags, magerrs :
        Shape ``[N_obj, N_b]``.
    context :
        Per-object context integers, or None to derive from finite mags.
    tensors :
        Template library; supplies error scaling and the global/forbidden contexts.
    offsets :
        Magnitude offsets applied as in ``adapt_mag`` (model convention).

    Returns
    -------
    ab, sab, busnorma, busul, nbul
        Flux, flux-error, fit mask, upper-limit mask, and upper-limit counts.
    """
    mags = np.asarray(mags, dtype=np.float64)
    magerrs = np.asarray(magerrs, dtype=np.float64)
    n_obj, n_b = mags.shape
    assert n_b == tensors.n_bands

    ab = mag2flux(mags)
    sab = np.where(ab > 0, ab * magerrs / 1.086, -9999.0)

    # rescale_flux_errors (ERR_SCALE in mag, ERR_FACTOR on flux err)
    for k in range(n_b):
        good = (sab[:, k] >= 0) & (ab[:, k] != 0.0)
        sab_mag = np.zeros(n_obj)
        sab_mag[good] = 1.086 * sab[good, k] / ab[good, k]
        scale = tensors.err_scale[k] if tensors.err_scale.size > 1 else tensors.err_scale[0]
        fac = tensors.err_factor[k] if tensors.err_factor.size > 1 else tensors.err_factor[0]
        sab_mag[good] = np.sqrt(sab_mag[good] ** 2 + scale**2)
        sab[good, k] = np.abs(ab[good, k]) * sab_mag[good] / 1.086
        sab[good, k] *= fac

    if offsets is not None:
        offsets = np.asarray(offsets, dtype=np.float64)
        for k in range(n_b):
            corr = offsets[k]
            use = (ab[:, k] > 0) | (sab[:, k] > 0)
            ab[use, k] = ab[use, k] * (10.0 ** (0.4 * corr))

    if context is None:
        context = np.zeros(n_obj, dtype=np.int64)
        for k in range(n_b):
            mask = (mags[:, k] > 0) & np.isfinite(mags[:, k])
            mask &= (magerrs[:, k] > 0) & np.isfinite(magerrs[:, k])
            context += mask.astype(np.int64) * (2**k)
    context = np.asarray(context, dtype=np.int64)

    busnorma = np.zeros((n_obj, n_b), dtype=np.float64)
    busul = np.zeros((n_obj, n_b), dtype=np.float64)
    nbul = np.zeros(n_obj, dtype=np.int32)

    gbcont = tensors.gbcont
    contforb = tensors.contforb
    for i in range(n_obj):
        cont = int(context[i])
        if gbcont >= 0:
            cont = gbcont
        for k in range(n_b):
            if cont <= 0:
                used = 1
            else:
                used = 1 if (cont & (1 << k)) else 0
            if contforb > 0 and (contforb & (1 << k)):
                used = 0
            if sab[i, k] < 0 and ab[i, k] < 0:
                used = 0
            if sab[i, k] == 0:
                used = 0
                ab[i, k] = -99.9
                sab[i, k] = -99.9
            bn = used
            if sab[i, k] < 0:
                bn = 0
            busnorma[i, k] = bn
            if used == 1 and sab[i, k] < 0:
                busul[i, k] = 1
                nbul[i] += 1

    return ab, sab, busnorma, busul, nbul


def fit_chi2_single(
    ab: np.ndarray,
    sab: np.ndarray,
    busnorma: np.ndarray,
    busul: np.ndarray,
    nbul: int,
    tensors: MagLibTensors,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return chi2[N_t], dm[N_t], chimin[3] for one object."""
    flux = tensors.flux
    invsab = busnorma / sab
    # Avoid div0 for unused bands (busnorma=0 already zeros invsab if sab finite)
    invsab = np.where(busnorma > 0, busnorma / sab, 0.0)
    invsabSq = invsab * invsab
    s2n = ab * invsab
    abinvsabSq = ab * invsabSq

    avmago = flux @ abinvsabSq
    avmagt = (flux * flux) @ invsabSq
    dm = np.where(avmagt > 0, avmago / avmagt, -999.0)

    inter = s2n[None, :] - dm[:, None] * flux * invsab[None, :]
    chi2 = np.sum(inter * inter, axis=1)

    if nbul > 0:
        for k in range(ab.size):
            if busul[k] == 1:
                bad = (dm * flux[:, k] > ab[k]) & (busnorma[k] == 1)
                chi2 = np.where(bad, HIGH_CHI2, chi2)

    prior = tensors.prior_lib
    z_lib = tensors.z
    types = tensors.types
    mag0 = tensors.mag0
    funz0 = tensors.funz0

    dm_ok = dm > 0
    if prior[0] < 0 and prior[1] < 0:
        abs_mag = mag0 - 2.5 * np.log10(np.where(dm_ok, dm, 1.0))
        abs_mag = np.where(z_lib < 1e-10, abs_mag - funz0, abs_mag)
        reject = dm_ok & (types == 0) & ((abs_mag <= prior[0]) | (abs_mag >= prior[1]))
        chi2 = np.where(reject, HIGH_CHI2, chi2)
    if prior[2] < 0 and prior[3] < 0:
        abs_mag = mag0 - 2.5 * np.log10(np.where(dm_ok, dm, 1.0))
        abs_mag = np.where(z_lib < 1e-10, abs_mag - funz0, abs_mag)
        reject = dm_ok & (types == 1) & ((abs_mag <= prior[2]) | (abs_mag >= prior[3]))
        chi2 = np.where(reject, HIGH_CHI2, chi2)

    chimin = np.full(3, HIGH_CHI2, dtype=np.float64)
    for t in range(3):
        m = types == t
        if np.any(m):
            chimin[t] = float(np.min(chi2[m]))
    return chi2, dm, chimin


def rm_discrepant(
    ab: np.ndarray,
    sab: np.ndarray,
    busnorma: np.ndarray,
    busul: np.ndarray,
    nbul: int,
    tensors: MagLibTensors,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Iteratively drop up to 2 bands while chi2 > threshold (matches C++)."""
    threshold = tensors.rm_discrepant
    chi2, dm, chimin = fit_chi2_single(ab, sab, busnorma, busul, nbul, tensors)
    improved = float(np.min(chimin))
    nb_removed = 0
    bus = busnorma.copy()

    while improved > threshold and nb_removed < 2:
        fl_dis = -1
        best = improved
        for k in range(ab.size):
            if bus[k] == 1:
                bus[k] = 0
                _, _, ch = fit_chi2_single(ab, sab, bus, busul, nbul, tensors)
                newmin = float(np.min(ch))
                if best > newmin:
                    best = newmin
                    fl_dis = k
                bus[k] = 1
        if fl_dis >= 0:
            bus[fl_dis] = 0
            chi2, dm, chimin = fit_chi2_single(ab, sab, bus, busul, nbul, tensors)
            improved = float(np.min(chimin))
            nb_removed += 1
        else:
            break

    return chi2, dm, chimin, bus


def bay_zg_pdf(chi2: np.ndarray, tensors: MagLibTensors) -> np.ndarray:
    """Marginalized BAY_ZG PDF on ``tensors.gridz``, trapezoid-normalized."""
    return bay_zg_pdf_batch(chi2[None, :], tensors)[0]


def bay_zg_pdf_from_index(chi2: np.ndarray, index: BayZgIndex) -> np.ndarray:
    """Batched BAY_ZG using a precomputed :class:`BayZgIndex`."""
    chi2 = np.asarray(chi2, dtype=np.float64)
    if chi2.ndim == 1:
        chi2 = chi2[None, :]
    n_obj = chi2.shape[0]
    pdf = np.zeros((n_obj, index.n_z), dtype=np.float64)
    if index.gal_idx.size == 0 or index.n_z == 0:
        return pdf

    chi2g = chi2[:, index.gal_idx]
    chimin0 = np.min(chi2g, axis=1, keepdims=True)
    ok = chi2g < HIGH_CHI2
    prob = np.where(ok, np.exp(-0.5 * (chi2g - chimin0)), 0.0)
    for b, cols in enumerate(index.bin_cols):
        if cols.size:
            pdf[:, b] = prob[:, cols].sum(axis=1)

    if index.n_z > 1 and index.dz.size:
        somme = np.sum((pdf[:, :-1] + pdf[:, 1:]) * 0.5 * index.dz[None, :], axis=1)
        good = somme > 0
        pdf[good] = pdf[good] / somme[good, None]
    return pdf


def bay_zg_pdf_batch(chi2: np.ndarray, tensors: MagLibTensors) -> np.ndarray:
    """Batched BAY_ZG: ``chi2[N_obj, N_t]`` → ``pdfs[N_obj, N_z]``."""
    return bay_zg_pdf_from_index(chi2, prepare_bay_zg_index(tensors))


def pdf_zmode_zmean(pdfs: np.ndarray, zgrid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized zmode / zmean from ``pdfs[N, N_z]`` (matches estimator loop).

    zmode uses the median of all bins tied for the maximum (same as the classic
    per-object Python loop). zmean is the probability-weighted redshift.
    """
    pdfs = np.asarray(pdfs, dtype=np.float64)
    zgrid = np.asarray(zgrid, dtype=np.float64)
    if pdfs.ndim == 1:
        pdfs = pdfs[None, :]
    denom = pdfs.sum(axis=1)
    mx = np.max(pdfs, axis=1, keepdims=True)
    # Empty / all-zero PDFs: no mode (classic loop hits every bin when max==0)
    valid = (denom > 0) & (mx[:, 0] > 0)
    tied = (pdfs == mx) & valid[:, None]
    cum = np.cumsum(tied, axis=1)
    n_hits = cum[:, -1]
    mid_lo = (n_hits - 1) // 2
    mid_hi = n_hits // 2

    def _kth_true_index(k_row: np.ndarray) -> np.ndarray:
        target = (k_row + 1)[:, None]
        ge = cum >= target
        return np.argmax(ge, axis=1)

    i_lo = _kth_true_index(mid_lo)
    i_hi = _kth_true_index(mid_hi)
    zmode = 0.5 * (zgrid[i_lo] + zgrid[i_hi])
    zmode = np.where(valid & (n_hits > 0), zmode, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        zmean = np.where(denom > 0, (pdfs * zgrid[None, :]).sum(axis=1) / denom, np.nan)
    return zmode, zmean


def process_batch(
    mags: np.ndarray,
    magerrs: np.ndarray,
    tensors: MagLibTensors,
    context: np.ndarray | None = None,
    offsets: np.ndarray | None = None,
    dtype=np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Process a batch of objects; returns ``pdfs[N_obj, N_z]`` and ``zgrid``."""
    tensors = tensors.astype(dtype) if dtype != np.float64 else tensors
    ab, sab, busnorma, busul, nbul = prepare_photometry(mags, magerrs, context, tensors, offsets=offsets)
    n_obj = ab.shape[0]
    pdfs = np.zeros((n_obj, len(tensors.gridz)), dtype=np.float64)
    for i in range(n_obj):
        chi2, _, _, _ = rm_discrepant(ab[i], sab[i], busnorma[i], busul[i], int(nbul[i]), tensors)
        pdfs[i] = bay_zg_pdf(chi2, tensors)
    return pdfs, tensors.gridz.astype(np.float64, copy=False)
