"""Filter convolution and AB mag/flux helpers matching LePHARE ``SED::integrateSED``.

Used by ``jax_full`` / ``cuda_full`` mag-lib builders. Numerics follow
``lephare/src/lib/SED.cpp`` (photon-weighted ``arean`` path used by
``compute_magnitudes``).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Speed of light in Angstrom / s (LePHARE convention in integrateSED).
C_LIGHT_A_S = 2.99792458e18


@dataclass
class FilterCurve:
    name: str
    lamb: np.ndarray
    trans: np.ndarray
    lmin: float
    lmax: float
    fcorr: float = 1.0


@dataclass
class SedSpectrum:
    nummod: int
    lamb: np.ndarray
    flux: np.ndarray
    age: float = 0.0
    luv: float = -999.0
    lopt: float = -999.0
    lnir: float = -999.0
    ltir: float = -999.0
    mass: float = -999.0
    sfr: float = -999.0
    zmet: float = -999.0


def ab_mag_to_flux(mags: np.ndarray) -> np.ndarray:
    """``pow(10, -0.4 * (mag + 48.6))`` as in ``PhotoZ`` flux fill.

    ``HIGH_MAG`` / ``INVALID_MAG`` underflow or overflow; treat them as no flux.
    """
    m = np.asarray(mags, dtype=np.float64)
    out = np.power(10.0, -0.4 * (m + 48.6))
    out = np.where((m >= 90.0) | (m <= -100.0) | ~np.isfinite(m), 0.0, out)
    return out


def flux_to_ab_mag(flux: np.ndarray) -> np.ndarray:
    flux = np.asarray(flux, dtype=np.float64)
    return -2.5 * np.log10(np.clip(flux, 1e-300, None)) - 48.6


def integrate_sed(
    sed_lamb: np.ndarray,
    sed_flux: np.ndarray,
    filt_lamb: np.ndarray,
    filt_trans: np.ndarray,
    lmin: float,
    lmax: float,
) -> tuple[float, float] | None:
    """Return ``(fmel, arean)`` or None if the SED does not cover the filter."""
    sed_lamb = np.asarray(sed_lamb, dtype=np.float64)
    sed_flux = np.asarray(sed_flux, dtype=np.float64)
    filt_lamb = np.asarray(filt_lamb, dtype=np.float64)
    filt_trans = np.asarray(filt_trans, dtype=np.float64)
    if sed_lamb.size < 2 or sed_lamb[0] >= lmin or sed_lamb[-1] <= lmax:
        return None
    mask_f = (filt_lamb >= lmin) & (filt_lamb <= lmax)
    mask_s = (sed_lamb >= lmin) & (sed_lamb <= lmax)
    grid = np.unique(np.concatenate([filt_lamb[mask_f], sed_lamb[mask_s]]))
    if grid.size < 2:
        return None
    sed_i = np.interp(grid, sed_lamb, sed_flux, left=0.0, right=0.0)
    tr_i = np.interp(grid, filt_lamb, filt_trans, left=0.0, right=0.0)
    arean = 0.0
    fmel = 0.0
    for k in range(grid.size - 1):
        lmean = 0.5 * (grid[k] + grid[k + 1])
        tmean = 0.5 * (tr_i[k] + tr_i[k + 1])
        fmean = 0.5 * (sed_i[k] + sed_i[k + 1])
        dlbd = grid[k + 1] - grid[k]
        conv = C_LIGHT_A_S / (lmean * lmean)
        arean += tmean * conv * dlbd
        fmel += fmean * tmean * dlbd
    return fmel, arean


def compute_magnitudes(
    sed_lamb: np.ndarray,
    sed_flux: np.ndarray,
    filters: Sequence[FilterCurve],
    dist_mod: float = 0.0,
) -> np.ndarray:
    """AB magnitudes matching ``SED::compute_magnitudes`` (HIGH_MAG=999 if invalid)."""
    mags = np.empty(len(filters), dtype=np.float64)
    for i, flt in enumerate(filters):
        integ = integrate_sed(sed_lamb, sed_flux, flt.lamb, flt.trans, flt.lmin, flt.lmax)
        if integ is None or integ[1] <= 0 or integ[0] <= 0:
            mags[i] = 999.0
            continue
        ratio = integ[0] / integ[1] * flt.fcorr
        mags[i] = -2.5 * np.log10(ratio) - 48.6 + dist_mod
    return mags


def compute_fluxes(
    sed_lamb: np.ndarray,
    sed_flux: np.ndarray,
    filters: Sequence[FilterCurve],
) -> np.ndarray:
    """Model fluxes as stored by PhotoZ: AB mag → ``10**(-0.4*(m+48.6))``."""
    return ab_mag_to_flux(compute_magnitudes(sed_lamb, sed_flux, filters, dist_mod=0.0))


def redshift_spectrum(sed_lamb: np.ndarray, sed_flux: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
    """``SED::redshift``: ``lamb*(1+z)``, ``flux/(1+z)``."""
    one_pz = 1.0 + z
    return sed_lamb * one_pz, sed_flux / one_pz


def apply_extinction(
    sed_lamb: np.ndarray,
    sed_flux: np.ndarray,
    ebv: float,
    ext_lamb: np.ndarray,
    ext_val: np.ndarray,
) -> np.ndarray:
    """``SED::applyExt`` continuum extinction."""
    if ebv <= 1e-20:
        return sed_flux
    k = np.interp(sed_lamb, ext_lamb, ext_val, left=0.0, right=0.0)
    return sed_flux * np.power(10.0, -0.4 * ebv * k)


def load_filters_from_config(config: dict) -> list[FilterCurve]:
    """Filters as C++ ``Mag`` sees them.

    ``Mag::Mag`` (``mag.cpp``) reads ``$LEPHAREWORK/filt/<FILTER_FILE>.dat``, which the filter
    step writes with 6 significant digits; ``FilterSvc.from_keymap`` keeps full precision and
    shifts every band integral by up to ~1e-6 relative.  Prefer the written file when present.
    """
    import os

    import lephare as lp

    keymap = lp.all_types_to_keymap(config)
    name = str(keymap["FILTER_FILE"].value).split(",")[0].strip() if "FILTER_FILE" in keymap else "filters"
    dat = Path(os.environ.get("LEPHAREWORK", "")) / "filt" / f"{name or 'filters'}.dat"
    if dat.is_file():
        flts = list(lp.read_filters_from_file(str(dat)))
    else:
        flts = list(lp.FilterSvc.from_keymap(keymap))
    out: list[FilterCurve] = []
    for f in flts:
        lamb = np.array([e.lamb for e in f.lamb_trans], dtype=np.float64)
        trans = np.array([e.val for e in f.lamb_trans], dtype=np.float64)
        fcorr = float(getattr(f, "fcorr", 1.0) or 1.0)
        out.append(
            FilterCurve(
                name=str(f.name),
                lamb=lamb,
                trans=trans,
                lmin=float(f.lmin()),
                lmax=float(f.lmax()),
                fcorr=fcorr,
            )
        )
    return out


def _gal_phys_count(raw: bytes) -> int:
    """Number of doubles after each GAL spectrum in ``GalSED::writeSED``.

    LePhare 1.0 writes 11 (``luv lopt lnir ltir mass sfr zmet tau d4000 qi[2]
    age``); older releases wrote 10 (no ``qi[2]``). Pick the layout that walks
    the file exactly.
    """
    import struct

    for n_phys in (11, 10):
        offset = 0
        while offset + 12 <= len(raw):
            nbw = int(struct.unpack_from("<q", raw, offset + 4)[0])
            if nbw <= 0:
                break
            offset += 12 + 16 * nbw + 8 * n_phys
        if offset == len(raw):
            return n_phys
    raise ValueError("GAL lib_bin layout not recognised (expected 10 or 11 phys doubles)")


def iter_sed_bin(path: str | Path, kind: str = "star") -> Iterable[SedSpectrum]:
    """Yield SEDs from a sedtolib ``lib_bin/*.bin`` (STAR/QSO/GAL layouts)."""
    import struct

    path = Path(path)
    raw = path.read_bytes()
    offset = 0
    n = len(raw)
    is_gal = kind.lower().startswith("g")
    n_phys = _gal_phys_count(raw) if is_gal else 0

    while offset + 4 + 8 <= n:
        nummod = int(struct.unpack_from("<i", raw, offset)[0])
        offset += 4
        nbw = int(struct.unpack_from("<q", raw, offset)[0])
        offset += 8
        if nbw <= 0 or offset + 16 * nbw > n:
            break
        lamb = np.frombuffer(raw, dtype="<f8", count=nbw, offset=offset).copy()
        offset += 8 * nbw
        flux = np.frombuffer(raw, dtype="<f8", count=nbw, offset=offset).copy()
        offset += 8 * nbw
        if not is_gal:
            yield SedSpectrum(nummod=nummod, lamb=lamb, flux=flux)
            continue
        if offset + 8 * n_phys > n:
            break
        phys = np.frombuffer(raw, dtype="<f8", count=n_phys, offset=offset)
        offset += 8 * n_phys
        yield SedSpectrum(
            nummod=nummod,
            lamb=lamb,
            flux=flux,
            age=float(phys[-1]),
            luv=float(phys[0]),
            lopt=float(phys[1]),
            lnir=float(phys[2]),
            ltir=float(phys[3]),
            mass=float(phys[4]),
            sfr=float(phys[5]),
            zmet=float(phys[6]),
        )


def load_extinction_law(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a LePHARE ``ext/*.dat`` law as ``(lambda, k_lambda)``."""
    path = Path(path)
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            rows.append((float(parts[0]), float(parts[1])))
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:, 0], arr[:, 1]


def device_ab_mag_to_flux(mags, backend: str = "numpy", dtype=np.float64):
    """AB mag → flux on numpy / jax / cupy."""
    return ab_mag_to_flux(np.asarray(mags, dtype=np.float64))


def device_compute_fluxes_batch(
    spectra: Sequence[SedSpectrum],
    filters: Sequence[FilterCurve],
    redshifts: Sequence[float] | None = None,
    dist_mods: Sequence[float] | None = None,
    backend: str = "numpy",
    dtype=np.float64,
) -> np.ndarray:
    """Compute ``[N_sed, N_band]`` fluxes; optional per-SED redshift/distMod.

    Host loop over SEDs (spectra lengths differ); band integrals use the shared
    NumPy kernel, then mag→flux is applied on the requested device.
    """
    n = len(spectra)
    n_b = len(filters)
    mags = np.empty((n, n_b), dtype=np.float64)
    zs = redshifts if redshifts is not None else [0.0] * n
    dms = dist_mods if dist_mods is not None else [0.0] * n
    for i, sed in enumerate(spectra):
        z = float(zs[i])
        lamb, flux = sed.lamb, sed.flux
        if z > 1e-10:
            lamb, flux = redshift_spectrum(lamb, flux, z)
        mags[i] = compute_magnitudes(lamb, flux, filters, dist_mod=float(dms[i]))
    return device_ab_mag_to_flux(mags, backend=backend, dtype=dtype)
