"""Export / load dense mag-library tensors from a LePHARE PhotoZ instance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class MagLibTensors:
    """Dense arrays for accelerated χ² / PDF kernels."""

    flux: np.ndarray  # [N_t, N_b] model fluxes
    z: np.ndarray  # [N_t] template redshifts
    types: np.ndarray  # [N_t] 0=GAL, 1=QSO, 2=STAR
    mag0: np.ndarray  # [N_t] abs-mag prior reference magnitude at z=0
    gridz: np.ndarray  # PDF / mag-lib redshift grid
    prior_lib: np.ndarray  # [4] MAG_ABS gal/agn bright/faint
    funz0: float
    err_scale: np.ndarray
    err_factor: np.ndarray
    gbcont: int
    contforb: int
    rm_discrepant: float
    mabs_ref: int
    n_bands: int
    nummod: np.ndarray | None = None  # [N_t] SED nummod (for MOD_STAR)
    age: np.ndarray | None = None  # [N_t] SED age (interp_lib is_same_model)
    ebv: np.ndarray | None = None  # [N_t] SED E(B-V)
    z_interp: bool = True  # match C++ Z_INTERP for Z_BEST / ZQ_BEST
    # Per-template SED metadata for galaxy physical properties (None if unknown).
    extlaw: np.ndarray | None = None  # [N_t] extinction law index (SED::extlawId)
    luv: np.ndarray | None = None  # [N_t] log NUV luminosity (SED::luv)
    lopt: np.ndarray | None = None  # [N_t] log optical luminosity (SED::lopt)
    lnir: np.ndarray | None = None  # [N_t] log NIR luminosity (SED::lnir)
    ltir: np.ndarray | None = None  # [N_t] log dust luminosity [Lsol] (SED::ltir)
    mass: np.ndarray | None = None  # [N_t] stellar mass [Msol] (SED::mass, -999 if undefined)
    sfr: np.ndarray | None = None  # [N_t] SFR [Msol/yr] (SED::sfr, -999 if undefined)
    index_z0: np.ndarray | None = None  # [N_t] row of the z=0 version of the template
    mag: np.ndarray | None = None  # [N_t, N_b] model AB mags incl. emission lines
    kcorr: np.ndarray | None = None  # [N_t, N_b] k-corrections mag(z) - mag(0) - DM(z)

    def has_phys(self) -> bool:
        """True when all per-template metadata for galaxy properties is present."""
        return all(getattr(self, f) is not None for f in PHYS_FIELDS)

    def astype(self, dtype) -> MagLibTensors:
        return MagLibTensors(
            flux=self.flux.astype(dtype, copy=False),
            z=self.z.astype(dtype, copy=False),
            types=self.types,  # keep int
            mag0=self.mag0.astype(dtype, copy=False),
            gridz=self.gridz.astype(dtype, copy=False),
            prior_lib=self.prior_lib.astype(dtype, copy=False),
            funz0=float(self.funz0),
            err_scale=self.err_scale.astype(dtype, copy=False),
            err_factor=self.err_factor.astype(dtype, copy=False),
            gbcont=self.gbcont,
            contforb=self.contforb,
            rm_discrepant=self.rm_discrepant,
            mabs_ref=self.mabs_ref,
            n_bands=self.n_bands,
            nummod=None if self.nummod is None else self.nummod.copy(),
            age=None if self.age is None else self.age.astype(dtype, copy=False),
            ebv=None if self.ebv is None else self.ebv.astype(dtype, copy=False),
            z_interp=bool(self.z_interp),
            **{f: _copy_or_none(getattr(self, f)) for f in PHYS_FIELDS},
        )


# Optional per-template metadata (float arrays keep float64; index arrays stay int).
PHYS_FIELDS = (
    "extlaw",
    "luv",
    "lopt",
    "lnir",
    "ltir",
    "mass",
    "sfr",
    "index_z0",
    "mag",
    "kcorr",
)
_PHYS_INT_FIELDS = ("extlaw", "index_z0")


def _copy_or_none(arr):
    return None if arr is None else np.array(arr, copy=True)


def _parse_doubles(cfg: dict[str, Any], key: str, default: str, n: int = -1) -> list[float]:
    raw = str(cfg.get(key, default))
    parts = [p.strip() for p in raw.replace(",", " ").split() if p.strip()]
    vals = [float(p) for p in parts]
    if n > 0 and len(vals) == 1:
        vals = vals * n
    return vals


def export_maglib_from_photoz(photz, config: dict[str, Any] | None = None) -> MagLibTensors:
    """Build MagLibTensors from an constructed ``lephare.PhotoZ`` instance."""
    import lephare as lp

    config = config or {}
    gridz = np.asarray(photz.gridz, dtype=np.float64)

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
    # Match C++: store bright/faint with bright = min
    gb0, gf0 = mag_abs[0], mag_abs[1]
    if gb0 > gf0:
        gb0, gf0 = gf0, gb0
    qb0, qf0 = mag_abs_agn[0], mag_abs_agn[1]
    if qb0 > qf0:
        qb0, qf0 = qf0, qb0
    prior_lib = np.array([gb0, gf0, qb0, qf0], dtype=np.float64)

    mabs_ref = int(float(str(config.get("MAG_REF", "3")).split(",")[0])) - 1
    mabs_ref = max(mabs_ref, 0)

    flux = np.asarray(photz.flux, dtype=np.float64)
    z = np.asarray(photz.zLib, dtype=np.float64)
    full = photz.fullLib
    n_t = int(flux.shape[0])
    if len(full) != n_t:
        raise RuntimeError(f"PhotoZ flux rows {n_t} != fullLib size {len(full)}")
    n_bands = int(photz.imagm)
    types = np.empty(n_t, dtype=np.int32)
    mag0 = np.empty(n_t, dtype=np.float64)
    nummod = np.empty(n_t, dtype=np.int32)
    age = np.empty(n_t, dtype=np.float64)
    ebv = np.empty(n_t, dtype=np.float64)
    extlaw = np.empty(n_t, dtype=np.int64)
    lum = np.empty((n_t, 4), dtype=np.float64)  # luv, lopt, lnir, ltir
    index_z0 = np.empty(n_t, dtype=np.int64)
    mag = np.empty((n_t, n_bands), dtype=np.float64)
    red = np.empty(n_t, dtype=np.float64)
    for i, sed in enumerate(full):
        if sed.is_gal():
            types[i] = 0
        elif sed.is_qso():
            types[i] = 1
        else:
            types[i] = 2
        i0 = int(sed.index_z0)
        if i0 < 0 or i0 >= n_t:
            i0 = i
        index_z0[i] = i0
        mag[i] = sed.mag
        mag_z0 = mag[i0] if i0 <= i else full[i0].mag
        mag0[i] = float(mag_z0[mabs_ref] if mabs_ref < len(mag_z0) else mag_z0[0])
        nummod[i] = int(sed.nummod)
        age[i] = float(getattr(sed, "age", 0.0) or 0.0)
        ebv[i] = float(getattr(sed, "ebv", 0.0) or 0.0)
        extlaw[i] = int(sed.extlawId)
        lum[i] = (sed.luv, sed.lopt, sed.lnir, sed.ltir)
        red[i] = float(sed.red)
    if int(np.sum(types == 0)) <= 0:
        raise RuntimeError("maglib export produced no GAL templates")
    if np.unique(z).size <= 1:
        raise RuntimeError("maglib export has no redshift diversity (unique z <= 1)")

    cosmo_vals = _parse_doubles(config, "COSMOLOGY", "70,0.3,0.7")
    if len(cosmo_vals) < 3:
        cosmo_vals = [70.0, 0.3, 0.7]
    lcdm = lp.cosmo(cosmo_vals[0], cosmo_vals[1], cosmo_vals[2])
    # z=0 distance-modulus shift of the absolute-magnitude prior (C++ folds it into SED::mag0)
    funz0 = float(lcdm.distMod((gridz[1] if len(gridz) > 1 else 0.01) / 20.0))
    # PhotoZ::read_lib -> GalSED::kcorrec: mag(z) - mag(z=0) - distMod(z)
    kcorr = mag - mag[index_z0] - np.asarray(lcdm.distMod(red), dtype=np.float64)[:, None]
    mass, sfr = _gal_mass_sfr_from_lib_bin(config, types, nummod, age)

    err_scale = np.asarray(_parse_doubles(config, "ERR_SCALE", "0.02", n_bands), dtype=np.float64)
    if err_scale.size == 1:
        err_scale = np.full(n_bands, err_scale[0], dtype=np.float64)
    err_factor = np.asarray(_parse_doubles(config, "ERR_FACTOR", "1.5", n_bands), dtype=np.float64)
    if err_factor.size == 1:
        err_factor = np.full(n_bands, err_factor[0], dtype=np.float64)

    gbcont = int(float(str(config.get("GLB_CONTEXT", "0")).split(",")[0]))
    contforb = int(float(str(config.get("FORB_CONTEXT", "-1")).split(",")[0]))
    rm_disc = float(str(config.get("RM_DISCREPANT_BD", "1e9")).split(",")[0])
    z_interp_raw = str(config.get("Z_INTERP", "YES")).strip().upper()
    z_interp = z_interp_raw in ("YES", "Y", "TRUE", "1")

    return MagLibTensors(
        flux=flux,
        z=z,
        types=types,
        mag0=mag0,
        gridz=gridz,
        prior_lib=prior_lib,
        funz0=funz0,
        err_scale=err_scale[:n_bands],
        err_factor=err_factor[:n_bands],
        gbcont=gbcont,
        contforb=contforb,
        rm_discrepant=rm_disc,
        mabs_ref=mabs_ref,
        n_bands=n_bands,
        nummod=nummod,
        age=age,
        ebv=ebv,
        z_interp=z_interp,
        extlaw=extlaw,
        luv=lum[:, 0].copy(),
        lopt=lum[:, 1].copy(),
        lnir=lum[:, 2].copy(),
        ltir=lum[:, 3].copy(),
        mass=mass,
        sfr=sfr,
        index_z0=index_z0,
        mag=mag,
        kcorr=kcorr,
    )


def _gal_mass_sfr_from_lib_bin(
    config: dict[str, Any], types: np.ndarray, nummod: np.ndarray, age: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """GAL stellar mass / SFR per template, joined from sedtolib ``lib_bin`` on (nummod, age).

    The ``lephare`` bindings do not expose ``SED::mass`` / ``SED::sfr``; MagGal copies
    both unchanged from the rest-frame SED, so the ``lib_bin`` record is authoritative.
    Returns ``(None, None)`` if the library file is not available.
    """
    import os
    import warnings

    from .mag_conv import iter_sed_bin

    names = str(config.get("GAL_LIB_IN", config.get("GAL_LIB", ""))).split(",")
    path = Path(os.environ.get("LEPHAREWORK", "")) / "lib_bin" / f"{names[0].strip()}.bin"
    n_t = types.size
    if not names[0].strip() or not path.is_file():
        return None, None
    table = {(s.nummod, s.age): (s.mass, s.sfr) for s in iter_sed_bin(path, kind="gal")}
    mass = np.full(n_t, -999.0)
    sfr = np.full(n_t, -999.0)
    missing = 0
    for i in np.flatnonzero(types == 0):
        hit = table.get((int(nummod[i]), float(age[i])))
        if hit is None:
            missing += 1
            continue
        mass[i], sfr[i] = hit
    if missing:
        warnings.warn(
            f"{missing} GAL templates have no (nummod, age) match in {path}; "
            "their MASS/SFR are left undefined (-999).",
            stacklevel=2,
        )
    return mass, sfr


def save_maglib(tensors: MagLibTensors, path: str | Path, compressed: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nummod = (
        tensors.nummod if tensors.nummod is not None else np.full(tensors.flux.shape[0], -99, dtype=np.int32)
    )
    age = tensors.age if tensors.age is not None else np.zeros(tensors.flux.shape[0], dtype=np.float64)
    ebv = tensors.ebv if tensors.ebv is not None else np.zeros(tensors.flux.shape[0], dtype=np.float64)
    (np.savez_compressed if compressed else np.savez)(
        path,
        flux=tensors.flux,
        z=tensors.z,
        types=tensors.types,
        mag0=tensors.mag0,
        gridz=tensors.gridz,
        prior_lib=tensors.prior_lib,
        funz0=np.array([tensors.funz0]),
        err_scale=tensors.err_scale,
        err_factor=tensors.err_factor,
        nummod=np.asarray(nummod, dtype=np.int32),
        age=np.asarray(age, dtype=np.float64),
        ebv=np.asarray(ebv, dtype=np.float64),
        meta=np.array(
            [
                tensors.gbcont,
                tensors.contforb,
                tensors.rm_discrepant,
                tensors.mabs_ref,
                tensors.n_bands,
                1.0 if tensors.z_interp else 0.0,
            ],
            dtype=np.float64,
        ),
        **{f: np.asarray(getattr(tensors, f)) for f in PHYS_FIELDS if getattr(tensors, f) is not None},
    )


def load_maglib(path: str | Path) -> MagLibTensors:
    data = np.load(path)
    meta = data["meta"]
    nummod = data["nummod"].astype(np.int32) if "nummod" in data.files else None
    n_t = int(data["flux"].shape[0])
    age = data["age"].astype(np.float64) if "age" in data.files else np.zeros(n_t)
    ebv = data["ebv"].astype(np.float64) if "ebv" in data.files else np.zeros(n_t)
    z_interp = True if meta.size < 6 else bool(meta[5] > 0.5)
    phys = {}
    for f in PHYS_FIELDS:
        if f in data.files:
            dt = np.int64 if f in _PHYS_INT_FIELDS else np.float64
            phys[f] = data[f].astype(dt)
    return MagLibTensors(
        flux=data["flux"],
        z=data["z"],
        types=data["types"].astype(np.int32),
        mag0=data["mag0"],
        gridz=data["gridz"],
        prior_lib=data["prior_lib"],
        funz0=float(data["funz0"][0]),
        err_scale=data["err_scale"],
        err_factor=data["err_factor"],
        gbcont=int(meta[0]),
        contforb=int(meta[1]),
        rm_discrepant=float(meta[2]),
        mabs_ref=int(meta[3]),
        n_bands=int(meta[4]),
        nummod=nummod,
        age=age,
        ebv=ebv,
        z_interp=z_interp,
        **phys,
    )


def cached_by_identity(cache: dict, obj, extra: tuple, build, max_entries: int = 4):
    """Memoize ``build()`` per (object identity, extra).

    Keys on ``id(obj)`` but only hits while the cached weak reference still points to
    ``obj``: a freed library's address can be reused by a new one of the same shape, and
    address-keyed caches then returned the old library's device arrays. At most
    ``max_entries`` are kept (each may hold device memory).
    """
    import weakref

    key = (id(obj),) + tuple(extra)
    hit = cache.get(key)
    if hit is not None and hit[0]() is obj:
        return hit[1]
    value = build()
    cache.pop(key, None)
    while len(cache) >= max_entries:
        cache.pop(next(iter(cache)))
    cache[key] = (weakref.ref(obj), value)
    return value
