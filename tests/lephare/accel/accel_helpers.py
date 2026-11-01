"""Shared helpers for the ``lephare.accel`` tests."""

import importlib
import os

import numpy as np
from astropy.table import Table

import lephare as lp

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "..", "data", "examples")
BANDS = [f"mag_{b}_lsst" for b in "ugrizy"]
ERR_BANDS = [f"mag_err_{b}_lsst" for b in "ugrizy"]

# Template sub-configs passed to ``lp.prepare`` (the rail_lephare defaults).
STAR_CONFIG = dict(LIB_ASCII="YES")
GAL_CONFIG = dict(
    LIB_ASCII="YES",
    MOD_EXTINC="18,26,26,33,26,33,26,33",
    EXTINC_LAW="SMC_prevot.dat,SB_calzetti.dat,SB_calzetti_bump1.dat,SB_calzetti_bump2.dat",
    EM_LINES="EMP_UV",
    EM_DISPERSION="0.5,0.75,1.,1.5,2.",
)
QSO_CONFIG = dict(
    LIB_ASCII="YES",
    MOD_EXTINC="0,1000",
    EB_V="0.,0.1,0.2,0.3",
    EXTINC_LAW="SB_calzetti.dat",
)


def set_lephare_env(lephare_dir=None, lephare_work=None):
    """Point lephare (Python and C++) at new data/work directories."""
    if lephare_dir is not None:
        os.environ["LEPHAREDIR"] = lephare_dir
    if lephare_work is not None:
        os.environ["LEPHAREWORK"] = lephare_work
    importlib.reload(lp)


def read_catalog(name):
    """Return (mags, magerrs, zspec) arrays from ``lsst_accel_<name>.fits``."""
    t = Table.read(os.path.join(EXAMPLES, f"lsst_accel_{name}.fits"))
    mags = np.column_stack([np.asarray(t[b], dtype=np.float64) for b in BANDS])
    magerrs = np.column_stack([np.asarray(t[b], dtype=np.float64) for b in ERR_BANDS])
    return mags, magerrs, np.asarray(t["redshift"], dtype=np.float64)


def lephare_input(mags, magerrs, zspec=None):
    """Build the standard ``lp.process`` input table (id, mag/err pairs, context, zspec)."""
    ng = mags.shape[0]
    table = Table()
    table["id"] = np.arange(ng)
    context = np.zeros(ng, dtype=np.int64)
    for n in range(mags.shape[1]):
        table[BANDS[n]] = mags[:, n]
        table[ERR_BANDS[n]] = magerrs[:, n]
        ok = (mags[:, n] > 0) & np.isfinite(mags[:, n]) & (magerrs[:, n] > 0) & np.isfinite(magerrs[:, n])
        context += ok * 2**n
    table["context"] = context
    table["zspec"] = np.full(ng, -99.0) if zspec is None else zspec
    table["string_data"] = " "
    return table
