"""Linear-operator MagGal vs the per-SED lephare v1.0.0 port (mag_physics)."""

from __future__ import annotations

import numpy as np
import pytest

from lephare.accel import maggal_linear as ml
from lephare.accel.mag_conv import FilterCurve
from lephare.accel.mag_physics import (
    OpaTable,
    apply_opa_cpp,
    compute_magnitudes_cpp,
    redshift_spectrum,
    warning_integrate_sed,
)


class _Setup:
    def __init__(self, filters, gridz, opa_all):
        self.filters = filters
        self.gridz = np.asarray(gridz, dtype=np.float64)
        self.grid_dm = 5 * np.log10(1 + 1e3 * self.gridz) + 1.0  # any per-z offset
        self.opa_all = opa_all


def _filter(lo, hi, n, name):
    lamb = np.linspace(lo, hi, n)
    trans = np.sin(np.linspace(0, np.pi, n)) ** 2
    return FilterCurve(name=name, lamb=lamb, trans=trans, lmin=lo, lmax=hi, fcorr=1.0)


def _reference(lamb, flux, setup):
    out = np.empty((setup.gridz.size, len(setup.filters)))
    for k, z in enumerate(setup.gridz):
        lam, f = warning_integrate_sed(lamb, flux, float(z), setup.filters)
        lam, f = apply_opa_cpp(lam, f, float(z), setup.opa_all)
        lam, f = redshift_spectrum(lam, f, float(z))
        out[k] = compute_magnitudes_cpp(lam, f, setup.filters, dist_mod=float(setup.grid_dm[k]))
    return out


@pytest.fixture
def setup():
    rng = np.random.default_rng(3)
    opa = []
    for i in range(81):
        ol = np.linspace(300.0, 1215.66, 40)
        opa.append(
            OpaTable(lamb=ol, trans=np.clip(1 - 0.012 * i * rng.random(40), 0, 1), lmin=ol[0], lmax=ol[-1])
        )
    filters = [_filter(3000, 4000, 57, "u"), _filter(5400, 7000, 81, "r"), _filter(9000, 12000, 33, "y")]
    return _Setup(filters, np.round(np.arange(0, 3.01, 0.05), 10), opa)


@pytest.mark.parametrize("case", ["regular", "duplicate_lambda", "narrow_sed"])
def test_continuum_linear_matches_per_sed_port(setup, case):
    rng = np.random.default_rng(7)
    lamb = np.sort(rng.uniform(100.0, 30000.0, 600))
    if case == "duplicate_lambda":
        lamb[300] = lamb[299]  # a step in the spectrum, as in some QSO templates
    if case == "narrow_sed":
        lamb = np.sort(rng.uniform(2500.0, 9000.0, 300))  # needs warning_integrateSED extension
    flux = rng.uniform(0.1, 2.0, lamb.size) * 1e-30
    got = ml.continuum_mags_linear([lamb], [flux], np.ones((1, setup.gridz.size), bool), setup)[0]
    ref = _reference(lamb, flux, setup)
    special = np.abs(ref) >= 999
    np.testing.assert_array_equal(np.abs(got) >= 999, special)
    np.testing.assert_array_equal(got[special], ref[special])
    np.testing.assert_allclose(got[~special], ref[~special], rtol=0, atol=1e-11)


def test_identity_cache_ignores_reused_addresses():
    """A freed library's address can be reused by a new one of the same shape; caches
    keyed on the address returned the old library's arrays."""
    import gc

    from lephare.accel.maglib import cached_by_identity

    cache: dict = {}
    for i in range(200):
        arr = np.full(4096, float(i))
        assert cached_by_identity(cache, arr, (), lambda arr=arr: float(arr.sum())) == 4096.0 * i
        del arr
        gc.collect()
    assert len(cache) <= 4
