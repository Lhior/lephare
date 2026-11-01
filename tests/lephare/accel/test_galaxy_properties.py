"""Galaxy physical-property helpers of lephare.accel vs the C++ PDF and cosmology code."""

from __future__ import annotations

import numpy as np
import pytest

import lephare as lp
from lephare.accel.physical import (
    PHYS_PDF_GRIDS,
    dist_mod,
    pdf_index,
    pdf_median_interval,
)


def _random_pdfs(rng, name, n_rows=60):
    lo, hi, n = PHYS_PDF_GRIDS[name]
    x = np.linspace(lo, hi, n)
    pdfs = np.zeros((n_rows, n))
    for i in range(n_rows):
        kind = i % 6
        if kind == 0:
            pdfs[i, rng.integers(n)] = 1.0
        elif kind == 2:
            pdfs[i, :2] = (1.0, 0.3)
        elif kind == 3:
            pdfs[i, -1] = 1.0
        elif kind > 3:
            c, w = rng.uniform(lo, hi), rng.uniform(0.05, 2.0)
            pdfs[i] = np.exp(-0.5 * ((x - c) / w) ** 2) * rng.uniform(0.1, 10.0)
            pdfs[i, rng.random(n) < 0.3] = 0.0
    return pdfs  # kind 1 rows stay empty


@pytest.mark.parametrize("name", list(PHYS_PDF_GRIDS))
def test_pdf_median_interval_matches_cpp(name):
    """levelCumu2x(0.5) + credible_interval(68) port vs ``lp.PDF``."""
    lo, hi, n = PHYS_PDF_GRIDS[name]
    pdfs = _random_pdfs(np.random.default_rng(0), name)
    med, inf, sup = pdf_median_interval(pdfs, name)
    for i, p in enumerate(pdfs):
        ref = lp.PDF(lo, hi, n)
        ref.vPDF = list(p)
        ref.normalization()
        m = ref.levelCumu2x(0.5)
        a, b = ref.credible_interval(68, m)
        np.testing.assert_allclose([med[i], inf[i], sup[i]], [m, a, b], rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", list(PHYS_PDF_GRIDS))
def test_pdf_index_matches_cpp(name):
    lo, hi, n = PHYS_PDF_GRIDS[name]
    rng = np.random.default_rng(1)
    step = (hi - lo) / (n - 1)
    vals = np.r_[
        rng.uniform(lo - 2, hi + 2, 500), lo, hi, np.nan, -np.inf, np.inf, lo + step * (np.arange(n) + 0.5)
    ]
    ref = lp.PDF(lo, hi, n)
    expected = [ref.index(float(v)) for v in vals]
    np.testing.assert_array_equal(pdf_index(vals, name), expected)


def test_dist_mod_matches_cpp():
    z = np.r_[0.0, 1e-12, np.random.default_rng(2).uniform(0, 7, 200)]
    expected = np.asarray(lp.cosmo(70, 0.3, 0.7).distMod(z))
    np.testing.assert_allclose(dist_mod(z, (70.0, 0.3, 0.7)), expected, rtol=0, atol=1e-12)
