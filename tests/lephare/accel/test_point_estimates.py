"""Batched device zbest vs the per-object scipy point estimate used by RAIL."""

from __future__ import annotations

import numpy as np
import pytest

from lephare.accel.dispatch import cuda_available, jax_available
from lephare.accel.point_estimates import simpson_weights, zbest_batched, zbest_on_backend


def _rail_zbest(pdf_vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """RAIL's PointEstimationMixin._calculate_best_point_estimate loop, verbatim math."""
    from scipy.integrate import simpson
    from scipy.optimize import minimize_scalar

    out = np.zeros(pdf_vals.shape[0])
    for i, pz in enumerate(pdf_vals):

        def risk(zx, pz=pz):
            dz = (zx - grid) / (1 + grid)
            return simpson(pz * (1 - 1 / (1 + (dz / 0.15) ** 2)), grid)

        out[i] = minimize_scalar(risk, bounds=(grid[0], grid[-1]), method="bounded").x
    return out


def _pdfs(n=400, nz=301, seed=0):
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 3.0, nz)
    pdfs = np.zeros((n, nz))
    for i in range(n):
        kind = i % 5
        if kind == 0:  # narrow
            pdfs[i] = np.exp(-0.5 * ((grid - rng.uniform(0, 3)) / rng.uniform(0.005, 0.05)) ** 2)
        elif kind == 1:  # bimodal (several local risk minima)
            for _ in range(2):
                pdfs[i] += rng.uniform(0.3, 1) * np.exp(
                    -0.5 * ((grid - rng.uniform(0, 3)) / rng.uniform(0.03, 0.2)) ** 2
                )
        elif kind == 2:  # peaked at an edge
            pdfs[i] = (
                np.exp(-grid / rng.uniform(0.05, 0.5)) if rng.random() < 0.5 else np.exp((grid - 3) / 0.1)
            )
        elif kind == 3:  # broad, noisy
            pdfs[i] = np.exp(-0.5 * ((grid - rng.uniform(0.5, 2.5)) / 0.6) ** 2) * rng.uniform(0.5, 1.5, nz)
        else:  # flat
            pdfs[i] = 1.0
    return grid, pdfs


def test_simpson_weights_match_scipy():
    from scipy.integrate import simpson

    for grid in (
        np.linspace(0, 3, 301),
        np.linspace(0, 3, 300),
        np.sort(np.random.default_rng(1).uniform(0, 3, 51)),
    ):
        y = np.random.default_rng(2).random(grid.size)
        assert np.isclose(simpson_weights(grid) @ y, simpson(y, x=grid), rtol=1e-13, atol=0)


@pytest.mark.parametrize(
    "backend",
    [
        "numpy",
        pytest.param(
            "cuda_hybrid", marks=pytest.mark.skipif(not cuda_available(), reason="cuda/cupy unavailable")
        ),
        pytest.param("jax_hybrid", marks=pytest.mark.skipif(not jax_available(), reason="jax not installed")),
    ],
)
def test_zbest_batched_matches_rail(backend):
    grid, pdfs = _pdfs()
    ref = _rail_zbest(pdfs, grid)
    got = (
        np.asarray(zbest_batched(pdfs, grid)) if backend == "numpy" else zbest_on_backend(pdfs, grid, backend)
    )
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-9)
