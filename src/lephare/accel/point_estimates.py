"""Batched RAIL point estimates on device.

``zbest`` is RAIL's ``PointEstimationMixin._calculate_best_point_estimate``: per object,
``scipy.optimize.minimize_scalar(risk, bounds=(z[0], z[-1]), method="bounded")`` with
``risk(zx) = simpson(p(z) * loss(zx, z), z)``. Here the same bounded Brent iteration
(``scipy.optimize._optimize._minimize_scalar_bounded``, xatol=1e-5, maxiter=500) runs for
all objects at once, one masked step per iteration, so every object takes the steps scipy
takes for it. Simpson's rule is linear in p(z); its weights come from ``scipy.integrate.simpson``
itself, so the quadrature is scipy's (including non-uniform grids).

Array code is functional (no in-place updates) so it runs on NumPy, CuPy or jax.numpy.
"""

from __future__ import annotations

import contextlib
import math

import numpy as np

_GOLDEN = 0.5 * (3.0 - math.sqrt(5.0))
_SQRT_EPS = math.sqrt(2.2e-16)


def simpson_weights(grid: np.ndarray) -> np.ndarray:
    """Weights ``w`` with ``simpson(y, grid) == w @ y`` (scipy's rule, any grid)."""
    from scipy.integrate import simpson

    grid = np.asarray(grid, dtype=np.float64)
    return np.asarray(simpson(np.eye(grid.size), x=grid, axis=-1), dtype=np.float64)


def zbest_batched(pdf_vals, grid, xp=np, gamma: float = 0.15, xatol: float = 1e-5, maxiter: int = 500):
    """RAIL ``zbest`` for every row of ``pdf_vals`` (``[N, len(grid)]``) on ``xp``'s device."""
    grid_np = np.asarray(grid, dtype=np.float64)
    w = xp.asarray(simpson_weights(grid_np))
    z = xp.asarray(grid_np)
    inv1pz = 1.0 / (1.0 + z)
    pw = xp.asarray(pdf_vals, dtype=xp.float64) * w[None, :]
    n = pw.shape[0]

    def risk(x):
        dz = (x[:, None] - z[None, :]) * inv1pz[None, :] / gamma
        return xp.sum(pw * (1.0 - 1.0 / (1.0 + dz * dz)), axis=1)

    lo, hi = float(grid_np[0]), float(grid_np[-1])
    a = xp.full(n, lo)
    b = xp.full(n, hi)
    xf = a + _GOLDEN * (b - a)
    nfc, fulc = xf, xf
    rat = xp.zeros(n)
    e = xp.zeros(n)
    fx = risk(xf)
    ffulc, fnfc = fx, fx
    num = 1
    xm = 0.5 * (a + b)
    tol1 = _SQRT_EPS * xp.abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1
    active = xp.abs(xf - xm) > (tol2 - 0.5 * (b - a))

    while num < maxiter and bool(xp.any(active)):
        # parabolic fit (scipy: only when |e| > tol1)
        para = xp.abs(e) > tol1
        r = (xf - nfc) * (fx - ffulc)
        q = (xf - fulc) * (fx - fnfc)
        p = (xf - fulc) * q - (xf - nfc) * r
        q = 2.0 * (q - r)
        p = xp.where(q > 0.0, -p, p)
        q = xp.abs(q)
        r_old = e
        accept = para & (xp.abs(p) < xp.abs(0.5 * q * r_old)) & (p > q * (a - xf)) & (p < q * (b - xf))
        rat_par = p / xp.where(q == 0.0, 1.0, q)
        x_par = xf + rat_par
        si_m = xp.sign(xm - xf) + ((xm - xf) == 0)
        rat_par = xp.where(((x_par - a) < tol2) | ((b - x_par) < tol2), tol1 * si_m, rat_par)
        # golden section otherwise
        e_gold = xp.where(xf >= xm, a - xf, b - xf)
        e_new = xp.where(accept, rat, e_gold)
        rat_new = xp.where(accept, rat_par, _GOLDEN * e_gold)

        si = xp.sign(rat_new) + (rat_new == 0)
        x = xf + si * xp.maximum(xp.abs(rat_new), tol1)
        fu = risk(x)

        better = fu <= fx
        a_new = xp.where(better, xp.where(x >= xf, xf, a), xp.where(x < xf, x, a))
        b_new = xp.where(better, xp.where(x >= xf, b, xf), xp.where(x < xf, b, x))
        c1 = ~better & ((fu <= fnfc) | (nfc == xf))
        c2 = ~better & ~c1 & ((fu <= ffulc) | (fulc == xf) | (fulc == nfc))
        fulc_new = xp.where(better | c1, nfc, xp.where(c2, x, fulc))
        ffulc_new = xp.where(better | c1, fnfc, xp.where(c2, fu, ffulc))
        nfc_new = xp.where(better, xf, xp.where(c1, x, nfc))
        fnfc_new = xp.where(better, fx, xp.where(c1, fu, fnfc))
        xf_new = xp.where(better, x, xf)
        fx_new = xp.where(better, fu, fx)

        upd = lambda new, old, active=active: xp.where(active, new, old)  # noqa: E731
        a, b = upd(a_new, a), upd(b_new, b)
        e, rat = upd(e_new, e), upd(rat_new, rat)
        fulc, ffulc = upd(fulc_new, fulc), upd(ffulc_new, ffulc)
        nfc, fnfc = upd(nfc_new, nfc), upd(fnfc_new, fnfc)
        xf, fx = upd(xf_new, xf), upd(fx_new, fx)
        num += 1
        xm = 0.5 * (a + b)
        tol1 = _SQRT_EPS * xp.abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1
        active = active & (xp.abs(xf - xm) > (tol2 - 0.5 * (b - a)))
    return xf


def device_array_module(backend: str):
    """``(xp, context)`` for the backend's device: CuPy for CUDA backends, jax.numpy
    (inside ``enable_x64``) for JAX backends, NumPy otherwise."""
    if backend.startswith("cuda"):
        try:
            import cupy as cp

            return cp, contextlib.nullcontext()
        except ImportError:  # pragma: no cover
            pass
    if backend.startswith("jax"):
        try:
            import jax.numpy as jnp
            from jax.experimental import enable_x64

            return jnp, enable_x64()
        except ImportError:  # pragma: no cover
            pass
    return np, contextlib.nullcontext()


def zbest_on_backend(pdf_vals: np.ndarray, grid: np.ndarray, backend: str) -> np.ndarray:
    xp, ctx = device_array_module(backend)
    with ctx:
        out = zbest_batched(pdf_vals, grid, xp=xp)
        if xp is np:
            return np.asarray(out, dtype=np.float64)
        if hasattr(out, "get"):
            return np.asarray(out.get(), dtype=np.float64)
        return np.asarray(out, dtype=np.float64)
