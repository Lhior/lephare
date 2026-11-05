"""MagGal continuum as a linear operator: ``fmel[c, z, b] = S[c, :] @ W[z, b, :]``.

For a fixed rest-frame wavelength grid, every step of lephare v1.0.0 ``GalMag``
(``applyOpa -> redshift -> reduce_memory -> integrateSED``, ``mag.cpp``) is linear in
the SED flux values: the union grid with the filter, the interpolation weights, the IGM
transmission and the trapezoid all depend on wavelengths only. So the weights ``W`` are
built once per (grid, z, band) with the same rules, and all SEDs sharing the grid (every
BC03 template and its emission-line / extinction variants) are integrated in one GEMM on
the device. Emission lines are linear in the line luminosities the same way.
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext as _null

import numpy as np

from .mag_physics import C_LIGHT_A_S, HIGH_MAG, INVALID_MAG, opa_index, opa_transmission


def _bound_scaled(L, s, q, upper: bool, xp=np):
    """Per row k, the number of entries of ``L * s[k]`` that are ``< q`` (lower_bound) or
    ``<= q`` (upper_bound), for sorted ``L`` and positive ``s``."""
    n = L.size
    side = "right" if upper else "left"
    idx = xp.searchsorted(L, q[None, :] / s[:, None], side=side)
    sk = s[:, None]
    for _ in range(4):
        prev = xp.clip(idx - 1, 0, n - 1)
        x = L[prev] * sk
        back = (idx > 0) & ((x > q) if upper else (x >= q))
        nxt = xp.clip(idx, 0, n - 1)
        y = L[nxt] * sk
        fwd = (idx < n) & ((y <= q) if upper else (y < q))
        if not (bool(back.any()) or bool(fwd.any())):
            break
        idx = idx - back.astype(idx.dtype) + fwd.astype(idx.dtype)
    return idx


def band_weights(rest_lamb, gridz, flt, igm_rows, lmin_all, lmax_all, xp=np):
    """``SED::integrateSED`` (lephare v1.0.0) for one filter at every redshift, as
    weights on the rest-frame flux points: ``r3[k] = W[k] @ flux_rest``.

    Per z (``mag.cpp`` ``GalMag::make_maglib``): ``warning_integrateSED`` adds a zero point
    at lambda=0 / 1e8(1+z) when some filter needs it; the IGM multiplies the flux, the SED is
    redshifted (``lambda*(1+z)``, ``flux/(1+z)``); the band is invalid if the SED starts after
    or ends before the filter; the integral runs over the union of the SED points inside the
    filter support and the filter points, both curves linearly interpolated
    (``interp_linear_point``), ``tmp = tmean*dl; r3 += tmp*fmean; r5 += tmp/lmean^2``.
    ``reduce_memory`` keeps every point this uses. Returns ``W [nz, n]``, ``r1 = c*r5 [nz]``
    and ``invalid [nz]``.
    """
    L = xp.asarray(rest_lamb, dtype=xp.float64)
    n, nz = int(L.size), int(np.asarray(gridz).size)
    s = 1.0 + xp.asarray(np.asarray(gridz, dtype=xp.float64))
    pre = L[0] * s > lmin_all
    post = L[-1] * s < lmax_all
    # extended observed grid; absent extension points become sentinels far outside every
    # filter, so they never bracket a filter point nor enter the union
    obs = xp.empty((nz, n + 2))
    obs[:, 0] = xp.where(pre, 0.0, -1.0)
    obs[:, 1:-1] = L[None, :] * s[:, None]  # SED::redshift: lamb * (1 + red)
    obs[:, -1] = xp.where(post, (1.0e8 * s) * s, 1.0e300)
    fac = xp.asarray(igm_rows, dtype=xp.float64) / s[:, None]  # applyOpa, then val / (1 + red)
    F = xp.asarray(np.asarray(flt.lamb, dtype=xp.float64))
    T = xp.asarray(np.asarray(flt.trans, dtype=xp.float64))
    f0, fn, nf = F[0], F[-1], int(F.size)
    front = xp.where(pre, 0.0, obs[:, 1])
    back = xp.where(post, obs[:, -1], obs[:, -2])
    invalid = (front > f0) | (back < fn)

    rows = xp.arange(nz)
    # merge positions of the union (stable: SED entries before filter entries on ties).
    # lower/upper_bound of F in each row of L*(1+z): search F/(1+z) in L, then fix the
    # few entries where that rounding disagrees with the exact products.
    lb_f = 1 + _bound_scaled(L, s, F, upper=False, xp=xp)  # +1: the leading 0 / sentinel entry
    ub_f = 1 + _bound_scaled(L, s, F, upper=True, xp=xp)
    # union members from the SED: real points inside the filter support; a repeated
    # wavelength enters once (std::set) and, as interp_linear_point uses lower_bound,
    # carries the value of its first occurrence
    obs_r = obs[:, 1:-1]
    dup = xp.zeros(obs_r.shape, dtype=bool)
    dup[:, 1:] = obs_r[:, 1:] == obs_r[:, :-1]
    in_rng = (obs_r >= f0) & (obs_r <= fn) & ~dup
    cumk = xp.cumsum(in_rng, axis=1)  # kept points up to and including each one
    c_k = cumk[:, -1]
    n_before_f = xp.searchsorted(F, obs_r, side="left")  # filter points < SED point
    pos_sed = (cumk - 1) + n_before_f
    ub_r = ub_f - 1  # real points <= F (the leading entry is always <= F)
    kept_le = xp.where(ub_r > 0, xp.take_along_axis(cumk, xp.clip(ub_r - 1, 0, n - 1), axis=1), 0)
    pos_flt = xp.arange(nf)[None, :] + kept_le
    width = int(c_k.max()) + nf
    lam_u = xp.full((nz, width), xp.inf)
    tv = xp.zeros((nz, width))
    rs, js = xp.nonzero(in_rng)
    ps = pos_sed[rs, js]
    lam_u[rs, ps] = obs[rs, js + 1]
    rf = xp.repeat(rows, nf)
    pf = pos_flt.ravel()
    lam_u[rf, pf] = xp.tile(F, nz)
    tv[rf, pf] = xp.tile(T, nz)
    # filter transmission at SED points: interp_linear_point over (F, T)
    o = obs[rs, js + 1]
    lbf = xp.clip(xp.searchsorted(F, o, side="left"), 1, nf - 1)
    t_f = (o - F[lbf - 1]) / (F[lbf] - F[lbf - 1])
    t_sed = T[lbf - 1] + t_f * (T[lbf] - T[lbf - 1])
    tv[rs, ps] = xp.where(o == f0, T[0], xp.where(o == fn, T[-1], t_sed))
    # SED value at filter points: interp_linear_point over the extended SED
    i1 = xp.clip(lb_f, 1, n + 1)
    x0 = xp.take_along_axis(obs, i1 - 1, axis=1)
    x1 = xp.take_along_axis(obs, i1, axis=1)
    t_s = (F[None, :] - x0) / (x1 - x0)  # y0 + t*(y1 - y0) == (1-t)*y0 + t*y1

    live = xp.isfinite(lam_u)
    lam_f = xp.where(live, lam_u, 0.0)
    pair = live[:, 1:] & live[:, :-1]
    dl = xp.where(pair, lam_f[:, 1:] - lam_f[:, :-1], 0.0)
    lmean = (lam_f[:, 1:] + lam_f[:, :-1]) / 2
    tmean = (tv[:, 1:] + tv[:, :-1]) / 2
    tmp = tmean * dl
    with np.errstate(divide="ignore", invalid="ignore") if xp is np else _null():
        r5 = xp.sum(xp.where(pair, tmp / xp.where(pair, lmean * lmean, 1.0), 0.0), axis=1)
    r1 = r5 * C_LIGHT_A_S
    u = xp.zeros_like(lam_u)  # r3 = sum_i tmp_i (s_i + s_{i+1}) / 2 = sum_p u_p s_p
    u[:, :-1] += 0.5 * tmp
    u[:, 1:] += 0.5 * tmp

    # scatter onto the flux sources (extended columns 1..n are rest points 0..n-1)
    W = xp.zeros((nz, n + 2))
    xp.add.at(W, (rs, js + 1), u[rs, ps])
    uf = u[rf, pf]
    xp.add.at(W, (rf, i1.ravel() - 1), uf * (1.0 - t_s.ravel()))
    xp.add.at(W, (rf, i1.ravel()), uf * t_s.ravel())
    return W[:, 1:-1] * fac, r1, invalid


def _pool():
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=max(1, min(16, os.cpu_count() or 1)))
    return _POOL


_POOL = None


def _weights_xp(xp):
    """Array module for building weights: in-place scatter needs NumPy or CuPy, so a
    jax.numpy GEMM gets NumPy-built weights."""
    return np if getattr(xp, "__name__", "").startswith("jax") else xp


def _to_host(xp, a):
    return np.asarray(a.get() if hasattr(a, "get") else a)


def grids_weights(grids: Sequence[np.ndarray], setup, gridz=None, igm: bool = True, xp=np):
    """``(W [nz, nb, n], r1 [nz, nb], invalid [nz, nb])`` per rest-frame grid, as ``xp``
    arrays (CuPy: built on the GPU; NumPy: every (grid, band) on a thread pool)."""
    gridz = np.asarray(setup.gridz if gridz is None else gridz, dtype=np.float64)
    lmin_all = min(float(np.asarray(f.lamb)[0]) for f in setup.filters)
    lmax_all = max(float(np.asarray(f.lamb)[-1]) for f in setup.filters)
    oidx = np.asarray([opa_index(float(z)) for z in gridz])
    nb = len(setup.filters)

    def igm_rows(L):
        if not igm:
            return np.ones((gridz.size, L.size))
        tab = {i: opa_transmission(L, setup.opa_all[i]) for i in np.unique(oidx)}
        return np.stack([tab[i] for i in oidx])

    grids = [np.asarray(L, dtype=np.float64) for L in grids]
    rows = [igm_rows(L) for L in grids]

    def one(g, b):
        return band_weights(grids[g], gridz, setup.filters[b], rows[g], lmin_all, lmax_all, xp=xp)

    tasks = [(g, b) for g in range(len(grids)) for b in range(nb)]
    if xp is np:
        res = dict(zip(tasks, _pool().map(lambda t: one(*t), tasks)))
    else:
        res = {t: one(*t) for t in tasks}
    out = []
    for g in range(len(grids)):
        W = xp.stack([res[(g, b)][0] for b in range(nb)], axis=1)
        r1 = _to_host(xp, xp.stack([res[(g, b)][1] for b in range(nb)], axis=1))
        invalid = _to_host(xp, xp.stack([res[(g, b)][2] for b in range(nb)], axis=1))
        out.append((W, r1, invalid))
    return out


def _matmul(xp, S, W):
    """``S @ W`` on ``xp``'s device, returned on the host."""
    return _to_host(xp, xp.asarray(S) @ xp.asarray(W))


def continuum_mags_linear(
    rest_l: Sequence[np.ndarray],
    rest_f: Sequence[np.ndarray],
    age_ok: np.ndarray,
    setup,
    xp=np,
) -> np.ndarray:
    """``(N_combo, N_z, N_band)`` AB mags, same contract as ``jax_continuum_mags_grid``
    (``HIGH_MAG`` where ``age_ok`` is False)."""
    n_c, n_z, n_b = len(rest_l), int(np.asarray(setup.gridz).size), len(setup.filters)
    out = np.full((n_c, n_z, n_b), HIGH_MAG, dtype=np.float64)
    groups: dict[bytes, list[int]] = defaultdict(list)
    for c, lam in enumerate(rest_l):
        groups[np.asarray(lam, dtype=np.float64).tobytes()].append(c)
    fcorr = np.asarray([float(f.fcorr) for f in setup.filters])
    dm = np.asarray(setup.grid_dm, dtype=np.float64)
    keys = list(groups)
    weights = grids_weights([np.frombuffer(k, dtype=np.float64) for k in keys], setup, xp=_weights_xp(xp))
    for key, (W, arean, invalid) in zip(keys, weights):
        members = groups[key]
        lam = np.frombuffer(key, dtype=np.float64)
        S = np.stack([np.asarray(rest_f[c], dtype=np.float64) for c in members])
        fmel = _matmul(xp, S, W.reshape(n_z * n_b, lam.size).T).reshape(len(members), n_z, n_b)
        with np.errstate(divide="ignore", invalid="ignore"):
            mag = -2.5 * np.log10(fmel / arean[None] * fcorr[None, None]) - 48.6 + dm[None, :, None]
        mag = np.where(fmel > 0, mag, HIGH_MAG)  # SED::compute_magnitudes
        mag = np.where(invalid[None], INVALID_MAG, mag)
        mem = np.asarray(members)
        out[mem] = np.where(age_ok[mem][:, :, None], mag, HIGH_MAG)
    return out


def em_line_basis(line_lamb: np.ndarray, nstep: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """``GalSED::generateEmSpectra`` grid for the given (present) lines, and one
    unit-luminosity Gaussian per line on it: ``spectrum = lv @ basis``."""
    from .mag_physics import PI

    ll = np.asarray(line_lamb, dtype=np.float64)
    nbsigma = 6.0
    sigma = 100.0 * ll / C_LIGHT_A_S / 2.355 * 1.0e13
    step = nbsigma * 2.0 * sigma / float(nstep)
    # C++ appends the samples line by line and looks for gaps in that order, then sorts
    pts = (ll - nbsigma * sigma)[:, None] + np.arange(nstep, dtype=np.float64)[None, :] * step[:, None]
    lamb = pts.ravel()
    basis = np.exp(-((lamb[None, :] - ll[:, None]) ** 2) / (2.0 * sigma[:, None] ** 2)) / (
        sigma[:, None] * np.sqrt(2.0 * PI)
    )
    k = np.arange(1, lamb.size - 1)
    ins = np.concatenate(
        [
            lamb[k][(lamb[k] + 1.0) < lamb[k + 1]] + 0.1,
            lamb[k][(lamb[k - 1] + 1.0) < lamb[k]] - 0.1,
            [0.0, 1.0e7],
        ]
    )
    lamb_all = np.concatenate([lamb, ins])
    order = np.argsort(lamb_all, kind="mergesort")
    basis_all = np.concatenate([basis, np.zeros((ll.size, ins.size))], axis=1)[:, order]
    return lamb_all[order], basis_all


def em_line_fluxes_linear(line_vals: Sequence[tuple[np.ndarray, np.ndarray]], setup, xp=np) -> np.ndarray:
    """Observed line fluxes ``fmel/arean * 10^(-0.4 DM)`` ``(N_combo, N_z, N_band)`` for
    rest-frame line lists ``(lamb, val)`` (after line extinction and EM_DISPERSION)."""
    n_c, n_z, n_b = len(line_vals), int(np.asarray(setup.gridz).size), len(setup.filters)
    out = np.zeros((n_c, n_z, n_b), dtype=np.float64)
    groups: dict[bytes, list[int]] = defaultdict(list)
    for c, (ll, lv) in enumerate(line_vals):
        on = np.asarray(lv) > 0
        if on.any():
            groups[np.asarray(ll, dtype=np.float64)[on].tobytes()].append(c)
    scale = np.power(10.0, -0.4 * np.asarray(setup.grid_dm, dtype=np.float64))
    keys = list(groups)
    bases = [em_line_basis(np.frombuffer(k, dtype=np.float64)) for k in keys]
    wxp = _weights_xp(xp)
    weights = grids_weights([el for el, _ in bases], setup, xp=wxp)
    for key, (_el, basis), (W, arean, invalid) in zip(keys, bases, weights):
        members = groups[key]
        G = wxp.einsum("kbp,jp->kbj", W, wxp.asarray(basis))  # unit-luminosity line fluxes
        lv = np.stack(
            [np.asarray(line_vals[c][1], dtype=np.float64)[np.asarray(line_vals[c][1]) > 0] for c in members]
        )
        nz, nb, nl = G.shape
        fmel = _matmul(xp, lv, (_to_host(wxp, G) if wxp is not xp else G).reshape(nz * nb, nl).T).reshape(
            len(members), nz, nb
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            val = fmel / arean[None] * scale[None, :, None]
        val = np.where((fmel > 0) & (arean[None] > 0) & ~invalid[None], val, 0.0)
        out[np.asarray(members)] = val
    return out


def star_mags_linear(spectra, setup, xp=np) -> np.ndarray:
    """``StarMag``: rest-frame (z=0, no IGM) AB mags ``[N_sed, N_band]`` with
    ``warning_integrateSED`` extension, one GEMM per wavelength grid."""
    n_b = len(setup.filters)
    out = np.empty((len(spectra), n_b), dtype=np.float64)
    groups: dict[bytes, list[int]] = defaultdict(list)
    for i, sp in enumerate(spectra):
        groups[np.asarray(sp.lamb, dtype=np.float64).tobytes()].append(i)
    keys = list(groups)
    fcorr = np.asarray([float(f.fcorr) for f in setup.filters])
    weights = grids_weights(
        [np.frombuffer(k, dtype=np.float64) for k in keys], setup, gridz=[0.0], igm=False, xp=_weights_xp(xp)
    )
    for key, (W, r1, invalid) in zip(keys, weights):
        mem = np.asarray(groups[key])
        S = np.stack([np.asarray(spectra[i].flux, dtype=np.float64) for i in mem])
        r3 = _matmul(xp, S, W[0].T)  # [n_mem, n_b]
        with np.errstate(divide="ignore", invalid="ignore"):
            mag = -2.5 * np.log10(r3 / r1[0][None] * fcorr[None]) - 48.6
        mag = np.where(r3 > 0, mag, HIGH_MAG)
        out[mem] = np.where(invalid[0][None], INVALID_MAG, mag)
    return out
