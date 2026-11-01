"""Parity tests for ``lephare.accel`` (NumPy reference, JAX and CUDA backends).

Library-based tests compare against the C++ code on the LSST fixture in
``conftest.py``; the rest use small synthetic template libraries.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from accel_helpers import EXAMPLES, lephare_input, read_catalog, set_lephare_env

import lephare as lp
from lephare.accel.dispatch import available_backends, cuda_available, jax_available
from lephare.accel.kernels_ref import process_batch
from lephare.accel.maglib import load_maglib, save_maglib

# Sum of object 0's BAY_ZG PDF (C++ lephare) sampled on linspace(0, 3, 300), as in rail_lephare.
GOLDEN_PDF_SUM = 99.66482986408954

DEVICE_KINDS = [
    pytest.param("jax", marks=pytest.mark.skipif(not jax_available(), reason="jax not installed")),
    pytest.param("cuda", marks=pytest.mark.skipif(not cuda_available(), reason="cuda/cupy unavailable")),
]


def _cpp_pdfs(cfg, mags, magerrs):
    """BAY_ZG PDFs and output table from classic ``lp.process``."""
    output, photozlist = lp.process(cfg, lephare_input(mags, magerrs), write_outputs=False)
    pdfs = np.array([np.array(p.pdfmap[11].vPDF) for p in photozlist])
    zgrid = np.array(photozlist[0].pdfmap[11].xaxis)
    return pdfs, zgrid, output


def test_available_backends_includes_lephare():
    assert "lephare" in available_backends()


def test_maglib_export_roundtrip(lsst_library, tmp_path):
    """Exported tensors survive save/load, including age and E(B-V)."""
    tensors = lsst_library
    path = tmp_path / "maglib.npz"
    save_maglib(tensors, path)
    tensors2 = load_maglib(path)
    assert np.allclose(tensors.flux, tensors2.flux)
    assert tensors.age is not None and tensors.age.shape == tensors.types.shape
    assert tensors.ebv is not None and tensors.ebv.shape == tensors.types.shape
    assert np.allclose(tensors.age, tensors2.age)
    assert np.allclose(tensors.ebv, tensors2.ebv)


def test_numpy_pdf_matches_cpp(lsst_env, lsst_library):
    """NumPy reference BAY_ZG matches ``lp.process`` object by object."""
    mags, magerrs, _ = read_catalog("test")
    mags, magerrs = mags[:32], magerrs[:32]
    pdf_cpp, z_cpp, _ = _cpp_pdfs(lsst_env, mags, magerrs)
    pdfs, zgrid = process_batch(mags, magerrs, lsst_library)
    assert np.allclose(zgrid, z_cpp)
    assert np.allclose(pdfs, pdf_cpp, rtol=1e-8, atol=1e-10)
    s = np.sum(np.interp(np.linspace(0, 3, 300), zgrid, pdfs[0] / np.trapezoid(pdfs[0], zgrid)))
    assert np.isclose(s, GOLDEN_PDF_SUM)


@pytest.mark.skipif(not jax_available(), reason="jax not installed")
def test_jax_hybrid_matches_cpp(lsst_env, lsst_library):
    """jax_hybrid BAY_ZG PDFs and best-fit columns match classic lephare."""
    from lephare.accel.dispatch import process_with_backend

    mags, magerrs, _ = read_catalog("test")
    pdf_cpp, _, out_cpp = _cpp_pdfs(lsst_env, mags, magerrs)
    result = process_with_backend("jax_hybrid", mags, magerrs, lsst_library, dtype=np.float64)
    pdfs, ancil = result[0], result[-1]
    pdfs = pdfs / pdfs.sum(axis=1, keepdims=True)
    pdf_cpp = pdf_cpp / pdf_cpp.sum(axis=1, keepdims=True)
    assert np.max(np.abs(pdfs - pdf_cpp)) < 1e-6
    for key in ("Z_BEST", "CHI_BEST", "ZQ_BEST", "CHI_QSO", "MOD_STAR", "CHI_STAR"):
        a = np.asarray(out_cpp[key], dtype=np.float64)
        b = np.asarray(ancil[key], dtype=np.float64)
        atol, rtol = (1e-3, 1e-6) if "CHI" in key else (1e-4, 1e-5)
        ok = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        assert np.all(ok), f"{key}: {(~ok).sum()} mismatches, cpp={a[~ok][:3]} jax={b[~ok][:3]}"


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("jax_hybrid", marks=DEVICE_KINDS[0].marks),
        pytest.param("cuda_hybrid", marks=DEVICE_KINDS[1].marks),
    ],
)
def test_device_cache_not_reused_across_libraries(lsst_library, backend):
    """A second library sharing the flux array but not mag0 must not hit the first one's device cache."""
    import dataclasses

    from lephare.accel.dispatch import process_with_backend

    mags, magerrs, _ = read_catalog("test")
    mags, magerrs = mags[:32], magerrs[:32]
    shifted = dataclasses.replace(lsst_library, mag0=lsst_library.mag0 + 4.0)
    process_with_backend(backend, mags, magerrs, lsst_library, dtype=np.float64)
    pdfs = process_with_backend(backend, mags, magerrs, shifted, dtype=np.float64)[0]
    ref, _ = process_batch(mags, magerrs, shifted)
    unshifted, _ = process_batch(mags, magerrs, lsst_library)
    assert not np.allclose(ref, unshifted)  # the shift moves templates across the MAG_ABS prior
    assert np.allclose(pdfs, ref, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("kind", DEVICE_KINDS)
def test_batched_chi2_matches_per_object(lsst_library, kind):
    """Object-batched χ² must match per-object χ² on a small photometry slice."""
    from lephare.accel.kernels_ref import prepare_photometry

    tensors = lsst_library
    mags, magerrs, _ = read_catalog("test")
    ab, sab, bus, _busul, _nbul = prepare_photometry(mags[:8], magerrs[:8], None, tensors)

    if kind == "jax":
        import jax
        import jax.numpy as jnp

        from lephare.accel.jax_backend import _FIT_CACHE, _build_fit_fns, chi2_batch_jax

        _FIT_CACHE.clear()
        fit_one = _build_fit_fns(tensors, np.float64, jax, jnp)["fit_one"]
        per = []
        for i in range(len(ab)):
            c2, _ = fit_one(jnp.asarray(ab[i]), jnp.asarray(sab[i]), jnp.asarray(bus[i]))
            per.append(np.asarray(jax.device_get(c2)))
        per = np.stack(per, axis=0)
        batched, _ = chi2_batch_jax(ab, sab, bus, tensors, dtype=np.float64)
    else:
        from lephare.accel.cuda_backend import chi2_cuda, chi2_cuda_batch

        lib = (tensors.flux,)
        tail = (tensors.types, tensors.mag0, tensors.z, tensors.prior_lib, tensors.funz0)
        per = np.stack(
            [chi2_cuda(*lib, ab[i], sab[i], bus[i], *tail, dtype=np.float64)[0] for i in range(len(ab))]
        )
        batched, _ = chi2_cuda_batch(*lib, ab, sab, bus, *tail, dtype=np.float64)

    assert batched.shape == per.shape
    assert np.allclose(batched, per, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", DEVICE_KINDS)
def test_device_pdf_matches_numpy(lsst_library, kind):
    """On-device BAY_ZG must match NumPy bay_zg_pdf_batch."""
    from lephare.accel.kernels_ref import bay_zg_pdf_batch, prepare_photometry

    tensors = lsst_library
    mags, magerrs, _ = read_catalog("test")
    ab, sab, bus, _busul, _nbul = prepare_photometry(mags[:8], magerrs[:8], None, tensors)

    if kind == "jax":
        from lephare.accel.jax_backend import chi2_batch_jax, pdf_batch_jax

        chi2, _ = chi2_batch_jax(ab, sab, bus, tensors, dtype=np.float64)
        device_pdf = pdf_batch_jax(chi2, tensors, dtype=np.float64)
    else:
        from lephare.accel.cuda_backend import chi2_cuda_batch, pdf_batch_cuda

        chi2, _ = chi2_cuda_batch(
            tensors.flux,
            ab,
            sab,
            bus,
            tensors.types,
            tensors.mag0,
            tensors.z,
            tensors.prior_lib,
            tensors.funz0,
            dtype=np.float64,
        )
        device_pdf = pdf_batch_cuda(chi2, tensors)

    ref = bay_zg_pdf_batch(chi2, tensors)
    assert device_pdf.shape == ref.shape
    assert np.allclose(device_pdf, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", DEVICE_KINDS)
def test_device_photometry_matches_numpy(lsst_library, kind):
    """Device prepare_photometry must allclose NumPy oracle."""
    from lephare.accel.kernels_ref import prepare_photometry

    tensors = lsst_library
    mags, magerrs, _ = read_catalog("test")
    mags, magerrs = mags[:32], magerrs[:32]
    ref = prepare_photometry(mags, magerrs, None, tensors)

    if kind == "jax":
        from lephare.accel.jax_backend import prepare_photometry_jax

        got = prepare_photometry_jax(mags, magerrs, tensors, dtype=np.float64)
    else:
        from lephare.accel.cuda_backend import prepare_photometry_cuda

        got = prepare_photometry_cuda(mags, magerrs, tensors, dtype=np.float64)

    for a, b, name in zip(got, ref, ("ab", "sab", "bus", "busul", "nbul")):
        assert np.allclose(a, b, rtol=1e-12, atol=1e-12), name


def test_star_filter_convolution_matches_cpp(lsst_env, lsst_library):
    """STAR integrateSED port matches C++ magnitudes within 1e-6."""
    from lephare.accel.mag_conv import compute_magnitudes, iter_sed_bin, load_filters_from_config

    filters = load_filters_from_config(lsst_env)
    star_lib = str(lsst_env.get("STAR_LIB_IN", lsst_env["STAR_LIB"])).split(",")[0].strip()
    path = os.path.join(os.environ["LEPHAREWORK"], "lib_bin", f"{star_lib}.bin")
    spectra = list(iter_sed_bin(path, kind="star"))
    assert len(spectra) >= 1

    sed = lp.StarSED("x")
    sed.readSEDBin(path)
    flts = list(lp.FilterSvc.from_keymap(lp.all_types_to_keymap(lsst_env)))
    sed.compute_magnitudes(flts)
    cpp = np.array(list(sed.mag), dtype=np.float64)
    py = compute_magnitudes(spectra[0].lamb, spectra[0].flux, filters)
    assert np.allclose(py, cpp, rtol=0, atol=1e-6)


@pytest.mark.parametrize("backend", DEVICE_KINDS)
def test_full_maglib_rebuild_parity(lsst_env, lsst_library, backend):
    """Device mag rebuild passes flux parity vs C++ export."""
    from lephare.accel.mag_builder import assert_maglib_parity, rebuild_maglib_cuda, rebuild_maglib_jax

    if backend == "jax":
        rebuilt = rebuild_maglib_jax(lsst_library, config=lsst_env)
    else:
        rebuilt = rebuild_maglib_cuda(lsst_library, config=lsst_env)
    assert_maglib_parity(lsst_library, rebuilt)


@pytest.mark.skipif(not jax_available(), reason="jax not installed")
def test_jax_template_tiles_match_numpy(lsst_library, monkeypatch):
    """Forced template-axis tiling matches NumPy BAY_ZG + ancil (FP64)."""
    monkeypatch.setenv("LEPHARE_ACCEL_TILE_AXIS", "template")
    monkeypatch.setenv("LEPHARE_ACCEL_TEMPLATE_TILE", "4096")

    from lephare.accel.bestfit import bestfit_ancil_from_chi2
    from lephare.accel.jax_backend import _FIT_CACHE, process_batch_jax
    from lephare.accel.kernels_ref import fit_chi2_single, prepare_photometry

    tensors = lsst_library
    mags, magerrs, _ = read_catalog("test")
    mags, magerrs = mags[:16], magerrs[:16]

    _FIT_CACHE.clear()
    pdf_j, z_j, _zm, _zmn, ancil_j = process_batch_jax(mags, magerrs, tensors, return_ancil=True)
    pdf_n, z_n = process_batch(mags, magerrs, tensors)
    assert np.allclose(z_j, z_n)
    assert np.allclose(pdf_j, pdf_n, rtol=1e-10, atol=1e-12)

    ab, sab, bus, busul, nbul = prepare_photometry(mags, magerrs, None, tensors)
    chi2s = [
        fit_chi2_single(ab[i], sab[i], bus[i], busul[i], int(nbul[i]), tensors)[0] for i in range(len(mags))
    ]
    ancil_n = bestfit_ancil_from_chi2(np.stack(chi2s), tensors)
    for key in ancil_n:
        assert np.allclose(ancil_j[key], ancil_n[key], rtol=1e-9, atol=1e-9, equal_nan=True), key


def test_pdf_zmode_zmean_vectorized():
    """Vectorized zmode/zmean matches classic median-of-max-bins definition."""
    from lephare.accel.kernels_ref import pdf_zmode_zmean

    zgrid = np.linspace(0, 3, 7)
    pdfs = np.zeros((3, 7), dtype=np.float64)
    pdfs[0, 2] = 1.0
    pdfs[1, 2] = 0.5
    pdfs[1, 3] = 0.5  # tie → median of z[2], z[3]
    pdfs[2, :] = 0.0
    zmode, zmean = pdf_zmode_zmean(pdfs, zgrid)
    assert np.isclose(zmode[0], zgrid[2])
    assert np.isclose(zmode[1], 0.5 * (zgrid[2] + zgrid[3]))
    assert np.isnan(zmode[2])
    assert np.isclose(zmean[0], zgrid[2])
    assert np.isclose(zmean[1], 0.5 * (zgrid[2] + zgrid[3]))
    assert np.isnan(zmean[2])


def test_improve_extremum_chi2_parabola():
    """Quadratic χ² extremum matches a known parabola vertex (dz ≤ 0.5)."""
    from lephare.accel.bestfit import improve_extremum_chi2

    # C++ skips parabola when |Δx| > 0.5 between consecutive bins
    xaxis = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
    chi2 = (xaxis - 0.75) ** 2 + 2.0
    z_imp, c_imp = improve_extremum_chi2(xaxis, chi2)
    assert np.isclose(z_imp, 0.75, atol=1e-6)
    assert np.isclose(c_imp, 2.0, atol=1e-6)


def _synthetic_9band_nz701_tensors(n_obj: int = 4):
    """Tiny maglib covering fused N_B=9 and Nz=701 (Cardinal FAIR grid)."""
    from lephare.accel.maglib import MagLibTensors

    rng = np.random.default_rng(0)
    n_b, n_z = 9, 701
    gridz = np.linspace(0.0, 7.0, n_z)
    n_gal, n_qso, n_star = 80, 16, 8
    n_t = n_gal + n_qso + n_star
    types = np.concatenate(
        [
            np.zeros(n_gal, dtype=np.int32),
            np.ones(n_qso, dtype=np.int32),
            np.full(n_star, 2, dtype=np.int32),
        ]
    )
    z = np.concatenate(
        [
            np.linspace(0.0, 7.0, n_gal),
            np.linspace(0.0, 7.0, n_qso),
            np.zeros(n_star),
        ]
    )
    flux = np.abs(rng.normal(1.0, 0.3, size=(n_t, n_b)))
    mag0 = np.full(n_t, -20.0)
    tensors = MagLibTensors(
        flux=flux,
        z=z,
        types=types,
        mag0=mag0,
        gridz=gridz,
        prior_lib=np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float64),
        funz0=0.0,
        err_scale=np.full(n_b, 0.02, dtype=np.float64),
        err_factor=np.full(n_b, 1.5, dtype=np.float64),
        gbcont=0,
        contforb=-1,
        rm_discrepant=1e9,
        mabs_ref=0,
        n_bands=n_b,
        nummod=np.arange(n_t, dtype=np.int32),
        z_interp=True,
    )
    mags = 20.0 + rng.normal(0.0, 0.4, size=(n_obj, n_b))
    magerrs = np.full((n_obj, n_b), 0.05)
    return tensors, mags, magerrs


@pytest.mark.skipif(not cuda_available(), reason="cuda/cupy unavailable")
def test_cuda_fused_9band_nz701_vs_numpy():
    """Fused CUDA N_B=9 + Nz=701 matches NumPy (dynamic smem + N_B compile)."""
    from lephare.accel.cuda_template import (
        _FUSE_KERNEL_CACHE,
        _LIB_CACHE,
        process_template_tiles_cuda,
    )

    tensors, mags, magerrs = _synthetic_9band_nz701_tensors()
    _LIB_CACHE.clear()
    _FUSE_KERNEL_CACHE.clear()
    pdf_c, zgrid_c, _, _, ancil_c = process_template_tiles_cuda(mags, magerrs, tensors, return_ancil=True)
    pdf_n, zgrid_n = process_batch(mags, magerrs, tensors)
    assert zgrid_c.shape == (701,)
    assert np.allclose(zgrid_c, zgrid_n)
    assert np.allclose(pdf_c, pdf_n, rtol=1e-9, atol=1e-11)
    from lephare.accel.bestfit import bestfit_ancil_from_chi2
    from lephare.accel.kernels_ref import fit_chi2_single, prepare_photometry

    ab, sab, bus, busul, nbul = prepare_photometry(mags, magerrs, None, tensors)
    chi2s = [
        fit_chi2_single(ab[i], sab[i], bus[i], busul[i], int(nbul[i]), tensors)[0] for i in range(len(mags))
    ]
    ancil_n = bestfit_ancil_from_chi2(np.stack(chi2s), tensors)
    for key in ("Z_BEST", "CHI_BEST"):
        assert np.allclose(ancil_c[key], ancil_n[key], rtol=1e-8, atol=1e-8, equal_nan=True), key


def test_mag_abs_dm_window_matches_log10():
    """dm window reject matches C++ ``mag0 - 2.5 log10(dm)`` (interior points)."""
    from lephare.accel.cuda_template import _mag_abs_dm_bounds

    mag0 = np.array([-20.0, -18.0, -22.0, -19.0])
    z_lib = np.array([0.0, 0.5, 1.0, 0.0])
    types = np.array([0, 0, 1, 2], dtype=np.int32)
    prior = np.array([-24.0, -5.0, -30.0, -5.0])
    funz0 = 0.12
    dm_min, dm_max = _mag_abs_dm_bounds(mag0, z_lib, types, prior, funz0, np.float64)
    rng = np.random.default_rng(0)
    dm = np.exp(rng.uniform(np.log(1e-4), np.log(1e4), size=2000))
    mag0_eff = np.where(z_lib < 1e-10, mag0 - funz0, mag0)
    for i, typ in enumerate(types):
        if typ == 2:
            assert dm_min[i] == 0.0
            continue
        p0, p1 = (prior[0], prior[1]) if typ == 0 else (prior[2], prior[3])
        abs_mag = mag0_eff[i] - 2.5 * np.log10(dm)
        reject_log = (abs_mag <= p0) | (abs_mag >= p1)
        reject_dm = (dm <= dm_min[i]) | (dm >= dm_max[i])
        # Boundaries can differ by 1 ulp; interior must match.
        interior = (np.abs(abs_mag - p0) > 1e-10) & (np.abs(abs_mag - p1) > 1e-10)
        assert np.array_equal(reject_log[interior], reject_dm[interior])


def test_fp32_amp_scale_ignores_bright_outliers():
    from lephare.accel.fp32_numerics import (
        FP32_CLIP_ABS,
        FP32_FLUX_SCALE_CAP,
        fp32_amp_scale,
        fp32_apply_scale,
    )

    typical = np.full((100, 9), 1.0e-40)
    typical[0, 0] = 4000.0
    scale = fp32_amp_scale(typical)
    assert scale == FP32_FLUX_SCALE_CAP
    scaled = fp32_apply_scale(typical, scale)
    assert float(np.max(np.abs(scaled))) <= FP32_CLIP_ABS
    compact = np.full((8, 9), 1.0)
    assert fp32_amp_scale(compact) == 1.0e4


@pytest.mark.skipif(not cuda_available(), reason="cuda/cupy unavailable")
def test_cuda_tiled_objs_matches_block_per_object(monkeypatch):
    """Object-chunk SMEM tiles match the legacy one-block-per-object fused path."""
    from lephare.accel.cuda_template import (
        _FUSE_KERNEL_CACHE,
        _LIB_CACHE,
        process_template_tiles_cuda,
    )

    tensors, mags, magerrs = _synthetic_9band_nz701_tensors(n_obj=12)
    _LIB_CACHE.clear()
    _FUSE_KERNEL_CACHE.clear()
    monkeypatch.delenv("LEPHARE_ACCEL_CUDA_TILE_OBJS", raising=False)
    pdf_a, _, _, _, ancil_a = process_template_tiles_cuda(mags, magerrs, tensors, return_ancil=True)
    _LIB_CACHE.clear()
    _FUSE_KERNEL_CACHE.clear()
    monkeypatch.setenv("LEPHARE_ACCEL_CUDA_TILE_OBJS", "1")
    pdf_b, _, _, _, ancil_b = process_template_tiles_cuda(mags, magerrs, tensors, return_ancil=True)
    assert np.allclose(pdf_a, pdf_b, rtol=1e-12, atol=1e-14)
    for key in ("Z_BEST", "CHI_BEST", "CHI_QSO", "CHI_STAR"):
        assert np.allclose(ancil_a[key], ancil_b[key], rtol=1e-10, atol=1e-10, equal_nan=True), key


@pytest.mark.skipif(not (cuda_available() and jax_available()), reason="cuda+jax unavailable")
def test_jax_fp32_vs_cuda_fp32_no_empty_pdfs(monkeypatch):
    """FP32 CUDA/JAX on 9-band tensors: no empty PDFs; Z_BEST close (not bitwise)."""
    from lephare.accel.dispatch import process_with_backend

    tensors, mags, magerrs = _synthetic_9band_nz701_tensors(n_obj=8)
    # Helper fluxes are O(1); MAG_ABS + FP32 need AB-like fluxes (~1e-29).
    from lephare.accel.maglib import MagLibTensors

    tensors = MagLibTensors(
        **{
            **tensors.__dict__,
            "flux": np.asarray(tensors.flux, dtype=np.float64) * 1.0e-29,
            "prior_lib": np.array([-24.0, -5.0, -30.0, -10.0], dtype=np.float64),
        }
    )
    monkeypatch.setenv("LEPHARE_ACCEL_TILE_AXIS", "template")
    monkeypatch.setenv("LEPHARE_ACCEL_CUDA_TILE_OBJS", "1")
    cuda = process_with_backend("cuda_full", mags, magerrs, tensors, dtype=np.float32)
    jax_out = process_with_backend("jax_full", mags, magerrs, tensors, dtype=np.float32)
    pdf_c, pdf_j = np.asarray(cuda[0]), np.asarray(jax_out[0])
    assert pdf_c.shape == pdf_j.shape
    assert np.all(np.isfinite(pdf_c)) and np.all(np.isfinite(pdf_j))
    assert np.all(np.nansum(np.abs(pdf_c), axis=1) > 0)
    assert np.all(np.nansum(np.abs(pdf_j), axis=1) > 0)
    zc = np.asarray(cuda[4]["Z_BEST"] if len(cuda) == 5 else cuda[2])
    zj = np.asarray(jax_out[4]["Z_BEST"] if len(jax_out) == 5 else jax_out[2])
    assert np.allclose(zc, zj, rtol=0.0, atol=0.05, equal_nan=True)


def _synthetic_autoadapt_tensors(n_b: int = 9, n_z: int = 5, seed: int = 0):
    from lephare.accel.maglib import MagLibTensors

    rng = np.random.default_rng(seed)
    gridz = np.linspace(0.0, 0.4, n_z)
    n_gal, n_qso, n_star = 24, 6, 10
    blocks = []
    types = []
    z = []
    nummod = []
    age = []
    ebv = []
    mid = 0
    for zv in gridz:
        for m in range(n_gal):
            blocks.append(rng.uniform(1e-32, 1e-28, size=n_b))
            types.append(0)
            z.append(zv)
            nummod.append(m + 1)
            age.append(1e9 * (m + 1))
            ebv.append(0.05 * (m % 3))
            mid += 1
        for m in range(n_qso):
            blocks.append(rng.uniform(1e-32, 1e-28, size=n_b))
            types.append(1)
            z.append(zv)
            nummod.append(100 + m)
            age.append(0.0)
            ebv.append(0.0)
    for m in range(n_star):
        blocks.append(rng.uniform(1e-32, 1e-28, size=n_b))
        types.append(2)
        z.append(0.0)
        nummod.append(200 + m)
        age.append(0.0)
        ebv.append(0.0)
    flux = np.vstack(blocks).astype(np.float64)
    n_t = flux.shape[0]
    mag0 = rng.uniform(-23.0, -18.0, size=n_t)
    return MagLibTensors(
        flux=flux,
        z=np.asarray(z, dtype=np.float64),
        types=np.asarray(types, dtype=np.int32),
        mag0=mag0,
        gridz=gridz.astype(np.float64),
        prior_lib=np.array([-24.0, -5.0, -30.0, -10.0], dtype=np.float64),
        funz0=0.01,
        err_scale=np.full(n_b, 0.02, dtype=np.float64),
        err_factor=np.full(n_b, 1.0, dtype=np.float64),
        gbcont=0,
        contforb=-1,
        rm_discrepant=1e9,
        mabs_ref=2,
        n_bands=n_b,
        nummod=np.asarray(nummod, dtype=np.int32),
        age=np.asarray(age, dtype=np.float64),
        ebv=np.asarray(ebv, dtype=np.float64),
        z_interp=True,
    )


def test_cuda_autoadapt_vs_numpy():
    """CUDA Z-fixed AUTO_ADAPT matches the NumPy reference."""
    if not cuda_available():
        pytest.skip("CUDA backend not available")
    from lephare.accel.autoadapt import calculate_offsets
    from lephare.accel.cuda_autoadapt import calculate_offsets_cuda

    tensors = _synthetic_autoadapt_tensors(n_b=9)
    rng = np.random.default_rng(1)
    n_obj, n_b = 80, tensors.n_bands
    mags = rng.uniform(18.0, 22.0, size=(n_obj, n_b))
    magerrs = rng.uniform(0.02, 0.08, size=(n_obj, n_b))
    zs = rng.uniform(0.05, 0.35, size=n_obj)
    cfg = {
        "AUTO_ADAPT": "YES",
        "ADAPT_BAND": "1",
        "ADAPT_LIM": "15,35",
        "ADAPT_ZBIN": "0.001,6",
        "APPLY_SYSSHIFT": "0.",
    }
    a0_np = calculate_offsets(mags, magerrs, zs, tensors, cfg, verbose=False)
    a0_cu = calculate_offsets_cuda(mags, magerrs, zs, tensors, cfg, verbose=False)
    assert np.allclose(a0_cu, a0_np, rtol=1e-12, atol=1e-12), (
        f"a0 cuda={a0_cu} numpy={a0_np} maxabs={np.max(np.abs(a0_cu - a0_np))}"
    )


# Dense-library checks need a pre-built fine-z LSST run (Nt ~ 210k templates).
# Point LEPHARE_ACCEL_DENSE_RUN_DIR at it, e.g. a LEPHAREWORK made with Z_STEP=0.0357,0,5.
DENSE_RUN_DIR = os.environ.get("LEPHARE_ACCEL_DENSE_RUN_DIR", "")


def _dense_tensors():
    from lephare.accel.maglib import MagLibTensors

    cache = os.path.join(DENSE_RUN_DIR, "maglib_tensors_types_only.npz")
    if os.path.isfile(cache):
        return load_maglib(cache)
    old_work = os.environ.get("LEPHAREWORK")
    set_lephare_env(lephare_work=DENSE_RUN_DIR)
    try:
        cfg = lp.keymap_to_string_dict(lp.read_config(os.path.join(EXAMPLES, "lsst_accel.para")))
        cfg["Z_STEP"] = "0.03571428571428571,0.0,5.0"
        photz = lp.PhotoZ(lp.all_types_to_keymap(cfg))
        n_t = len(photz.fullLib)
        if n_t < 100_000:
            pytest.skip(f"expected dense library, got Nt={n_t}")
        n_b = int(photz.imagm)
        # Priors off -> mag0 unused; keeps export fast and matches the speed-bench config.
        tensors = MagLibTensors(
            flux=np.asarray(photz.flux, dtype=np.float64),
            z=np.asarray(photz.zLib, dtype=np.float64),
            types=np.array(
                [0 if s.is_gal() else (1 if s.is_qso() else 2) for s in photz.fullLib], dtype=np.int32
            ),
            mag0=np.zeros(n_t, dtype=np.float64),
            gridz=np.asarray(photz.gridz, dtype=np.float64),
            prior_lib=np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float64),
            funz0=0.0,
            err_scale=np.full(n_b, 0.02, dtype=np.float64),
            err_factor=np.full(n_b, 1.5, dtype=np.float64),
            gbcont=0,
            contforb=-1,
            rm_discrepant=1e9,
            mabs_ref=0,
            n_bands=n_b,
            nummod=np.arange(n_t, dtype=np.int32),
            z_interp=True,
        )
    finally:
        set_lephare_env(lephare_work=old_work)
    save_maglib(tensors, cache)
    return tensors


def _check_dense_against_numpy(run_backend):
    from lephare.accel.bestfit import bestfit_ancil_from_chi2
    from lephare.accel.kernels_ref import fit_chi2_single, prepare_photometry

    tensors = _dense_tensors()
    mags, magerrs, _ = read_catalog("test")
    mags, magerrs = mags[:12], magerrs[:12]
    pdf_d, _, _, _, ancil_d = run_backend(mags, magerrs, tensors)
    pdf_n, _ = process_batch(mags, magerrs, tensors)
    assert np.allclose(pdf_d, pdf_n, rtol=1e-9, atol=1e-11)

    ab, sab, bus, busul, nbul = prepare_photometry(mags, magerrs, None, tensors)
    chi2s = [
        fit_chi2_single(ab[i], sab[i], bus[i], busul[i], int(nbul[i]), tensors)[0] for i in range(len(mags))
    ]
    ancil_n = bestfit_ancil_from_chi2(np.stack(chi2s), tensors)
    for key in ("Z_BEST", "CHI_BEST", "ZQ_BEST", "CHI_QSO", "CHI_STAR"):
        assert np.allclose(ancil_d[key], ancil_n[key], rtol=1e-8, atol=1e-8, equal_nan=True), key


@pytest.mark.slow
@pytest.mark.skipif(not DENSE_RUN_DIR, reason="LEPHARE_ACCEL_DENSE_RUN_DIR not set")
@pytest.mark.skipif(not jax_available(), reason="jax not installed")
def test_jax_dense_maglib_template_tiles_vs_numpy(monkeypatch):
    """Dense Nt~210k: template-tile JAX vs NumPy on a small object batch."""
    from lephare.accel.jax_backend import _FIT_CACHE, process_batch_jax

    monkeypatch.setenv("LEPHARE_ACCEL_TILE_AXIS", "template")
    _FIT_CACHE.clear()
    _check_dense_against_numpy(lambda m, e, t: process_batch_jax(m, e, t, return_ancil=True))


@pytest.mark.slow
@pytest.mark.skipif(not DENSE_RUN_DIR, reason="LEPHARE_ACCEL_DENSE_RUN_DIR not set")
@pytest.mark.skipif(not cuda_available(), reason="cuda/cupy unavailable")
def test_cuda_fused_template_tiles_vs_numpy(monkeypatch):
    """Dense fused CUDA template path matches NumPy BAY_ZG (FP64)."""
    from lephare.accel.cuda_template import _FUSE_KERNEL_CACHE, _LIB_CACHE, process_template_tiles_cuda

    monkeypatch.setenv("LEPHARE_ACCEL_TILE_AXIS", "template")
    _LIB_CACHE.clear()
    _FUSE_KERNEL_CACHE.clear()
    _check_dense_against_numpy(lambda m, e, t: process_template_tiles_cuda(m, e, t, return_ancil=True))


def _bc03_scale_tensors(n_obj: int):
    """Synthetic maglib with BC03-like fluxes (~1e-40, below FP32 normal range)."""
    from lephare.accel.maglib import MagLibTensors

    tensors, mags, magerrs = _synthetic_9band_nz701_tensors(n_obj=n_obj)
    tensors = MagLibTensors(
        **{**tensors.__dict__, "flux": np.asarray(tensors.flux, dtype=np.float64) * 1.0e-40}
    )
    return tensors, mags, magerrs


@pytest.mark.skipif(not cuda_available(), reason="cuda/cupy unavailable")
def test_cuda_fp32_small_batch_fits_bc03_scale_fluxes(monkeypatch):
    """Small FP32 batches (object-axis sized) must still scale fluxes into FP32 range."""
    from lephare.accel.dispatch import process_with_backend

    monkeypatch.delenv("LEPHARE_ACCEL_TILE_AXIS", raising=False)
    tensors, mags, magerrs = _bc03_scale_tensors(n_obj=4)
    ref = process_with_backend("cuda_hybrid", mags, magerrs, tensors, dtype=np.float64)
    got = process_with_backend("cuda_hybrid", mags, magerrs, tensors, dtype=np.float32)
    assert np.all(np.asarray(got[4]["CHI_BEST"]) < 1e8)
    assert np.all(np.nansum(np.asarray(got[0]), axis=1) > 0)
    np.testing.assert_allclose(got[4]["Z_BEST"], ref[4]["Z_BEST"], atol=0.05)
    np.testing.assert_allclose(got[4]["CHI_BEST"], ref[4]["CHI_BEST"], rtol=1e-3, atol=0.05)


@pytest.mark.skipif(not jax_available(), reason="jax not installed")
@pytest.mark.parametrize("axis", ["object", "template"])
def test_jax_fp32_chi2_full_precision(monkeypatch, axis):
    """JAX FP32 best-fit χ² on BC03-scale fluxes: the object path must scale its ancil
    photometry, and GEMMs must not run as TF32 (base - avmago²/avmagt cancels)."""
    from lephare.accel.dispatch import process_with_backend

    monkeypatch.setenv("LEPHARE_ACCEL_TILE_AXIS", axis)
    tensors, mags, magerrs = _bc03_scale_tensors(n_obj=16)
    magerrs = np.full_like(magerrs, 0.002)  # high S/N: chi2 ~ 1e3-1e4, cancellation-heavy
    ref = process_with_backend("jax_hybrid", mags, magerrs, tensors, dtype=np.float64)
    got = process_with_backend("jax_hybrid", mags, magerrs, tensors, dtype=np.float32)
    c64 = np.asarray(ref[4]["CHI_BEST"], dtype=np.float64)
    c32 = np.asarray(got[4]["CHI_BEST"], dtype=np.float64)
    np.testing.assert_allclose(c32, c64, rtol=1e-4, atol=0.05)
