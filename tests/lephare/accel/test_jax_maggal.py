"""JAX MagGal FP64 parity vs C++ ``GalMag.make_maglib`` / ``QSOMag.make_maglib``."""

from __future__ import annotations

import os

import numpy as np
import pytest
from accel_helpers import set_lephare_env

import lephare as lp
from lephare.accel.dispatch import jax_available
from lephare.accel.jax_maggal import (
    ensure_filt_and_sedtolib,
    maggal_one_gal,
    maggal_one_qso,
    maggal_setup_from_config,
)
from lephare.accel.mag_conv import iter_sed_bin
from lephare.accel.mag_physics import ab_mag_to_flux

pytestmark = pytest.mark.skipif(not jax_available(), reason="jax not installed")


def _cfg(lsst_env: dict) -> dict:
    cfg = dict(lsst_env)
    cfg["VERBOSE"] = "NO"
    cfg["LIB_ASCII"] = "NO"
    return cfg


def _prep_libbin(cfg: dict, tmp_path):
    run = str(tmp_path / "jax_maggal_work")
    os.makedirs(run, exist_ok=True)
    set_lephare_env(lephare_work=run)
    ensure_filt_and_sedtolib(cfg)
    return run


def _galmag(cfg: dict):
    km = lp.all_types_to_keymap(cfg)
    km["t"] = lp.keyword("t", "GAL")
    km["c"] = lp.keyword("c", "")
    mag = lp.GalMag(km)
    return mag


def _qsomag(cfg: dict):
    km = lp.all_types_to_keymap(cfg)
    km["t"] = lp.keyword("t", "QSO")
    km["c"] = lp.keyword("c", "")
    mag = lp.QSOMag(km)
    return mag


def _first_galsed(cfg: dict):
    name = str(cfg.get("GAL_LIB_IN", cfg.get("GAL_LIB", "LSST_GAL_BIN"))).split()[0]
    path = os.path.join(os.environ["LEPHAREWORK"], "lib_bin", f"{name}.bin")
    sed = lp.GalSED("first")
    sed.readSEDBin(path)
    return sed, path


def _first_qsosed(cfg: dict):
    name = str(cfg.get("QSO_LIB_IN", cfg.get("QSO_LIB", "LSST_QSO_BIN"))).split()[0]
    path = os.path.join(os.environ["LEPHAREWORK"], "lib_bin", f"{name}.bin")
    sed = lp.QSOSED("first")
    sed.readSEDBin(path)
    return sed, path


def _cpp_mags(seds, sum_lines: bool) -> np.ndarray:
    rows = []
    for s in seds:
        if sum_lines:
            s.sumEmLines()
        rows.append(np.asarray(s.mag, dtype=np.float64))
    return np.vstack(rows)


def test_jax_maggal_continuum_matches_cpp(lsst_env, tmp_path):
    cfg = _cfg(lsst_env)
    cfg["EM_LINES"] = "NO"
    cfg["EB_V"] = "0.,0.1"
    cfg["EXTINC_LAW"] = "SMC_prevot.dat"
    cfg["MOD_EXTINC"] = "0,1000"
    cfg["Z_STEP"] = "0.5,0.,1.5"
    _prep_libbin(cfg, tmp_path)

    cpp_sed, path = _first_galsed(cfg)
    gmag = _galmag(cfg)
    cpp = gmag.make_maglib(cpp_sed)
    cpp_mag = _cpp_mags(cpp, sum_lines=False)

    spec = next(iter_sed_bin(path, kind="gal"))
    setup = maggal_setup_from_config(cfg)
    out = maggal_one_gal(spec, setup, integrator="cpp", em_lines=False)
    assert out["mag"].shape[0] == cpp_mag.shape[0]
    assert np.allclose(out["mag"], cpp_mag, rtol=1e-8, atol=1e-10)

    jax_out = maggal_one_gal(spec, setup, integrator="jax", em_lines=False)
    assert jax_out["mag"].shape == out["mag"].shape
    assert np.allclose(jax_out["mag"], cpp_mag, rtol=1e-8, atol=1e-10)


def test_jax_maggal_igm_highz_and_qso(lsst_env, tmp_path):
    cfg = _cfg(lsst_env)
    cfg["EM_LINES"] = "NO"
    cfg["EB_V"] = "0."
    cfg["EXTINC_LAW"] = "SMC_prevot.dat"
    cfg["MOD_EXTINC"] = "0,0"
    cfg["Z_STEP"] = "1.5,0.,3."
    _prep_libbin(cfg, tmp_path)

    cpp_sed, path = _first_galsed(cfg)
    cpp_mag = _cpp_mags(_galmag(cfg).make_maglib(cpp_sed), sum_lines=False)
    spec = next(iter_sed_bin(path, kind="gal"))
    setup = maggal_setup_from_config(cfg)
    out = maggal_one_gal(spec, setup, integrator="cpp", em_lines=False)
    assert out["z"].max() >= 2.9
    assert np.allclose(out["mag"], cpp_mag, rtol=1e-8, atol=1e-10)

    qsed, qpath = _first_qsosed(cfg)
    qcpp = _cpp_mags(_qsomag(cfg).make_maglib(qsed), sum_lines=False)
    qspec = next(iter_sed_bin(qpath, kind="qso"))
    qout = maggal_one_qso(qspec, setup, integrator="cpp")
    assert qout["mag"].shape[0] == qcpp.shape[0]
    assert np.allclose(qout["mag"], qcpp, rtol=1e-8, atol=1e-10)


def test_jax_maggal_emlines_emp_uv(lsst_env, tmp_path):
    cfg = _cfg(lsst_env)
    cfg["EM_LINES"] = "EMP_UV"
    cfg["EM_DISPERSION"] = "1."
    cfg["EB_V"] = "0.,0.1"
    cfg["EXTINC_LAW"] = "SMC_prevot.dat"
    cfg["MOD_EXTINC"] = "0,1000"
    cfg["Z_STEP"] = "0.5,0.,1.0"
    _prep_libbin(cfg, tmp_path)

    cpp_sed, path = _first_galsed(cfg)
    cpp_list = _galmag(cfg).make_maglib(cpp_sed)
    cpp_mag = _cpp_mags(cpp_list, sum_lines=True)
    spec = next(iter_sed_bin(path, kind="gal"))
    setup = maggal_setup_from_config(cfg)
    out = maggal_one_gal(spec, setup, integrator="cpp", em_lines=True)
    assert out["mag"].shape[0] == cpp_mag.shape[0]
    assert np.allclose(out["mag"], cpp_mag, rtol=1e-8, atol=1e-10)
    jax_out = maggal_one_gal(spec, setup, integrator="jax", em_lines=True)
    assert np.allclose(jax_out["mag"], cpp_mag, rtol=1e-8, atol=1e-10)
    flux = ab_mag_to_flux(out["mag"])
    assert np.all(np.isfinite(flux))


@pytest.mark.slow
def test_jax_maggal_gpu_batch_occupancy(lsst_env, tmp_path):
    """One ``(N_combo, N_z)`` kernel should keep the GPU busy vs tiny per-z launches."""
    import shutil
    import subprocess
    import time

    import jax
    import jax.numpy as jnp

    cfg = _cfg(lsst_env)
    cfg["EM_LINES"] = "NO"
    cfg["EB_V"] = "0."
    cfg["EXTINC_LAW"] = "SMC_prevot.dat"
    cfg["MOD_EXTINC"] = "0,0"
    cfg["Z_STEP"] = "0.2,0.,2."
    _prep_libbin(cfg, tmp_path)
    from lephare.accel.jax_maggal import jax_continuum_mags_grid, maggal_setup_from_config

    setup = maggal_setup_from_config(cfg)
    path = os.path.join(
        os.environ["LEPHAREWORK"],
        "lib_bin",
        f"{str(cfg.get('GAL_LIB_IN', cfg.get('GAL_LIB'))).split()[0]}.bin",
    )
    specs = list(iter_sed_bin(path, kind="gal"))[:8]
    rest_l = [s.lamb for s in specs for _ in range(16)]
    rest_f = [s.flux for s in specs for _ in range(4)]
    from lephare.accel.jax_maggal import _pad_spectra

    pad_l, pad_f = _pad_spectra(rest_l, rest_f)
    n_c, n_z = pad_l.shape[0], int(setup.gridz.size)
    age_ok = np.ones((n_c, n_z), dtype=bool)
    jax_continuum_mags_grid(pad_l[:2], pad_f[:2], age_ok[:2], setup)
    jax.block_until_ready(jnp.zeros(1))

    smi = shutil.which("nvidia-smi")
    proc = None
    log = tmp_path / "gpu_util.csv"
    if smi:
        proc = subprocess.Popen(
            [smi, "dmon", "-s", "u", "-d", "1", "-c", "12"],
            stdout=log.open("w"),
            stderr=subprocess.STDOUT,
        )
    t_end = time.perf_counter() + 8.0
    mags = None
    while time.perf_counter() < t_end:
        mags = jax_continuum_mags_grid(pad_l, pad_f, age_ok, setup)
    jax.block_until_ready(jnp.asarray(mags))
    elapsed = 8.0
    if proc is not None:
        proc.wait(timeout=20)
    assert mags.shape == (n_c, n_z, len(setup.filters))
    assert elapsed > 0
    if smi and log.is_file():
        text = log.read_text()
        utils = []
        for line in text.splitlines():
            if not line.strip() or line.strip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3 and parts[1].replace(".", "", 1).isdigit():
                try:
                    utils.append(float(parts[1]))
                except ValueError:
                    continue
        if utils:
            peak = max(utils)
            assert peak >= 10.0, (
                f"GPU SM util peak {peak}% too low (was ~2% with tiny launches); dmon log:\n{text}"
            )


def test_iter_sed_bin_reads_every_gal_record(lsst_env, tmp_path):
    """The Python lib_bin reader walks every GAL record (11 trailing doubles since LePHARE 1.1)."""
    cfg = _cfg(lsst_env)
    _prep_libbin(cfg, tmp_path)
    _, path = _first_galsed(cfg)
    specs = list(iter_sed_bin(path, kind="gal"))
    # sedtolib writes one .phys row per record: age luv lopt ... (6 significant digits)
    phys = np.loadtxt(path.replace(".bin", ".phys"), comments="#", ndmin=2)
    assert len(specs) == phys.shape[0] > 1
    assert np.allclose([sp.age for sp in specs], phys[:, 0], rtol=1e-5)
    assert np.allclose([sp.luv for sp in specs], phys[:, 1], rtol=1e-5, atol=1e-4)
