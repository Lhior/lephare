"""Fixtures for ``lephare.accel`` parity tests.

Tests that compare against the C++ library need the LSST filters and COSMOS
templates from the lephare-data repository. These are fetched once into
``LEPHARE_ACCEL_DATA_DIR`` (default: the user cache used by ``lephare``), not
into ``tests/data``. The tests are skipped when the files cannot be fetched.
"""

import os

import pytest
from accel_helpers import EXAMPLES, GAL_CONFIG, QSO_CONFIG, STAR_CONFIG, set_lephare_env
from platformdirs import user_cache_dir

import lephare as lp


@pytest.fixture(scope="module")
def lsst_env(tmp_path_factory):
    """LSST config with auxiliary data available; restores the lephare env afterwards."""
    old_dir, old_work = os.environ.get("LEPHAREDIR"), os.environ.get("LEPHAREWORK")
    data_dir = os.environ.get("LEPHARE_ACCEL_DATA_DIR") or os.path.join(user_cache_dir("lephare"), "data")
    work_dir = str(tmp_path_factory.mktemp("accel_work"))
    os.makedirs(data_dir, exist_ok=True)
    set_lephare_env(data_dir, work_dir)
    keymap = lp.read_config(os.path.join(EXAMPLES, "lsst_accel.para"))
    try:
        lp.data_retrieval.get_auxiliary_data(
            lephare_dir=data_dir, keymap=keymap, additional_files=["examples/output.para"]
        )
    except Exception as exc:  # noqa: BLE001 - any download failure means "no data"
        set_lephare_env(old_dir, old_work)
        pytest.skip(f"LePHARE auxiliary data unavailable: {exc}")
    cfg = lp.keymap_to_string_dict(keymap)
    # Sparse grid used by the rail_lephare parity fixture (zmin=0, zmax=5, 6 bins).
    cfg["Z_STEP"] = "1.0,0.0,5.0"
    # Output keys matching this C++ version (lephare-data's output.para can be newer).
    cfg["PARA_OUT"] = os.path.join(EXAMPLES, "output.para")
    yield cfg
    set_lephare_env(old_dir, old_work)


@pytest.fixture(scope="module")
def lsst_library(lsst_env):
    """Run classic ``lp.prepare`` once and export the dense template tensors."""
    from lephare.accel.maglib import export_maglib_from_photoz

    lp.prepare(lsst_env, star_config=STAR_CONFIG, gal_config=GAL_CONFIG, qso_config=QSO_CONFIG)
    photz = lp.PhotoZ(lp.all_types_to_keymap(lsst_env))
    return export_maglib_from_photoz(photz, lsst_env)
