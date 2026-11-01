"""GPU batch sizing knobs, out-of-memory backoff and the vectorized best-fit helpers."""

import numpy as np
import pytest

from lephare.accel import gpu_util
from lephare.accel.bestfit import HIGH_CHI2, improve_extremum_chi2, improve_extremum_chi2_rows


@pytest.fixture
def free_8gb(monkeypatch):
    monkeypatch.setattr(gpu_util, "_gpu_free_bytes", lambda: 8 * 1024**3)
    for k in ("LEPHARE_ACCEL_CHI2_MEM_FRACTION", "LEPHARE_ACCEL_CHI2_MAX_CHUNK", "LEPHARE_ACCEL_CHI2_CHUNK"):
        monkeypatch.delenv(k, raising=False)


def test_obj_chunk_defaults_fill_vram_uncapped(free_8gb):
    per_obj = 1024
    assert gpu_util.suggest_obj_chunk(per_obj) == int(8 * 1024**3 * 0.8) // per_obj
    # the old 100k cap is gone by default
    assert gpu_util.suggest_chi2_chunk_size(10, dtype=np.float32) > 100_000


def test_obj_chunk_env_overrides(free_8gb, monkeypatch):
    monkeypatch.setenv("LEPHARE_ACCEL_CHI2_MEM_FRACTION", "0.25")
    assert gpu_util.suggest_obj_chunk(1024) == int(8 * 1024**3 * 0.25) // 1024
    monkeypatch.setenv("LEPHARE_ACCEL_CHI2_MAX_CHUNK", "5000")
    assert gpu_util.suggest_obj_chunk(1024) == 5000
    monkeypatch.setenv("LEPHARE_ACCEL_CHI2_MAX_CHUNK", "0")
    assert gpu_util.suggest_obj_chunk(1024) == int(8 * 1024**3 * 0.25) // 1024
    monkeypatch.setenv("LEPHARE_ACCEL_CHI2_CHUNK", "777")
    assert gpu_util.suggest_chi2_chunk_size(10) == 777


class OutOfMemoryError(Exception):
    pass


def test_oom_backoff_halves_and_covers_every_row(monkeypatch):
    monkeypatch.setattr(gpu_util, "free_gpu_pools", lambda: None)
    seen = []

    def fn(s, e):
        if e - s > 300:
            raise OutOfMemoryError("out of memory allocating")
        seen.append((s, e))

    chunk = gpu_util.run_with_oom_backoff(fn, 1000, 1000, min_chunk=64, tag="t")
    assert chunk == 250
    covered = np.zeros(1000, dtype=int)
    for s, e in seen:
        covered[s:e] += 1
    assert np.all(covered == 1)
    assert gpu_util.STATS["t"] == {"chunk": 250, "oom_retries": 2, "n_obj": 1000}


def test_oom_backoff_reraises_other_errors_and_floor(monkeypatch):
    monkeypatch.setattr(gpu_util, "free_gpu_pools", lambda: None)

    def bad(s, e):
        raise ValueError("not memory")

    with pytest.raises(ValueError):
        gpu_util.run_with_oom_backoff(bad, 10, 10)

    def always_oom(s, e):
        raise OutOfMemoryError("out of memory")

    with pytest.raises(OutOfMemoryError):
        gpu_util.run_with_oom_backoff(always_oom, 1000, 1000, min_chunk=100)


def test_improve_extremum_rows_bit_identical_to_scalar():
    rng = np.random.default_rng(1)
    x = np.linspace(0.0, 3.0, 301)
    c = rng.normal(50.0, 10.0, (3000, 301))
    c[:100] = HIGH_CHI2  # nothing valid
    c[100:200, 0] = -1.0  # minimum at the first node
    c[200:300, -1] = -1.0  # minimum at the last node
    c[300:400, 100] = -1.0
    c[300:400, 101] = HIGH_CHI2  # neighbour at HIGH_CHI2
    c[400:450] = 7.0  # flat row
    c[450:500, 150] = np.nan
    c[500:550, 150:153] = 1.0  # aligned points
    xs, ys = improve_extremum_chi2_rows(x, c)
    ref = np.array([improve_extremum_chi2(x, row) for row in c])
    np.testing.assert_array_equal(xs, ref[:, 0])
    np.testing.assert_array_equal(ys, ref[:, 1])
    # coarse grid: steps > 0.5 skip the parabola
    xg = np.array([0.0, 0.2, 0.9, 1.0, 1.1])
    cg = rng.normal(5.0, 1.0, (500, 5))
    xs, ys = improve_extremum_chi2_rows(xg, cg)
    ref = np.array([improve_extremum_chi2(xg, row) for row in cg])
    np.testing.assert_array_equal(xs, ref[:, 0])
    np.testing.assert_array_equal(ys, ref[:, 1])
