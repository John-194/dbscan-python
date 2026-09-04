"""Boundary and argument-validation tests for the dbscan.DBSCAN C extension."""

import numpy as np
import pytest

from dbscan import DBSCAN


@pytest.mark.parametrize("sequential", [True, False])
def test_empty_input_returns_empty_arrays(sequential):
    import dbscan
    if hasattr(dbscan, "set_sequential"):
        dbscan.set_sequential(sequential)
    elif sequential:
        pytest.skip("no sequential mode in this build")
    labels, core = DBSCAN(np.zeros((0, 3)), eps=0.3, min_samples=5)
    assert labels.shape == (0,)
    assert core.shape == (0,)


def test_transposed_float32_input():
    """Production layout: float32, F-contiguous, built by transposing three columns."""
    p = np.random.default_rng(0).normal(size=(500, 3)).astype(np.float32)
    X = np.asarray((p[:, 0], p[:, 1], p[:, 2])).T
    labels, core = DBSCAN(X, eps=0.3, min_samples=5)
    assert labels.shape == (500,)
    assert core.shape == (500,)
    ref_labels, ref_core = DBSCAN(np.ascontiguousarray(X, np.float64), eps=0.3, min_samples=5)
    assert np.array_equal(labels, ref_labels)
    assert np.array_equal(core, ref_core)


@pytest.mark.parametrize("eps", [0.0, -1.0, float("nan"), float("inf")])
def test_bad_eps_raises(eps):
    with pytest.raises(ValueError):
        DBSCAN(np.zeros((10, 3)), eps=eps, min_samples=5)


@pytest.mark.parametrize("min_samples", [0, -3])
def test_bad_min_samples_raises(min_samples):
    with pytest.raises(ValueError):
        DBSCAN(np.zeros((10, 3)), eps=0.3, min_samples=min_samples)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_coordinate_raises(bad):
    X = np.zeros((10, 3))
    X[7, 1] = bad
    with pytest.raises(ValueError):
        DBSCAN(X, eps=0.3, min_samples=5)
