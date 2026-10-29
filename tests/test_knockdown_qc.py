"""Knockdown design quality control."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from bio.knockdown import drop_failed_knockdowns, fill_dropouts, knockdown_z, scale_genes


def _design(n_genes: int, reps: int = 2) -> np.ndarray:
    return -np.repeat(np.eye(n_genes), reps, axis=1)


def test_fill_dropouts_uses_replicate_then_gene_median():
    P = _design(3)
    Y = np.array(
        [
            [-2.0, -2.2, 0.1, 0.3, 0.2, 0.0],
            [0.5, -15.0, -1.0, -1.4, 0.4, 0.6],
            [0.1, 0.2, 0.3, 0.1, -14.0, -16.0],
        ]
    )
    filled, mask = fill_dropouts(Y, P, below=-10)
    assert mask.sum() == 3
    assert filled[1, 1] == 0.5
    assert np.isclose(filled[2, 4], np.median([0.1, 0.2, 0.3, 0.1]))
    assert np.isclose(filled[2, 5], filled[2, 4])
    assert np.array_equal(filled[~mask], Y[~mask])


def test_fill_dropouts_noop():
    Y = np.ones((2, 4))
    filled, mask = fill_dropouts(Y, _design(2), below=-10)
    assert not mask.any()
    assert np.array_equal(filled, Y)


def test_knockdown_z_sign_and_failed_drop():
    rng = np.random.default_rng(0)
    n = 6
    P = _design(n)
    Y = rng.normal(0, 0.3, (n, 2 * n))
    Y[P != 0] = -3.0
    Y[4, P[4] != 0] = 0.5
    z = knockdown_z(Y, P)
    assert np.all(z[[0, 1, 2, 3, 5]] < -5)
    assert z[4] > 0

    Y2, P2, failed = drop_failed_knockdowns(Y, P, max_z=0.0)
    assert failed.tolist() == [False, False, False, False, True, False]
    assert Y2.shape == (n, 2 * n - 2)
    assert not np.any(P2[4])
    assert np.array_equal(Y2, Y[:, np.any(P[[0, 1, 2, 3, 5]] != 0, axis=0)])


def test_scale_genes_unit_sd_without_centering():
    Y = np.array([[1.0, 3.0, 5.0], [0.0, 0.0, 0.0]])
    scaled = scale_genes(Y)
    assert np.isclose(scaled[0].std(), 1.0)
    assert np.allclose(scaled[0] / Y[0], scaled[0, 0] / Y[0, 0])
    assert np.array_equal(scaled[1], Y[1])
