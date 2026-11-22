"""RidgeCO matches GeneSPIDER: ridge of P on Y, then a hard cutoff.

Self-regression of Y on Y is a symmetric Gram system, so every off-diagonal
link that survives a cutoff has its reverse. RidgeCO must not do that.
"""

import numpy as np

from src.datastruct.Dataset import Dataset
from src.methods.lsco import LSCO
from src.methods.ridge import RidgeCO


def _knockdown(n: int = 8, reps: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    true = rng.normal(scale=0.4, size=(n, n))
    true[np.abs(true) < 0.55] = 0.0
    np.fill_diagonal(true, -1.2)
    samples = n * reps
    design = np.zeros((n, samples))
    for gene in range(n):
        design[gene, gene * reps : (gene + 1) * reps] = -1.0
    response = -np.linalg.inv(true) @ design
    response = response + rng.normal(scale=0.02, size=response.shape)
    dataset = Dataset()
    dataset._Y = response
    dataset._P = design
    return dataset


def _reverse_fraction(weights: np.ndarray) -> float:
    off = np.array(weights, dtype=float, copy=True)
    np.fill_diagonal(off, 0.0)
    present = off != 0
    directed = int(present.sum())
    if directed == 0:
        return 0.0
    return float((present & present.T).sum()) / directed


def test_ridgeco_cutoff_is_not_a_symmetric_self_regression():
    dataset = _knockdown()
    zeta = np.array([0.05, 0.2, 0.8])
    stack, used = RidgeCO(dataset, threshold_range=zeta)

    assert stack.shape == (8, 8, 3)
    assert np.allclose(used, zeta)
    # A stricter cutoff never restores a link.
    present = np.abs(stack) > 0
    assert present[:, :, 1].sum() <= present[:, :, 0].sum()
    assert present[:, :, 2].sum() <= present[:, :, 1].sum()

    dense = stack[:, :, 0]
    assert np.median(np.diag(dense)) < 0
    # Same sign convention as LSCO (A @ Y ≈ -P), not a symmetric self-fit.
    als, _ = LSCO(dataset)
    assert np.corrcoef(dense.ravel(), np.asarray(als).ravel())[0, 1] > 0.5
    assert _reverse_fraction(dense) < 1.0
    # The strict slice still has links, and far fewer of them are reciprocal
    # than a symmetric solution (which stays at 1).
    strict = stack[:, :, 1]
    assert int(np.count_nonzero(strict) - np.count_nonzero(np.diag(strict))) >= 4
    assert _reverse_fraction(strict) < 0.75
