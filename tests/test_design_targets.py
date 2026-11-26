import numpy as np
import pytest
import sparselink.methods  # noqa: F401
from sparselink import get_method

from bio.io import design_targets


def _knockdown_data(n: int = 6, replicates: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = np.where(rng.random((n, n)) < 0.3, rng.normal(0, 0.3, (n, n)), 0.0)
    np.fill_diagonal(A, -1.0)
    P = -np.tile(np.eye(n), replicates)
    Y = -np.linalg.solve(A, P)
    return A, Y, P


@pytest.mark.parametrize(
    ("method", "kwargs", "tol"),
    [
        ("lsco", {}, 1e-8),
        ("ridge", {"alpha": 1e-10}, 1e-6),
        ("lasso", {"alpha": 1e-8}, 1e-3),
        ("elastic_net", {"alpha": 1e-8}, 1e-3),
    ],
)
def test_design_methods_fit_A_Y_equals_minus_P(method, kwargs, tol):
    A, Y, P = _knockdown_data()
    fit = get_method(method)(**kwargs).fit(Y.T, design_targets(method, P.T))
    np.testing.assert_allclose(fit.adjacency_matrix, A, atol=tol)


def test_methods_without_a_design_get_none():
    _, _, P = _knockdown_data()
    assert design_targets("genie3", P.T) is None
    assert design_targets("lasso", None) is None


@pytest.mark.parametrize(("method", "l1_ratio"), [("lasso", 1.0), ("elastic_net", 0.5)])
def test_penalty_path_matches_one_fit_per_penalty(method, l1_ratio):
    from methods.lasso import penalty_path

    A, Y, P = _knockdown_data()
    Y = Y + np.random.default_rng(1).normal(0, 0.05, Y.shape)
    targets = design_targets(method, P.T)
    alphas = np.array([0.01, 0.1, 0.001])
    stack = penalty_path(Y.T, targets, alphas, l1_ratio)
    for k, alpha in enumerate(alphas):
        single = get_method(method)(alpha=alpha).fit(Y.T, targets).adjacency_matrix
        np.testing.assert_allclose(stack[:, :, k], single, atol=1e-4)
