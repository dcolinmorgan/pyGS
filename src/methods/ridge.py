"""Ridge regression with cutoff (GeneSPIDER ridgeco)."""

import numpy as np
from numpy.linalg import LinAlgError

from datastruct.Dataset import Dataset


def _expression_and_design(dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(dataset, "Y"):
        Y, P = dataset.Y, dataset.P
    elif hasattr(dataset, "data") and dataset.data is not None:
        Y, P = dataset.data.Y, dataset.data.P
    else:
        raise ValueError("Dataset must contain Y and P matrices")

    if Y is None or P is None:
        raise ValueError("Dataset must contain Y and P matrices")
    Y = np.asarray(Y, dtype=float)
    P = np.asarray(P, dtype=float)
    if Y.shape != P.shape:
        raise ValueError("Y and P must have the same shape")
    return Y, P


def _ridge_path(
    Y: np.ndarray, P: np.ndarray, penalties: np.ndarray
) -> list[np.ndarray]:
    """MATLAB ``ridge(..., scaled=1)`` of every row of P on Y, for each penalty.

    Predictors are centered and scaled to sample standard deviation 1, and the
    coefficients are mapped back to that original scale. They are negated so
    ``A @ Y`` approximates ``-P``, the same sign convention as LSCO.
    """
    predictors = Y.T
    targets = P.T
    centered_x = predictors - predictors.mean(axis=0)
    centered_y = targets - targets.mean(axis=0)
    scale = centered_x.std(axis=0, ddof=1)
    scale = np.where((scale == 0.0) | ~np.isfinite(scale), 1.0, scale)
    scaled = centered_x / scale
    left, singular, right = np.linalg.svd(scaled, full_matrices=False)
    projected = left.T @ centered_y
    networks: list[np.ndarray] = []
    for penalty in penalties:
        shrink = singular / (singular * singular + float(penalty))
        coef_scaled = right.T @ (shrink[:, None] * projected)
        coef = coef_scaled / scale[:, None]
        networks.append(-coef.T)
    return networks


def _reconstruction_r2(weights: np.ndarray, Y: np.ndarray, P: np.ndarray) -> float:
    """R² of Y against ``-A⁺ P``, the score ridgeco maximises over penalties."""
    fitted = -(np.linalg.pinv(weights) @ P)
    total = float(np.sum((Y - np.mean(Y)) ** 2))
    if total == 0.0:
        return 0.0
    residual = float(np.sum((Y - fitted) ** 2))
    return 1.0 - residual / total


def RidgeCO(
    dataset: Dataset,
    threshold_range: np.ndarray | None = None,
    penalties: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer A by ridge regression of P on Y, then cut small coefficients.

    GeneSPIDER ``ridgeco`` fits one ridge network per penalty, keeps the
    penalty with the best reconstruction of Y, and removes entries with
    ``|A| <= zeta``. ``threshold_range`` is that cutoff grid, used raw
    (GeneSPIDER ``rawZeta = 1``). ``penalties`` are the L2 penalties tried;
    they default to the cutoff grid when that grid is positive.

    Args:
        dataset: Dataset with Y and P, both genes × samples.
        threshold_range: Absolute cutoffs. None returns the selected dense network.
        penalties: Ridge penalties. None uses ``threshold_range`` or a short log grid.

    Returns:
        ``(networks, cutoffs)``. ``networks`` is genes × genes × cutoffs, or
        the dense genes × genes matrix when no cutoffs are given. ``cutoffs``
        is the penalty that was kept when no cutoffs are given.
    """
    Y, P = _expression_and_design(dataset)
    if penalties is None:
        if threshold_range is not None and np.all(np.asarray(threshold_range) > 0):
            penalties = np.asarray(threshold_range, dtype=float)
        else:
            penalties = np.logspace(-4, 0, 16)
    penalties = np.atleast_1d(np.asarray(penalties, dtype=float))
    if penalties.size == 0:
        raise ValueError("RidgeCO needs at least one penalty")

    best_score = -np.inf
    best: np.ndarray | None = None
    best_penalty = float(penalties[0])
    for penalty, weights in zip(penalties, _ridge_path(Y, P, penalties), strict=True):
        score = _reconstruction_r2(weights, Y, P)
        if score > best_score:
            best_score = score
            best = weights
            best_penalty = float(penalty)
    if best is None:
        raise LinAlgError("RidgeCO did not produce a network")

    if threshold_range is None:
        return best, np.array([best_penalty])

    cuts = np.atleast_1d(np.asarray(threshold_range, dtype=float))
    n_genes = best.shape[0]
    stack = np.zeros((n_genes, n_genes, cuts.size))
    magnitude = np.abs(best)
    for index, cut in enumerate(cuts):
        masked = best.copy()
        masked[magnitude <= cut] = 0.0
        stack[:, :, index] = masked
    return stack, cuts
