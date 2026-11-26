"""LASSO network inference — delegates to sparselink."""

import numpy as np

from datastruct.Dataset import Dataset
from bio.io import design_targets
from sparselink import get_method


def penalty_path(
    X: np.ndarray,
    targets: np.ndarray,
    alphas: np.ndarray,
    l1_ratio: float = 1.0,
) -> np.ndarray:
    """LASSO (``l1_ratio=1``) or elastic net at every penalty, as a stack of networks.

    Row i of slice k solves ``X @ A[i] ≈ targets[:, i]`` at penalty ``alphas[k]``,
    the same fit as sparselink's ``fit(X, targets)``. Each target's path is fitted
    from the largest penalty down, starting from the previous solution.
    """
    from sklearn.linear_model import enet_path

    X = np.asarray(X, dtype=float)
    targets = np.asarray(targets, dtype=float)
    alphas = np.asarray(alphas, dtype=float)
    order = np.argsort(alphas)[::-1]
    gram = X.T @ X
    stack = np.zeros((targets.shape[1], X.shape[1], len(alphas)))
    for i in range(targets.shape[1]):
        y = targets[:, i]
        _, coefs, _ = enet_path(
            X, y, l1_ratio=l1_ratio, alphas=alphas[order],
            precompute=gram, Xy=X.T @ y, max_iter=10000, tol=1e-4,
        )
        stack[i][:, order] = coefs
    return stack


def Lasso(
    dataset: Dataset,
    alpha_range: np.ndarray | None = None,
    cv: int = 5,
    tol: float = 1e-4,
    max_iter: int = 10000,
    use_covariance: bool | None = None,
    **kwargs: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer network matrix A using LASSO regression via sparselink.

    Args:
        dataset: Dataset object containing Y (expression data) and P (perturbations)
        alpha_range: Array of alpha values to try. If None, uses logspace(-6, -1, 30)
        cv: Number of folds for cross-validation (unused, kept for API compat)
        tol: Convergence tolerance (unused, kept for API compat)
        max_iter: Maximum iterations (unused, kept for API compat)
        use_covariance: Whether to use Gram matrix (unused, kept for API compat)
        **kwargs: Additional arguments (e.g., threshold_range)

    Returns:
        Tuple of (3D network array, alpha values used)
    """
    if alpha_range is None and "threshold_range" in kwargs:
        alpha_range = np.asarray(kwargs["threshold_range"])

    # Extract data from dataset
    if hasattr(dataset, "Y"):
        Y, P = dataset.Y, dataset.P
    elif hasattr(dataset, "data") and dataset.data is not None:
        Y, P = dataset.data.Y, dataset.data.P
    else:
        raise ValueError("Dataset must contain Y and P matrices")

    if Y is None or P is None:
        raise ValueError("Dataset must contain Y and P matrices")
    if Y.shape[0] != P.shape[0]:
        raise ValueError("Y and P must have same number of rows (genes)")

    n_genes = Y.shape[0]

    if alpha_range is None:
        alpha_range = np.logspace(-6, -1, 30)

    # sparselink expects (samples x features)
    X = Y.T  # (n_samples, n_genes)

    Afit = np.zeros((n_genes, n_genes, len(alpha_range)))
    method_cls = get_method("lasso")
    targets = design_targets("lasso", P.T)

    for j, alpha in enumerate(alpha_range):
        method = method_cls(alpha=alpha)
        result = method.fit(X, targets)
        Afit[:, :, j] = result.adjacency_matrix

    return Afit, alpha_range
