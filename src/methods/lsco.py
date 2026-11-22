"""Least Squares with Cutoff (LSCO) — delegates to sparselink."""

import numpy as np
from numpy import linalg
from sparselink import get_method

from datastruct.Dataset import Dataset


def LSCO(
    dataset: Dataset,
    threshold_range: np.ndarray | None = None,
    tol: float = 1e-8,
    rcond: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer network matrix A using least squares with thresholding via sparselink.

    Args:
        dataset: Dataset object containing Y (expression data) and P (perturbations)
        threshold_range: Array of threshold values (normalized 0-1).
        tol: Tolerance (unused, kept for API compat)
        rcond: Cut-off ratio (unused, kept for API compat)

    Returns:
        Tuple of (network array, thresholds or MSE)
    """
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

    # Use sparselink LSCO with threshold=0 to get raw matrix
    method = get_method("lsco")(threshold=0.0)
    # sparselink expects (samples x features); y = perturbation (samples x features)
    result = method.fit(Y.T, P.T)
    Als = result.adjacency_matrix

    if threshold_range is None:
        # Return unthresholded with MSE
        try:
            A_pinv = linalg.pinv(Als, rcond=1e-15)
            Y_hat = -A_pinv @ P
            mse = np.mean((Y - Y_hat) ** 2)
        except Exception:
            mse = 0.0
        return Als, mse

    # Apply thresholding
    nonzero_abs = np.abs(Als[Als != 0])
    if len(nonzero_abs) == 0:
        return np.zeros_like(Als), np.array([0.0])

    zeta_min = np.min(nonzero_abs) - np.finfo(float).eps
    zeta_max = np.max(nonzero_abs) + 10 * np.finfo(float).eps
    delta = zeta_max - zeta_min

    actual_thresholds = threshold_range * delta + zeta_min
    n_genes = Als.shape[0]
    estA_3d = np.zeros((n_genes, n_genes, len(actual_thresholds)))

    for i, threshold in enumerate(actual_thresholds):
        Atmp = Als.copy()
        Atmp[np.abs(Atmp) <= threshold] = 0
        estA_3d[:, :, i] = Atmp

    return estA_3d, actual_thresholds
