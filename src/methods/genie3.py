"""GENIE3 network inference — delegates to sparselink."""

from typing import Any

import numpy as np
from sparselink import get_method

from analyze.Data import Data
from datastruct.Dataset import Dataset


def GENIE3(
    dataset: Dataset | Data | Any,
    threshold_range: np.ndarray | list[float] | None = None,
    n_estimators: int = 100,
    max_features: str | int | float = "sqrt",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """GENIE3 network inference via sparselink.

    Parameters
    ----------
    dataset : Dataset or Data object
        Input dataset containing gene expression data
    threshold_range : array-like, optional
        Range of threshold values for sparsification
    n_estimators : int
        Number of trees in the random forest
    max_features : str or int
        Number of features to consider for best split
    random_state : int
        Random seed

    Returns
    -------
    Afit : numpy.ndarray
        3D array of inferred networks (n_genes × n_genes × n_thresholds)
    threshold_range : numpy.ndarray
        Array of threshold values used
    """
    # Extract expression matrix Y (genes × samples)
    if hasattr(dataset, "Y") and dataset.Y is not None:
        Y = dataset.Y
    elif hasattr(dataset, "data") and dataset.data is not None:
        data = dataset.data
        if hasattr(data, "Y") and data.Y is not None:
            Y = data.Y
        elif hasattr(data, "data"):
            Y = data.data
        else:
            Y = data
    else:
        Y = dataset

    if not isinstance(Y, np.ndarray):
        raise ValueError("Could not extract expression matrix Y from dataset")

    # sparselink expects (samples x features)
    n_genes, n_samples = Y.shape
    if n_samples < 3:
        # Not enough samples to fit — return zeros
        if threshold_range is None:
            zeta = np.logspace(-6, 0, 30)
        else:
            zeta = np.asarray(threshold_range)
        return np.zeros((n_genes, n_genes, len(zeta))), zeta

    method = get_method("genie3")(
        n_estimators=n_estimators,
        max_features=str(max_features),
        random_state=random_state,
    )
    result = method.fit(Y.T)
    importance_matrix = result.adjacency_matrix

    # Create threshold range
    if threshold_range is None:
        zeta = np.logspace(-6, 0, 30)
    else:
        zeta = np.asarray(threshold_range)

    pos_vals = importance_matrix[importance_matrix > 0]
    imp_min = np.min(pos_vals) if pos_vals.size > 0 else 0
    imp_max = np.max(importance_matrix)

    if imp_max > imp_min:
        threshold_range_scaled = imp_min + zeta * (imp_max - imp_min)
    else:
        threshold_range_scaled = zeta * imp_max

    Afit = importance_matrix[:, :, np.newaxis] * (
        importance_matrix[:, :, np.newaxis] >= threshold_range_scaled
    )

    return Afit, threshold_range_scaled
