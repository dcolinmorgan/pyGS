"""Context Likelihood of Relatedness (CLR) — delegates to sparselink."""

from typing import Any

import numpy as np
from sklearn.metrics import mutual_info_score
from sparselink import get_method

from analyze.Data import Data
from datastruct.Dataset import Dataset


def mutual_information_matrix(data: np.ndarray) -> np.ndarray:
    """Calculate mutual information matrix between all pairs of genes.

    Parameters
    ----------
    data : numpy.ndarray
        Gene expression data (genes × samples)

    Returns
    -------
    numpy.ndarray
        Mutual information matrix (genes × genes)
    """
    n_genes, n_samples = data.shape
    n_bins = max(2, min(int(np.sqrt(n_samples)), 10))
    discretized = np.zeros_like(data, dtype=int)
    for i in range(n_genes):
        ranks = np.argsort(np.argsort(data[i, :]))
        discretized[i, :] = (ranks * n_bins) // n_samples

    mi_matrix = np.zeros((n_genes, n_genes))
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            mi = mutual_info_score(discretized[i, :], discretized[j, :])
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi
    return mi_matrix


def clr_transform(mi_matrix: np.ndarray) -> np.ndarray:
    """Apply CLR transformation to a mutual information matrix.

    Parameters
    ----------
    mi_matrix : numpy.ndarray
        Mutual information matrix

    Returns
    -------
    numpy.ndarray
        CLR-transformed matrix
    """
    n_genes = mi_matrix.shape[0]
    mask = np.ones((n_genes, n_genes), dtype=bool)
    np.fill_diagonal(mask, 0)

    means = np.zeros(n_genes)
    stds = np.zeros(n_genes)
    for i in range(n_genes):
        row = mi_matrix[i, mask[i, :]]
        means[i] = np.mean(row)
        stds[i] = np.std(row)
    stds[stds == 0] = 1.0

    Z = (mi_matrix - means[:, np.newaxis]) / stds[:, np.newaxis]
    clr_matrix = np.sqrt(Z**2 + Z.T**2)
    np.fill_diagonal(clr_matrix, 0)
    return clr_matrix


def CLR(
    dataset: Dataset | Data | Any,
    threshold_range: np.ndarray | list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """CLR network inference via sparselink.

    Parameters
    ----------
    dataset : Dataset or Data object
        Input dataset containing gene expression data
    threshold_range : array-like, optional
        Range of threshold values for sparsification

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

    # sparselink CLR expects (samples x features)
    method = get_method("clr")(threshold=0.0)
    result = method.fit(Y.T)
    clr_matrix = result.adjacency_matrix

    # Create threshold range
    if threshold_range is None:
        zeta = np.logspace(-6, 0, 30)
    else:
        zeta = np.asarray(threshold_range)

    pos_vals = clr_matrix[clr_matrix > 0]
    clr_min = np.min(pos_vals) if pos_vals.size > 0 else 0
    clr_max = np.max(clr_matrix)

    if clr_max > clr_min:
        threshold_range_scaled = clr_min + zeta * (clr_max - clr_min)
    else:
        threshold_range_scaled = zeta * clr_max

    Afit = clr_matrix[:, :, np.newaxis] * (
        clr_matrix[:, :, np.newaxis] >= threshold_range_scaled
    )

    return Afit, threshold_range_scaled
