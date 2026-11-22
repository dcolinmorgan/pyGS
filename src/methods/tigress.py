"""TIGRESS network inference — delegates to sparselink."""

from typing import Any

import numpy as np
from sklearn.linear_model import Lars, LassoLarsIC
from sparselink import get_method

from analyze.Data import Data
from datastruct.Dataset import Dataset


def tigress_single_gene(
    target_expr: np.ndarray,
    predictor_expr: np.ndarray,
    n_bootstrap: int = 100,
    alpha_range: np.ndarray | list[float] | None = None,
    random_state: int = 42,
) -> np.ndarray:
    """Run TIGRESS stability selection for a single target gene.

    Parameters
    ----------
    target_expr : numpy.ndarray
        Expression of target gene (samples,)
    predictor_expr : numpy.ndarray
        Expression of predictor genes (samples × n_predictors)
    n_bootstrap : int
        Number of bootstrap samples
    alpha_range : array-like, optional
        Unused, kept for API compat
    random_state : int
        Random seed

    Returns
    -------
    scores : numpy.ndarray
        Stability scores for each predictor
    """
    n_samples, n_predictors = predictor_expr.shape
    if n_samples < 3 or n_predictors == 0:
        return np.zeros(n_predictors)

    selection_counts = np.zeros(n_predictors)
    rng = np.random.RandomState(random_state)

    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        try:
            model = LassoLarsIC(criterion="bic", max_iter=500)
            model.fit(predictor_expr[idx], target_expr[idx])
            selection_counts += (np.abs(model.coef_) > 1e-10).astype(float)
        except Exception:
            try:
                model = Lars(n_nonzero_coefs=min(5, n_predictors), fit_intercept=True)
                model.fit(predictor_expr[idx], target_expr[idx])
                selection_counts += (np.abs(model.coef_) > 1e-10).astype(float)
            except Exception:
                continue

    return selection_counts / n_bootstrap


def TIGRESS(
    dataset: Dataset | Data | Any,
    threshold_range: np.ndarray | list[float] | None = None,
    n_bootstrap: int = 50,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """TIGRESS network inference via sparselink.

    Parameters
    ----------
    dataset : Dataset or Data object
        Input dataset containing gene expression data
    threshold_range : array-like, optional
        Range of threshold values for sparsification
    n_bootstrap : int
        Number of bootstrap samples for stability selection
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

    n_genes, n_samples = Y.shape

    # Create threshold range
    if threshold_range is None:
        zeta = np.logspace(-6, 0, 30)
    else:
        zeta = np.asarray(threshold_range)

    if n_samples < 3:
        return np.zeros((n_genes, n_genes, len(zeta))), zeta

    # sparselink expects (samples x features)
    method = get_method("tigress")(n_bootstrap=n_bootstrap, random_state=random_state)
    result = method.fit(Y.T)
    stability_matrix = result.adjacency_matrix

    pos_vals = stability_matrix[stability_matrix > 0]
    stab_min = np.min(pos_vals) if pos_vals.size > 0 else 0
    stab_max = np.max(stability_matrix)

    if stab_max > stab_min:
        threshold_range_scaled = stab_min + zeta * (stab_max - stab_min)
    else:
        threshold_range_scaled = zeta * stab_max if stab_max > 0 else zeta * 0.5

    Afit = stability_matrix[:, :, np.newaxis] * (
        stability_matrix[:, :, np.newaxis] >= threshold_range_scaled
    )

    return Afit, threshold_range_scaled


def TIGRESS_base(
    dataset: Dataset | Data | Any,
    random_state: int = 42,
    **kwargs: Any,
) -> np.ndarray:
    """TIGRESS base learner without stability selection (for NestBoot).

    Parameters
    ----------
    dataset : Dataset or Data object
        Input dataset containing gene expression data
    random_state : int
        Random seed

    Returns
    -------
    A : numpy.ndarray
        Adjacency matrix (n_genes × n_genes)
    """
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

    # Use sparselink TIGRESS with n_bootstrap=1 as base learner
    method = get_method("tigress")(n_bootstrap=1, random_state=random_state)
    result = method.fit(Y.T)
    return result.adjacency_matrix
