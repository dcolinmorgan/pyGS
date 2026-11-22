"""Prior-guided LASSO — uses prior network to reduce penalty on known edges."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Lasso
from sparselink.base import InferenceMethod
from sparselink.types import InferenceResult, InputData


class PriorLassoMethod(InferenceMethod):
    """LASSO with reduced regularization on edges supported by prior knowledge."""
    name = "prior_lasso"

    def __init__(self, alpha: float = 0.01, prior_weight: float = 0.5, **kwargs: Any) -> None:
        super().__init__(alpha=alpha, prior_weight=prior_weight, **kwargs)
        self.alpha = alpha
        self.prior_weight = prior_weight

    def fit(self, X: InputData, y: InputData | None = None, prior_matrix: np.ndarray | None = None, **kwargs: Any) -> InferenceResult:
        X_arr = self._to_array(X)
        n_features = X_arr.shape[1]
        targets = self._to_array(y) if y is not None else X_arr
        A = np.zeros((n_features, n_features))

        for j in range(n_features):
            # Compute effective alpha per-feature based on prior
            if prior_matrix is not None:
                # Lower alpha for edges with prior support
                prior_strength = np.abs(prior_matrix[:, j])
                effective_alpha = self.alpha * (1.0 - self.prior_weight * np.clip(prior_strength, 0, 1))
                # Use minimum effective alpha (most conservative single value)
                alpha_j = float(np.min(effective_alpha[effective_alpha > 0])) if np.any(effective_alpha > 0) else self.alpha
            else:
                alpha_j = self.alpha

            model = Lasso(alpha=alpha_j, fit_intercept=False, max_iter=10000)
            model.fit(X_arr, targets[:, j])
            A[j, :] = model.coef_

        np.fill_diagonal(A, 0.0)
        return InferenceResult(adjacency_matrix=np.abs(A))
