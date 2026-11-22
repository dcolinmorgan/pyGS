from unittest.mock import MagicMock

import numpy as np
import pytest

from src.datastruct.Dataset import Dataset
from src.methods.lasso import Lasso


class TestLasso:
    @pytest.fixture
    def synthetic_data(self):
        # Create a small synthetic dataset
        # 5 genes, 10 samples
        np.random.seed(42)
        Y = np.random.rand(5, 10)
        P = np.random.rand(5, 10) # Perturbations
        dataset = MagicMock(spec=Dataset)
        dataset.Y = Y
        dataset.P = P
        dataset.data = None
        return dataset, Y, P

    def test_lasso_basic(self, synthetic_data):
        dataset, Y, P = synthetic_data

        # Run Lasso
        # Use small alpha range for speed
        alpha_range = np.logspace(-4, -1, 5)
        Afit, alphas = Lasso(dataset, alpha_range=alpha_range)

        # Check output shapes
        n_genes = Y.shape[0]
        assert Afit.shape == (n_genes, n_genes, 5)
        assert len(alphas) == 5

    def test_lasso_input_validation(self):
        dataset = MagicMock(spec=Dataset)
        dataset.Y = np.random.rand(5, 10)
        dataset.P = None # Missing P

        with pytest.raises(ValueError):
            Lasso(dataset)

        dataset.P = np.random.rand(4, 10) # Mismatched dimensions
        with pytest.raises(ValueError):
            Lasso(dataset)

    def test_lasso_parallel(self, synthetic_data):
        # Test if parallel execution path works (mocking joblib if needed, but here just running it)
        # We can't easily force parallel execution if joblib isn't installed or if USE_PARALLEL is False
        # But we can run the function and ensure it doesn't crash
        dataset, _, _ = synthetic_data
        alpha_range = np.logspace(-4, -1, 2)
        Afit, _ = Lasso(dataset, alpha_range=alpha_range)
        assert Afit.shape[2] == 2
