from unittest.mock import MagicMock

import numpy as np
import pytest

from src.datastruct.Dataset import Dataset
from src.methods.tigress import TIGRESS, tigress_single_gene


class TestTIGRESS:
    @pytest.fixture
    def synthetic_data(self):
        # Create a small synthetic dataset
        # 5 genes, 10 samples
        np.random.seed(42)
        Y = np.random.rand(5, 10)
        dataset = MagicMock(spec=Dataset)
        dataset.Y = Y
        dataset.data = None
        return dataset, Y

    def test_tigress_basic(self, synthetic_data):
        dataset, Y = synthetic_data

        # Run TIGRESS with small bootstrap for speed
        Afit, thresholds = TIGRESS(dataset, n_bootstrap=5, random_state=42)

        # Check output shapes
        n_genes = Y.shape[0]
        assert Afit.shape == (n_genes, n_genes, 30) # Default 30 thresholds
        assert len(thresholds) == 30

        # Check values
        assert np.all(Afit >= 0)
        assert np.all(Afit <= 1) # Stability scores are probabilities [0, 1]

    def test_tigress_single_gene(self):
        target = np.random.rand(10)
        predictors = np.random.rand(10, 4)

        scores = tigress_single_gene(target, predictors, n_bootstrap=5)

        assert len(scores) == 4
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)

    def test_tigress_small_sample_size(self):
        dataset = MagicMock(spec=Dataset)
        dataset.Y = np.random.rand(5, 2) # Only 2 samples

        Afit, _ = TIGRESS(dataset, n_bootstrap=5)

        assert np.all(Afit == 0)
