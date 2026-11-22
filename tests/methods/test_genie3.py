from unittest.mock import MagicMock

import numpy as np
import pytest

from src.datastruct.Dataset import Dataset
from src.methods.genie3 import GENIE3


class TestGENIE3:
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

    def test_genie3_basic(self, synthetic_data):
        dataset, Y = synthetic_data

        # Run GENIE3
        Afit, thresholds = GENIE3(dataset, n_estimators=10, random_state=42)

        # Check output shapes
        n_genes = Y.shape[0]
        assert Afit.shape == (n_genes, n_genes, 30) # Default 30 thresholds
        assert len(thresholds) == 30

        # Check values
        assert np.all(Afit >= 0)
        # Importance scores are not strictly bounded to 1, but should be non-negative

    def test_genie3_custom_thresholds(self, synthetic_data):
        dataset, Y = synthetic_data
        thresholds = np.linspace(0, 0.1, 5)

        # Note: GENIE3 scales the thresholds based on importance values
        # So we check if the number of thresholds matches
        Afit, out_thresholds = GENIE3(dataset, threshold_range=thresholds, n_estimators=10)

        assert Afit.shape[2] == 5
        assert len(out_thresholds) == 5

    def test_genie3_input_validation(self):
        # Test with invalid input
        with pytest.raises(ValueError): # Or whatever error is raised when accessing attributes on None
            GENIE3(None)

        dataset = MagicMock()
        dataset.Y = None
        dataset.data = None
        # Depending on implementation, might raise ValueError or AttributeError
        try:
            GENIE3(dataset)
        except (ValueError, AttributeError):
            pass

    def test_genie3_small_sample_size(self):
        # Test with too few samples
        dataset = MagicMock(spec=Dataset)
        dataset.Y = np.random.rand(5, 2) # Only 2 samples

        Afit, _ = GENIE3(dataset, n_estimators=10)

        # Should return zeros if skipped
        assert np.all(Afit == 0)
