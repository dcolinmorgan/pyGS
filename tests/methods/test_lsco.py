from unittest.mock import MagicMock

import numpy as np
import pytest

from src.datastruct.Dataset import Dataset
from src.methods.lsco import LSCO


class TestLSCO:
    @pytest.fixture
    def synthetic_data(self):
        np.random.seed(42)
        Y = np.random.rand(5, 10)
        P = np.random.rand(5, 10)
        dataset = MagicMock(spec=Dataset)
        dataset.Y = Y
        dataset.P = P
        dataset.data = None
        return dataset, Y, P

    def test_lsco_basic(self, synthetic_data):
        dataset, Y, P = synthetic_data

        # Run LSCO without thresholding
        Als, mse = LSCO(dataset)

        assert Als.shape == (5, 5)
        assert isinstance(mse, float)

    def test_lsco_thresholding(self, synthetic_data):
        dataset, Y, P = synthetic_data
        thresholds = np.linspace(0, 1, 5)

        Afit, out_thresholds = LSCO(dataset, threshold_range=thresholds)

        assert Afit.shape == (5, 5, 5)
        assert len(out_thresholds) == 5

    def test_lsco_input_validation(self):
        dataset = MagicMock(spec=Dataset)
        dataset.Y = np.random.rand(5, 10)
        dataset.P = None

        with pytest.raises(ValueError):
            LSCO(dataset)
