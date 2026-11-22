from unittest.mock import MagicMock

import numpy as np
import pytest

from src.datastruct.Dataset import Dataset
from src.methods.clr import CLR, clr_transform, mutual_information_matrix


class TestCLR:
    @pytest.fixture
    def synthetic_data(self):
        np.random.seed(42)
        Y = np.random.rand(5, 20) # More samples for MI
        dataset = MagicMock(spec=Dataset)
        dataset.Y = Y
        dataset.data = None
        return dataset, Y

    def test_clr_basic(self, synthetic_data):
        dataset, Y = synthetic_data

        Afit, thresholds = CLR(dataset)

        n_genes = Y.shape[0]
        assert Afit.shape == (n_genes, n_genes, 30)
        assert len(thresholds) == 30
        assert np.all(Afit >= 0)

    def test_mi_calculation(self, synthetic_data):
        _, Y = synthetic_data
        mi = mutual_information_matrix(Y)

        assert mi.shape == (5, 5)
        assert np.allclose(mi, mi.T) # Symmetric
        assert np.all(np.diag(mi) == 0) # Diagonal zeroed

    def test_clr_transform(self):
        mi = np.array([
            [0, 1, 0.5],
            [1, 0, 0.2],
            [0.5, 0.2, 0]
        ])

        clr = clr_transform(mi)

        assert clr.shape == (3, 3)
        assert np.all(np.diag(clr) == 0)
        assert np.all(clr >= 0)

    def test_clr_input_validation(self):
        with pytest.raises(ValueError):
            CLR(None)
