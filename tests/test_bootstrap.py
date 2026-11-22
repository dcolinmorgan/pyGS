import numpy as np
import pytest

from bootstrap.utils import NetworkUtils
from methods.nestboot import Nestboot, NetworkData


def test_pyGS_initialization() -> None:
    """Test NetworkBootstrap class initialization."""
    data = NetworkData(
        Y=np.random.random((10, 10)),
        names=[f"Node_{i}" for i in range(10)],
        N=10,
        M=10
    )
    nb = Nestboot(data)
    assert nb.data == data
    assert nb.logger is not None

def test_matrix_operations() -> None:
    """Test matrix AND/OR operations."""
    test_matrix = np.array([
        [True, False, True],
        [False, True, False],
        [True, True, False]
    ])

    result_or = NetworkUtils.matrix_or(test_matrix)
    result_and = NetworkUtils.matrix_and(test_matrix)

    assert isinstance(result_or, np.ndarray)
    assert isinstance(result_and, np.ndarray)

@pytest.mark.parametrize("matrix,init,expected_bins", [
    (np.array([[0.1, 0.2], [0.3, 0.4]]), 4, 5),  # 4 bins + 1 edge
    (np.array([[0.5, 0.6], [0.7, 0.8]]), 5, 6),  # 5 bins + 1 edge
])
def test_calc_bin_freq(
    matrix: np.ndarray,
    init: int,
    expected_bins: int
) -> None:
    """Test frequency binning calculation.

    Args:
        matrix: Test input matrix
        init: Number of bins
        expected_bins: Expected number of bin edges
    """
    freq, bins = NetworkUtils.calc_bin_freq(matrix, init)
    assert len(bins) == expected_bins
    assert np.isclose(np.sum(freq), 1.0)
