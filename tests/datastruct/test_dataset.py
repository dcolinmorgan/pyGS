import numpy as np

from datastruct.Dataset import Dataset
from datastruct.Network import Network


def test_dataset_init():
    ds = Dataset()
    assert ds.Y is None
    assert ds.N == 0

    net = Network(np.eye(3))
    ds = Dataset(net)
    assert ds.network is net
    assert ds.Y is not None
    assert ds.Y.shape == (3, 3)

def test_dataset_properties():
    ds = Dataset()
    ds.Y = np.zeros((5, 10))
    assert ds.N == 5
    assert ds.M == 10

    assert len(ds.gene_names) == 5
    assert ds.gene_names[0] == "G1"

def test_dataset_populate_network():
    A = np.eye(2)
    net = Network(A)
    ds = Dataset()
    ds.populate(net)

    assert ds.network is net
    assert ds.P is not None
    # Y = A @ P, P is eye(2) -> Y = A
    assert np.array_equal(ds.Y, A)

def test_dataset_true_response():
    A = np.eye(2)
    net = Network(A)
    ds = Dataset(net)

    # true_response uses G @ P
    # G = -pinv(A) = -I
    # P = I
    # res = -I
    res = ds.true_response()
    assert np.allclose(res, -np.eye(2))

def test_dataset_name_generation():
    net = Network(np.eye(2))
    net.network = "TestNet-ID123"
    ds = Dataset(net)

    assert "ID123" in ds.dataset
    assert "N2" in ds.dataset
