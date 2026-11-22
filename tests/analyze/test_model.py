import numpy as np

from analyze.Model import Model
from datastruct.Network import Network


def test_model_init():
    A = np.eye(3)
    net = Network(A)
    model = Model(net)

    assert model.network == net.network
    assert model.interampatteness == 1.0  # cond(I) = 1
    assert model.networkComponents == 3   # 3 disconnected nodes with self-loops

def test_model_properties():
    # Create a simple directed graph: 0 -> 1 -> 2
    A = np.zeros((3, 3))
    A[1, 0] = 1
    A[2, 1] = 1
    net = Network(A)
    model = Model(net)

    assert model.networkComponents == 3 # Strongly connected components: {0}, {1}, {2}
    # Path lengths: 0->1 (1), 1->2 (1), 0->2 (2). Others inf.
    # Model implementation handles inf?
    # _calc_path_lengths returns median/mean of reachable paths.
    # Reachable: (0,1), (1,2), (0,2). Lengths: 1, 1, 2.
    # Median: 1. Mean: 1.333
    assert model.medianPathLength == 1.0
    assert np.isclose(model.meanPathLength, 4/3)

    assert model.DD == 2/3 # Mean degree. In+Out.
    # Node 0: Out=1, In=0 -> 1
    # Node 1: Out=1, In=1 -> 2
    # Node 2: Out=0, In=1 -> 1
    # Mean: 4/3?
    # _calc_degree: np.sum(A, axis=1) for directed. This is Out-degree?
    # A[i, j] means j -> i? Or i -> j?
    # Network.py: G = -pinv(A). Usually A_ij is effect of j on i.
    # If A_ij means j -> i (column to row), then axis=1 is row sum (In-degree).
    # Let's check Model._calc_degree: np.sum(A, axis=1).
    # If A[1,0]=1, then 0->1. Row 1 has sum 1.
    # So it calculates In-degree.
    # Node 0: 0. Node 1: 1. Node 2: 1. Mean: 2/3.
    assert np.isclose(model.DD, 2/3)

def test_model_clustering():
    # Triangle: 0->1, 1->2, 2->0
    A = np.zeros((3, 3))
    A[1, 0] = 1
    A[2, 1] = 1
    A[0, 2] = 1
    net = Network(A)
    model = Model(net)

    # Clustering coefficient for directed graph?
    # Model uses custom implementation or networkx.
    # Custom: sum(sub_A) / (k*(k-1))
    # Node 0: neighbors (incoming): 2. k=1. k*(k-1)=0. -> 0.
    # All have k=1 (in-degree). So all 0.
    assert model.CC == 0.0
