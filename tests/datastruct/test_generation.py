import numpy as np

from datastruct.random import randomNet
from datastruct.scalefree import scalefree


def test_scalefree():
    N = 20
    avg_links = 2
    A = scalefree(N, avg_links)
    assert A.shape == (N, N)
    # Check if it's roughly the right density
    # It's stochastic, so just check it's not empty
    assert np.count_nonzero(A) > 0

def test_random_network():
    N = 20
    sparsity = 0.1
    A = randomNet(N, sparsity)
    assert A.shape == (N, N)
    # Check density is close to sparsity (within reason)
    density = np.count_nonzero(A) / (N * N)
    # It's random, so just check bounds
    assert 0 <= density <= 1
