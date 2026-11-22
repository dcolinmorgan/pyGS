import numpy as np

from datastruct.Experiment import Experiment
from datastruct.Network import Network


def test_experiment_init():
    exp = Experiment()
    assert exp._G is None
    assert exp._P is None

def test_experiment_with_network():
    A = np.eye(3)
    net = Network(A)
    exp = Experiment(net)
    assert exp._G is not None
    assert exp._P is not None
    assert exp._Y is not None
    assert exp._Y.shape == (3, 3)

def test_experiment_noise():
    A = np.eye(3)
    net = Network(A)
    exp = Experiment(net, scale=0.1)
    noise = exp.noise()
    assert noise.shape == (3, 3)
    # Check if noise is not all zeros (highly unlikely)
    assert np.any(noise != 0)

def test_experiment_signal():
    A = np.eye(3)
    net = Network(A)
    exp = Experiment(net)
    signal = exp.signal()
    # G is -pinv(A) = -I
    # P is I
    # Signal = G @ P = -I
    expected = -np.eye(3)
    assert np.allclose(signal, expected)
