from unittest.mock import MagicMock, patch

import numpy as np

from datastruct.Network import Network


def test_network_init():
    net = Network()
    assert net.A is None
    assert net.N == 0

    A = np.eye(3)
    net = Network(A)
    assert np.array_equal(net.A, A)
    assert net.N == 3
    assert net.G is not None
    assert net.G.shape == (3, 3)

def test_network_properties():
    A = np.array([[0, 1], [0, 0]])
    net = Network(A)

    assert net.nnz() == 1
    assert net.size() == (2, 2)
    assert net.size(1) == 2

    # Check names generation
    assert len(net.names) == 2
    assert net.names[0] == "G1"

    # Check sign and logical
    assert np.array_equal(net.sign(), A)
    assert np.array_equal(net.logical(), A.astype(bool))

def test_network_matmul():
    A = np.eye(2)
    net = Network(A)
    # G should be -pinv(A) = -I
    p = np.array([1, 2])
    res = net @ p
    expected = -p.reshape(-1, 1)
    assert np.allclose(res, expected)

def test_network_populate():
    A = np.eye(2)
    net1 = Network(A)
    net1.description = "Test Net"

    net2 = Network()
    net2.populate(net1)

    assert np.array_equal(net2.A, net1.A)
    assert net2.description == "Test Net"

    net3 = Network()
    net3.populate(A)
    assert np.array_equal(net3.A, A)

def test_network_fetch():
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "obj_data": {
                "A": [[1, 0], [0, 1]],
                "names": ["A", "B"],
                "network": "TestNet"
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        net = Network.fetch("http://example.com/net.json")
        assert net.N == 2
        assert net.names == ["A", "B"]
        assert net.network == "TestNet"

def test_network_show(capsys):
    net = Network(np.eye(2))
    net.show()
    captured = capsys.readouterr()
    assert "Network Matrix:" in captured.out
    assert "# Nodes: 2" in captured.out

def test_network_view(capsys):
    net = Network(np.eye(2))
    net.view()
    captured = capsys.readouterr()
    # It falls back to NetworkX if graphistry is not installed
    assert "Plotting with NetworkX/Matplotlib..." in captured.out or "Opening Graphistry visualization..." in captured.out
