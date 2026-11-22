import numpy as np

from analyze.CompareModels import CompareModels
from datastruct.Network import Network


def test_compare_models_single():
    ref_A = np.eye(2)
    ref_net = Network(ref_A)

    pred_A = np.eye(2)
    pred_net = Network(pred_A)

    cm = CompareModels(ref_net, pred_net)

    assert cm.TP[0] == 2
    assert cm.FP[0] == 0
    assert cm.FN[0] == 0
    assert cm.TN[0] == 2 # Diagonals are 1, off-diagonals 0.
    # Wait, ref_binary has 2 True, 2 False.
    # pred_binary has 2 True, 2 False.
    # TP = 2. TN = 2.

    assert cm.F1[0] == 1.0
    assert cm.AUROC[0] == 1.0

def test_compare_models_multiple():
    ref_A = np.eye(2)
    ref_net = Network(ref_A)

    # Model 1: Perfect
    m1 = np.eye(2)
    # Model 2: All zeros
    m2 = np.zeros((2, 2))

    # Stack: (2, 2, 2)
    net_list = np.dstack((m1, m2))

    cm = CompareModels(ref_net, net_list)

    assert len(cm.F1) == 2
    assert cm.F1[0] == 1.0
    assert cm.F1[1] == 0.0 # No TP

    assert cm.TP[0] == 2
    assert cm.TP[1] == 0

def test_compare_models_list():
    ref_A = np.eye(2)
    ref_net = Network(ref_A)

    m1 = Network(np.eye(2))
    m2 = Network(np.zeros((2, 2)))

    cm = CompareModels(ref_net, [m1, m2])

    assert cm.F1[0] == 1.0
    assert cm.F1[1] == 0.0
