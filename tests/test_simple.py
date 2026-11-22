import sys

sys.path.append("src")  # Add src/ to path
import numpy as np

from analyze.CompareModels import CompareModels
from analyze.Data import Data
from analyze.Model import Model
from datastruct.Dataset import Dataset
from datastruct.Experiment import Experiment
from datastruct.Network import Network


def test_basic_functionality():
    A = np.array([[0, 1, 0.5, -1], [-0.3, 0.9, 0, 1], [1,0, 0, 0],[0,0.1,-0.2,-0.9]])
    net = Network(A)
    print(f"Network A shape: {net.A.shape}")
    exp = Experiment(net)
    exp.gaussian()
    print(f"Experiment E shape: {exp._E.shape}, P shape: {exp._P.shape}")
    print(f"Experiment Y: {exp.noiseY()}")
    ds = Dataset(exp)
    print(f"Dataset Y: {ds.Y}")
    model = Model(net)
    print(f"Condition Number: {model.interampatteness}")
    print(f"Small-worldness: {model.proximity_ratio}")
    data = Data(ds)
    print(f"SNR_L: {data.SNR_L}")
    B = np.array([[1,0, 0, 0.9],[0,0.1,-0.6,0.2],[0, 0, -0.5, 0], [0, -0.2, 0, 1]])
    net2 = Network(B)
    comp = CompareModels(net, net2)
    print(f"Absolute Frobenius Norm: {comp.afronorm}")
    print(f"F1 Score: {comp.F1}")
    print(f"MCC : {comp.MCC}")

if __name__ == "__main__":
    test_basic_functionality()
