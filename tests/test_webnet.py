import sys

import pytest

sys.path.append("src")  # Add src/ to path
from analyze.CompareModels import CompareModels
from analyze.Data import Data
from analyze.Model import Model
from datastruct.Dataset import Dataset
from datastruct.Experiment import Experiment
from datastruct.Network import Network


@pytest.mark.web
def test_basic_functionality():
    networkA = Network.from_json_url('https://bitbucket.org/sonnhammergrni/gs-networks/raw/0b3a66e67d776eadaa5d68667ad9c8fbac12ef85/scalefree/N50/Tjarnberg-D20131127-scalefree-N50-L231-ID461522.json')
    print(f"Network A shape: {networkA.A.shape}")

    exp = Experiment(networkA)
    exp.gaussian()
    print(f"Experiment E shape: {exp._E.shape}, P shape: {exp._P.shape}")
    print(f"Experiment Y: {exp.noiseY()}")
    ds = Dataset(exp)
    print(f"Dataset Y: {ds.Y}")
    model = Model(networkA)
    print(f"Condition Number: {model.interampatteness}")
    print(f"Small-worldness: {model.proximity_ratio}")
    data = Data(ds)
    print(f"SNR_L: {data.SNR_L}")

    networkB = Network.from_json_url('https://bitbucket.org/sonnhammergrni/gs-networks/raw/0b3a66e67d776eadaa5d68667ad9c8fbac12ef85/scalefree/N50/Tjarnberg-D20131127-scalefree-N50-L232-ID279215.json')


    comp = CompareModels(networkA, networkB)
    print(f"Absolute Frobenius Norm: {comp.afronorm}")
    print(f"F1 Score: {comp.F1}")
    print(f"MCC : {comp.MCC}")

if __name__ == "__main__":
    test_basic_functionality()
