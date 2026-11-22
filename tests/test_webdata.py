import sys

import pytest

sys.path.append("src")  # Add src/ to path
from analyze.CompareModels import CompareModels
from analyze.Data import Data
from datastruct.Network import Network
from methods.lasso import Lasso
from methods.lsco import LSCO


@pytest.mark.web
def test_basic_functionality():
    dataA = Data.from_json_url('https://bitbucket.org/sonnhammergrni/gs-datasets/raw/d2047430263f5ffe473525c74b4318f723c23b0e/N50/Tjarnberg-ID252384-D20151111-N50-E150-SNR10-IDY252384.json')
    lasso_net,alpha=Lasso(dataA.data)
    lasso_net=Network(lasso_net)
    true_net = Network.from_json_url('https://bitbucket.org/sonnhammergrni/gs-networks/raw/0b3a66e67d776eadaa5d68667ad9c8fbac12ef85/random/N50/Tjarnberg-D20150910-random-N50-L158-ID252384.json')
    comp = CompareModels(true_net, lasso_net)
    print(f"lasso F1 Score: {comp.F1}")
    print(f"lasso MCC : {comp.MCC}")

    lsco_net,alpha=LSCO(dataA.data)
    lsco_net=Network(lsco_net)
    comp = CompareModels(lsco_net, true_net)
    print(f"lsco F1 Score: {comp.F1}")
    print(f"lsco MCC : {comp.MCC}")


if __name__ == "__main__":
    test_basic_functionality()
