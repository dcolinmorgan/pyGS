"""Expression loading for local files and knockdown tables."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from bio.io import load_inference_inputs


def test_samples_by_genes_csv(tmp_path):
    path = tmp_path / "expr.csv"
    pd.DataFrame({"G1": [1.0, 2.0], "G2": [3.0, 4.0]}, index=["s1", "s2"]).to_csv(path)
    X, names, P = load_inference_inputs(str(path))
    assert names == ["G1", "G2"]
    assert X.shape == (2, 2)
    assert P is None
    assert X[0, 0] == 1.0


def test_knockdown_tsv_named_csv(tmp_path):
    """GeneSnake ENCODE layout: genes × experiments, replicate columns share a name."""
    genes = ["AATF", "ABCF1"]
    columns = ["AATF", "AATF", "ABCF1", "ABCF1"]
    values = np.arange(8, dtype=float).reshape(2, 4)
    frame = pd.DataFrame(values, index=genes, columns=columns)
    path = tmp_path / "ymatrix.csv"
    frame.to_csv(path, sep="\t")

    X, names, P = load_inference_inputs(str(path))
    assert names == genes
    assert X.shape == (4, 2)
    assert np.allclose(X, values.T)
    assert P.shape == (4, 2)
    assert np.allclose(P, np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float))


def test_replicate_suffix_name_dot_1():
    """pandas may rename the second replicate GENE -> GENE.1."""
    from bio.io import _perturbation_matrix

    genes = ["AATF", "ABCF1"]
    design = _perturbation_matrix(genes, ["AATF", "AATF.1", "ABCF1", "ABCF1.1"])
    assert design.shape == (2, 4)
    assert design.sum(axis=1).tolist() == [2.0, 2.0]
    assert design.sum(axis=0).tolist() == [1.0, 1.0, 1.0, 1.0]


def test_file_url(tmp_path):
    path = tmp_path / "expr.csv"
    pd.DataFrame({"G1": [1.0]}, index=["s1"]).to_csv(path)
    X, names, P = load_inference_inputs(path.as_uri())
    assert names == ["G1"]
    assert X.shape == (1, 1)
    assert P is None
