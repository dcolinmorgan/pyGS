"""``pygs nestboot --zeta`` writes the files ``plot_support.py`` reads."""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from bench import cli

COLUMNS = [
    "slice", "zeta", "bin", "support_from", "measured_links", "measured_sd",
    "shuffled_links", "shuffled_sd", "measured_overlap", "shuffled_overlap",
    "tail_area_ratio",
]


def _plot_support():
    path = Path(__file__).resolve().parents[1] / "nestboot_report" / "plot_support.py"
    spec = importlib.util.spec_from_file_location("plot_support", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_zeta_run_writes_support_table_for_plot_support(tmp_path):
    rng = np.random.default_rng(0)
    genes = ["G1", "G2", "G3", "G4"]
    columns = [gene for gene in genes for _ in range(2)]
    values = rng.normal(size=(4, 8))
    for j, gene in enumerate(columns):
        values[genes.index(gene), j] -= 3.0
    data = tmp_path / "ymatrix.csv"
    pd.DataFrame(values, index=genes, columns=columns).to_csv(data, sep="\t")
    run = tmp_path / "run.npy"

    cli._cmd_nestboot(argparse.Namespace(
        file=str(data), method="lsco", nest_runs=3, boot_runs=4,
        fdr=0.05, seed=1, zeta="0.01,0.1", output=str(run),
    ))

    table = pd.read_csv(tmp_path / "run_support.tsv", sep="\t")
    assert list(table.columns) == COLUMNS
    # 2 sparsities x 3 outer-run bins.
    assert len(table) == 6
    assert np.allclose(sorted(table["zeta"].unique()), [0.01, 0.1])
    # Every off-diagonal link lands in one bin per outer run.
    per_slice = table.groupby("slice")[["measured_links", "shuffled_links"]].sum()
    assert np.allclose(per_slice.to_numpy(), 4 * 3)

    support = np.load(tmp_path / "run_support.npz")
    assert support["run_histograms"].shape == (3, 2, 5, 2)
    assert support["fdr_curve"].shape == (2, 3)

    loaded, archive, links = _plot_support().support_table(run)
    assert len(loaded) == 6
    assert archive is not None and links.shape == (2,)


def test_dropout_and_failed_knockdown_flags(tmp_path):
    rng = np.random.default_rng(0)
    genes = ["G1", "G2", "G3", "G4"]
    columns = [gene for gene in genes for _ in range(2)]
    values = rng.normal(size=(4, 8))
    for j, gene in enumerate(columns):
        values[genes.index(gene), j] += 3.0 if gene == "G4" else -3.0
    values[0, 2] = -50.0
    data = tmp_path / "ymatrix.csv"
    pd.DataFrame(values, index=genes, columns=columns).to_csv(data, sep="\t")
    run = tmp_path / "run.npy"

    cli._cmd_nestboot(argparse.Namespace(
        file=str(data), method="lsco", nest_runs=3, boot_runs=4,
        fdr=0.05, seed=1, zeta="0.01,0.1", output=str(run),
        dropout_below=-10.0, max_knockdown_z=0.0,
    ))

    support = np.load(tmp_path / "run_support.npz")
    assert support["failed_knockdowns"].tolist() == [False, False, False, True]
    assert int(support["dropouts_filled"]) == 1
    assert support["knockdown_z"][3] > 0 > support["knockdown_z"][:3].max()
    summary = (tmp_path / "run.txt").read_text()
    assert "knockdowns removed (own-gene z > 0.0): 1 [G4]; samples used: 6" in summary


def test_lasso_targets_follow_design_zeta_and_no_design_options(tmp_path, monkeypatch):
    import methods.lasso

    rng = np.random.default_rng(0)
    genes = ["G1", "G2", "G3", "G4"]
    columns = [gene for gene in genes for _ in range(2)]
    values = rng.normal(size=(4, 8))
    data = tmp_path / "ymatrix.csv"
    pd.DataFrame(values, index=genes, columns=columns).to_csv(data, sep="\t")
    calls = []

    def fake_path(X, targets, alphas, l1_ratio=1.0):
        calls.append((X, targets, np.asarray(alphas)))
        return np.zeros((X.shape[1], X.shape[1], len(alphas)))

    monkeypatch.setattr(methods.lasso, "penalty_path", fake_path)

    def run(**options):
        calls.clear()
        cli._cmd_nestboot(argparse.Namespace(
            file=str(data), method="lasso", nest_runs=2, boot_runs=1,
            fdr=0.05, seed=1, zeta="0.01,0.1", output=str(tmp_path / "run.npy"), **options,
        ))
        return calls[0]

    X, targets, alphas = run()
    assert set(np.unique(targets)) == {0.0, 1.0} and np.allclose(alphas, [0.01, 0.1])
    assert X.min() < -1 or X.max() > 1

    X, targets, _ = run(no_design=True)
    assert targets is X

    X, targets, alphas = run(zeta_cutoff=True, alpha=1e-4)
    assert set(np.unique(targets)) == {0.0, 1.0} and np.allclose(alphas, [1e-4])
    assert X.min() >= -1 and X.max() <= 1
