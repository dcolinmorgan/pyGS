"""``pygs nestboot -m lasso --zeta`` must return one network per zeta.

A LASSO fit of gene-shuffled data is often all zeros at the default penalty.
That fit used to come back as a single matrix, while a fit that still had
links came back as one slice per zeta, and NestBoot aborted on the mismatch.
"""

import argparse

import numpy as np
import pandas as pd

from bench import cli


def _knockdown_table(path, values, genes):
    columns = [gene for gene in genes for _ in range(values.shape[1] // len(genes))]
    pd.DataFrame(values, index=genes, columns=columns).to_csv(path, sep="\t")


def test_lasso_nestboot_zeta_survives_an_empty_shuffled_fit(tmp_path):
    """Measured LASSO still has links; the gene-shuffled null does not.

    That is the K562 failure: one fit is (genes, genes, n_zeta) and the next
    is a single matrix, and NestBoot raises on the shape mismatch.
    """
    rng = np.random.default_rng(1)
    genes = [f"G{i}" for i in range(6)]
    reps = 3
    true = rng.normal(scale=0.5, size=(6, 6))
    true[np.abs(true) < 0.6] = 0.0
    np.fill_diagonal(true, -1.5)
    design = np.zeros((6, 6 * reps))
    for gene in range(6):
        design[gene, gene * reps : (gene + 1) * reps] = -1.0
    values = -np.linalg.inv(true) @ design
    values = values + rng.normal(scale=0.05, size=values.shape)
    data = tmp_path / "ymatrix.tsv"
    _knockdown_table(data, values, genes)
    run = tmp_path / "lasso.npy"

    cli._cmd_nestboot(
        argparse.Namespace(
            file=str(data),
            method="lasso",
            nest_runs=2,
            boot_runs=2,
            fdr=0.05,
            seed=0,
            zeta="0.001,0.01,0.1",
            output=str(run),
        )
    )

    stack = np.load(run)
    assert stack.shape == (6, 6, 3)
    assert np.isfinite(stack).all()


def test_lasso_nestboot_keeps_a_slice_per_zeta_when_a_fit_is_empty(tmp_path):
    rng = np.random.default_rng(0)
    genes = ["G1", "G2", "G3", "G4"]
    columns = [gene for gene in genes for _ in range(2)]
    # Knockdown-shaped, but tiny: default LASSO (alpha 0.01) zeros every
    # off-diagonal weight, including on the gene-shuffled null.
    values = rng.normal(scale=1e-3, size=(4, 8))
    for j, gene in enumerate(columns):
        values[genes.index(gene), j] -= 1e-3
    data = tmp_path / "ymatrix.tsv"
    pd.DataFrame(values, index=genes, columns=columns).to_csv(data, sep="\t")
    run = tmp_path / "lasso.npy"

    cli._cmd_nestboot(
        argparse.Namespace(
            file=str(data),
            method="lasso",
            nest_runs=2,
            boot_runs=2,
            fdr=0.05,
            seed=0,
            zeta="-4:0:5",
            output=str(run),
        )
    )

    stack = np.load(run)
    assert stack.shape == (4, 4, 5)
    assert np.isfinite(stack).all()
