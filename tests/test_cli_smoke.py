"""Tiny end-to-end runs of the installed ``pygs`` command."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _pygs() -> str:
    found = shutil.which("pygs")
    if found:
        return found
    sibling = Path(sys.executable).with_name("pygs")
    if sibling.is_file():
        return str(sibling)
    raise AssertionError("pygs is not on PATH; install the project editable")


def _run(args: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [_pygs(), *args],
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _knockdown_table(path: Path, n_genes: int = 6, replicates: int = 2, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(n_genes)]
    network = rng.normal(scale=0.4, size=(n_genes, n_genes))
    network[np.abs(network) < 0.55] = 0.0
    np.fill_diagonal(network, -1.2)
    samples = n_genes * replicates
    design = np.zeros((n_genes, samples))
    for gene in range(n_genes):
        design[gene, gene * replicates : (gene + 1) * replicates] = -1.0
    values = -np.linalg.pinv(network) @ design
    values = values + rng.normal(scale=0.02, size=values.shape)
    columns = [gene for gene in genes for _ in range(replicates)]
    pd.DataFrame(values, index=genes, columns=columns).to_csv(path, sep="\t")


def test_pygs_nestboot_smoke(tmp_path):
    data = tmp_path / "expr.tsv"
    out = tmp_path / "nestboot.npy"
    _knockdown_table(data)
    completed = _run(
        [
            "nestboot",
            str(data),
            "-m",
            "lsco",
            "--nest-runs",
            "2",
            "--boot-runs",
            "2",
            "--zeta",
            "0.05,0.2,0.8",
            "--seed",
            "0",
            "--output",
            str(out),
        ],
        tmp_path,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    stack = np.load(out)
    assert stack.shape == (6, 6, 3)
    assert np.isfinite(stack).all()


def test_pygs_bench_fast_smoke(tmp_path):
    out = tmp_path / "bench.json"
    completed = _run(
        [
            "bench",
            "--tier",
            "fast",
            "--data",
            "generated",
            "--snr",
            "10",
            "--n-nodes",
            "8",
            "--n-datasets",
            "1",
            "--topologies",
            "random",
            "--sparsity",
            "0.25",
            "--nestboot",
            "--nest-runs",
            "2",
            "--boot-runs",
            "2",
            "--seed",
            "0",
            "--output",
            str(out),
        ],
        tmp_path,
        timeout=240,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert out.is_file()
    text = out.read_text()
    assert '"method"' in text
    assert "lasso" in text
