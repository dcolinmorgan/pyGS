#!/usr/bin/env python3
"""Summarize real-data `pygs nestboot` runs, one row per FDR-controlled sparsity.

Usage: summarize_runs.py RUN.npy [RUN.npy ...] [--out table.tsv]

Gene SD and knockdown z come from the real table with dropouts (< -10)
filled, so every run is described against the same reference.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bio.io import load_inference_inputs  # noqa: E402
from bio.knockdown import fill_dropouts, knockdown_z  # noqa: E402

REAL = ROOT / ".gs_cache" / "downloads" / "134c84bebbd65203.csv"


def reference() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, names, P = load_inference_inputs(str(REAL))
    Y, _ = fill_dropouts(X.T, P.T, below=-10)
    return np.asarray(names), Y.std(axis=1), knockdown_z(Y, P.T) > 0


def summarize(run: Path, genes: np.ndarray, sd: np.ndarray, failed: np.ndarray) -> list[dict]:
    xnet = np.load(run) != 0
    z = np.load(run.with_name(run.stem + "_support.npz"))
    rows = []
    for k in np.flatnonzero(z["fdr_reached"]):
        net = xnet[:, :, k]
        out_degree = net.sum(axis=0)
        mean_net = z["mean_support"][:, :, k] >= z["support_cutoff"][k] - 1e-9
        top = np.argsort(-out_degree, kind="stable")[:5]
        rows.append(
            {
                "run": run.stem,
                "slice": int(k),
                "zeta": float(z["zetavec"][k]),
                "cutoff": float(z["support_cutoff"][k]),
                "estimated_fdr": float(z["fdr_at_cutoff"][k]),
                "links": int(net.sum()),
                "mean_links": int(mean_net.sum()),
                "rho_outdegree_sd": float(spearmanr(out_degree, sd)[0]) if net.any() else np.nan,
                "links_into_failed": int(net[failed].sum()),
                "top_regulators": ", ".join(f"{genes[i]} {out_degree[i]}" for i in top if out_degree[i]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    genes, sd, failed = reference()
    table = pd.DataFrame([r for run in args.runs for r in summarize(run, genes, sd, failed)])
    with pd.option_context("display.width", 250, "display.max_colwidth", 70):
        print(table.round(3).to_string(index=False))
    if args.out:
        table.to_csv(args.out, sep="\t", index=False)


if __name__ == "__main__":
    main()
