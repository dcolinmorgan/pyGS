#!/usr/bin/env python3
"""Score `pygs nestboot` networks against a simulated true network.

Usage: score_synthetic.py RUN.npy TRUE.npy TRUTH.json [--label NAME]

For every sparsity with a tail at the FDR, prints and appends to
synthetic_scores.tsv: links, true FDR and recall of the every-outer-run
network and of the mean-support network, sign agreement on true links, and
links pointing into genes whose knockdown had no effect. A run with no
FDR-controlled sparsity gets one row with an empty slice and 0 links.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
COLUMNS = [
    "label", "slice", "zeta", "cutoff", "estimated_fdr", "links", "true_fdr", "recall",
    "sign_agreement", "links_into_failed", "mean_links", "mean_true_fdr", "mean_recall",
]


def score(net: np.ndarray, true: np.ndarray) -> tuple[int, float, float]:
    links = int(net.sum())
    tp = int((net & true).sum())
    fdr = 1 - tp / links if links else np.nan
    return links, fdr, tp / true.sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("true", type=Path)
    parser.add_argument("truth", type=Path)
    parser.add_argument("--label", default=None)
    parser.add_argument("--out", type=Path, default=HERE / "synthetic_scores.tsv")
    args = parser.parse_args()

    W = np.load(args.true)
    true = W != 0
    meta = json.loads(args.truth.read_text())
    failed = np.isin([f"G{i:03d}" for i in range(W.shape[0])], meta["failed_knockdowns"])
    xnet = np.load(args.run) != 0
    signed = np.load(args.run.with_stem(args.run.stem + "_signed"))
    z = np.load(args.run.with_name(args.run.stem + "_support.npz"))
    off = ~np.eye(W.shape[0], dtype=bool)

    rows = []
    for k in np.flatnonzero(z["fdr_reached"]):
        net = xnet[:, :, k] & off
        mean_net = (z["mean_support"][:, :, k] >= z["support_cutoff"][k] - 1e-9) & off
        links, fdr, recall = score(net, true)
        m_links, m_fdr, m_recall = score(mean_net, true)
        hits = net & true
        rows.append(
            {
                "label": args.label or args.run.stem,
                "slice": int(k),
                "zeta": float(z["zetavec"][k]),
                "cutoff": float(z["support_cutoff"][k]),
                "estimated_fdr": float(z["fdr_at_cutoff"][k]),
                "links": links,
                "true_fdr": fdr,
                "recall": recall,
                "sign_agreement": float((np.sign(signed[:, :, k][hits]) == np.sign(W[hits])).mean())
                if hits.any()
                else np.nan,
                "links_into_failed": int(net[failed].sum()),
                "mean_links": m_links,
                "mean_true_fdr": m_fdr,
                "mean_recall": m_recall,
            }
        )
    if not rows:
        rows.append({"label": args.label or args.run.stem, "slice": np.nan, "links": 0})
    table = pd.DataFrame(rows).reindex(columns=COLUMNS)
    print(table.round(3).to_string(index=False))
    new = not args.out.exists() or args.out.stat().st_size == 0
    table.to_csv(args.out, sep="\t", index=False, mode="a", header=new)


if __name__ == "__main__":
    main()
