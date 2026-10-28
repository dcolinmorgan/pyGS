#!/usr/bin/env python3
"""Plot the SNR benchmark table as boxplots.

Reads the CSV written by ``python_benchmark_snr_all_methods.py`` (or the
MATLAB twin). Each row must already be one network: F1 and MCC are the best
score over the sparsity/zeta sweep, and AUROC is from the continuous scores.

Expected columns: dataset, method, auroc, f1_score (or f1), mcc.
SNR is read from a ``snr`` column or from names such as ``SNR0.01-...``.

The chapter figure also includes randomized TIGRESS, LASSO and LSCO. Those
rows are not in the stored CSVs, and they cannot be recovered from the image.
If a later run adds them, name the methods ``random tigress``, ``random lasso``
and ``random lsco`` and this script will include them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

METHOD_ORDER = [
    "tigress",
    "random tigress",
    "lasso",
    "random lasso",
    "nestboot lasso",
    "lsco",
    "random lsco",
    "nestboot lsco",
]
SNR_ORDER = ["0.01", "0.1", "1"]
METRICS = [
    ("auroc", "maxAUROC"),
    ("f1", "maxF1"),
    ("mcc", "maxMCC"),
]


def _snr_label(value: str) -> str:
    return f"{float(value):g}"


def load_scores(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [column.strip().lower() for column in frame.columns]
    if "f1" not in frame.columns and "f1_score" in frame.columns:
        frame = frame.rename(columns={"f1_score": "f1"})
    missing = {"method", "auroc", "f1", "mcc"} - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(sorted(missing))}")

    if "snr" not in frame.columns:
        if "dataset" not in frame.columns:
            raise SystemExit(f"{path} needs an snr column or a dataset name containing SNR")
        frame["snr"] = frame["dataset"].astype(str).str.extract(r"SNR([0-9.]+)", expand=False)
    frame["snr"] = frame["snr"].map(lambda value: _snr_label(str(value)))
    frame["method"] = (
        frame["method"]
        .astype(str)
        .str.lower()
        .str.replace(r"\s*\+\s*", " ", regex=True)
        .str.replace("_", " ")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return frame.dropna(subset=["snr"])


def plot_scores(frame: pd.DataFrame, output: Path) -> None:
    long = frame.melt(
        id_vars=["method", "snr"],
        value_vars=["auroc", "f1", "mcc"],
        var_name="metric",
        value_name="score",
    )
    long["metric"] = long["metric"].map(dict(METRICS))
    present = [method for method in METHOD_ORDER if method in set(long["method"])]
    extra = sorted(set(long["method"]) - set(present))
    present.extend(extra)
    snrs = [snr for snr in SNR_ORDER if snr in set(long["snr"])]
    snrs.extend(sorted(set(long["snr"]) - set(snrs), key=float))

    sns.set_theme(style="whitegrid", context="talk")
    grid = sns.catplot(
        data=long,
        x="metric",
        y="score",
        hue="method",
        col="snr",
        col_order=snrs,
        kind="box",
        hue_order=present,
        order=[label for _, label in METRICS],
        height=4.2,
        aspect=0.9,
        sharey=True,
        fliersize=3,
        legend_out=True,
    )
    grid.set_axis_labels("Metric", "Score")
    grid.set_titles("SNR = {col_name}")
    grid.set(ylim=(-0.05, 1.05))
    grid.figure.suptitle(
        "TIGRESS, LASSO and LSCO, with and without NestBoot",
        y=1.03,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(grid.figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        nargs="?",
        default="benchmark/results/python_snr_all_methods_results.csv",
        help="Benchmark score table",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="benchmark/plots/snr_bootstrap_comparison_from_csv.png",
        help="Output figure. Does not overwrite the chapter figure.",
    )
    args = parser.parse_args()
    plot_scores(load_scores(Path(args.csv)), Path(args.output))
    print(args.output)


if __name__ == "__main__":
    main()
