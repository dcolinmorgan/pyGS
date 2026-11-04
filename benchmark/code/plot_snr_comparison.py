#!/usr/bin/env python3
"""Plot the SNR benchmark table as boxplots.

Reads the CSV written by ``pygs bench``. A row may be one regularization
setting. AUROC is the threshold-free score of the setting with the best F1.
F1 and MCC are still the best values over the sweep. The axis therefore says
AUROC, not maxAUROC.

Expected columns: dataset, method, auroc, f1_score (or f1), mcc.
SNR is read from an ``snr_l`` column when every row has one, else from
``snr`` or from names such as ``SNR0.01-...``. ``pygs bench-gs`` JSON
records the filename SNR in ``snr`` and SNR_L in ``snr_l``.

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
from matplotlib.patches import Patch

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
    ("auroc", "AUROC"),
    ("f1", "maxF1"),
    ("mcc", "maxMCC"),
]


def _snr_label(value: str) -> str:
    return f"{float(value):g}"


def load_scores(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        frame = pd.read_json(path)
    else:
        frame = pd.read_csv(path)
    frame.columns = [column.strip().lower() for column in frame.columns]
    if "f1" not in frame.columns and "f1_score" in frame.columns:
        frame = frame.rename(columns={"f1_score": "f1"})
    missing = {"method", "auroc", "f1", "mcc"} - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(sorted(missing))}")

    if "error" in frame.columns:
        frame = frame[frame["error"].isna()].copy()
    if "snr_l" in frame.columns and frame["snr_l"].notna().all():
        frame["snr"] = frame["snr_l"]
    if "snr" not in frame.columns:
        if "dataset" not in frame.columns:
            raise SystemExit(f"{path} needs an snr column or a dataset name containing SNR")
        frame["snr"] = frame["dataset"].astype(str).str.extract(r"SNR([0-9.]+)", expand=False)
    frame["snr"] = frame["snr"].map(lambda value: _snr_label(str(value)))
    frame["method"] = (
        frame["method"]
        .astype(str)
        .str.lower()
        .str.replace(r"\s*[+:]\s*", " ", regex=True)
        .str.replace("_", " ")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return frame.dropna(subset=["snr"])


def collapse_sparsity(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per network. AUROC belongs to the setting with the best F1."""
    if "alpha" not in frame.columns or frame["alpha"].isna().all():
        return frame
    keys = [column for column in ("dataset", "method", "snr", "topology") if column in frame.columns]
    rows = []
    for _, part in frame.groupby(keys, dropna=False):
        at_f1 = part.loc[part["f1"].idxmax()]
        row = {key: at_f1[key] for key in keys}
        row["auroc"] = float(at_f1["auroc"])
        row["f1"] = float(at_f1["f1"])
        row["mcc"] = float(part["mcc"].max())
        rows.append(row)
    return pd.DataFrame(rows)


RUNS = ["direct", "NestBoot"]
RUN_COLORS = {"direct": "#9CB4D8", "NestBoot": "#E8833A"}


def split_runs(long: pd.DataFrame) -> pd.DataFrame:
    """Base method and run ("direct" or "NestBoot") from names such as ``nestboot lsco``."""
    nested = long["method"].str.startswith("nestboot ")
    return long.assign(
        base=long["method"].str.removeprefix("nestboot "),
        run=nested.map({True: "NestBoot", False: "direct"}),
    )


def plot_split(long: pd.DataFrame, snrs: list[str], output: Path) -> None:
    """One split violin per method: direct on one half, NestBoot on the other.

    Rows are metrics and columns SNR levels, so the figure grows downward with
    the number of methods. Methods never run through NestBoot, such as
    TIGRESS, have only the direct half. Each run is also drawn as a point,
    because a violin needs several networks to show a distribution.
    """
    long = split_runs(long)
    bases = sorted(set(long["base"]))
    sns.set_theme(style="whitegrid", context="notebook")
    grid = sns.FacetGrid(
        long,
        row="metric",
        col="snr",
        row_order=[label for _, label in METRICS],
        col_order=snrs,
        height=0.42 * len(bases) + 1.0,
        aspect=0.85,
        sharex=True,
        sharey=True,
    )
    common = dict(x="score", y="base", hue="run", order=bases, hue_order=RUNS, palette=RUN_COLORS, legend=False)
    grid.map_dataframe(
        sns.violinplot, split=True, cut=0, inner="quart", density_norm="width", linewidth=0.8, **common
    )
    grid.map_dataframe(
        sns.stripplot, dodge=True, jitter=0.08, size=3.5, edgecolor="black", linewidth=0.4, **common
    )
    grid.set_titles(row_template="{row_name}", col_template="SNR_L = {col_name}")
    grid.set_axis_labels("Score", "")
    grid.set(xlim=(-0.05, 1.05))
    handles = [Patch(facecolor=RUN_COLORS[run], edgecolor="black", label=run) for run in RUNS]
    grid.figure.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    grid.figure.suptitle("Direct inference and NestBoot on GeneSPIDER networks", y=1.03)
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(grid.figure)


def plot_scores(frame: pd.DataFrame, output: Path) -> None:
    frame = collapse_sparsity(frame)
    long = frame.melt(
        id_vars=["method", "snr"],
        value_vars=["auroc", "f1", "mcc"],
        var_name="metric",
        value_name="score",
    )
    long["metric"] = long["metric"].map(dict(METRICS))
    methods = set(long["method"])
    snrs = [snr for snr in SNR_ORDER if snr in set(long["snr"])]
    snrs.extend(sorted(set(long["snr"]) - set(snrs), key=float))
    if not methods <= set(METHOD_ORDER):
        plot_split(long, snrs, output)
        return
    present = [method for method in METHOD_ORDER if method in methods]

    sns.set_theme(style="whitegrid", context="talk")
    grid = sns.catplot(
        data=long,
        x="metric",
        y="score",
        hue="method",
        row="snr",
        row_order=snrs,
        kind="box",
        hue_order=present,
        order=[label for _, label in METRICS],
        height=3.4,
        aspect=2.4,
        sharey=True,
        fliersize=3,
        legend_out=True,
    )
    grid.set_axis_labels("Metric", "Score")
    grid.set_titles("SNR = {row_name}")
    grid.set(ylim=(-0.05, 1.05))
    grid.figure.suptitle(
        "TIGRESS, LASSO and LSCO, with and without NestBoot",
        y=1.02,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(grid.figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        nargs="?",
        default="benchmark/plots/snr_figure.csv",
        help="Benchmark score table",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="benchmark/plots/snr_figure.png",
        help="Output figure.",
    )
    args = parser.parse_args()
    plot_scores(load_scores(Path(args.csv)), Path(args.output))
    print(args.output)


if __name__ == "__main__":
    main()
