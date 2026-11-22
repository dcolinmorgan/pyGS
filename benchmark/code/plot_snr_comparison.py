#!/usr/bin/env python3
"""Plot the SNR benchmark table as paired direct vs NestBoot violins.

Reads the CSV written by ``pygs bench``. A row may be one regularization
setting. AUROC is the threshold-free score of the setting with the best F1.
F1 and MCC are still the best values over the sweep. The axis therefore says
AUROC, not maxAUROC.

Expected columns: dataset, method, auroc, f1_score (or f1), mcc.
SNR is read from an ``snr_l`` column when every row has one, else from
``snr`` or from names such as ``SNR0.01-...``. ``pygs bench --data stored`` JSON
records the filename SNR in ``snr`` and SNR_L in ``snr_l``.

Methods named ``nestboot <method>`` are the NestBoot half of ``<method>``, and
``random <method>`` rows (gene-wise shuffled expression) are drawn as that
method's shuffled-data median.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


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
            raise SystemExit(
                f"{path} needs an snr column or a dataset name containing SNR"
            )
        frame["snr"] = (
            frame["dataset"].astype(str).str.extract(r"SNR([0-9.]+)", expand=False)
        )
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
    keys = [
        column
        for column in ("dataset", "method", "snr", "topology")
        if column in frame.columns
    ]
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
RUN_COLORS = {"direct": "#4C78A8", "NestBoot": "#E8833A"}
SHUFFLED_COLOR = "0.35"
PAIRED_METRICS = [
    ("auroc", "AUROC", 0.5),
    ("f1", "max F1", None),
    ("mcc", "max MCC", 0.0),
]


def split_runs(long: pd.DataFrame) -> pd.DataFrame:
    """Base method and run from names such as ``nestboot lsco`` or ``random lsco``.

    The run is "direct", "NestBoot", or "shuffled" for ``random`` rows.
    """
    shuffled = long["method"].str.startswith("random ")
    base = long["method"].str.removeprefix("random ")
    nested = base.str.startswith("nestboot ")
    run = np.where(shuffled, "shuffled", np.where(nested, "NestBoot", "direct"))
    return long.assign(base=base.str.removeprefix("nestboot "), run=run)


def _method_tick(name: str) -> str:
    """Two lines for long names so neighbouring labels do not cross."""
    words = name.split()
    if len(words) < 2:
        return name
    split = max(1, len(words) // 2)
    return " ".join(words[:split]) + "\n" + " ".join(words[split:])


def _row_limits(
    frame: pd.DataFrame, key: str, reference: float | None
) -> tuple[float, float, float]:
    """Data range, plus headroom inside the axes for the paired-change labels."""
    low = float(np.nanmin(frame[key]))
    bottom = min(low - 0.05, 0.0 if key != "auroc" else low - 0.05)
    top = 1.02
    head = 0.20
    return bottom, top, top + head


def draw_split(frame: pd.DataFrame, snrs: list[str]) -> plt.Figure:
    """Paired half violins. Change labels sit in headroom inside each panel.

    Rows are AUROC, max F1 and max MCC; columns are SNR_L levels. The left
    half is the direct run and the right half NestBoot, each network a point,
    with a grey line joining a network's two scores when the table names its
    dataset. The number above each method is the median paired change
    (NestBoot minus direct). A dashed line across a method is its median on
    shuffled expression. Methods are in alphabetical order.
    """
    frame = split_runs(frame)
    paired = "dataset" in frame.columns
    bases = sorted(set(frame["base"]))
    rows, cols = len(PAIRED_METRICS), len(snrs)
    # Wide enough that the longest method name ("neighborhood", "correlation")
    # fits in its own slot instead of crossing the next name.
    width = max(3.4 * cols, 1.7 * len(bases) * cols + 1.6)
    height = 3.6 * rows + 1.8
    with plt.rc_context({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12}):
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(width, height),
            sharex=True,
            sharey="row",
            squeeze=False,
        )
        rng = np.random.default_rng(0)
        for r, (key, label, reference) in enumerate(PAIRED_METRICS):
            bottom, data_top, ylim_top = _row_limits(frame, key, reference)
            for c, snr in enumerate(snrs):
                ax = axes[r, c]
                part = frame[frame["snr"] == snr]
                for i, base in enumerate(bases):
                    runs = {
                        run: part[(part["base"] == base) & (part["run"] == run)]
                        for run in RUNS
                    }
                    shuffled = part[
                        (part["base"] == base) & (part["run"] == "shuffled")
                    ][key].dropna()
                    if len(shuffled):
                        level = float(np.median(shuffled))
                        ax.plot(
                            [i - 0.4, i + 0.4],
                            [level, level],
                            color=SHUFFLED_COLOR,
                            ls="--",
                            lw=1.4,
                            zorder=4,
                        )
                    for run, side, shift in (
                        ("direct", "low", -1),
                        ("NestBoot", "high", 1),
                    ):
                        values = runs[run][key].dropna().to_numpy(dtype=float)
                        if len(values) == 0:
                            continue
                        if len(values) > 1 and np.ptp(values) > 0:
                            body = ax.violinplot(
                                values,
                                positions=[i],
                                side=side,
                                widths=0.8,
                                showextrema=False,
                            )["bodies"][0]
                            body.set_facecolor(RUN_COLORS[run])
                            body.set_edgecolor("none")
                            body.set_alpha(0.35)
                        median = float(np.median(values))
                        ax.plot(
                            [i, i + shift * 0.36],
                            [median, median],
                            color=RUN_COLORS[run],
                            lw=2.4,
                            solid_capstyle="butt",
                            zorder=4,
                        )
                    if paired and len(runs["direct"]) and len(runs["NestBoot"]):
                        both = runs["direct"].merge(
                            runs["NestBoot"], on="dataset", suffixes=("_d", "_n")
                        )
                        x_d = i - 0.17 + rng.uniform(-0.04, 0.04, len(both))
                        x_n = i + 0.17 + rng.uniform(-0.04, 0.04, len(both))
                        for xd, xn, yd, yn in zip(
                            x_d, x_n, both[f"{key}_d"], both[f"{key}_n"], strict=True
                        ):
                            ax.plot(
                                [xd, xn],
                                [yd, yn],
                                color="0.55",
                                lw=0.6,
                                alpha=0.6,
                                zorder=2,
                            )
                        ax.scatter(
                            x_d,
                            both[f"{key}_d"],
                            s=14,
                            color=RUN_COLORS["direct"],
                            edgecolor="white",
                            linewidth=0.4,
                            zorder=3,
                        )
                        ax.scatter(
                            x_n,
                            both[f"{key}_n"],
                            s=14,
                            color=RUN_COLORS["NestBoot"],
                            edgecolor="white",
                            linewidth=0.4,
                            zorder=3,
                        )
                        change = float(np.median(both[f"{key}_n"] - both[f"{key}_d"]))
                        color = (
                            "#2e7d32"
                            if change > 0.01
                            else "#c62828"
                            if change < -0.01
                            else "0.45"
                        )
                        ax.text(
                            i,
                            data_top + 0.03,
                            f"{change:+.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            color=color,
                            clip_on=True,
                            gid="delta",
                            zorder=5,
                        )
                    else:
                        for run, shift in (("direct", -0.17), ("NestBoot", 0.17)):
                            values = runs[run][key].dropna()
                            ax.scatter(
                                i + shift + rng.uniform(-0.04, 0.04, len(values)),
                                values,
                                s=14,
                                color=RUN_COLORS[run],
                                edgecolor="white",
                                linewidth=0.4,
                                zorder=3,
                            )
                if reference is not None:
                    ax.axhline(reference, color="0.6", ls="--", lw=0.9, zorder=1)
                ax.set_ylim(bottom, ylim_top)
                candidates = np.array([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ticks = candidates[
                    (candidates >= bottom - 1e-9) & (candidates <= data_top + 1e-9)
                ]
                if len(ticks):
                    ax.set_yticks(ticks)
                ax.grid(axis="y", color="0.9", lw=0.8)
                ax.set_axisbelow(True)
                for spine in ("top", "right"):
                    ax.spines[spine].set_visible(False)
                if r == 0:
                    ax.set_title(f"SNR$_L$ = {snr}", pad=8)
                    ax.title.set_gid("title")
                if c == 0:
                    ax.set_ylabel(label)
                    ax.yaxis.label.set_gid("ylabel")
            axes[r, 0].set_ylim(bottom, ylim_top)
        for ax in axes[-1]:
            ax.set_xticks(range(len(bases)))
            ax.set_xticklabels(
                [_method_tick(name) for name in bases],
                rotation=0,
                ha="center",
                fontsize=7,
            )
            ax.set_xlim(-0.6, len(bases) - 0.4)
            ax.tick_params(axis="x", pad=2)
            for tick in ax.get_xticklabels():
                tick.set_gid("xtick")
        handles = [
            Patch(facecolor=RUN_COLORS[run], alpha=0.6, label=run) for run in RUNS
        ]
        if paired:
            handles.append(
                plt.Line2D([], [], color="0.55", lw=0.8, label="same network")
            )
        if (frame["run"] == "shuffled").any():
            handles.append(
                plt.Line2D(
                    [],
                    [],
                    color=SHUFFLED_COLOR,
                    ls="--",
                    lw=1.4,
                    label="shuffled data: median",
                )
            )
        legend = fig.legend(
            handles=handles,
            loc="upper center",
            ncol=len(handles),
            frameon=False,
            bbox_to_anchor=(0.5, 0.995),
        )
        for text in legend.get_texts():
            text.set_gid("legend")
        fig.text(
            0.01,
            0.012,
            "Numbers: median paired change, NestBoot minus direct.",
            fontsize=9,
            color="0.35",
            ha="left",
            va="bottom",
            gid="footnote",
        )
        fig.subplots_adjust(
            left=0.055, right=0.99, bottom=0.12, top=0.90, hspace=0.32, wspace=0.18
        )
    return fig


def plot_split(frame: pd.DataFrame, snrs: list[str], output: Path) -> None:
    """Draw ``draw_split`` and save it."""
    fig = draw_split(frame, snrs)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def plot_scores(frame: pd.DataFrame, output: Path) -> None:
    frame = collapse_sparsity(frame)
    snrs = sorted(set(frame["snr"]), key=float)
    plot_split(frame, snrs, output)


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
