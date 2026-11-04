#!/usr/bin/env python3
"""Link frequency and outer-run overlap against bootstrap support.

Reads the table written beside a NestBoot run (``<stem>_support.tsv``).
Without ``--zeta`` every sparsity is drawn, one panel each. With ``--zeta``
only the nearest sparsity is drawn. Support 0 is left off the plot. Overlap
is the fraction of links seen in any outer run that are seen in every outer run.

From the pyGS root:

  uv run python nestboot_report/plot_support.py \\
      nestboot_report/runs_100x100/real.npy -o support.png

  uv run python nestboot_report/plot_support.py \\
      nestboot_report/runs_100x100/real.npy --zeta 0.158 -o support.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MEASURED, SHUFFLED, CUT = "#1f5aa6", "#c8312b", "#d9a400"


def support_table(path: Path) -> tuple[pd.DataFrame, dict | None, np.ndarray | None]:
    """Return the bin table, the support archive, and links kept at each zeta."""
    if path.suffix == ".npy":
        table = pd.read_csv(path.with_name(path.stem + "_support.tsv"), sep="\t")
        archive = dict(np.load(path.with_name(path.stem + "_support.npz")))
        links = (np.load(path) != 0).sum(axis=(0, 1))
        return table, archive, links
    return pd.read_csv(path, sep="\t"), None, None


def _slice(table: pd.DataFrame, zeta: float, archive: dict | None, links: np.ndarray | None):
    zetas = np.sort(table["zeta"].unique())
    chosen = float(zetas[np.argmin(np.abs(np.log(zetas / zeta)))])
    part = table[np.isclose(table["zeta"], chosen)].sort_values("support_from")
    if archive is not None:
        index = int(np.argmin(np.abs(np.log(archive["zetavec"] / chosen))))
        cutoff = float(archive["support_cutoff"][index])
        reached = bool(archive["fdr_reached"][index])
        n_links = None if links is None else int(links[index])
    else:
        under = part[part["tail_area_ratio"] <= 0.05]
        cutoff = float(under["support_from"].min()) if len(under) else float("nan")
        reached = bool(len(under))
        n_links = None
    return chosen, part, cutoff, reached, n_links


def _draw(
    ax: plt.Axes,
    part: pd.DataFrame,
    cutoff: float,
    reached: bool,
    label_cutoff: bool,
    full: bool = False,
) -> plt.Axes:
    # Support 0 is left off. Frequency is divided by every bin, support 0
    # included, as in NB_FDR's binned_freq. full=True scales frequency to 0–1.
    keep = part["bin"].to_numpy() > 1
    drawn = part.loc[keep]
    ax.plot(drawn["support_from"], drawn["measured_overlap"], color=MEASURED, ls=":", lw=1.5, label="measured: overlap")
    ax.plot(drawn["support_from"], drawn["shuffled_overlap"], color=SHUFFLED, ls=":", lw=1.5, label="shuffled: overlap")
    if reached:
        ax.axvline(
            cutoff, color=CUT, ls="--", lw=1.3,
            label=f"cutoff {cutoff:.2f} (FDR 5%)" if label_cutoff else "FDR cutoff",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    right = ax.twinx()
    for side, color in (("measured", MEASURED), ("shuffled", SHUFFLED)):
        counts = part[f"{side}_links"].to_numpy(dtype=float)
        frequency = counts / counts.sum() if counts.sum() else counts
        right.plot(drawn["support_from"], frequency[keep], color=color, lw=1.5, label=f"{side}: link frequency")
    if full:
        right.set_ylim(0, 1)
    else:
        right.set_ylim(bottom=0)
    return right


def plot_support(
    table: pd.DataFrame,
    zeta: float | None,
    output: Path,
    archive: dict | None,
    links: np.ndarray | None,
) -> None:
    zetas = [zeta] if zeta is not None else [float(value) for value in np.sort(table["zeta"].unique())]
    many = len(zetas) > 1
    cols = 1 if not many else min(3, len(zetas))
    rows = int(np.ceil(len(zetas) / cols))
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(11, 7.5) if not many else (4.6 * cols, 3.3 * rows),
        squeeze=False,
    )
    freq = None
    first_freq = None
    for i, (ax, value) in enumerate(zip(axes.ravel(), zetas)):
        chosen, part, cutoff, reached, n_links = _slice(table, value, archive, links)
        freq = _draw(ax, part, cutoff, reached, label_cutoff=not many)
        if first_freq is None:
            first_freq = freq
        if not many:
            # Full support range, including the 0.99–1 bin, in the region above
            # the overlap curve. The legend sits above the inset.
            inset = ax.inset_axes([0.24, 0.55, 0.52, 0.24])
            inset.set_facecolor("white")
            inset_freq = _draw(inset, part, cutoff, reached, label_cutoff=False, full=True)
            inset.set_xlabel("support", fontsize=7)
            inset.set_ylabel("overlap", fontsize=7)
            inset_freq.set_ylabel("frequency", fontsize=7)
            inset.tick_params(labelsize=6)
            inset_freq.tick_params(labelsize=6)
            handles = list(ax.get_lines()) + list(freq.get_lines())
            ax.legend(
                handles,
                [line.get_label() for line in handles],
                loc="upper center",
                bbox_to_anchor=(0.5, 1.0),
                ncol=3,
                frameon=True,
                framealpha=1,
                edgecolor="0.6",
                fontsize=7,
                borderaxespad=0.2,
            )
        status = f"cutoff {cutoff:.2f}" if reached else "not FDR-controlled"
        if i % cols == 0:
            ax.set_ylabel("overlap")
        if i % cols == cols - 1 or i == len(zetas) - 1:
            freq.set_ylabel("frequency")
        if i >= len(zetas) - cols:
            ax.set_xlabel("support")
        print(f"zeta {chosen:.3g}: {status}")
    for ax in axes.ravel()[len(zetas):]:
        ax.axis("off")
    if many:
        handles = list(axes.ravel()[0].get_lines()) + list(first_freq.get_lines())
        axes.ravel()[0].legend(
            handles,
            [line.get_label() for line in handles],
            loc="upper right",
            frameon=True,
            framealpha=1,
            edgecolor="0.6",
            fontsize=6,
        )
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.12, top=0.96)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150 if many else 200)
    plt.close(fig)
    print(output)


def plot_near_degree(table, archive, links, output: Path, per_gene: float = 3.0, band: float = 2.5) -> None:
    """Every sparsity whose FDR cutoff is reached, including the current figure."""
    current = 0.1585
    zetas = []
    notes = []
    for value in np.sort(table["zeta"].unique()):
        chosen, part, cutoff, reached, n_links = _slice(table, value, archive, links)
        if n_links is None or not reached:
            continue
        density = n_links / 232
        zetas.append(chosen)
        notes.append((density, reached, cutoff, n_links, abs(np.log(chosen / current)) < 0.05))
    cols = 5
    rows = int(np.ceil(len(zetas) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.5 * rows), squeeze=False)
    first = None
    first_freq = None
    for i, (ax, value, note) in enumerate(zip(axes.ravel(), zetas, notes)):
        density, reached, cutoff, n_links, is_current = note
        chosen, part, cutoff, reached, n_links = _slice(table, value, archive, links)
        freq = _draw(ax, part, cutoff, reached, label_cutoff=False)
        inset = ax.inset_axes([0.28, 0.50, 0.46, 0.22])
        inset.set_facecolor("white")
        _draw(inset, part, cutoff, reached, label_cutoff=False, full=True)
        inset.tick_params(labelsize=5)
        inset.set_xlabel("")
        mark = "current, " if is_current else ""
        ax.set_title(f"{mark}zeta {chosen:.3g}: {density:.1f}/gene, cutoff {cutoff:.2f}", fontsize=8)
        if is_current:
            for spine in ax.spines.values():
                spine.set_color("#d9a400")
                spine.set_linewidth(2)
        ax.set_xlabel("support")
        ax.set_ylabel("overlap")
        freq.set_ylabel("frequency")
        if first is None:
            first, first_freq = ax, freq
        print(f"zeta {chosen:.3g}: {density:.1f}/gene cutoff {cutoff:.2f}{' current' if is_current else ''}")
    for ax in axes.ravel()[len(zetas):]:
        ax.axis("off")
    handles = list(first.get_lines()) + list(first_freq.get_lines())
    fig.legend(handles, [line.get_label() for line in handles], loc="upper center", ncol=5, frameon=True, fontsize=8)
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.04, top=0.90, wspace=0.45, hspace=0.45)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=Path, help="NestBoot .npy, or the _support.tsv itself")
    parser.add_argument("--zeta", type=float, default=None, help="One sparsity. Omit to plot every sparsity")
    parser.add_argument("-o", "--output", type=Path, default=Path("support.png"))
    args = parser.parse_args()
    table, archive, links = support_table(args.run)
    plot_support(table, args.zeta, args.output, archive, links)


if __name__ == "__main__":
    main()
