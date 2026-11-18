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
    table = pd.read_csv(path, sep="\t")
    beside = path.with_suffix(".npz")
    if not (path.stem.endswith("_support") and beside.exists()):
        return table, None, None
    archive = dict(np.load(beside))
    run = path.with_name(path.stem.removesuffix("_support") + ".npy")
    if "links" in archive:
        links = archive["links"]
    elif run.exists():
        links = (np.load(run) != 0).sum(axis=(0, 1))
    else:
        links = None
    return table, archive, links


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


def _axes_points(ax: plt.Axes, lines: list) -> np.ndarray:
    """Points along every line in ``ax``'s axes fraction, dense enough to catch crossings."""
    to_axes = ax.transAxes.inverted()
    blocks = []
    for line in lines:
        xy = np.asarray(line.get_xydata(), dtype=float)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) == 0:
            continue
        pts = (line.get_transform() + to_axes).transform(xy)
        if len(pts) > 1:
            steps = np.linspace(0, 1, 20)[:, None, None]
            pts = (pts[:-1] + steps * (pts[1:] - pts[:-1])).reshape(-1, 2)
        blocks.append(pts)
    return np.vstack(blocks) if blocks else np.empty((0, 2))


def _free_box(points: np.ndarray, size: float, pad: tuple[float, float, float, float]) -> tuple[int, float, float]:
    """Lines crossed and lower-left corner of the clearest size x size box.

    ``pad`` (left, right, bottom, top) reserves room for the box's own tick
    labels. Ties go to the higher box, then to the one nearer the middle.
    """
    left, right, bottom, top = pad
    best = None
    for y0 in np.linspace(bottom, 1 - size - top, 30):
        for x0 in np.linspace(left, 1 - size - right, 40):
            hits = int(np.count_nonzero(
                (points[:, 0] > x0 - left) & (points[:, 0] < x0 + size + right)
                & (points[:, 1] > y0 - bottom) & (points[:, 1] < y0 + size + top)
            ))
            score = (hits, -y0, abs(x0 + size / 2 - 0.5))
            if best is None or score < best[0]:
                best = (score, x0, y0)
    return best[0][0], best[1], best[2]


SINGLE_STYLE = {
    "font.size": 15,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
}
ZOOMED = "0.86"


def _plot_single(table, zeta, output: Path, archive, links, inset: bool = True) -> None:
    """One sparsity, with an inset showing the full frequency range unless ``inset`` is False.

    The inset has the main panel's aspect ratio and sits where it crosses the
    fewest lines; grey marks the frequency range the main panel zooms into.
    The legend is above the panel.
    """
    from matplotlib.patches import Patch

    with plt.rc_context(SINGLE_STYLE):
        fig, ax = plt.subplots(figsize=(11, 8.2))
        fig.subplots_adjust(left=0.09, right=0.90, bottom=0.10, top=0.84)
        chosen, part, cutoff, reached, n_links = _slice(table, zeta, archive, links)
        freq = _draw(ax, part, cutoff, reached, label_cutoff=True)
        for line in ax.get_lines() + freq.get_lines():
            line.set_linewidth(2.2)
        ax.set_xlabel("support")
        ax.set_ylabel("overlap")
        freq.set_ylabel("frequency")
        fig.canvas.draw()

        if inset:
            points = _axes_points(ax, ax.get_lines() + freq.get_lines())
            pad = (0.075, 0.085, 0.10, 0.02)
            found = None
            for size in (0.40, 0.36, 0.32, 0.28, 0.24):
                hits, x0, y0 = _free_box(points, size, pad)
                if found is None or hits < found[0]:
                    found = (hits, x0, y0, size)
                if hits == 0:
                    break
            _, x0, y0, size = found

            inset = ax.inset_axes([x0, y0, size, size])
            inset.set_facecolor("white")
            inset_freq = _draw(inset, part, cutoff, reached, label_cutoff=False, full=True)
            zoom_top = freq.get_ylim()[1] / inset_freq.get_ylim()[1] * inset.get_ylim()[1]
            inset.axhspan(0, zoom_top, color=ZOOMED, lw=0, zorder=0)
            for axis, label in ((inset, "overlap"), (inset_freq, "frequency")):
                axis.set_ylabel(label, fontsize=13)
                axis.tick_params(labelsize=11)
            inset.set_xlabel("support", fontsize=13)

        handles = ax.get_lines() + freq.get_lines()
        if inset:
            handles.append(Patch(facecolor=ZOOMED, label="inset: range of main panel"))
        ax.legend(
            handles,
            [h.get_label() for h in handles],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=3,
            frameon=False,
            columnspacing=1.4,
            handlelength=2.2,
        )
        status = f"cutoff {cutoff:.2f}" if reached else "not FDR-controlled"
        print(f"zeta {chosen:.3g}: {status}")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        plt.close(fig)
    print(output)


def plot_support(
    table: pd.DataFrame,
    zeta: float | None,
    output: Path,
    archive: dict | None,
    links: np.ndarray | None,
    inset: bool = True,
) -> None:
    zetas = [float(value) for value in np.sort(table["zeta"].unique())]
    if zeta is not None or len(zetas) == 1:
        _plot_single(table, zetas[0] if zeta is None else zeta, output, archive, links, inset)
        return
    cols = min(3, len(zetas))
    rows = int(np.ceil(len(zetas) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.3 * rows), squeeze=False)
    freq = None
    first_freq = None
    for i, (ax, value) in enumerate(zip(axes.ravel(), zetas)):
        chosen, part, cutoff, reached, n_links = _slice(table, value, archive, links)
        freq = _draw(ax, part, cutoff, reached, label_cutoff=False)
        if first_freq is None:
            first_freq = freq
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
    fig.savefig(output, dpi=150)
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
    parser.add_argument(
        "run", type=Path,
        help="NestBoot .npy, or a _support.tsv (the _support.npz beside it is read when present)",
    )
    parser.add_argument("--zeta", type=float, default=None, help="One sparsity. Omit to plot every sparsity")
    parser.add_argument("-o", "--output", type=Path, default=Path("support.png"))
    parser.add_argument("--no-inset", action="store_true", help="With --zeta, leave out the full-range inset")
    args = parser.parse_args()
    table, archive, links = support_table(args.run)
    plot_support(table, args.zeta, args.output, archive, links, inset=not args.no_inset)


if __name__ == "__main__":
    main()
