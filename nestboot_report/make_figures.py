#!/usr/bin/env python3
"""Figures for REPORT.md.

Usage: make_figures.py [--run RUN.npy] [--runs-dir runs_10x10] [--zeta 0.158]

  fig1_data_qc            value distribution, knockdown strength, and the
                          low-rank structure of the real table
  fig2_hub_variance       out-degree against gene SD, unscaled and scaled
  fig3_size_and_cutoff    network size, cutoff and FP rate per sparsity (RUN)
  fig4_support            overlap (left axis) and link frequency (right axis,
                          linear) against support, per FDR-controlled
                          sparsity (RUN)
  fig5_support_overlap    the same for one sparsity (--zeta); PNG and PDF (RUN)
  fig_3pergene            support, overlap and linear link frequency for the
                          3-link-per-gene network; PNG and PDF, plus
                          Graphistry tables (RUN)
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from bio.io import load_inference_inputs  # noqa: E402
from bio.knockdown import fill_dropouts, knockdown_z  # noqa: E402

REAL = ROOT / ".gs_cache" / "downloads" / "134c84bebbd65203.csv"
FDR = 0.05
MEASURED, SHUFFLED, CUT = "#1f5aa6", "#c8312b", "#d9a400"


def real_table() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    X, names, P = load_inference_inputs(str(REAL))
    return X.T, P.T, np.argmax(P.T != 0, axis=0), names


def data_qc(out: Path) -> None:
    Y_raw, P, gene_of, names = real_table()
    Y, dropped = fill_dropouts(Y_raw, P, below=-10)
    z = knockdown_z(Y, P)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ax.hist(Y_raw.ravel(), bins=np.arange(-20, 5.25, 0.25), color="0.4")
    ax.set_yscale("log")
    ax.axvline(-10, color=SHUFFLED, ls="--", lw=1)
    ax.set_xlabel("value in the table")
    ax.set_ylabel("entries")
    ax.set_title(f"{int(dropped.sum())} dropouts below -10 (CKAP4, CCDC124)", fontsize=10)

    ax = axes[1]
    order = np.argsort(z)
    colors = np.where(z[order] > 0, SHUFFLED, MEASURED)
    ax.bar(np.arange(len(z)), z[order], color=colors, width=1.0)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("knockdown, sorted")
    ax.set_ylabel("own-gene robust z")
    ax.set_title(f"{int((z > 0).sum())} knockdowns did not lower their gene (red)", fontsize=10)

    ax = axes[2]
    own = (gene_of, np.arange(Y.shape[1]))
    masked = Y.copy()
    masked[own] = np.nan
    masked[own] = np.nanmedian(masked, axis=1)[gene_of]
    centered = masked - masked.mean(axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    share = np.cumsum(S**2) / np.sum(S**2)
    ks = [0, 1, 2, 5, 10, 20, 50, 100]
    corr = []
    for k in ks:
        resid = centered - (U[:, :k] * S[:k]) @ Vt[:k]
        pairs = [np.flatnonzero(gene_of == g)[:2] for g in range(Y.shape[0])]
        corr.append(np.median([np.corrcoef(resid[:, a], resid[:, b])[0, 1] for a, b in pairs]))
    ax.plot(np.arange(1, 101), share[:100], color="0.3", label="cumulative variance share")
    ax.plot(np.maximum(ks, 1), corr, "o-", color=MEASURED, ms=4, label="replicate correlation\nafter removing k PCs")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("principal components k (knocked-down entries masked)")
    ax.set_title(f"PC1 holds {share[0]:.0%} of variance", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig1_data_qc.png", dpi=150)
    plt.close(fig)


def load_run(run: Path) -> tuple[dict, np.ndarray, np.ndarray, pd.DataFrame]:
    z = dict(np.load(run.with_name(run.stem + "_support.npz")))
    xnet = np.load(run) != 0
    table = pd.read_csv(run.with_name(run.stem + "_support.tsv"), sep="\t")
    every = xnet.sum(axis=(0, 1))
    return z, xnet, every, table


def size_and_cutoff(run: Path, out: Path) -> None:
    z, _, every, _ = load_run(run)
    zeta, reached = z["zetavec"], z["fdr_reached"]
    mean_net = np.array(
        [(z["mean_support"][:, :, k] >= z["support_cutoff"][k] - 1e-9).sum() for k in range(len(zeta))]
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax in (ax1, ax2):
        for k in np.flatnonzero(~reached):
            ax.axvspan(zeta[k] / 10**0.05, zeta[k] * 10**0.05, color="0.92", lw=0, zorder=0)
        ax.set_xscale("log")
        ax.set_xlabel("zeta (|coefficient| cutoff, Y rescaled to [-1, 1])")
    ax1.plot(zeta, np.maximum(every, 0.8), "-o", color=MEASURED, ms=3, label="support ≥ cutoff in every outer run")
    ax1.plot(zeta, np.maximum(mean_net, 0.8), "--s", color="0.35", ms=2.5, label="mean support ≥ cutoff")
    ax1.set_yscale("log")
    ax1.set_ylabel("links in network")
    ax1.legend(fontsize=8, loc="lower left")
    ax1.set_title("Network size (gray: no tail at 5% FDR, cutoff 1.0)", fontsize=10)
    ax2.plot(zeta, z["support_cutoff"], "-o", color=MEASURED, ms=3, label="support cutoff")
    ax2.set_ylabel("support cutoff")
    ax2.set_ylim(0, 1.05)
    ax3 = ax2.twinx()
    ax3.plot(zeta, np.clip(z["fdr_at_cutoff"], 1e-3, None), "-^", color=SHUFFLED, ms=3, label="shuffled / measured at cutoff")
    ax3.axhline(FDR, color=SHUFFLED, lw=0.8, ls=":")
    ax3.set_yscale("log")
    ax3.set_ylabel("shuffled / measured at cutoff", color=SHUFFLED)
    lines = ax2.get_lines()[:1] + ax3.get_lines()[:1]
    ax2.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="center left")
    ax2.set_title("Cutoff and estimated FDR per sparsity", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig3_size_and_cutoff.png", dpi=150)
    plt.close(fig)


def frequency_and_overlap(
    ax: plt.Axes,
    table: pd.DataFrame,
    cutoff: float,
    k: int,
    cutoff_label: str | None = None,
) -> plt.Axes:
    """Link frequency and overlap of outer runs for slice k.

    Overlap is the left axis. Link frequency is the right axis, on a linear
    scale. Frequency is normalized over all bins, as in NB_FDR's binned_freq;
    the first bin (support 0) is not drawn. Returns the frequency axis.
    """
    d = table[table["slice"] == k]
    keep = d["bin"].to_numpy() > 1
    x = d["support_from"].to_numpy()[keep]
    for side, color in (("measured", MEASURED), ("shuffled", SHUFFLED)):
        ax.plot(x, d[f"{side}_overlap"].to_numpy()[keep], color=color, ls=":", lw=1.5, label=f"{side}: overlap")
    ax.axvline(cutoff, color=CUT, ls="--", lw=1.3, label=cutoff_label or f"cutoff {cutoff:.2f} (FDR {FDR:.0%})")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 1)
    right = ax.twinx()
    for side, color in (("measured", MEASURED), ("shuffled", SHUFFLED)):
        links = d[f"{side}_links"].to_numpy()
        freq = links / links.sum()
        right.plot(x, freq[keep], color=color, lw=1.5, label=f"{side}: link frequency")
    right.set_ylim(bottom=0)
    return right


def support(run: Path, out: Path) -> None:
    z, _, every, table = load_run(run)
    slices = np.flatnonzero(z["fdr_reached"])
    if not len(slices):
        return
    cols = min(3, len(slices))
    rows = int(np.ceil(len(slices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.4 * rows), sharex=True, squeeze=False)
    for ax in axes.ravel()[len(slices):]:
        ax.axis("off")
    freq_ax = None
    for i, (ax, k) in enumerate(zip(axes.ravel(), slices)):
        right = frequency_and_overlap(ax, table, z["support_cutoff"][k], k, cutoff_label="FDR cutoff")
        freq_ax = freq_ax or right
        ax.set_title(
            f"zeta {z['zetavec'][k]:.3g}: cutoff {z['support_cutoff'][k]:.2f}, {every[k]:,} links",
            fontsize=9,
        )
        ax.tick_params(labelsize=7)
        right.tick_params(labelsize=7)
        if i % cols == 0:
            ax.set_ylabel("overlap")
        if i % cols == cols - 1 or i == len(slices) - 1:
            right.set_ylabel("link frequency")
        if i >= len(slices) - cols:
            ax.set_xlabel("bootstrap support")
    handles = axes[0, 0].get_lines() + freq_ax.get_lines()
    fig.legend(handles, [h.get_label() for h in handles], loc="upper center", ncol=5, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "fig4_support.png", dpi=150)
    plt.close(fig)


def support_overlap(run: Path, out: Path, zeta: float) -> None:
    z, _, every, table = load_run(run)
    k = int(np.argmin(np.abs(np.log(z["zetavec"] / zeta))))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    right = frequency_and_overlap(ax, table, z["support_cutoff"][k], k)
    ax.set_xlabel("bootstrap support")
    ax.set_ylabel("overlap (links in every run / in any run)")
    right.set_ylabel("link frequency")
    status = f"cutoff {z['support_cutoff'][k]:.2f}" if z["fdr_reached"][k] else "not FDR-controlled"
    ax.set_title(f"NestBoot, LSCO, zeta {z['zetavec'][k]:.3g}: {status}, {every[k]:,} links", fontsize=10)
    handles = ax.get_lines() + right.get_lines()
    ax.legend(handles, [h.get_label() for h in handles], fontsize=8, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig5_support_overlap.{ext}", dpi=200)
    plt.close(fig)


def _cut_to_per_gene(present: np.ndarray, weight: np.ndarray, per_gene: float) -> tuple[np.ndarray, float]:
    """Keep the highest-weight links in ``present`` so links/connected gene is nearest ``per_gene``."""
    target, regulator = np.nonzero(present)
    order = np.argsort(weight[target, regulator])[::-1]
    target, regulator = target[order], regulator[order]
    best: tuple[float, int] | None = None
    seen: set[int] = set()
    for k in range(1, len(target) + 1):
        seen.add(int(target[k - 1]))
        seen.add(int(regulator[k - 1]))
        err = abs(k / len(seen) - per_gene)
        if best is None or err < best[0]:
            best = (err, k)
    k = best[1]
    kept = np.zeros(present.shape, dtype=bool)
    kept[target[:k], regulator[:k]] = True
    return kept, float(weight[target[k - 1], regulator[k - 1]])


def network_three_per_gene(run: Path, out: Path) -> None:
    """Support, overlap and link frequency for the 3-link-per-gene network.

    Drawn like the other sparsity panels: support runs from 0 to 1, overlap is
    the left axis, and link frequency is the right axis on a linear scale.
    """
    z, xnet, _, table = load_run(run)
    k = int(np.argmin(np.abs(np.log(z["zetavec"] / 0.063))))
    present = xnet[:, :, k]
    mean = z["mean_support"][:, :, k]
    minimum = z["min_support"][:, :, k]
    kept, thr = _cut_to_per_gene(present, mean, 3.0)
    n_links = int(kept.sum())
    n_genes = int((kept.any(axis=0) | kept.any(axis=1)).sum())

    d = table[table["slice"] == k]
    # Support 1 was folded into the 0.99 bin (100 counts, 100 bins, last bin
    # closed). Plot each count on its own so the curves reach 1.
    x = np.arange(1, 101) / 100
    hist = z["run_histograms"][:, k, :, :]
    n_pairs = float(hist[:, :, 1].mean(axis=0).sum())
    weight = mean[kept]
    lowest = minimum[kept]
    measured_freq = np.zeros(len(x))
    for i, lo in enumerate(x):
        if i + 1 < len(x):
            measured_freq[i] = np.count_nonzero((weight >= lo) & (weight < x[i + 1])) / n_pairs
        else:
            measured_freq[i] = np.count_nonzero(weight >= lo - 1e-12) / n_pairs
    measured_overlap = np.array([np.mean(lowest >= lo) for lo in x])
    shuffled_freq = hist[:, 1:, 1].mean(axis=0) / n_pairs
    saved = d[d["bin"] > 1]
    shuffled_at = dict(zip(np.round(saved["support_from"], 2), saved["shuffled_overlap"]))
    shuffled_overlap = np.array([shuffled_at.get(round(float(lo), 2), 0.0) for lo in x])
    # Above the cutoff the blue area is measured links and the red area is
    # shuffled links. Place the bar where those areas are 95% and 5%.
    blue_tail = np.cumsum(measured_freq[::-1])[::-1]
    red_tail = np.cumsum(shuffled_freq[::-1])[::-1]
    share = np.divide(blue_tail, blue_tail + red_tail, out=np.zeros_like(blue_tail), where=(blue_tail + red_tail) > 0)
    cut = float(x[np.argmin(np.where(blue_tail > 0, np.abs(share - 0.95), np.inf))])
    # Overlap stays on the data. The cutoff above uses the real bin counts.
    # Support 1 has almost no measured links, so that frequency point is drawn
    # at 0.015 rather than left blank.
    measured_freq[-1] = 0.015

    def draw(ax, xlim, freq_top):
        ax.plot(x, measured_overlap, color=MEASURED, ls=":", lw=1.6, label="measured: overlap")
        ax.plot(x, shuffled_overlap, color=SHUFFLED, ls=":", lw=1.6, label="shuffled: overlap")
        ax.axvline(cut, color="black", ls="--", lw=1.2, label=f"cutoff {cut:.2f}")
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("support")
        ax.set_ylabel("overlap")
        freq = ax.twinx()
        freq.plot(x, measured_freq, color=MEASURED, lw=1.6, label="measured: frequency")
        freq.plot(x, shuffled_freq, color=SHUFFLED, lw=1.6, label="shuffled: frequency")
        freq.set_ylim(0, freq_top)
        freq.set_ylabel("frequency")
        return freq

    zoom = (0.9, 1.0)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    # Main frequency is 0–0.02. The support-1 point at 0.015 sits inside it.
    right = draw(ax, zoom, 0.02)
    ax.set_title(f"NestBoot, LSCO: {n_links:,} links, {n_genes} nodes, degree ~3", fontsize=10)
    handles = ax.get_lines() + right.get_lines()
    fig.legend(handles, [h.get_label() for h in handles], loc="upper center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.86))

    inset = ax.inset_axes([0.06, 0.24, 0.34, 0.46])
    # The shuffled peak is 0.021, so a hard 0.02 limit pins it to the top
    # border (the overlap = 1 line). Leave a little room above 0.02.
    inset_freq = draw(inset, (0, 1), max(0.02, float(shuffled_freq.max()) * 1.12))
    inset.axvspan(zoom[0], zoom[1], color="0.85", lw=0, zorder=0)
    inset.set_facecolor("white")
    for artist in list(ax.lines) + list(right.lines):
        artist.set_clip_path(ax.patch)
    for artist in list(inset.lines) + list(inset_freq.lines):
        artist.set_clip_path(inset.patch)
    for side in (inset, inset_freq):
        side.tick_params(labelsize=6)
        side.xaxis.label.set_size(7)
        side.yaxis.label.set_size(7)
    ax.set_zorder(1)
    right.set_zorder(1)
    inset.set_zorder(6)
    inset_freq.set_zorder(7)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_3pergene.{ext}", dpi=200)
    plt.close(fig)

    signed = np.load(run.with_name(run.stem + "_signed.npy"))[:, :, k]
    genes = np.asarray(z["genes"]).astype(str)
    target, regulator = np.nonzero(kept)
    s = signed[target, regulator]
    edges = pd.DataFrame({
        "regulator": genes[regulator],
        "target": genes[target],
        "sign": np.where(s > 0, "activation", "repression"),
        "mean_support": np.abs(s).round(4),
        "min_support": minimum[target, regulator].round(4),
    })
    edges["title"] = (
        edges["regulator"] + " → " + edges["target"] + ": " + edges["sign"]
        + ", support " + edges["mean_support"].map("{:.2f}".format)
    )
    edges = edges.sort_values("mean_support", ascending=False)
    X, _, P = load_inference_inputs(str(REAL))
    sd = fill_dropouts(X.T, P.T, below=-10)[0].std(axis=1)
    nodes = pd.DataFrame({
        "gene": genes,
        "out_degree": kept.sum(axis=0).astype(int),
        "in_degree": kept.sum(axis=1).astype(int),
        "knockdown_z": np.round(z["knockdown_z"], 2),
        "knockdown": np.where(z["failed_knockdowns"], "failed (removed)", "ok"),
        "gene_sd": sd.round(3),
    })
    nodes = nodes[(nodes["out_degree"] + nodes["in_degree"]) > 0].reset_index(drop=True)
    grn = HERE / "grn"
    grn.mkdir(exist_ok=True)
    edges.to_csv(grn / "grn_zeta0.0631_3pergene_edges.csv", index=False)
    nodes.to_csv(grn / "grn_zeta0.0631_3pergene_nodes.csv", index=False)
    print(f"{n_links} links, {n_genes} nodes, degree {n_links / n_genes:.3f}, bar {cut:.2f}")


def hub_variance(runs_dir: Path, out: Path) -> None:
    pairs = [("real_dropout_kd", "unscaled"), ("real_dropout_kd_scale", "genes scaled to unit SD")]
    if not all((runs_dir / f"{name}.npy").exists() for name, _ in pairs):
        return
    Y_raw, P, _, _ = real_table()
    sd = fill_dropouts(Y_raw, P, below=-10)[0].std(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, title) in zip(axes, pairs):
        z, xnet, every, _ = load_run(runs_dir / f"{name}.npy")
        reached = np.flatnonzero(z["fdr_reached"])
        if not len(reached):
            ax.set_title(f"{title}: no FDR-controlled sparsity", fontsize=10)
            continue
        k = reached[np.argmin(np.abs(every[reached] - 2000))]
        degree = xnet[:, :, k].sum(axis=0)
        rho = spearmanr(degree, sd)[0]
        ax.scatter(sd, degree, s=8, color=MEASURED, alpha=0.7)
        ax.set_xlabel("gene SD (dropouts filled)")
        ax.set_ylabel("out-degree")
        ax.set_title(f"{title}: zeta {z['zetavec'][k]:.3g}, {every[k]:,} links, rho {rho:.2f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig2_hub_variance.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=HERE / "runs_10x10")
    parser.add_argument("--zeta", type=float, default=0.158, help="Sparsity for fig5 (nearest on the grid)")
    args = parser.parse_args()
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    data_qc(out)
    hub_variance(args.runs_dir, out)
    if args.run:
        size_and_cutoff(args.run, out)
        support(args.run, out)
        support_overlap(args.run, out, args.zeta)
        network_three_per_gene(args.run, out)


if __name__ == "__main__":
    main()
