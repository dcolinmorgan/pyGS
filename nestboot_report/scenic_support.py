#!/usr/bin/env python3
"""NestBoot support, FDR cutoff and outer-run overlap for SCENIC+ bootstrap runs.

Usage:

  uv run python nestboot_report/scenic_support.py \\
      benchmark/cat_scenic.gz benchmark/cat_null_scenic.gz \\
      -o nestboot_report/runs_scenic/scenic
  uv run python nestboot_report/plot_support.py \\
      nestboot_report/runs_scenic/scenic_support.tsv -o scenic_support.png

Each input stacks SCENIC+ runs as rows of ``TF, Gene, importance_x_rho,
init``, with ``init`` naming the run (``run_55``); the second input is the
same pipeline on gene-shuffled data. A link listed more than once in a run
keeps its largest |importance_x_rho| and that entry's sign. Runs are sorted by
number and the first ``outer x inner`` of each input are used, ``inner``
consecutive runs per outer run. At each zeta a link is present in a run when
|importance_x_rho| > zeta. Support, the cutoff, the FDR curve and the overlap
then follow GeneSPIDER's NB_FDR as in ``pygs nestboot``.

Writes ``<out>_support.tsv`` and ``<out>_support.npz``, the files
``plot_support.py`` reads, and ``<out>_edges.tsv``: links kept at every
FDR-controlled zeta, with sign, mean support and minimum support.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bench.cli import _support_rows  # noqa: E402
from methods.nestboot import Nestboot  # noqa: E402


def load_runs(path: Path) -> pd.DataFrame:
    """One row per (run, TF, Gene), the entry with the largest |weight|."""
    table = pd.read_csv(path, usecols=["Gene", "TF", "importance_x_rho", "init"])
    table["run"] = table["init"].str.extract(r"(\d+)", expand=False).astype(int)
    table["size"] = table["importance_x_rho"].abs()
    table = table.sort_values("size", ascending=False, kind="stable")
    return table.drop_duplicates(["run", "TF", "Gene"])[["run", "TF", "Gene", "importance_x_rho"]]


def weight_matrix(table: pd.DataFrame, runs: list[int], pairs: pd.Index) -> np.ndarray:
    """Signed weights, runs x links, 0 where a run does not list the link."""
    keys = pd.MultiIndex.from_arrays([table["TF"], table["Gene"]])
    column = pairs.get_indexer(keys)
    row = pd.Index(runs).get_indexer(table["run"])
    use = row >= 0
    weights = np.zeros((len(runs), len(pairs)), dtype=np.float32)
    weights[row[use], column[use]] = table["importance_x_rho"].to_numpy()[use]
    return weights


def nested_support(
    weights: tuple[np.ndarray, np.ndarray], zeta: np.ndarray, n_outer: int, n_inner: int, universe: int
) -> tuple[np.ndarray, ...]:
    """``Nestboot._nested_fdr`` inputs over the listed links, measured then shuffled."""
    n_links = weights[0].shape[1]
    run_hist = np.zeros((n_outer, len(zeta), n_inner + 1, 2))
    lo = np.zeros((2, n_links, 1, len(zeta)), dtype=np.int16)
    hi = np.zeros_like(lo)
    total = np.zeros((n_links, 1, len(zeta)), dtype=np.int32)
    positive = np.zeros_like(total)
    for side, matrix in enumerate(weights):
        size = np.abs(matrix)
        for p, cut in enumerate(zeta):
            present = size > cut
            counts = present.reshape(n_outer, n_inner, n_links).sum(axis=1)
            for r in range(n_outer):
                run_hist[r, p, :, side] = np.bincount(counts[r], minlength=n_inner + 1)
            lo[side, :, 0, p] = counts.min(axis=0)
            hi[side, :, 0, p] = counts.max(axis=0)
            if side == 0:
                total[:, 0, p] = present.sum(axis=0)
                positive[:, 0, p] = (present & (matrix > 0)).sum(axis=0)
    # Links neither input lists have support 0 in every run.
    run_hist[:, :, 0, :] += universe - n_links
    return run_hist, lo, hi, total, positive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("measured", type=Path, help="SCENIC+ runs on the measured data (.csv or .gz)")
    parser.add_argument("shuffled", type=Path, help="SCENIC+ runs on gene-shuffled data")
    parser.add_argument("--outer", type=int, default=6, help="Outer runs (default 6)")
    parser.add_argument("--inner", type=int, default=8, help="Runs per outer run (default 8)")
    parser.add_argument(
        "--zeta", default="-4:-1:31",
        help="|importance_x_rho| cutoffs LO:HI:N, logspace(LO, HI, N). Write --zeta=-4:-1:31",
    )
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("-o", "--output", type=Path, default=Path("nestboot_report/runs_scenic/scenic"))
    args = parser.parse_args()

    lo_exp, hi_exp, n_zeta = args.zeta.split(":")
    zeta = np.logspace(float(lo_exp), float(hi_exp), int(n_zeta))
    needed = args.outer * args.inner

    tables = [load_runs(args.measured), load_runs(args.shuffled)]
    chosen = []
    for name, table in zip(("measured", "shuffled"), tables):
        runs = sorted(table["run"].unique())
        if len(runs) < needed:
            raise SystemExit(f"{name}: {len(runs)} runs, {args.outer} x {args.inner} needs {needed}")
        chosen.append(runs[:needed])
        print(f"{name}: {len(runs)} runs, using {needed} (runs {runs[0]}-{runs[needed - 1]})")

    pairs = pd.MultiIndex.from_frame(
        pd.concat(tables)[["TF", "Gene"]].drop_duplicates().sort_values(["TF", "Gene"])
    )
    tfs = set(pairs.get_level_values(0))
    genes = set(pairs.get_level_values(1))
    universe = len(tfs) * len(genes) - len(tfs & genes)
    mask = np.asarray(pairs.get_level_values(0) != pairs.get_level_values(1))[:, None]
    print(f"{len(pairs)} listed links of {universe} possible ({len(tfs)} TFs x {len(genes)} genes)")

    weights = tuple(weight_matrix(t, r, pairs) for t, r in zip(tables, chosen))
    run_hist, lo, hi, total, positive = nested_support(weights, zeta, args.outer, args.inner, universe)
    results = Nestboot()._nested_fdr(
        run_hist, lo, hi, total, positive, args.outer, args.inner, args.fdr, mask=mask
    )

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    reached = np.any(results.fdr_curve[:, :-1] <= args.fdr, axis=1)
    kept = np.asarray(results.xnet, dtype=bool)[:, 0, :]
    np.savez_compressed(
        out.with_name(out.stem + "_support.npz"),
        zetavec=zeta,
        support_cutoff=np.asarray(results.support, dtype=float),
        fdr_at_cutoff=np.asarray(results.fp_rate, dtype=float),
        fdr_curve=results.fdr_curve,
        fdr_reached=reached,
        run_histograms=run_hist,
        links=kept.sum(axis=0),
        tf=pairs.get_level_values(0).to_numpy(dtype=str),
        target=pairs.get_level_values(1).to_numpy(dtype=str),
        mean_support=np.asarray(results.mean_support, dtype=np.float32)[:, 0, :],
        min_support=np.asarray(results.min_ab, dtype=np.float32)[:, 0, :],
        runs_measured=np.asarray(chosen[0]),
        runs_shuffled=np.asarray(chosen[1]),
    )
    pd.DataFrame(_support_rows(results, zeta)).to_csv(
        out.with_name(out.stem + "_support.tsv"), sep="\t", index=False
    )
    edges = []
    for p in np.flatnonzero(reached):
        index = np.flatnonzero(kept[:, p])
        edges.append(pd.DataFrame({
            "zeta": zeta[p],
            "support_cutoff": results.support[p],
            "tf": pairs.get_level_values(0)[index],
            "target": pairs.get_level_values(1)[index],
            "sign": np.where(np.asarray(results.ssum)[index, 0, p] > 0, "activation", "repression"),
            "mean_support": np.asarray(results.mean_support)[index, 0, p],
            "min_support": np.asarray(results.min_ab)[index, 0, p],
        }))
    edge_table = pd.concat(edges) if edges else pd.DataFrame()
    edge_table.to_csv(out.with_name(out.stem + "_edges.tsv"), sep="\t", index=False)

    for p, cut in enumerate(zeta):
        status = f"cutoff {results.support[p]:.2f}, {int(kept[:, p].sum())} links" if reached[p] else "not FDR-controlled"
        print(f"zeta {cut:.3g}: {status}")
    print(f"Wrote {out.stem}_support.tsv, {out.stem}_support.npz and {out.stem}_edges.tsv in {out.parent}")


if __name__ == "__main__":
    main()
