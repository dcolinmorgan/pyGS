"""Edge and node tables from a NestBoot ``.npy`` stack.

``pygs nestboot`` writes three files:

* ``out.npy`` — binary network, genes × genes × zeta
* ``out_signed.npy`` — the same links with sign
* ``out_support.npz`` — zeta grid, support, gene names, knockdown flags

Row ``i``, column ``j`` is the link regulator ``j`` → target ``i``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _cut_to_per_gene(
    present: np.ndarray, weight: np.ndarray, per_gene: float
) -> np.ndarray:
    """Keep the highest-weight links nearest ``per_gene`` links per gene."""
    target, regulator = np.nonzero(present)
    order = np.argsort(weight[target, regulator])[::-1]
    target, regulator = target[order], regulator[order]
    best: tuple[float, int] | None = None
    seen: set[int] = set()
    for count in range(1, len(target) + 1):
        seen.add(int(target[count - 1]))
        seen.add(int(regulator[count - 1]))
        error = abs(count / len(seen) - per_gene)
        if best is None or error < best[0]:
            best = (error, count)
    if best is None:
        return present
    kept = np.zeros(present.shape, dtype=bool)
    kept[target[: best[1]], regulator[: best[1]]] = True
    return kept


def nestboot_tables(
    run: Path,
    zeta: float,
    per_gene: float | None = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Edges and nodes for the sparsity nearest ``zeta``.

    ``per_gene`` keeps the highest-support links until the density is nearest
    that many links per connected gene. ``None`` keeps the full FDR network.
    """
    run = Path(run)
    support_path = run.with_name(run.stem + "_support.npz")
    signed_path = run.with_name(run.stem + "_signed.npy")
    if not support_path.exists() or not signed_path.exists():
        raise FileNotFoundError(
            f"{run.name} needs {support_path.name} and {signed_path.name} beside it"
        )

    support = np.load(support_path)
    binary = np.load(run) != 0
    signed = np.load(signed_path)
    index = int(np.argmin(np.abs(np.log(support["zetavec"] / zeta))))
    genes = np.asarray(support["genes"]).astype(str)
    present = binary[:, :, index]
    if per_gene:
        present = _cut_to_per_gene(present, support["mean_support"][:, :, index], per_gene)

    target, regulator = np.nonzero(present)
    sign = signed[target, regulator, index]
    edges = pd.DataFrame({
        "regulator": genes[regulator],
        "target": genes[target],
        "sign": np.where(sign > 0, "activation", "repression"),
        "mean_support": np.abs(sign).round(4),
        "min_support": support["min_support"][target, regulator, index].round(4),
    })
    edges["title"] = (
        edges["regulator"] + " → " + edges["target"] + ": " + edges["sign"]
        + ", support " + edges["mean_support"].map("{:.2f}".format)
    )
    nodes = pd.DataFrame({
        "gene": genes,
        "out_degree": present.sum(axis=0),
        "in_degree": present.sum(axis=1),
        "knockdown_z": np.round(support["knockdown_z"], 2),
        "knockdown": np.where(support["failed_knockdowns"], "failed (removed)", "ok"),
    })
    nodes = nodes[(nodes["out_degree"] + nodes["in_degree"]) > 0].reset_index(drop=True)
    info = {
        "zeta": float(support["zetavec"][index]),
        "cutoff": float(support["support_cutoff"][index]),
        "fdr_reached": bool(support["fdr_reached"][index]),
        "links": len(edges),
    }
    return edges, nodes, info
