"""Edge and node tables from a NestBoot ``.npy`` stack.

``pygs nestboot`` writes three files:

* ``out.npy`` — binary network, genes × genes × zeta
* ``out_signed.npy`` — the same links with sign
* ``out_support.npz`` — zeta grid, support, gene names, knockdown flags

Row ``i``, column ``j`` is the link regulator ``j`` → target ``i``.

``pygs nestboot --edges edges.tsv nodes.tsv`` writes the tables at the end of
a run; ``pygs plot out.npy --zeta Z -o edges.tsv`` writes them for one
sparsity of a finished run.
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
    edges["title"] = [
        f"{reg} → {tgt}: {sgn}, support {sup:.2f}"
        for reg, tgt, sgn, sup in zip(
            edges["regulator"], edges["target"], edges["sign"], edges["mean_support"]
        )
    ]
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


def nestboot_tables_all(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Edges and nodes of the full FDR network at every FDR-controlled sparsity.

    One block of rows per sparsity, marked by ``zeta`` and ``support_cutoff``
    columns. Sparsities where no support cutoff meets the FDR are left out.
    """
    support = np.load(Path(run).with_name(Path(run).stem + "_support.npz"))
    edge_blocks, node_blocks = [], []
    for zeta, reached in zip(support["zetavec"], support["fdr_reached"]):
        if not reached:
            continue
        edges, nodes, info = nestboot_tables(run, float(zeta), per_gene=None)
        for block in (edges, nodes):
            block.insert(0, "support_cutoff", round(info["cutoff"], 4))
            block.insert(0, "zeta", float(f"{info['zeta']:.4g}"))
        edge_blocks.append(edges)
        node_blocks.append(nodes)
    if not edge_blocks:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(edge_blocks, ignore_index=True), pd.concat(node_blocks, ignore_index=True)


def write_table(frame: pd.DataFrame, path: Path) -> None:
    """Tab-separated for ``.tsv`` and ``.txt``, comma-separated otherwise."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    frame.to_csv(path, sep=sep, index=False)


def nodes_path_for(edges_path: Path) -> Path:
    """``edges.tsv`` → ``edges_nodes.tsv``."""
    edges_path = Path(edges_path)
    return edges_path.with_name(edges_path.stem + "_nodes" + edges_path.suffix)
