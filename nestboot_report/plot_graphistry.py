#!/usr/bin/env python3
"""Export one sparsity of a NestBoot network as edge and node tables and plot it in Graphistry.

Usage, from the pyGS root:
  uv run --with graphistry python nestboot_report/plot_graphistry.py [--zeta 0.063] [--per-gene 3] [--run RUN.npy] [--no-plot]

Writes <out>/grn_zeta<zeta>_3pergene_edges.csv (regulator, target, sign, mean and
minimum support) and _nodes.csv (degrees, knockdown z, gene SD). --per-gene 3
keeps the highest-support links of that sparsity until the network is nearest
3 links per connected gene (678 links at zeta 0.063). --per-gene 0 keeps the
full FDR network.
To upload, export GRAPHISTRY_USERNAME (the password is prompted for unless
GRAPHISTRY_PASSWORD is set), or GRAPHISTRY_PERSONAL_KEY_ID and
GRAPHISTRY_PERSONAL_KEY_SECRET; GRAPHISTRY_SERVER defaults to
hub.graphistry.com. The plot URL is printed.
"""

import argparse
import getpass
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from bio.io import load_inference_inputs  # noqa: E402
from bio.knockdown import fill_dropouts  # noqa: E402

REAL = ROOT / ".gs_cache" / "downloads" / "134c84bebbd65203.csv"
ACTIVATION, REPRESSION = "#2e7d32", "#c62828"


def tables(run: Path, zeta: float, per_gene: float | None = 3.0) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Edges and nodes for the sparsity nearest zeta, plus gene SD from the K562 table."""
    from bio.nestboot_tables import nestboot_tables

    edges, nodes, info = nestboot_tables(run, zeta, per_gene)
    X, names, P = load_inference_inputs(str(REAL))
    sd = pd.Series(fill_dropouts(X.T, P.T, below=-10)[0].std(axis=1), index=names)
    nodes = nodes.copy()
    nodes["gene_sd"] = nodes["gene"].map(sd).round(3)
    info["per_gene"] = per_gene
    return edges, nodes, info


def register() -> bool:
    import graphistry

    server = os.environ.get("GRAPHISTRY_SERVER", "hub.graphistry.com")
    if os.environ.get("GRAPHISTRY_PERSONAL_KEY_ID"):
        graphistry.register(
            api=3, server=server, protocol="https",
            personal_key_id=os.environ["GRAPHISTRY_PERSONAL_KEY_ID"],
            personal_key_secret=os.environ["GRAPHISTRY_PERSONAL_KEY_SECRET"],
        )
    elif os.environ.get("GRAPHISTRY_USERNAME"):
        username = os.environ["GRAPHISTRY_USERNAME"]
        password = os.environ.get("GRAPHISTRY_PASSWORD") or getpass.getpass(f"Graphistry password for {username}: ")
        graphistry.register(api=3, server=server, protocol="https", username=username, password=password)
    else:
        return False
    return True


def plotter(edges: pd.DataFrame, nodes: pd.DataFrame):
    import graphistry

    return (
        graphistry.edges(edges, "regulator", "target")
        .nodes(nodes, "gene")
        .bind(point_title="gene", edge_title="title")
        .encode_edge_color("sign", categorical_mapping={"activation": ACTIVATION, "repression": REPRESSION})
        .encode_point_color("knockdown", categorical_mapping={"ok": "#1f5aa6", "failed (removed)": "#f39c12"})
        .encode_point_size("out_degree")
        .settings(url_params={"play": 3000, "pointSize": 0.6})
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, default=HERE / "runs_100x100" / "real.npy")
    parser.add_argument("--zeta", type=float, default=0.063, help="Sparsity (nearest on the grid)")
    parser.add_argument("--per-gene", type=float, default=3.0, help="Links per gene to keep (0 = full FDR network)")
    parser.add_argument("--out", type=Path, default=HERE / "grn")
    parser.add_argument("--no-plot", action="store_true", help="Write the tables only")
    args = parser.parse_args()

    per_gene = args.per_gene if args.per_gene and args.per_gene > 0 else None
    edges, nodes, info = tables(args.run, args.zeta, per_gene)
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "_3pergene" if per_gene else ""
    stem = args.out / f"grn_zeta{info['zeta']:.3g}{suffix}"
    edges.to_csv(f"{stem}_edges.csv", index=False)
    nodes.to_csv(f"{stem}_nodes.csv", index=False)
    status = f"cutoff {info['cutoff']:.2f}" if info["fdr_reached"] else "not FDR-controlled"
    degree = info["links"] / len(nodes)
    print(f"zeta {info['zeta']:.3g}: {info['links']} links, {len(nodes)} genes, {degree:.2f} per gene, {status}")
    print(f"wrote {stem}_edges.csv and {stem}_nodes.csv")

    if args.no_plot:
        return
    if not register():
        print("No Graphistry credentials in the environment; see the docstring. Tables only.")
        return
    if per_gene:
        name = f"NestBoot GRN, ENCODE K562, LSCO zeta {info['zeta']:.3g}, {degree:.0f} links per gene"
        description = (
            f"{info['links']} highest-support links ({degree:.1f} per gene, {len(nodes)} genes) "
            f"from the zeta {info['zeta']:.3g} network, {status}, 100 × 100 nested bootstrap"
        )
    else:
        name = f"NestBoot GRN, ENCODE K562, LSCO zeta {info['zeta']:.3g}"
        description = f"{info['links']} links, {status}, FDR 5%, 100 × 100 nested bootstrap"
    url = plotter(edges, nodes).plot(name=name, description=description, render="url")
    print(url)


if __name__ == "__main__":
    main()
