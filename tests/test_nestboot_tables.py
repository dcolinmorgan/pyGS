"""NestBoot array to edge and node tables."""

import numpy as np
import pandas as pd

from bio.nestboot_tables import nestboot_tables


def test_tables_keep_the_highest_support_links(tmp_path):
    genes = np.array(["A", "B", "C"])
    binary = np.zeros((3, 3, 1))
    signed = np.zeros((3, 3, 1))
    mean = np.zeros((3, 3, 1))
    minimum = np.zeros((3, 3, 1))
    # Column j regulates row i. B→A support 0.9, C→A support 0.2, A→B support 0.5.
    for target, regulator, weight in ((0, 1, 0.9), (0, 2, 0.2), (1, 0, -0.5)):
        binary[target, regulator, 0] = 1
        signed[target, regulator, 0] = weight
        mean[target, regulator, 0] = abs(weight)
        minimum[target, regulator, 0] = abs(weight) - 0.1
    run = tmp_path / "out.npy"
    np.save(run, binary)
    np.save(tmp_path / "out_signed.npy", signed)
    np.savez(
        tmp_path / "out_support.npz",
        zetavec=np.array([0.063]),
        support_cutoff=np.array([0.94]),
        fdr_reached=np.array([True]),
        mean_support=mean,
        min_support=minimum,
        genes=genes,
        knockdown_z=np.array([-1.0, -2.0, 0.4]),
        failed_knockdowns=np.array([False, False, True]),
    )

    edges, nodes, info = nestboot_tables(run, zeta=0.1, per_gene=1.0)

    assert info["zeta"] == 0.063
    # Two strongest links are B→A and A→B: two links on two genes is exactly 1 per gene.
    assert info["links"] == 2
    kept = set(zip(edges.regulator, edges.target))
    assert kept == {("B", "A"), ("A", "B")}
    assert set(nodes["gene"]) == {"A", "B"}
    assert edges.set_index(["regulator", "target"]).loc[("A", "B"), "sign"] == "repression"


def _two_zeta_run(tmp_path):
    genes = np.array(["A", "B", "C"])
    binary = np.zeros((3, 3, 2))
    binary[0, 1, :] = 1
    binary[1, 0, 1] = 1
    signed = binary * 0.9
    run = tmp_path / "run.npy"
    np.save(run, binary)
    np.save(tmp_path / "run_signed.npy", signed)
    np.savez(
        tmp_path / "run_support.npz",
        zetavec=np.array([0.01, 0.1]),
        support_cutoff=np.array([1.0, 0.8]),
        fdr_reached=np.array([False, True]),
        mean_support=np.abs(signed),
        min_support=np.abs(signed),
        genes=genes,
        knockdown_z=np.zeros(3),
        failed_knockdowns=np.zeros(3, dtype=bool),
    )
    return run


def test_all_tables_keep_only_fdr_controlled_sparsities(tmp_path):
    from bio.nestboot_tables import nestboot_tables_all, write_table

    edges, nodes = nestboot_tables_all(_two_zeta_run(tmp_path))
    assert set(edges["zeta"]) == {0.1}
    assert list(edges.columns[:2]) == ["zeta", "support_cutoff"]
    assert set(zip(edges.regulator, edges.target)) == {("B", "A"), ("A", "B")}
    assert set(nodes["gene"]) == {"A", "B"}

    write_table(edges, tmp_path / "edges.tsv")
    assert (tmp_path / "edges.tsv").read_text().splitlines()[0].startswith("zeta\tsupport_cutoff\t")


def test_nestboot_output_names_tables_and_arrays(tmp_path):
    from pathlib import Path

    import pytest

    from bench.cli import _nestboot_outputs

    assert _nestboot_outputs("out.npy") == (Path("out.npy"), None, None)
    assert _nestboot_outputs("res/edges.tsv") == (
        Path("res/edges.npy"), Path("res/edges.tsv"), Path("res/edges_nodes.tsv")
    )
    assert _nestboot_outputs("edges.tsv", "nodes.tsv")[2] == Path("nodes.tsv")
    with pytest.raises(ValueError):
        _nestboot_outputs("out.npy", "nodes.tsv")
