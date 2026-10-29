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
