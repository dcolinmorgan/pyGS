"""GeneSPIDER-style synthetic networks and expression.

One replicate is one network plus the expression matrix simulated from it.
``--replicates N`` draws N of those, at seeds ``seed``, ``seed + 1``, ...
Every replicate uses the same size, topology, sparsity and SNR_L. Inside a
replicate, each gene is knocked down ``experiments_per_gene`` times (the
default 3 matches the benchmark).

SNR_L is GeneSPIDER's definition (smallest signal singular value over the
noise scale). It is not the filename number: the stored files call SNR_L 1
``SNR1000``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bench.snr_benchmark import (
    knockdown_design,
    simulate_expression,
    snr_l,
    stable_network,
)


def generate_replicates(
    *,
    n_genes: int,
    topology: str,
    sparsity: float,
    snr: float,
    seed: int,
    replicates: int = 1,
    experiments_per_gene: int = 3,
) -> list[dict[str, Any]]:
    """``replicates`` independent networks and their expression at ``snr``.

    Each item has ``A`` (genes × genes), ``Y`` and ``P`` (genes × experiments),
    ``X`` (experiments × genes, the orientation the CLI writes) and ``snr_l``,
    the SNR_L the noise was scaled to.
    """
    if replicates < 1:
        raise ValueError("replicates must be at least 1")
    if experiments_per_gene < 1:
        raise ValueError("experiments_per_gene must be at least 1")
    if snr <= 0:
        raise ValueError("SNR_L must be positive")
    n_samples = experiments_per_gene * n_genes
    drawn: list[dict[str, Any]] = []
    for index in range(replicates):
        replicate_seed = seed + index
        network = stable_network(n_genes, topology, sparsity, replicate_seed)
        design = knockdown_design(n_genes, n_samples)
        expression = simulate_expression(
            network,
            design,
            snr,
            np.random.default_rng(replicate_seed + 1),
        )
        residual = expression - ((-np.linalg.pinv(network)) @ design)
        measured = snr_l(network, design, float(np.var(residual)))
        drawn.append(
            {
                "A": network,
                "Y": expression,
                "P": design,
                "X": expression.T,
                "snr": float(snr),
                "snr_l": measured,
                "seed": replicate_seed,
                "topology": topology,
                "replicate": index + 1,
            }
        )
    return drawn
