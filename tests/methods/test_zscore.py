"""Perturbation Z-score inference, the DREAM / GeneSPIDER method."""

import numpy as np

from methods.zscore import ZScore, perturbation_zscores


def _knockdown(n_genes: int = 6, replicates: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Gene j knocked down in its own block, and gene j+1 responds there only."""
    experiments = n_genes * replicates
    expression = np.zeros((n_genes, experiments))
    design = np.zeros((n_genes, experiments))
    rng = np.random.default_rng(0)
    expression += rng.normal(0, 0.05, expression.shape)
    for gene in range(n_genes):
        columns = slice(gene * replicates, (gene + 1) * replicates)
        design[gene, columns] = -1.0
        expression[gene, columns] -= 1.0
        expression[(gene + 1) % n_genes, columns] += 3.0
    return expression, design


def test_perturbation_zscore_points_at_the_responding_gene():
    expression, design = _knockdown()
    scores = perturbation_zscores(expression, design)
    assert scores.shape == (6, 6)
    assert np.all(np.diag(scores) == 0)
    # Column 0 is the perturbation of gene 0. Gene 1 is the one that moves.
    assert int(np.argmax(scores[:, 0])) == 1
    assert scores[1, 0] > 5


def test_zscore_threshold_range_sparsifies():
    expression, design = _knockdown()

    class _Data:
        Y = expression
        P = design

    dense, _ = ZScore(_Data())
    stack, cutoffs = ZScore(_Data(), threshold_range=[0.0, 0.5, 1.0])
    assert dense.shape == (6, 6)
    assert stack.shape == (6, 6, 3)
    assert len(cutoffs) == 3
    assert np.count_nonzero(stack[:, :, 0]) >= np.count_nonzero(stack[:, :, -1])
