"""Perturbation Z-score GRN inference.

This is the Z-score method used in the DREAM and GeneSPIDER benchmarks, not a
rescaling of another method's scores. For each perturbed gene, every gene's
expression in those experiments is compared with its background (the
experiments where that gene was not perturbed). The resulting z-score is the
weight of the link from the perturbed gene.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from datastruct.Dataset import Dataset


def perturbation_zscores(Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Signed perturbation z-scores, shape (genes, genes).

    Column ``j`` is the effect of perturbing gene ``j``. ``Y`` and ``P`` are
    both genes × experiments. The diagonal is left at 0.
    """
    expression = np.asarray(Y, dtype=float)
    design = np.asarray(P, dtype=float)
    if expression.ndim != 2 or expression.shape != design.shape:
        raise ValueError("Y and P must be genes × experiments and share a shape")
    n_genes = expression.shape[0]
    scores = np.zeros((n_genes, n_genes), dtype=float)
    for target in range(n_genes):
        perturbed = np.abs(design[target]) > 0
        if not np.any(perturbed):
            continue
        background = ~perturbed
        n_background = int(np.count_nonzero(background))
        if n_background >= 2:
            center = expression[:, background].mean(axis=1)
            scale = expression[:, background].std(axis=1, ddof=1)
        elif n_background == 1:
            center = expression[:, background].reshape(-1)
            scale = np.ones(n_genes)
        else:
            center = np.median(expression, axis=1)
            scale = expression.std(axis=1, ddof=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
        effect = expression[:, perturbed].mean(axis=1)
        scores[:, target] = (effect - center) / scale
    np.fill_diagonal(scores, 0.0)
    return scores


def ZScore(
    dataset: Dataset | Any,
    threshold_range: np.ndarray | list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray | float]:
    """Infer a network from perturbation z-scores.

    Without ``threshold_range`` the continuous z-score matrix is returned, which
    is what a threshold-free score (AUROC) needs. With a range, each value is a
    fraction of the largest |z| and that fraction of links is kept, the same
    zeta sweep GeneSPIDER runs.
    """
    if hasattr(dataset, "Y"):
        expression, design = dataset.Y, dataset.P
    elif hasattr(dataset, "data") and dataset.data is not None:
        expression, design = dataset.data.Y, dataset.data.P
    else:
        raise ValueError("Dataset must contain Y and P matrices")
    if expression is None or design is None:
        raise ValueError("Dataset must contain Y and P matrices")

    scores = perturbation_zscores(np.asarray(expression), np.asarray(design))
    if threshold_range is None:
        return scores, float(np.mean(scores**2))

    fractions = np.asarray(threshold_range, dtype=float)
    largest = float(np.max(np.abs(scores)))
    n_genes = scores.shape[0]
    stack = np.zeros((n_genes, n_genes, len(fractions)))
    cutoffs = fractions * largest
    for index, cutoff in enumerate(cutoffs):
        kept = scores.copy()
        kept[np.abs(kept) < cutoff] = 0.0
        stack[:, :, index] = kept
    return stack, cutoffs
