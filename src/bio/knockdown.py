"""Quality control for knockdown designs before network inference.

Y and P are genes × samples. A knockdown sample has one nonzero entry in its
column of P, at the knocked-down gene; replicates share the same column.
"""

from __future__ import annotations

import numpy as np


def _groups(P: np.ndarray) -> np.ndarray:
    """Group index per sample; samples with identical P columns are replicates."""
    _, inverse = np.unique(np.asarray(P).T, axis=0, return_inverse=True)
    return np.asarray(inverse).ravel()


def fill_dropouts(
    Y: np.ndarray, P: np.ndarray, below: float
) -> tuple[np.ndarray, np.ndarray]:
    """Replace values below ``below`` and return ``(Y, mask of replaced)``.

    A replaced value takes the mean of the gene's remaining replicates of the
    same perturbation, or the gene's median over all remaining samples when
    every replicate is replaced.
    """
    Y = np.array(Y, dtype=float)
    mask = Y < below
    if not mask.any():
        return Y, mask
    groups = _groups(P)
    kept = np.where(mask, np.nan, Y)
    gene_median = np.nanmedian(kept, axis=1)
    for i, j in zip(*np.nonzero(mask)):
        replicates = kept[i, groups == groups[j]]
        replicates = replicates[~np.isnan(replicates)]
        Y[i, j] = replicates.mean() if replicates.size else gene_median[i]
    return Y, mask


def knockdown_z(Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Robust z of each gene's mean level in its own knockdowns.

    The reference is the gene's median and MAD (scaled to SD) over the samples
    that do not perturb it. Negative means the knockdown lowered its target.
    Genes without a knockdown sample get NaN.
    """
    Y = np.asarray(Y, dtype=float)
    own = np.asarray(P) != 0
    z = np.full(Y.shape[0], np.nan)
    for i in range(Y.shape[0]):
        if not own[i].any() or own[i].all():
            continue
        other = Y[i, ~own[i]]
        median = np.median(other)
        mad = 1.4826 * np.median(np.abs(other - median))
        if mad > 0:
            z[i] = (Y[i, own[i]].mean() - median) / mad
    return z


def drop_failed_knockdowns(
    Y: np.ndarray, P: np.ndarray, max_z: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove the samples of knockdowns whose ``knockdown_z`` exceeds ``max_z``.

    Returns ``(Y, P, failed)`` with ``failed`` a boolean mask over genes. A
    failed gene keeps its row: it can still regulate, but with no perturbation
    left its row of A = -P @ pinv(Y) is zero, so it gets no regulators.
    """
    z = knockdown_z(Y, P)
    failed = np.nan_to_num(z, nan=-np.inf) > max_z
    keep = ~np.any(np.asarray(P)[failed] != 0, axis=0)
    return np.asarray(Y)[:, keep], np.asarray(P)[:, keep], failed


def scale_genes(Y: np.ndarray) -> np.ndarray:
    """Divide each gene by its standard deviation, without centering.

    Values stay relative to the control, and LSCO coefficients become the
    effect of a one-SD change in the regulator: pinv(D⁻¹Y) = pinv(Y) D.
    """
    Y = np.asarray(Y, dtype=float)
    sd = Y.std(axis=1, keepdims=True)
    return Y / np.where(sd > 0, sd, 1.0)
