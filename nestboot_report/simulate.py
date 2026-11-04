#!/usr/bin/env python3
"""Simulate a knockdown dataset with a known network and the real design.

GeneSPIDER steady-state model Y = -A⁻¹P + C + E, with A = W - I for a stable
scale-free W (about 3 regulators per gene, spectral radius 0.8), 2 replicates
per knockdown and P = -strength at the knocked-down gene. Simulated gene i
stands in for real gene i. Knockdown strength is set so a gene's own shift
matches the real median shift (mean in its knockdowns minus median
elsewhere), and the noise SD so the median knockdown z
(bio.knockdown.knockdown_z) equals the real data's.

Scenarios:
  ideal       C = 0, every knockdown at the median strength
  confounded  C = the real data's top 65 principal components (own-gene
              entries masked), on the same genes and samples: shared
              structure that no network link produced. 65 components make
              the replicate correlation match the real data's
  realistic   confounded, and each knockdown has its real gene's own shift;
              the 14 real knockdowns with z > 0 have no effect

The confounder carries the real genes' unequal variances, so the noise SD is
the same for every gene.

Writes <out>/<scenario>.csv in the real table's layout (genes × samples,
columns named by the knocked-down gene), <scenario>_true.npy (W, target ×
regulator) and <scenario>_truth.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bio.io import load_inference_inputs  # noqa: E402
from bio.knockdown import fill_dropouts, knockdown_z  # noqa: E402
from sparselink.bench.synthetic import generate_network  # noqa: E402

REAL = ROOT / ".gs_cache" / "downloads" / "134c84bebbd65203.csv"
N_PCS = 65


def replicate_correlation(Y: np.ndarray, gene_of: np.ndarray) -> float:
    """Median correlation of replicate profiles, leaving out the knocked-down gene."""
    corr = []
    for g in np.unique(gene_of):
        a, b = np.flatnonzero(gene_of == g)[:2]
        keep = np.arange(Y.shape[0]) != g
        corr.append(np.corrcoef(Y[keep, a], Y[keep, b])[0, 1])
    return float(np.median(corr))


def real_data() -> tuple[np.ndarray, np.ndarray]:
    """Real Y (dropouts filled) with samples ordered gene by gene, replicates adjacent."""
    X, _, P = load_inference_inputs(str(REAL))
    Y, _ = fill_dropouts(X.T, P.T, below=-10)
    gene_of = np.argmax(P.T != 0, axis=0)
    order = np.argsort(gene_of, kind="stable")
    return Y[:, order], gene_of[order]


def low_rank(Y: np.ndarray, gene_of: np.ndarray, k: int) -> np.ndarray:
    """Top-k principal components of Y with the knocked-down entries masked."""
    masked = Y.copy()
    own = (gene_of, np.arange(Y.shape[1]))
    masked[own] = np.nan
    masked[own] = np.nanmedian(masked, axis=1)[gene_of]
    centered = masked - masked.mean(axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return (U[:, :k] * S[:k]) @ Vt[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scenario", choices=["ideal", "confounded", "realistic"], required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "synthetic")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    Y_real, gene_of = real_data()
    n = Y_real.shape[0]
    P_label = np.zeros((n, 2 * n))
    P_label[gene_of, np.arange(2 * n)] = -1.0
    target_z = float(np.nanmedian(knockdown_z(Y_real, P_label)))

    own = P_label != 0
    shift = np.array(
        [Y_real[i, own[i]].mean() - np.median(Y_real[i, ~own[i]]) for i in range(n)]
    )
    real_z = knockdown_z(Y_real, P_label)

    W = generate_network(n, topology="scalefree", sparsity=7 / n, seed=args.seed)
    G = np.linalg.inv(np.eye(n) - W)  # -A⁻¹ with A = W - I
    strength = np.full(n, -np.median(shift) / np.median(np.diag(G)))
    failed = np.zeros(n, dtype=bool)
    confounder = np.zeros_like(Y_real)
    if args.scenario != "ideal":
        confounder = low_rank(Y_real, gene_of, N_PCS)
    if args.scenario == "realistic":
        failed = real_z > 0
        strength = np.where(failed, 0.0, np.maximum(-shift, 0.0) / np.diag(G))
    signal = G @ -np.repeat(np.diag(strength), 2, axis=1) + confounder
    E = rng.standard_normal(signal.shape)

    def median_z(sigma: float) -> float:
        return float(np.nanmedian(knockdown_z(signal + sigma * E, P_label)))

    lo, hi = 1e-4, 10.0
    for _ in range(50):
        sigma = np.sqrt(lo * hi)
        if median_z(sigma) < target_z:
            lo = sigma
        else:
            hi = sigma
    Y = signal + sigma * E

    genes = [f"G{i:03d}" for i in range(n)]
    columns = [genes[g] for g in gene_of]
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(Y, index=genes, columns=columns).to_csv(args.out / f"{args.scenario}.csv")
    np.save(args.out / f"{args.scenario}_true.npy", W)
    network_part = G @ -np.repeat(np.diag(strength), 2, axis=1)
    truth = {
        "scenario": args.scenario,
        "seed": args.seed,
        "genes": n,
        "true_links": int(np.count_nonzero(W)),
        "noise_sigma": sigma,
        "median_knockdown_z": median_z(sigma),
        "real_median_knockdown_z": target_z,
        "replicate_correlation_without_own_gene": replicate_correlation(Y, gene_of),
        "real_replicate_correlation_without_own_gene": replicate_correlation(Y_real, gene_of),
        "variance_share_network": float(network_part.var() / Y.var()),
        "variance_share_confounder": float(confounder.var() / Y.var()),
        "failed_knockdowns": [genes[i] for i in np.flatnonzero(failed)],
    }
    (args.out / f"{args.scenario}_truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    print(json.dumps(truth))


if __name__ == "__main__":
    main()
