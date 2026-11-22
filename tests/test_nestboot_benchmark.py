#!/usr/bin/env python3
"""Verify NestBoot improves over base methods on GeneSpider data.

Run: .venv/bin/python -B tests/test_nestboot_benchmark.py

Expected: NestBoot should match or improve F1/MCC for methods with
alpha sweep (lasso, lsco, elastic_net) at high SNR.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")

# Minimal test: 1 dataset, SNR=1000, fast methods
METHODS_TO_TEST = ["lasso", "lsco", "elastic_net"]
NEST_RUNS = 5
BOOT_RUNS = 5
FDR = 0.05


def load_test_data():
    """Load a cached GeneSpider N50 dataset at SNR=1000."""
    cache = Path(".gs_cache")
    # Find a SNR=1000 dataset
    for f in sorted(cache.glob("N50_*SNR1000*.json")):
        with open(f) as fh:
            ds = json.load(fh)["obj_data"]
        # Find matching network
        net_name = ds["network"]
        net_file = cache / f"{net_name}.json"
        if net_file.exists():
            with open(net_file) as fh:
                net = json.load(fh)["obj_data"]
            X = np.array(ds["Y"]).T  # samples x genes
            P = np.array(ds["P"]).T
            A_true = np.array(net["A"])
            return X, P, A_true, f.stem
    raise FileNotFoundError("No cached GeneSpider N50/SNR=1000 data. Run benchmark first.")


def run_direct(method_name, X, P, A_true):
    """Run method with oracle alpha sweep (same as benchmark)."""
    from bench.genespider import _fit_best
    best_adj, elapsed = _fit_best(method_name, X, A_true, P)
    from sparselink.bench.metrics import evaluate
    return evaluate(A_true, best_adj), elapsed


def run_nestboot(method_name, X, P, A_true, gene_names):
    """Run method through NestBoot."""
    from sparselink import get_method
    from sparselink.bench.metrics import evaluate

    from analyze.Data import Data
    from bench.genespider import ALPHA_SWEEP
    from config import AnalysisConfig
    from datastruct.Dataset import Dataset
    from datastruct.Network import Network
    from methods.nestboot import Nestboot

    n_genes = X.shape[1]

    ds = Dataset()
    ds._Y = X.T
    ds._P = P.T
    ds._network = Network(A_true)
    ds._names = gene_names
    data_obj = Data(ds)

    method_cls = get_method(method_name)
    has_native_alpha = method_name in ALPHA_SWEEP
    alpha_param = ALPHA_SWEEP.get(method_name, "alpha")
    alpha_range = np.logspace(-3, 0, 10) if has_native_alpha else None

    def _infer(dataset, **kwargs):
        Y = dataset.data.Y
        P_mat = dataset.data.P
        X_in = Y.T
        P_in = P_mat.T if P_mat is not None else None
        if has_native_alpha:
            slices = []
            for alpha in alpha_range:
                result = method_cls(**{alpha_param: float(alpha)}).fit(X_in, P_in)
                adj = result.adjacency_matrix.copy()
                scores = np.abs(adj)
                np.fill_diagonal(scores, 0.0)
                nonzero = scores[scores > 0]
                if len(nonzero) > 0:
                    thr = np.percentile(nonzero, 50)
                    adj[scores < thr] = 0.0
                slices.append(adj)
            return np.stack(slices, axis=2)
        else:
            result = method_cls().fit(X_in, P_in)
            return result.adjacency_matrix

    config = AnalysisConfig(
        total_runs=NEST_RUNS * BOOT_RUNS,
        inner_group_size=BOOT_RUNS,
        fdr_threshold=FDR,
    )
    nb = Nestboot(config)
    nb.logger.setLevel(50)  # suppress

    t0 = time.perf_counter()
    nb_results = nb.run_nestboot(
        dataset=data_obj,
        inference_method=_infer,
        nest_runs=NEST_RUNS,
        boot_runs=BOOT_RUNS,
        seed=42,
    )
    elapsed = time.perf_counter() - t0

    sxnet = nb_results.sxnet
    if hasattr(sxnet, "ndim") and sxnet.ndim == 3:
        # Pick best slice
        from sklearn.metrics import roc_auc_score
        mask = ~np.eye(n_genes, dtype=bool)
        y_true = (A_true[mask] != 0).astype(int)
        best_auroc = -1.0
        best_adj = np.abs(sxnet[:, :, 0])
        for k in range(sxnet.shape[2]):
            candidate = np.abs(sxnet[:, :, k])
            np.fill_diagonal(candidate, 0.0)
            try:
                auroc = float(roc_auc_score(y_true, candidate[mask]))
            except ValueError:
                auroc = 0.5
            if auroc > best_auroc:
                best_auroc = auroc
                best_adj = candidate
        combined = best_adj
    else:
        combined = np.abs(sxnet)
    np.fill_diagonal(combined, 0.0)
    return evaluate(A_true, combined), elapsed


def run_scenicplus_direct(X, P, A_true, gene_names):
    """Run SCENIC+ without NestBoot."""
    from bench.genespider import _fit_pygs_method
    best_adj, elapsed = _fit_pygs_method("scenicplus", X, A_true, P)
    from sparselink.bench.metrics import evaluate
    return evaluate(A_true, best_adj), elapsed


def main():
    print("=" * 70)
    print("NestBoot Verification Test")
    print("=" * 70)

    try:
        X, P, A_true, ds_name = load_test_data()
    except FileNotFoundError as e:
        print(f"SKIP: {e}")
        sys.exit(0)

    n_genes = X.shape[1]
    gene_names = [f"G{i}" for i in range(n_genes)]
    print(f"Dataset: {ds_name}")
    print(f"Shape: {X.shape[0]} samples × {n_genes} genes")
    print(f"NestBoot: {NEST_RUNS}×{BOOT_RUNS} runs, FDR={FDR}")
    print("-" * 70)
    print(f"{'Method':<20} {'Mode':<10} {'AUROC':>7} {'F1':>7} {'MCC':>7} {'Time':>7}")
    print("-" * 70)

    results = []
    for method in METHODS_TO_TEST:
        # Direct (oracle)
        metrics_d, t_d = run_direct(method, X, P, A_true)
        print(f"{method:<20} {'direct':<10} {metrics_d.auroc:7.3f} {metrics_d.f1:7.3f} {metrics_d.mcc:7.3f} {t_d:6.1f}s")

        # NestBoot
        metrics_nb, t_nb = run_nestboot(method, X, P, A_true, gene_names)
        print(f"{method:<20} {'nestboot':<10} {metrics_nb.auroc:7.3f} {metrics_nb.f1:7.3f} {metrics_nb.mcc:7.3f} {t_nb:6.1f}s")

        delta_f1 = metrics_nb.f1 - metrics_d.f1
        delta_mcc = metrics_nb.mcc - metrics_d.mcc
        status = "✓" if delta_f1 >= -0.05 else "✗"  # allow small regression
        print(f"  {'→ Δ':<28} {'':>7} {delta_f1:+7.3f} {delta_mcc:+7.3f}        {status}")
        results.append((method, delta_f1, delta_mcc))
        print()

    # SCENIC+ test
    print("-" * 70)
    print("SCENIC+ (direct only — verifies it runs without error)")
    try:
        metrics_sc, t_sc = run_scenicplus_direct(X, P, A_true, gene_names)
        print(f"{'scenicplus':<20} {'direct':<10} {metrics_sc.auroc:7.3f} {metrics_sc.f1:7.3f} {metrics_sc.mcc:7.3f} {t_sc:6.1f}s")
        print("  ✓ SCENIC+ runs successfully")
    except Exception as e:
        print(f"  ✗ SCENIC+ failed: {e}")

    print("=" * 70)
    # Summary
    passes = sum(1 for _, df1, _ in results if df1 >= -0.05)
    print(f"RESULT: {passes}/{len(results)} methods pass (NestBoot ≥ direct - 0.05 F1)")
    if passes == len(results):
        print("✓ ALL PASS")
    else:
        print("✗ SOME FAILURES — NestBoot may need more runs or tuning")


if __name__ == "__main__":
    main()
