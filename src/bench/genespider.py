#!/usr/bin/env python3
"""Benchmark sparselink methods against real GeneSpider datasets.

Downloads datasets and networks from the Sonnhammer GRNi Bitbucket repos
and runs all methods against them.

Usage:
    python benchmark_genespider.py --tier fast --timeout 60
    python benchmark_genespider.py --tier fast,medium --sizes N50
"""

from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests

from sparselink import list_methods
from sparselink.bench.metrics import evaluate
from sparselink.registry import get_method

import sparselink.methods  # noqa: F401

DATASET_COMMIT = "d2047430263f5ffe473525c74b4318f723c23b0e"
NETWORK_COMMIT = "0b3a66e67d776eadaa5d68667ad9c8fbac12ef85"
DATASET_BASE = f"https://api.bitbucket.org/2.0/repositories/sonnhammergrni/gs-datasets/src/{DATASET_COMMIT}"
DATASET_RAW = f"https://bitbucket.org/sonnhammergrni/gs-datasets/raw/{DATASET_COMMIT}"
NETWORK_BASE = f"https://api.bitbucket.org/2.0/repositories/sonnhammergrni/gs-networks/src/{NETWORK_COMMIT}"
NETWORK_RAW = f"https://bitbucket.org/sonnhammergrni/gs-networks/raw/{NETWORK_COMMIT}"

TOPOLOGIES = ["random", "scalefree", "smallworld"]
SIZES = ["N10", "N50", "N100"]

TIERS: dict[str, list[str]] = {
    "fast": [
        "lasso", "lsco", "elastic_net", "ridge", "partial_correlation",
        "clr", "genie3", "neighborhood_selection",
    ],
    "medium": [
        "glasso", "bdeu", "bge", "granger_causality",
        "transfer_entropy", "pcmci",
    ],
    "slow": ["tigress", "glasso_stars", "pc", "fci"],
    "very_slow": ["notears", "dag_gnn"],
    "scenicplus": ["scenicplus"],
}

# Methods handled by pyGS directly (not sparselink)
PYGS_METHODS: set[str] = {"scenicplus", "panda"}

# Scored directly but never wrapped in NestBoot, with the reason shown in the TUI
NESTBOOT_EXCLUDED: dict[str, str] = {
    "tigress": "already bootstrapped (stability selection)",
}

CACHE_DIR = Path(".gs_cache")


@dataclass
class RunResult:
    method: str
    dataset_name: str
    network_name: str
    topology: str
    n_genes: int
    snr: int
    auroc: float
    aupr: float
    precision: float
    recall: float
    f1: float
    fdr: float
    mcc: float
    r2: float
    elapsed_sec: float
    error: str | None = None


def _fetch_json(url: str, cache_key: str) -> dict:
    """Fetch JSON with local file cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    with open(cache_file, "w") as f:
        json.dump(data, f)
    return data


VALID_SNRS = {10, 1000, 100000}


def _list_datasets(size: str) -> list[dict]:
    """List all datasets for a given size, return parsed metadata."""
    url = f"{DATASET_BASE}/{size}/?pagelen=100"
    results = []
    while url:
        resp = requests.get(url, timeout=30).json()
        for v in resp.get("values", []):
            path = v["path"]
            m = re.search(r"ID(\d+).*N(\d+)-E(\d+)-SNR(\d+)-IDY", path)
            if m:
                snr = int(m.group(4))
                if snr not in VALID_SNRS:
                    continue
                results.append({
                    "path": path,
                    "network_id": m.group(1),
                    "n_genes": int(m.group(2)),
                    "n_experiments": int(m.group(3)),
                    "snr": snr,
                })
        url = resp.get("next")
    return results


def complete_networks(
    datasets: list[dict[str, Any]], size: str, n_networks: int
) -> tuple[list[tuple[dict[str, Any], tuple[Any, ...]]], list[str]]:
    """Load whole networks: every stored SNR file of each, in listing order.

    A network is kept only when it has a file at each of VALID_SNRS and all of
    them load, so every network is scored at every noise level. Returns the
    loaded (metadata, load_dataset result) pairs and one reason per skipped
    network. ``n_networks`` 0 keeps every complete network.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for meta in datasets:
        groups.setdefault(meta["network_id"], []).append(meta)
    loaded: list[tuple[dict[str, Any], tuple[Any, ...]]] = []
    skipped: list[str] = []
    kept = 0
    for network_id, metas in groups.items():
        if n_networks and kept >= n_networks:
            break
        stored = sorted(meta["snr"] for meta in metas)
        if set(stored) != VALID_SNRS:
            skipped.append(f"ID{network_id}: stored only at filename SNR {stored}")
            continue
        try:
            pairs = [(meta, load_dataset(meta, size)) for meta in sorted(metas, key=lambda m: m["snr"])]
        except Exception as exc:
            skipped.append(f"ID{network_id}: {str(exc)[:120]}")
            continue
        loaded.extend(pairs)
        kept += 1
    return loaded, skipped


def _topology_by_id(size: str) -> dict[str, str]:
    """Map a GeneSPIDER network id to random, scalefree, or smallworld."""
    mapping: dict[str, str] = {}
    for topo in TOPOLOGIES:
        url = f"{NETWORK_BASE}/{topo}/{size}/?pagelen=100"
        while url:
            resp = requests.get(url, timeout=30).json()
            for item in resp.get("values", []):
                match = re.search(r"ID(\d+)", item["path"])
                if match:
                    mapping[match.group(1)] = topo
            url = resp.get("next")
    return mapping


def select_datasets(
    size: str,
    topologies: list[str],
    snrs: list[int],
    n_per_topology: int,
) -> list[tuple[dict, str]]:
    """Real GeneSPIDER files only: expression, knockdown design, and true network.

    Keeps the first ``n_per_topology`` network ids in each topology (0 keeps
    every id) and every requested SNR of those networks.
    """
    topo_of = _topology_by_id(size)
    wanted = set(snrs)
    picked: dict[str, list[str]] = {topo: [] for topo in topologies}
    chosen: list[tuple[dict, str]] = []
    for ds in sorted(_list_datasets(size), key=lambda item: (item["network_id"], item["snr"])):
        topo = topo_of.get(ds["network_id"])
        if topo not in picked or ds["snr"] not in wanted:
            continue
        ids = picked[topo]
        if ds["network_id"] not in ids:
            if n_per_topology > 0 and len(ids) >= n_per_topology:
                continue
            ids.append(ds["network_id"])
        chosen.append((ds, topo))
    return chosen


def select_networks(
    size: str,
    topologies: list[str],
    n_per_topology: int,
) -> list[tuple[list[dict], str]]:
    """One group of dataset files per GeneSPIDER network.

    Keeps the first ``n_per_topology`` network ids in each topology. Every
    stored SNR of that network is included so the caller can use the file
    when it matches and the knockdown design otherwise.
    """
    topo_of = _topology_by_id(size)
    grouped: dict[str, list[dict]] = {}
    for ds in sorted(_list_datasets(size), key=lambda item: (item["network_id"], item["snr"])):
        grouped.setdefault(ds["network_id"], []).append(ds)
    picked: dict[str, list[str]] = {topo: [] for topo in topologies}
    chosen: list[tuple[list[dict], str]] = []
    for network_id, rows in grouped.items():
        topo = topo_of.get(network_id)
        if topo not in picked:
            continue
        ids = picked[topo]
        if n_per_topology > 0 and len(ids) >= n_per_topology:
            continue
        ids.append(network_id)
        chosen.append((rows, topo))
    return chosen


def _find_network(network_name: str) -> str | None:
    """Find the raw URL for a network by name across topologies."""
    for topo in TOPOLOGIES:
        # Parse N size from network name
        nm = re.search(r"N(\d+)", network_name)
        if not nm:
            continue
        size = f"N{nm.group(1)}"
        url = f"{NETWORK_RAW}/{topo}/{size}/{network_name}.json"
        try:
            resp = requests.head(url, timeout=10)
            if resp.status_code == 200:
                return url
        except Exception:
            continue
    return None


def stored_snr_l(ds_meta: dict[str, object]) -> float | None:
    """SNR_L recorded in a dataset file. The filename SNR is a different scale."""
    path = str(ds_meta["path"])
    key = path.replace("/", "_").replace(".json", "")
    value = _fetch_json(f"{DATASET_RAW}/{path}", key)["obj_data"].get("SNR_L")
    if value is None:
        return None
    return float(np.asarray(value).reshape(-1)[0])


def load_dataset(ds_meta: dict, size: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, str, str]:
    """Load a GeneSpider dataset + its network. Returns (X, A_true, topology, net_name)."""
    ds_key = ds_meta["path"].replace("/", "_").replace(".json", "")
    ds_data = _fetch_json(f"{DATASET_RAW}/{ds_meta['path']}", ds_key)
    obj = ds_data["obj_data"]

    Y = np.array(obj["Y"])  # (n_genes x n_experiments)
    X = Y.T  # (samples x features) for sparselink
    P = np.array(obj["P"]).T if "P" in obj else None  # (samples x features)

    network_name = obj["network"]

    # Find and load the network
    net_url = _find_network(network_name)
    if net_url is None:
        raise ValueError(f"Network {network_name} not found")

    net_key = network_name.replace("/", "_")
    net_data = _fetch_json(net_url, net_key)
    A_true = np.array(net_data["obj_data"]["A"])

    # Determine topology from URL
    topology = "unknown"
    for t in TOPOLOGIES:
        if t in (net_url or ""):
            topology = t
            break

    return X, P, A_true, topology, network_name


ALPHA_SWEEP: dict[str, str] = {
    "lasso": "alpha",
    "lsco": "threshold",
    "elastic_net": "alpha",
    "ridge": "alpha",
    "neighborhood_selection": "alpha",
    "glasso": "alpha",
}
ALPHA_RANGE = np.logspace(-4, 1, 30)


class _Timeout(Exception):
    pass


def _alarm_handler(signum: int, frame: object) -> None:
    raise _Timeout()


def _fit_best(method_name: str, X: np.ndarray, A_true: np.ndarray, P: np.ndarray | None = None) -> tuple:
    """Fit method, sweeping alpha if applicable. Returns (best_adj, elapsed)."""
    if method_name in PYGS_METHODS:
        return _fit_pygs_method(method_name, X, A_true, P)

    method_cls = get_method(method_name)
    if method_name in ALPHA_SWEEP:
        param = ALPHA_SWEEP[method_name]
        best_auroc = -1.0
        best_adj = None
        t0 = time.perf_counter()
        for alpha in ALPHA_RANGE:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = method_cls(**{param: float(alpha)}).fit(X, P)
            adj = result.adjacency_matrix
            try:
                from sklearn.metrics import roc_auc_score
                mask = ~np.eye(adj.shape[0], dtype=bool)
                y_true = (A_true[mask] != 0).astype(int)
                y_scores = np.abs(adj[mask])
                auroc = float(roc_auc_score(y_true, y_scores))
            except ValueError:
                auroc = 0.5
            if auroc > best_auroc:
                best_auroc = auroc
                best_adj = adj
        elapsed = time.perf_counter() - t0
        return best_adj, elapsed
    else:
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = method_cls().fit(X, P)
        elapsed = time.perf_counter() - t0
        return result.adjacency_matrix, elapsed


def _fit_pygs_method(method_name: str, X: np.ndarray, A_true: np.ndarray, P: np.ndarray | None = None) -> tuple:
    """Fit a pyGS-native method (e.g., SCENIC+). Returns (best_adj, elapsed)."""
    from datastruct.Dataset import Dataset
    from analyze.Data import Data

    n_genes = X.shape[1]
    n_samples = X.shape[0]
    gene_names = [f"G{i}" for i in range(n_genes)]

    # Build Dataset (X is samples×genes, Dataset wants genes×samples)
    ds = Dataset()
    ds._Y = X.T
    ds._P = P.T if P is not None else np.eye(n_genes, n_samples)
    ds._names = gene_names
    data_obj = Data(ds)

    if method_name == "scenicplus":
        from methods.scenicplus import SCENICPLUS
        threshold_range = np.logspace(-6, 0, 30)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = SCENICPLUS(
                dataset=data_obj,
                nested_boot=False,
                threshold_range=threshold_range,
                var_names=gene_names,
                n_cpu=1,
            )
        elapsed = time.perf_counter() - t0
        adj_3d, thresholds = result

        # Pick best alpha (oracle) like other methods
        from sklearn.metrics import roc_auc_score
        mask = ~np.eye(n_genes, dtype=bool)
        y_true = (A_true[mask] != 0).astype(int)
        best_auroc = -1.0
        best_adj = adj_3d[:, :, 0]
        for k in range(adj_3d.shape[2]):
            candidate = np.abs(adj_3d[:, :, k])
            np.fill_diagonal(candidate, 0.0)
            y_scores = candidate[mask]
            try:
                auroc = float(roc_auc_score(y_true, y_scores))
            except ValueError:
                auroc = 0.5
            if auroc > best_auroc:
                best_auroc = auroc
                best_adj = candidate
        return best_adj, elapsed
    elif method_name == "panda":
        from methods.panda import PANDA
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adj, _ = PANDA(dataset=data_obj, var_names=gene_names)
        elapsed = time.perf_counter() - t0
        np.fill_diagonal(adj, 0.0)
        return np.abs(adj), elapsed
    else:
        raise ValueError(f"Unknown pyGS method: {method_name}")


def run_single(
    method_name: str, X: np.ndarray, A_true: np.ndarray,
    ds_meta: dict, topology: str, net_name: str, timeout: int,
    P: np.ndarray | None = None,
) -> RunResult:
    base = dict(
        method=method_name, dataset_name=ds_meta["path"],
        network_name=net_name, topology=topology,
        n_genes=ds_meta["n_genes"], snr=ds_meta["snr"],
    )
    fail = dict(auroc=0, aupr=0, precision=0, recall=0, f1=0, fdr=1, mcc=0, r2=0)

    old_handler = None
    try:
        if timeout > 0:
            import signal
            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(timeout)

        best_adj, elapsed = _fit_best(method_name, X, A_true, P)

        if timeout > 0:
            signal.alarm(0)

        metrics = evaluate(A_true, best_adj)
        return RunResult(
            **base, auroc=metrics.auroc, aupr=metrics.aupr,
            precision=metrics.precision, recall=metrics.recall,
            f1=metrics.f1, fdr=metrics.fdr, mcc=metrics.mcc, r2=metrics.r2,
            elapsed_sec=round(elapsed, 4),
        )
    except _Timeout:
        return RunResult(**base, **fail, elapsed_sec=float(timeout), error="TIMEOUT")
    except Exception as e:
        if timeout > 0:
            import signal
            signal.alarm(0)
        return RunResult(**base, **fail, elapsed_sec=0, error=str(e)[:120])
    finally:
        if old_handler is not None:
            import signal
            signal.signal(signal.SIGALRM, old_handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark on GeneSpider datasets")
    parser.add_argument("--sizes", default="N50", help="Comma-separated: N10,N50,N100")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--tier", default="fast")
    parser.add_argument("--max-datasets", type=int, default=0,
                        help="Max datasets per size (0=all)")
    parser.add_argument("--output", "-o", default="benchmark_genespider.json")
    args = parser.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",")]
    selected_tiers = [t.strip() for t in args.tier.split(",")]
    methods: list[str] = []
    for t in selected_tiers:
        methods.extend(TIERS.get(t, []))
    registered = set(list_methods())
    methods = [m for m in methods if m in registered or m in PYGS_METHODS]

    print(f"Methods: {methods}")
    print(f"Sizes: {sizes}")
    print(f"Timeout: {args.timeout}s\n")

    results: list[RunResult] = []
    wall_start = time.perf_counter()

    for size in sizes:
        print(f"=== {size} ===")
        print("Fetching dataset list...")
        datasets = _list_datasets(size)
        if args.max_datasets > 0:
            datasets = datasets[:args.max_datasets]
        print(f"Found {len(datasets)} datasets")

        total = len(datasets) * len(methods)
        done = 0

        for ds_meta in datasets:
            try:
                X, P, A_true, topology, net_name = load_dataset(ds_meta, size)
            except Exception as e:
                print(f"  SKIP {ds_meta['path']}: {e}")
                continue

            snr_label = ds_meta["snr"]
            for method_name in methods:
                done += 1
                tag = f"[{done}/{total}] {method_name:22s} {topology}/SNR={snr_label}/N={ds_meta['n_genes']}"
                print(f"  {tag}", end=" … ", flush=True)
                r = run_single(method_name, X, A_true, ds_meta, topology, net_name, args.timeout, P)
                if r.error:
                    print(r.error)
                else:
                    print(f"AUROC={r.auroc:.3f} MCC={r.mcc:.3f} F1={r.f1:.3f} ({r.elapsed_sec:.2f}s)")
                results.append(r)

    wall_elapsed = time.perf_counter() - wall_start

    # Summary
    print(f"\n{'=' * 105}")
    print(f"{'METHOD':25s} {'AUROC':>7s} {'AUPR':>7s} {'F1':>7s} {'MCC':>7s} "
          f"{'R²':>7s} {'FDR':>7s} {'TIME':>9s} {'OK':>4s}")
    print("-" * 105)
    for m in methods:
        ok = [r for r in results if r.method == m and r.error is None]
        if ok:
            a = {k: np.mean([getattr(r, k) for r in ok])
                 for k in ("auroc", "aupr", "f1", "mcc", "r2", "fdr", "elapsed_sec")}
            print(f"  {m:25s} {a['auroc']:7.3f} {a['aupr']:7.3f} {a['f1']:7.3f} {a['mcc']:7.3f} "
                  f"{a['r2']:7.3f} {a['fdr']:7.3f} {a['elapsed_sec']:8.2f}s {len(ok):>4d}")
        else:
            errs = [r for r in results if r.method == m and r.error is not None]
            print(f"  {m:25s} {'—':>7s} {'—':>7s} {'—':>7s} {'—':>7s} "
                  f"{'—':>7s} {'—':>7s} {'—':>9s} {0:>4d} ({len(errs)} err)")
    print(f"{'=' * 105}")
    print(f"Wall time: {wall_elapsed / 60:.1f} min")

    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
