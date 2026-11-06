"""Benchmark inference against known true networks across SNR_L levels.

``pygs bench-gs --snr`` loads the GeneSPIDER network JSON as the true GRN and
the dataset JSON as the expression and knockdown design. Only stored files
are used. The N50 collection has three levels, SNR_L 0.01, 1, and 100,
stored under filenames SNR10, SNR1000, and SNR100000. A requested SNR with
no matching file stops the run before any inference. ``pygs bench-gs``
without ``--snr`` scores every stored file of the chosen networks.

``pygs bench --snr`` generates the data instead. Each network has the
requested size, topology and edge density, with a negative diagonal shifted
until A + A.T is negative definite and cond(A) <= N/2. The design repeats
single-gene knockdowns, and Y = -pinv(A) @ P + E with E scaled so SNR_L is
each requested value. Any positive SNR works.

Every inferred network is written next to the score table.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2

ZETA = np.logspace(-6, 0, 30)
CHI_ALPHA = 0.95


def parse_snrs(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value.strip()]


def snr_l(A: np.ndarray, P: np.ndarray, noise_var: float) -> float:
    """GeneSPIDER SNR_L: smallest signal singular value over the noise scale."""
    signal = (-np.linalg.pinv(A)) @ P
    sigma = float(np.min(np.linalg.svd(signal, compute_uv=False)))
    chi = float(chi2.ppf(CHI_ALPHA, P.size))
    return sigma / math.sqrt(chi * noise_var)


def noise_var_for_snr(A: np.ndarray, P: np.ndarray, snr: float) -> float:
    signal = (-np.linalg.pinv(A)) @ P
    sigma = float(np.min(np.linalg.svd(signal, compute_uv=False)))
    chi = float(chi2.ppf(CHI_ALPHA, P.size))
    if snr <= 0:
        raise ValueError("SNR must be positive")
    return (sigma ** 2) / (chi * snr ** 2)


def simulate_expression(
    A: np.ndarray,
    P: np.ndarray,
    snr: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Y = G @ P + E, with E scaled so SNR_L equals ``snr``. G = -pinv(A)."""
    signal = (-np.linalg.pinv(A)) @ P
    variance = noise_var_for_snr(A, P, snr)
    noise = rng.normal(0.0, math.sqrt(variance), size=P.shape)
    return signal + noise


TOPOLOGIES = ("random", "scalefree", "smallworld")


def stable_network(n_nodes: int, topology: str, sparsity: float, seed: int) -> np.ndarray:
    """Signed links on a negative diagonal, stable and with cond(A) <= N/2.

    The diagonal shift makes the symmetric part negative definite, so every
    eigenvalue of A has a negative real part. N/2 is GeneSPIDER's low
    interampatteness bound.
    """
    from sparselink.bench.synthetic import generate_network

    if topology not in TOPOLOGIES:
        raise ValueError(f"Unknown topology {topology!r}; use one of {', '.join(TOPOLOGIES)}")
    structure = generate_network(n_nodes, topology=topology, sparsity=sparsity, seed=seed) != 0
    np.fill_diagonal(structure, False)
    rng = np.random.default_rng(seed)
    weights = np.where(structure, rng.standard_normal(structure.shape), 0.0)
    top = float(np.max(np.linalg.eigvalsh((weights + weights.T) / 2)))
    margin = 0.1
    while True:
        A = weights - (top + margin) * np.eye(n_nodes)
        if np.linalg.cond(A) <= n_nodes / 2:
            return A
        margin *= 2


def knockdown_design(n_nodes: int, n_samples: int) -> np.ndarray:
    """Single-gene knockdowns, gene j in every experiment j mod N."""
    P = np.zeros((n_nodes, n_samples))
    P[np.arange(n_samples) % n_nodes, np.arange(n_samples)] = 1.0
    return P


def match_snr(requested: float, files: list[dict]) -> dict | None:
    """Stored file whose SNR_L or filename SNR is the requested value."""
    hits = []
    for record in files:
        snr_l = record.get("snr_l")
        filename_snr = float(record["filename_snr"])
        if snr_l is not None and math.isclose(requested, float(snr_l), rel_tol=1e-2, abs_tol=1e-4):
            hits.append((0, abs(float(snr_l) - requested), record))
        elif math.isclose(requested, filename_snr, rel_tol=1e-6, abs_tol=0.0):
            hits.append((1, 0.0, record))
    if not hits:
        return None
    hits.sort(key=lambda item: (item[0], item[1]))
    return hits[0][2]


def shuffle_genes(Y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute each gene's experiments independently."""
    shuffled = np.empty_like(Y)
    for row in range(Y.shape[0]):
        shuffled[row] = rng.permutation(Y[row])
    return shuffled


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in text)


def _snr_label(snr: float) -> str:
    return f"{snr:g}"


def _tiers(args: argparse.Namespace) -> list[str]:
    tiers = str(getattr(args, "tier", None) or "fast")
    return [tier.strip() for tier in tiers.split(",") if tier.strip()]


def _methods(args: argparse.Namespace) -> list[str]:
    from sparselink import list_methods

    import sparselink.methods  # noqa: F401
    from bench.genespider import TIERS

    if getattr(args, "methods", None):
        names = [name.strip() for name in args.methods.split(",") if name.strip()]
    else:
        names = []
        for tier in _tiers(args):
            names.extend(TIERS.get(tier, []))
    names.extend(getattr(args, "extra_methods", None) or [])
    registered = set(list_methods())
    return [name for name in dict.fromkeys(names) if name in registered]


def use_nestboot(args: argparse.Namespace) -> bool:
    """``--nestboot``, or ``nestboot`` in the tier list as ``bench-gs`` allows."""
    return bool(getattr(args, "nestboot", False)) or "nestboot" in _tiers(args)


IMAGE_SUFFIXES = {".png", ".pdf", ".svg", ".jpg", ".jpeg"}


def output_paths(output: str | Path) -> dict[str, Path]:
    """Score table, figure and saved-run folder for ``-o``.

    ``-o results.json`` writes results.json, results.csv, results.png and
    results_runs/. ``-o Figure_3.png`` writes the figure there and the tables
    beside it as Figure_3.json and Figure_3.csv.
    """
    path = Path(output)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        figure, table = path, path.with_suffix(".json")
    elif path.suffix.lower() == ".csv":
        figure, table = path.with_suffix(".png"), path.with_suffix(".json")
    else:
        figure, table = path.with_suffix(".png"), path
    return {
        "json": table,
        "csv": table.with_suffix(".csv"),
        "figure": figure,
        "runs": table.with_name(table.stem + "_runs"),
    }


def _size(args: argparse.Namespace) -> str:
    n_nodes = getattr(args, "n_nodes", None)
    if n_nodes:
        return f"N{int(n_nodes)}"
    return str(getattr(args, "sizes", "N50"))


def _arms(methods: list[str], args: argparse.Namespace) -> list[tuple[str, str, bool]]:
    """(label, base method, shuffled). NestBoot is a separate pass."""
    arms = [(name, name, False) for name in methods]
    if getattr(args, "randomize", False):
        arms.extend((f"random {name}", name, True) for name in methods)
    return arms


def _read_dataset(meta: dict) -> dict:
    from bench.genespider import DATASET_RAW, _fetch_json

    key = meta["path"].replace("/", "_").replace(".json", "")
    obj = _fetch_json(f"{DATASET_RAW}/{meta['path']}", key)["obj_data"]
    snr_l = obj.get("SNR_L")
    if snr_l is not None:
        snr_l = float(np.asarray(snr_l).reshape(-1)[0])
    return {
        "Y": np.asarray(obj["Y"], dtype=float),
        "P": np.asarray(obj["P"], dtype=float),
        "snr_l": snr_l,
        "filename_snr": float(meta["snr"]),
        "network": obj["network"],
        "name": meta["path"].rsplit("/", 1)[-1],
    }


def _read_network(network_name: str) -> np.ndarray:
    from bench.genespider import _fetch_json, _find_network

    url = _find_network(network_name)
    if url is None:
        raise FileNotFoundError(f"GeneSPIDER network {network_name} was not found")
    obj = _fetch_json(url, network_name.replace("/", "_"))["obj_data"]
    return np.asarray(obj["A"], dtype=float)


def _gene_dataset(Y: np.ndarray, P: np.ndarray):
    from datastruct.Dataset import Dataset

    dataset = Dataset()
    dataset._Y = Y
    dataset._P = P
    return dataset


def infer_stack(
    method_name: str,
    Y: np.ndarray,
    P: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Weight stack shaped (genes, genes, sparsity)."""
    dataset = _gene_dataset(Y, P)
    if method_name == "lasso":
        from methods.lasso import Lasso

        stack, used = Lasso(dataset, alpha_range=ZETA)
    elif method_name == "lsco":
        from methods.lsco import LSCO

        stack, used = LSCO(dataset, threshold_range=ZETA)
    elif method_name == "tigress":
        from methods.tigress import TIGRESS

        stack, used = TIGRESS(dataset, random_state=seed)
    else:
        from sparselink import get_method

        from bench.genespider import ALPHA_SWEEP

        method_cls = get_method(method_name)
        param = ALPHA_SWEEP.get(method_name)
        settings = ZETA if param is not None else np.array([np.nan])
        pieces = []
        for alpha in settings:
            kwargs = {param: float(alpha)} if param is not None else {}
            pieces.append(method_cls(**kwargs).fit(Y.T, None if P is None else P.T).adjacency_matrix)
        stack = np.stack(pieces, axis=2)
        used = settings
    stack = np.asarray(stack, dtype=float)
    if stack.ndim == 2:
        stack = stack[:, :, None]
    return stack, np.atleast_1d(np.asarray(used, dtype=float))


def nestboot_slices(stack: np.ndarray) -> np.ndarray:
    """Sparse slices for NestBoot from methods whose weights are dense.

    NestBoot counts how often a link is nonzero, so dense weights would give
    every link full support. A sweep keeps the upper half of the nonzero
    weights in each slice; a single network is cut at the 10th to 90th
    percentile instead. ``pygs bench-gs --nestboot`` uses the same rule.
    """
    slices = []
    if stack.shape[2] == 1:
        weights = stack[:, :, 0]
        scores = np.abs(weights)
        np.fill_diagonal(scores, 0.0)
        nonzero = scores[scores > 0]
        if len(nonzero) == 0:
            return stack
        for cut in np.linspace(np.percentile(nonzero, 10), np.percentile(nonzero, 90), 10):
            slices.append(np.where(scores < cut, 0.0, weights))
        return np.stack(slices, axis=2)
    for index in range(stack.shape[2]):
        weights = stack[:, :, index].copy()
        scores = np.abs(weights)
        np.fill_diagonal(scores, 0.0)
        nonzero = scores[scores > 0]
        if len(nonzero) > 0:
            weights[scores < np.percentile(nonzero, 50)] = 0.0
        slices.append(weights)
    return np.stack(slices, axis=2)


def comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (dataset, method), in the form the TUI comparison table reads.

    AUROC and F1 come from the sparsity with the best F1 and MCC is the best
    over sparsities, as in the boxplot.
    """
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    summary = []
    for (dataset, method), part in frame.groupby(["dataset", "method"], sort=False):
        at_f1 = part.loc[part["f1_score"].idxmax()]
        summary.append({
            "method": str(method).replace("nestboot ", "nestboot:", 1),
            "dataset_name": dataset,
            "topology": at_f1["topology"],
            "snr": float(at_f1["snr"]),
            "auroc": float(at_f1["auroc"]),
            "f1": float(at_f1["f1_score"]),
            "mcc": float(part["mcc"].max()),
            "error": None,
        })
    return summary


def nestboot_stack(
    method_name: str,
    Y: np.ndarray,
    P: np.ndarray,
    A: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    import logging

    from analyze.Data import Data
    from config import AnalysisConfig
    from datastruct.Dataset import Dataset
    from datastruct.Network import Network
    from methods.nestboot import Nestboot

    n_genes = Y.shape[0]
    dataset = Dataset()
    dataset._Y = Y
    dataset._P = P
    dataset._network = Network(A)
    dataset._names = [f"G{i}" for i in range(n_genes)]
    data = Data(dataset)

    def _infer(incoming: Data, **kwargs: object) -> np.ndarray:
        matrix = incoming.data.Y  # type: ignore[union-attr]
        design = incoming.data.P  # type: ignore[union-attr]
        stack, _ = infer_stack(method_name, matrix, design, int(getattr(args, "seed", 42)))
        if method_name in {"lasso", "lsco"}:
            return stack
        return nestboot_slices(stack)

    config = AnalysisConfig(
        total_runs=args.nest_runs * args.boot_runs,
        inner_group_size=args.boot_runs,
        fdr_threshold=args.fdr,
    )
    nb = Nestboot(config)
    nb.logger.setLevel(logging.WARNING)
    result = nb.run_nestboot(
        dataset=data,
        inference_method=_infer,
        nest_runs=args.nest_runs,
        boot_runs=args.boot_runs,
        seed=int(getattr(args, "seed", 42)),
    )
    stack = np.asarray(result.sxnet, dtype=float)
    if stack.ndim == 2:
        stack = stack[:, :, None]
    from bench.genespider import ALPHA_SWEEP

    # NestBoot drops trailing sparsities with no links, so a zeta stack can come back shorter.
    if method_name in ALPHA_SWEEP and stack.shape[2] <= len(ZETA):
        alpha = ZETA[: stack.shape[2]]
    else:
        alpha = np.arange(stack.shape[2], dtype=float)
    return stack, alpha


def _score_rows(
    A: np.ndarray,
    stack: np.ndarray,
    alpha: np.ndarray,
    *,
    dataset: str,
    network: str,
    method: str,
    snr: float,
    topology: str,
    source: str,
    adjacency_file: str,
) -> list[dict]:
    from sparselink.bench.metrics import evaluate

    rows = []
    for index in range(stack.shape[2]):
        weights = np.array(stack[:, :, index], dtype=float, copy=True)
        metrics = evaluate(A, weights)
        used = None if index >= len(alpha) else float(alpha[index])
        rows.append({
            "dataset": dataset,
            "network": network,
            "method": method,
            "snr": snr,
            "topology": topology,
            "source": source,
            "alpha": used,
            "auroc": metrics.auroc,
            "aupr": metrics.aupr,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1,
            "fdr": metrics.fdr,
            "mcc": metrics.mcc,
            "r2": metrics.r2,
            "adjacency_file": adjacency_file,
        })
    return rows


def _save_stack(
    path: Path,
    stack: np.ndarray,
    alpha: np.ndarray,
    Y: np.ndarray,
    P: np.ndarray,
    A: np.ndarray,
    **meta: object,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "adjacency": stack,
        "alpha": alpha,
        "Y": Y,
        "P": P,
        "A": A,
    }
    payload.update({key: np.asarray(value) for key, value in meta.items()})
    np.savez_compressed(path, **payload)


def _checkpoint(rows: list[dict], csv_path: Path, json_path: Path) -> None:
    frame = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2))


def _stored_conditions(
    args: argparse.Namespace, snrs: list[float], topologies: list[str]
) -> tuple[list[dict[str, Any]], str]:
    """Every (network, SNR) pair from stored files. A missing SNR stops the run.

    Without ``--snr``, every stored file of the chosen networks is used.
    """
    from bench.cli import ROSE, console
    from bench.genespider import select_networks

    if not snrs:
        return _all_stored_conditions(args, topologies)
    size = _size(args)
    if "," in size:
        console.print(f"  [{ROSE}]--snr takes one size; got {size}.[/]")
        raise SystemExit(2)
    chosen = select_networks(size, topologies, int(args.n_datasets))
    if not chosen:
        console.print(f"  [{ROSE}]No GeneSPIDER networks for {size} and {topologies}.[/]")
        raise SystemExit(1)

    conditions = []
    for network_index, (metas, topology) in enumerate(chosen):
        files = [_read_dataset(meta) for meta in metas]
        network_name = files[0]["network"]
        A = _read_network(network_name)
        for snr_index, snr in enumerate(snrs):
            stored = match_snr(snr, files)
            if stored is None:
                available = ", ".join(
                    _snr_label(float(record["snr_l"]))
                    for record in files
                    if record.get("snr_l") is not None
                )
                console.print(
                    f"  [{ROSE}]No stored GeneSPIDER file for SNR {_snr_label(snr)}[/] "
                    f"on {network_name}. Available SNR_L values: {available}. "
                    "The N50 files are 0.01, 1, and 100 (filenames SNR10, SNR1000, SNR100000). "
                    "`pygs bench --snr` generates data at any SNR."
                )
                raise SystemExit(1)
            conditions.append({
                "topology": topology,
                "network": network_name,
                "A": A,
                "snr": snr,
                "Y": stored["Y"],
                "P": stored["P"],
                "source": "genespider",
                "dataset": stored["name"],
                "seed_offset": 1000 * network_index + 10 * snr_index,
                "note": f"stored file {stored['name']}",
            })
    return conditions, f"GeneSPIDER {size} networks ({args.n_datasets} per topology)"


def _all_stored_conditions(
    args: argparse.Namespace, topologies: list[str]
) -> tuple[list[dict[str, Any]], str]:
    """Each stored file of the chosen networks, at its own SNR_L, for every size.

    ``--max-networks N`` takes N whole networks, each at every stored SNR.
    ``--max-datasets N`` takes the first N files in listing order. With
    neither, every complete network is used. Files that fail to load are
    reported and skipped.
    """
    from bench.cli import ORANGE, console
    from bench.genespider import (
        _list_datasets,
        complete_networks,
        load_dataset,
        stored_snr_l,
    )

    max_networks = int(getattr(args, "max_networks", 0) or 0)
    max_datasets = int(getattr(args, "max_datasets", 0) or 0)
    sizes = [size.strip() for size in str(getattr(args, "sizes", "N50")).split(",") if size.strip()]
    conditions: list[dict[str, Any]] = []
    for size in sizes:
        listed = _list_datasets(size)
        if max_networks or not max_datasets:
            loaded, skipped = complete_networks(listed, size, max_networks)
        else:
            loaded, skipped = [], []
            for meta in listed[:max_datasets]:
                try:
                    loaded.append((meta, load_dataset(meta, size)))
                except Exception as exc:
                    skipped.append(f"{meta['path'].rsplit('/', 1)[-1]}: {str(exc)[:120]}")
        for meta, (X, P, A, topology, network_name) in loaded:
            name = meta["path"].rsplit("/", 1)[-1]
            snr = stored_snr_l(meta)
            if snr is None:
                skipped.append(f"{name}: no SNR_L stored in the file")
                continue
            if topology not in topologies:
                continue
            conditions.append({
                "topology": topology,
                "network": network_name,
                "A": np.asarray(A, dtype=float),
                "snr": snr,
                "Y": np.asarray(X, dtype=float).T,
                "P": None if P is None else np.asarray(P, dtype=float).T,
                "source": "genespider",
                "dataset": name,
                "seed_offset": 10 * len(conditions),
                "note": f"stored file {name}",
            })
        for reason in skipped:
            console.print(f"  [{ORANGE}]Skipped {reason}[/]")
    if max_networks:
        chosen = f"{max_networks} networks per size"
    elif max_datasets:
        chosen = f"first {max_datasets} files per size"
    else:
        chosen = "every complete network"
    return conditions, f"GeneSPIDER {', '.join(sizes)} networks ({chosen}, every stored SNR)"


def _simulated_conditions(
    args: argparse.Namespace, snrs: list[float], topologies: list[str]
) -> tuple[list[dict[str, Any]], str]:
    """Generated networks, each scored at every requested SNR."""
    from bench.cli import ROSE, console

    unknown = [name for name in topologies if name not in TOPOLOGIES]
    if unknown:
        console.print(f"  [{ROSE}]Unknown topologies {unknown}; use {', '.join(TOPOLOGIES)}.[/]")
        raise SystemExit(2)
    if any(snr <= 0 for snr in snrs):
        console.print(f"  [{ROSE}]SNR values must be positive.[/]")
        raise SystemExit(2)
    n_nodes = int(args.n_nodes)
    n_samples = int(getattr(args, "n_samples", None) or 3 * n_nodes)
    sparsity = float(args.sparsity)
    seed = int(getattr(args, "seed", 42))
    P = knockdown_design(n_nodes, n_samples)

    conditions = []
    network_index = 0
    for topology in topologies:
        for _ in range(int(args.n_datasets)):
            network_seed = seed + 1000 * network_index
            A = stable_network(n_nodes, topology, sparsity, network_seed)
            links = int(np.count_nonzero(A) - np.count_nonzero(np.diag(A)))
            network_name = f"sim-{topology}-N{n_nodes}-L{links}-seed{network_seed}"
            for snr_index, snr in enumerate(snrs):
                offset = 1000 * network_index + 10 * snr_index
                Y = simulate_expression(A, P, snr, np.random.default_rng(seed + offset + 1))
                conditions.append({
                    "topology": topology,
                    "network": network_name,
                    "A": A,
                    "snr": snr,
                    "Y": Y,
                    "P": P,
                    "source": "simulated",
                    "dataset": f"{network_name}-SNR{_snr_label(snr)}",
                    "seed_offset": offset,
                    "note": f"generated, {n_nodes} genes × {n_samples} experiments",
                })
            network_index += 1
    return conditions, (
        f"generated N{n_nodes} networks at sparsity {sparsity:g} ({args.n_datasets} per topology)"
    )


def run_snr_benchmark(args: argparse.Namespace, simulate: bool = False) -> None:
    """Score every method on each (network, SNR) condition.

    ``simulate`` generates networks and expression from the flags. Otherwise
    the stored GeneSPIDER files are used.
    """
    from bench.cli import (
        DIM,
        GREEN,
        ROSE,
        TEAL,
        _render_comparison,
        _render_results,
        _write_snr_figure,
        console,
    )

    methods = _methods(args)
    if not methods:
        console.print(f"  [{ROSE}]No valid methods selected.[/]")
        return
    if simulate and not getattr(args, "snr", None):
        console.print(f"  [{ROSE}]--snr is required to generate data.[/]")
        return

    snrs = parse_snrs(args.snr) if getattr(args, "snr", None) else []
    topologies = [
        name.strip()
        for name in str(getattr(args, "topologies", ",".join(TOPOLOGIES))).split(",")
        if name.strip()
    ]
    arms = _arms(methods, args)
    from bench.genespider import NESTBOOT_EXCLUDED

    nested = use_nestboot(args)
    nest_methods = [name for name in methods if name not in NESTBOOT_EXCLUDED] if nested else []
    build = _simulated_conditions if simulate else _stored_conditions
    conditions, description = build(args, snrs, topologies)
    if not conditions:
        console.print(f"  [{ROSE}]No datasets to score.[/]")
        return
    n_networks = len({condition["network"] for condition in conditions})
    if not snrs:
        snrs = sorted({float(condition["snr"]) for condition in conditions})

    paths = output_paths(args.output)
    run_dir, csv_path, json_path = paths["runs"], paths["csv"], paths["json"]
    per_condition = len(arms) + len(nest_methods)
    console.print(f"  [{TEAL}]Methods[/]     {', '.join(method for _, method, shuffled in arms if not shuffled)}")
    if getattr(args, "randomize", False):
        console.print(f"  [{TEAL}]Random[/]      gene-wise shuffle of each method")
    if nest_methods:
        console.print(
            f"  [{TEAL}]NestBoot[/]    {', '.join(nest_methods)} ({args.nest_runs}×{args.boot_runs})"
        )
    if nested:
        for name in methods:
            if name in NESTBOOT_EXCLUDED:
                console.print(f"  [{DIM}]{name} is scored directly only: {NESTBOOT_EXCLUDED[name]}[/]")
    console.print(f"  [{TEAL}]Networks[/]    {n_networks} {description}")
    console.print(f"  [{TEAL}]SNR[/]         {', '.join(_snr_label(snr) for snr in snrs)}")
    console.print(
        f"  [{TEAL}]Saved runs[/]  {run_dir}  ({len(conditions)} conditions × {per_condition} arms)"
    )
    console.print()

    rows: list[dict] = []
    seed = int(getattr(args, "seed", 42))
    done = 0
    for condition in conditions:
        topology = condition["topology"]
        network_name = condition["network"]
        A = condition["A"]
        snr = condition["snr"]
        Y = condition["Y"]
        P = condition["P"]
        source = condition["source"]
        dataset_name = condition["dataset"]
        console.print(
            f"  [{GREEN}]{topology} {network_name} SNR {_snr_label(snr)}[/] "
            f"[{DIM}]{condition['note']}[/]"
        )
        condition_seed = seed + condition["seed_offset"]
        jobs: list[tuple[str, str, bool, bool]] = [
            (label, method, shuffled, False) for label, method, shuffled in arms
        ]
        jobs.extend((f"nestboot {method}", method, False, True) for method in nest_methods)
        for job_index, (label, method, shuffled, nested) in enumerate(jobs):
            done += 1
            console.print(
                f"    [{DIM}]{done}/{len(conditions) * per_condition}[/] {label} {topology}/SNR={_snr_label(snr)}"
            )
            expression = Y
            if shuffled:
                expression = shuffle_genes(
                    Y, np.random.default_rng(condition_seed + 17 + job_index)
                )
            try:
                if nested:
                    stack, alpha = nestboot_stack(method, expression, P, A, args)
                else:
                    stack, alpha = infer_stack(method, expression, P, condition_seed + job_index)
            except Exception as exc:
                error_path = json_path.with_name(json_path.stem + "_errors.json")
                errors = []
                if error_path.exists():
                    errors = json.loads(error_path.read_text())
                errors.append({
                    "dataset": dataset_name,
                    "network": network_name,
                    "method": label,
                    "snr": snr,
                    "topology": topology,
                    "source": source,
                    "error": str(exc)[:200],
                })
                error_path.write_text(json.dumps(errors, indent=2))
                console.print(f"      [{ROSE}]{exc}[/]")
                continue
            filename = (
                f"{_slug(network_name)}__{_slug(label)}__snr_{_snr_label(snr)}.npz"
            )
            path = run_dir / filename
            _save_stack(
                path, stack, alpha, expression, P, A,
                method=label, snr=snr, topology=topology,
                network=network_name, source=source, dataset=dataset_name,
            )
            rows.extend(_score_rows(
                A, stack, alpha,
                dataset=dataset_name,
                network=network_name,
                method=label,
                snr=snr,
                topology=topology,
                source=source,
                adjacency_file=str(path),
            ))
            _checkpoint(rows, csv_path, json_path)

    _checkpoint(rows, csv_path, json_path)
    summary = comparison_rows(rows)
    console.print()
    if any(row["method"].startswith("nestboot:") for row in summary):
        _render_comparison(summary)
    else:
        _render_results(summary, f"SNR benchmark ({len(summary)} runs)")
    console.print(f"\n  [{DIM}]Scores saved to {csv_path}[/]")
    console.print(f"  [{DIM}]Inferred networks saved to {run_dir}[/]")
    if getattr(args, "plot", False) or getattr(args, "figure", False):
        figure = paths["figure"]
        _write_snr_figure(csv_path, figure)
        console.print(f"  [{DIM}]Figure saved to {figure}[/]")
