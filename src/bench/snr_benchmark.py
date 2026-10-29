"""Benchmark inference against GeneSPIDER true networks.

``pygs bench --snr`` loads the network JSON as the true GRN and the dataset
JSON as the knockdown design. A requested SNR uses the stored expression
when that SNR is already in a file. Otherwise expression is simulated from
the same network and the same design, at that SNR. Every inferred network
is written next to the score table.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

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


def _methods(args: argparse.Namespace) -> list[str]:
    from sparselink import list_methods
    from sparselink.bench.run_benchmark import TIERS

    import sparselink.methods  # noqa: F401

    if getattr(args, "methods", None):
        names = [name.strip() for name in args.methods.split(",") if name.strip()]
    else:
        names = []
        for tier in str(getattr(args, "tier", "fast")).split(","):
            names.extend(TIERS.get(tier.strip(), []))
    registered = set(list_methods())
    return [name for name in names if name in registered]


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
        return stack

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
    alpha = ZETA[: stack.shape[2]] if stack.shape[2] == len(ZETA) else np.arange(stack.shape[2], dtype=float)
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


def run_snr_benchmark(args: argparse.Namespace) -> None:
    from bench.cli import DIM, GREEN, ORANGE, ROSE, TEAL, _write_snr_figure, console
    from bench.genespider import select_networks

    methods = _methods(args)
    if not methods:
        console.print(f"  [{ROSE}]No valid methods selected.[/]")
        return
    if not getattr(args, "snr", None):
        console.print(f"  [{ROSE}]--snr is required.[/]")
        return

    snrs = parse_snrs(args.snr)
    topologies = [name.strip() for name in args.topologies.split(",") if name.strip()]
    size = _size(args)
    arms = _arms(methods, args)
    nest_methods = [name for name in methods if name in {"lasso", "lsco"}] if args.nestboot else []
    chosen = select_networks(size, topologies, int(args.n_datasets))
    if not chosen:
        console.print(f"  [{ROSE}]No GeneSPIDER networks for {size} and {topologies}.[/]")
        return

    run_dir = Path(args.output).with_name(Path(args.output).stem + "_runs")
    csv_path = Path(args.output).with_suffix(".csv")
    json_path = Path(args.output)
    conditions = len(chosen) * len(snrs)
    per_condition = len(arms) + len(nest_methods)
    console.print(f"  [{TEAL}]Methods[/]     {', '.join(method for _, method, shuffled in arms if not shuffled)}")
    if getattr(args, "randomize", False):
        console.print(f"  [{TEAL}]Random[/]      gene-wise shuffle of each method")
    if nest_methods:
        console.print(
            f"  [{TEAL}]NestBoot[/]    {', '.join(nest_methods)} ({args.nest_runs}×{args.boot_runs})"
        )
    console.print(
        f"  [{TEAL}]Networks[/]    {len(chosen)} GeneSPIDER {size} networks "
        f"({args.n_datasets} per topology)"
    )
    console.print(f"  [{TEAL}]SNR[/]         {', '.join(_snr_label(snr) for snr in snrs)}")
    console.print(
        f"  [{TEAL}]Saved runs[/]  {run_dir}  ({conditions} conditions × {per_condition} arms)"
    )
    console.print()

    rows: list[dict] = []
    seed = int(getattr(args, "seed", 42))
    done = 0
    for network_index, (metas, topology) in enumerate(chosen):
        files = [_read_dataset(meta) for meta in metas]
        network_name = files[0]["network"]
        A = _read_network(network_name)
        design = max(files, key=lambda record: float(record["snr_l"] or 0.0))
        for snr_index, snr in enumerate(snrs):
            stored = match_snr(snr, files)
            if stored is None:
                rng = np.random.default_rng(seed + 1000 * network_index + snr_index)
                Y = simulate_expression(A, design["P"], snr, rng)
                P = design["P"]
                source = "simulated"
                dataset_name = f"{network_name}__snr{_snr_label(snr)}"
                console.print(
                    f"  [{ORANGE}]{topology} {network_name} SNR {_snr_label(snr)}[/] "
                    f"[{DIM}]simulated from the GeneSPIDER network and knockdown design[/]"
                )
            else:
                Y = stored["Y"]
                P = stored["P"]
                source = "genespider"
                dataset_name = stored["name"]
                console.print(
                    f"  [{GREEN}]{topology} {network_name} SNR {_snr_label(snr)}[/] "
                    f"[{DIM}]stored file {stored['name']}[/]"
                )
            condition_seed = seed + 1000 * network_index + 10 * snr_index
            jobs: list[tuple[str, str, bool, bool]] = [
                (label, method, shuffled, False) for label, method, shuffled in arms
            ]
            jobs.extend((f"nestboot {method}", method, False, True) for method in nest_methods)
            for job_index, (label, method, shuffled, nested) in enumerate(jobs):
                done += 1
                console.print(
                    f"    [{DIM}]{done}/{conditions * per_condition}[/] {label} {topology}/SNR={_snr_label(snr)}"
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
    console.print(f"\n  [{DIM}]Scores saved to {csv_path}[/]")
    console.print(f"  [{DIM}]Inferred networks saved to {run_dir}[/]")
    if getattr(args, "plot", False):
        figure = Path(args.output).with_suffix(".png")
        _write_snr_figure(csv_path, figure)
        console.print(f"  [{DIM}]Figure saved to {figure}[/]")
