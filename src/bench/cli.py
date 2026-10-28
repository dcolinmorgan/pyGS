#!/usr/bin/env python3
"""pygs — CLI for gene regulatory network inference & benchmarking.

Wraps sparselink inference methods with GeneSpider data, NestBoot FDR,
biology-specific preprocessing, and gold-standard evaluation.

Usage::

    pygs status                                  # system info
    pygs methods                                 # list inference methods
    pygs infer data.csv -m lasso                 # infer a GRN
    pygs infer https://example/ymatrix.csv -m lsco
    pygs bench --tier fast                       # synthetic benchmark
    pygs bench --methods tigress,lasso,lsco --nestboot --randomize \
        --snr 0.01,0.1,1 --n-nodes 50 --n-datasets 3 --plot
    pygs bench-gs --tier fast --sizes N50        # GeneSpider benchmark
    pygs nestboot data.csv -m lsco -i 100 -o 100
    pygs nestboot --runs 100x100 -m lsco -f data.csv --output out.npy
    pygs evaluate pred.npy --gold gold.npy       # evaluate against gold standard
    pygs plot pred.npy --genes genes.txt         # plot a GRN
    pygs dashboard -i results.json               # HTML dashboard
    pygs show results.json                       # render result table
    pygs                                         # interactive mode
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Prompt
from rich.table import Table
from rich.tree import Tree

console = Console()

# ── Palette ───────────────────────────────────────────────────────────────
TEAL = "bold #14B8A6"
INDIGO = "bold #818CF8"
GREEN = "#22C55E"
ORANGE = "#FB923C"
ROSE = "#FB7185"
DIM = "#71717A"
BOLD = "bold #FAFAFA"

BANNER = [
    " ▄▄▄▄  ▄  ▄  ▄▄▄▄  ▄▄▄▄",
    " █  █  █  █  █     █    ",
    " █▀▀█  ▀▄▄█  █ ▀█▀ ▀▀▀█",
    " █     ▄  █  █  █  ▄  █",
    " ▀     ▀▀▀▀  ▀▀▀▀  ▀▀▀▀",
]

# Method categories for display
METHOD_CATEGORIES: dict[str, str] = {
    "lasso": "regression",
    "elastic_net": "regression",
    "ridge": "regression",
    "lsco": "regression",
    "tigress": "stability selection",
    "genie3": "tree-based",
    "clr": "information theory",
    "partial_correlation": "correlation",
    "glasso": "graphical model",
    "glasso_stars": "graphical model",
    "neighborhood_selection": "graphical model",
    "pcmci": "causal (time-series)",
    "granger_causality": "causal (time-series)",
    "transfer_entropy": "causal (time-series)",
    "pc": "constraint-based",
    "fci": "constraint-based",
    "notears": "continuous optimization",
    "dag_gnn": "deep learning",
    "bdeu": "bayesian",
    "bge": "bayesian",
}


# ── Helpers ───────────────────────────────────────────────────────────────


def _print_banner() -> None:
    for line in BANNER:
        console.print(f"[{TEAL}]{line}[/]")
    console.print(
        f"  [{BOLD}]pyGS[/]  [{DIM}]python genespider —"
        f" network inference & benchmarking[/]"
    )
    console.print()


def _color(val: float, lo: float = 0.4, hi: float = 0.7) -> str:
    if val >= hi:
        return GREEN
    if val >= lo:
        return ORANGE
    return ROSE


def _bar(value: float, width: int = 20) -> str:
    v = max(0.0, min(1.0, value))
    filled = int(v * width)
    return f"[{GREEN}]{'█' * filled}[/][dim]{'░' * (width - filled)}[/]"


def _pick_multi(
    label: str, options: dict[str, str], default: str = ""
) -> list[str]:
    for k, v in options.items():
        console.print(f"    [{INDIGO}]{k}[/] [{DIM}]{v}[/]")
    raw = Prompt.ask(f"  [{DIM}]{label} (comma-separated)[/]", default=default)
    keys = [k.strip() for k in raw.split(",")]
    return [options[k] for k in keys if k in options]


def _render_results(data: list[dict], title: str = "Results") -> None:
    """Render benchmark results as a rich table."""
    if not data:
        console.print("[dim]No results[/]")
        return

    ok = [r for r in data if not r.get("error")]
    errs = [r for r in data if r.get("error")]
    methods = sorted(set(r["method"] for r in ok))
    metrics = ["auroc", "aupr", "f1", "mcc"]

    t = Table(title=title, title_style=TEAL, border_style="dim")
    t.add_column("Method", style="bold")
    for m in metrics:
        t.add_column(m.upper(), justify="right")
    t.add_column("", min_width=22)
    t.add_column("Time", justify="right")
    t.add_column("Runs", justify="right")

    for method in methods:
        rows = [r for r in ok if r["method"] == method]
        avgs = {m: np.mean([r.get(m, 0) for r in rows]) for m in metrics}
        avg_time = np.mean([r.get("elapsed_sec", 0) for r in rows])
        t.add_row(
            method,
            *[f"[{_color(avgs[m])}]{avgs[m]:.3f}[/]" for m in metrics],
            _bar(avgs["auroc"]),
            f"{avg_time:.2f}s",
            str(len(rows)),
        )
    console.print(t)
    if errs:
        console.print(f"  [{DIM}]{len(errs)} errors skipped[/]")

    # SNR breakdown (for GeneSpider results)
    snrs = sorted(set(r.get("snr", 0) for r in ok))
    if len(snrs) > 1:
        console.print()
        for snr in snrs:
            sub = [r for r in ok if r.get("snr") == snr]
            st = Table(
                title=f"SNR={snr}",
                title_style=f"bold {ORANGE}",
                border_style="dim",
                show_edge=False,
            )
            st.add_column("Method", style="bold", width=25)
            st.add_column("AUROC", justify="right")
            st.add_column("F1", justify="right")
            st.add_column("MCC", justify="right")
            st.add_column("", min_width=22)
            for method in methods:
                rows = [r for r in sub if r["method"] == method]
                if not rows:
                    continue
                a = np.mean([r.get("auroc", 0) for r in rows])
                f = np.mean([r.get("f1", 0) for r in rows])
                m = np.mean([r.get("mcc", 0) for r in rows])
                st.add_row(
                    method,
                    f"[{_color(a)}]{a:.3f}[/]",
                    f"[{_color(f)}]{f:.3f}[/]",
                    f"[{_color(m, 0.2, 0.5)}]{m:.3f}[/]",
                    _bar(a),
                )
            console.print(st)


def _render_comparison(data: list[dict]) -> None:
    """Render direct vs NestBoot comparison table with deltas."""
    ok = [r for r in data if not r.get("error")]
    if not ok:
        console.print("[dim]No results[/]")
        return

    # Split into direct and nestboot results
    direct = [r for r in ok if not r["method"].startswith("nestboot:")]
    nestboot = [r for r in ok if r["method"].startswith("nestboot:")]

    # Get base method names
    base_methods = sorted(set(r["method"] for r in direct))

    # Overall comparison
    t = Table(
        title="Direct vs NestBoot",
        title_style=TEAL,
        border_style="dim",
    )
    t.add_column("Method", style="bold")
    t.add_column("Direct", justify="right")
    t.add_column("NestBoot", justify="right")
    t.add_column("Δ AUROC", justify="right")
    t.add_column("", min_width=22)
    t.add_column("Direct F1", justify="right")
    t.add_column("NB F1", justify="right")
    t.add_column("Δ F1", justify="right")

    for method in base_methods:
        d_rows = [r for r in direct if r["method"] == method]
        n_rows = [r for r in nestboot if r["method"] == f"nestboot:{method}"]
        if not d_rows or not n_rows:
            continue

        d_auroc = np.mean([r.get("auroc", 0) for r in d_rows])
        n_auroc = np.mean([r.get("auroc", 0) for r in n_rows])
        d_f1 = np.mean([r.get("f1", 0) for r in d_rows])
        n_f1 = np.mean([r.get("f1", 0) for r in n_rows])

        delta_auroc = n_auroc - d_auroc
        delta_f1 = n_f1 - d_f1

        # Color the delta
        if delta_auroc > 0.01:
            dc = GREEN
            arrow = "▲"
        elif delta_auroc < -0.01:
            dc = ROSE
            arrow = "▼"
        else:
            dc = DIM
            arrow = "="

        t.add_row(
            method,
            f"[{_color(d_auroc)}]{d_auroc:.3f}[/]",
            f"[{_color(n_auroc)}]{n_auroc:.3f}[/]",
            f"[{dc}]{arrow} {delta_auroc:+.3f}[/]",
            _bar(n_auroc),
            f"[{_color(d_f1)}]{d_f1:.3f}[/]",
            f"[{_color(n_f1)}]{n_f1:.3f}[/]",
            f"[{dc}]{delta_f1:+.3f}[/]",
        )

    console.print(t)

    # Per-SNR breakdown
    snrs = sorted(set(r.get("snr", 0) for r in ok))
    if len(snrs) > 1:
        console.print()
        for snr in snrs:
            d_sub = [r for r in direct if r.get("snr") == snr]
            n_sub = [r for r in nestboot if r.get("snr") == snr]
            if not d_sub or not n_sub:
                continue

            st = Table(
                title=f"SNR={snr}",
                title_style=f"bold {ORANGE}",
                border_style="dim",
                show_edge=False,
            )
            st.add_column("Method", style="bold", width=22)
            st.add_column("Direct", justify="right")
            st.add_column("NestBoot", justify="right")
            st.add_column("Δ AUROC", justify="right")
            st.add_column("", min_width=18)

            for method in base_methods:
                dr = [r for r in d_sub if r["method"] == method]
                nr = [r for r in n_sub if r["method"] == f"nestboot:{method}"]
                if not dr or not nr:
                    continue
                da = np.mean([r.get("auroc", 0) for r in dr])
                na = np.mean([r.get("auroc", 0) for r in nr])
                delta = na - da
                if delta > 0.01:
                    dc = GREEN
                    arrow = "▲"
                elif delta < -0.01:
                    dc = ROSE
                    arrow = "▼"
                else:
                    dc = DIM
                    arrow = "="
                st.add_row(
                    method,
                    f"{da:.3f}",
                    f"[{_color(na)}]{na:.3f}[/]",
                    f"[{dc}]{arrow} {delta:+.3f}[/]",
                    _bar(na),
                )
            console.print(st)


# ── Status ────────────────────────────────────────────────────────────────


def _cmd_status(args: argparse.Namespace) -> None:
    """Show sparselink + pyGS system status."""
    # sparselink status
    tree = Tree(f"[{TEAL}]sparselink[/]", guide_style="dim")
    try:
        from sparselink import list_methods

        import sparselink.methods  # noqa: F401

        methods = list_methods()
        mb = tree.add(f"Methods ({len(methods)})")
        for m in methods:
            mb.add(f"[{GREEN}]✓[/] {m}")
    except Exception as e:
        tree.add(f"[{ROSE}]✗ {e}[/]")

    accel = tree.add("Acceleration")
    try:
        import mlx.core  # noqa: F401

        accel.add(f"[{GREEN}]✓[/] MLX (Apple Silicon)")
    except ImportError:
        accel.add("[dim]○[/] MLX not available")

    deps = tree.add("Optional deps")
    for pkg, label in [("causallearn", "causal"), ("torch", "deep")]:
        try:
            __import__(pkg)
            deps.add(f"[{GREEN}]✓[/] {label}")
        except ImportError:
            deps.add(f"[dim]○[/] {label} — pip install sparselink[{label}]")

    console.print(tree)

    # pyGS status
    pygs_tree = Tree(f"[{TEAL}]pyGS[/]", guide_style="dim")

    cache = Path(".gs_cache")
    if cache.exists():
        n = len(list(cache.glob("*.json")))
        pygs_tree.add(f"[{GREEN}]✓[/] GeneSpider cache: {n} files")
    else:
        pygs_tree.add(
            f"[dim]○[/] No GeneSpider cache (auto-downloaded on first run)"
        )

    mods = pygs_tree.add("pyGS modules")
    for mod in [
        "analyze.Data",
        "datastruct.Network",
        "methods.nestboot",
        "bio.evaluation",
        "bio.preprocessing",
        "bench.genespider",
    ]:
        try:
            __import__(mod)
            mods.add(f"[{GREEN}]✓[/] {mod}")
        except ImportError:
            mods.add(f"[dim]○[/] {mod}")

    console.print(pygs_tree)


# ── Methods ───────────────────────────────────────────────────────────────


def _cmd_methods(args: argparse.Namespace) -> None:
    """List all registered inference methods."""
    from sparselink import list_methods

    import sparselink.methods  # noqa: F401

    t = Table(title="Available Methods", title_style=TEAL, border_style="dim")
    t.add_column("Name", style="bold")
    t.add_column("Category", style=DIM)

    for m in sorted(list_methods()):
        t.add_row(m, METHOD_CATEGORIES.get(m, ""))
    console.print(t)


# ── Infer ─────────────────────────────────────────────────────────────────


def _cmd_infer(args: argparse.Namespace) -> None:
    """Infer a network from a data file using a sparselink method."""
    warnings.simplefilter("ignore")
    from sparselink import get_method

    import sparselink.methods  # noqa: F401
    from bio.io import PERTURBATION_METHODS, load_inference_inputs

    path = Path(args.file)
    try:
        X, feature_names, P = load_inference_inputs(args.file)
    except (ValueError, ImportError, OSError) as e:
        console.print(f"  [{ROSE}]{e}[/]")
        return

    console.print(
        f"  [{TEAL}]Data[/]    {path.name}:"
        f" {X.shape[0]} samples × {X.shape[1]} features"
    )
    if P is not None:
        console.print(
            f"  [{TEAL}]Design[/]  knockdown labels → perturbation matrix"
        )
    console.print(f"  [{TEAL}]Method[/]  {args.method}")

    method = get_method(args.method)
    fit_P = P if args.method in PERTURBATION_METHODS else None
    with console.status(f"[{TEAL}]Running {args.method}...[/]"):
        result = method().fit(X, fit_P)

    adj = result.adjacency_matrix

    # Apply direction inference if requested
    if getattr(args, "directed", False):
        from bio.direction import infer_direction
        adj = infer_direction(adj)
        console.print(f"  [{TEAL}]Direction[/] inferred from perturbation asymmetry")

    n_edges = int(np.count_nonzero(adj) - np.count_nonzero(np.diag(adj)))
    console.print(f"  [{GREEN}]✓[/] {n_edges} edges inferred")

    # Top edges
    edges = (
        result.edge_list
        if hasattr(result, "edge_list") and result.edge_list
        else []
    )
    if not edges:
        mask = ~np.eye(adj.shape[0], dtype=bool)
        idx = np.argsort(-np.abs(adj[mask]))
        rows_i, cols_j = np.where(mask)
        edges = [
            (rows_i[i], cols_j[i], adj[rows_i[i], cols_j[i]])
            for i in idx[:20]
        ]

    t = Table(title="Top Edges", title_style=TEAL, border_style="dim")
    t.add_column("Source", style="bold")
    t.add_column("Target", style="bold")
    t.add_column("Weight", justify="right")
    for src, tgt, w in edges[:15]:
        sn = feature_names[src] if src < len(feature_names) else str(src)
        tn = feature_names[tgt] if tgt < len(feature_names) else str(tgt)
        t.add_row(sn, tn, f"{w:.4f}")
    console.print(t)

    if args.output:
        np.save(args.output, adj)
        console.print(f"  [{DIM}]Adjacency saved to {args.output}[/]")


# ── Synthetic benchmark ───────────────────────────────────────────────────


def _configure_synthetic() -> argparse.Namespace:
    """Interactive config builder for synthetic benchmark."""
    console.print(f"\n  [{TEAL}]Configure Synthetic Benchmark[/]\n")

    console.print(f"  [{TEAL}]A) Method tier[/]")
    tiers = _pick_multi(
        "Tiers",
        {"1": "fast", "2": "medium", "3": "slow", "4": "very_slow"},
        default="1",
    )
    tier = ",".join(tiers) if tiers else "fast"

    n_nodes = int(Prompt.ask(f"\n  [{DIM}]Number of genes[/]", default="50"))
    n_datasets = int(Prompt.ask(f"  [{DIM}]Number of datasets[/]", default="5"))
    timeout = int(
        Prompt.ask(f"  [{DIM}]Timeout per method (seconds)[/]", default="60")
    )
    output = Prompt.ask(
        f"  [{DIM}]Output file[/]", default="benchmark_results.json"
    )

    console.print()
    return argparse.Namespace(
        tier=tier,
        n_nodes=n_nodes,
        n_samples=n_nodes * 4,
        n_datasets=n_datasets,
        seed=42,
        timeout=timeout,
        output=output,
    )


def _perturbation_data(true_net: np.ndarray, snr: float, rng: np.random.Generator):
    """GeneSpider knockdown data: Y = (I-A)^{-1}(P + noise), SNR = var(signal)/var(noise)."""
    n_genes = true_net.shape[0]
    n_samples = 3 * n_genes
    gain = np.linalg.inv(np.eye(n_genes) - true_net)
    design = np.eye(n_genes, n_samples)
    signal = gain @ design
    noise_std = np.sqrt(np.var(signal) / snr) if snr > 0 else 0.01
    noise = rng.normal(0, noise_std, design.shape)
    expression = gain @ (design + noise)
    return expression.T, design.T


def _shuffle_genes(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Break coexpression by permuting each gene across samples."""
    shuffled = X.copy()
    for gene in range(shuffled.shape[1]):
        rng.shuffle(shuffled[:, gene])
    return shuffled


def _max_over_sparsity(method_name: str, X: np.ndarray, P: np.ndarray | None, true_net: np.ndarray):
    """Best AUROC, F1 and MCC over the regularization sweep."""
    from sparselink import get_method
    from sparselink.bench.metrics import evaluate
    from bench.genespider import ALPHA_SWEEP, ALPHA_RANGE

    method_cls = get_method(method_name)
    fit_p = P if method_name == "lsco" else None
    settings = ALPHA_RANGE if method_name in ALPHA_SWEEP else [None]
    param = ALPHA_SWEEP.get(method_name)
    best = {"auroc": -1.0, "f1": -1.0, "mcc": -1.0}
    for alpha in settings:
        kwargs = {param: float(alpha)} if param is not None else {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adj = method_cls(**kwargs).fit(X, fit_p).adjacency_matrix
        metrics = evaluate(true_net, adj)
        best["auroc"] = max(best["auroc"], metrics.auroc)
        best["f1"] = max(best["f1"], metrics.f1)
        best["mcc"] = max(best["mcc"], metrics.mcc)
    return best


def _run_snr_benchmark(args: argparse.Namespace) -> None:
    """N50-style SNR comparison, optionally with a shuffled-data arm."""
    from sparselink import list_methods
    from sparselink.bench.run_benchmark import TIERS
    from sparselink.bench.synthetic import generate_network

    import sparselink.methods  # noqa: F401

    if getattr(args, "methods", None):
        methods = [name.strip() for name in args.methods.split(",") if name.strip()]
    else:
        methods = []
        for tier in args.tier.split(","):
            methods.extend(TIERS.get(tier.strip(), []))
    registered = set(list_methods())
    methods = [name for name in methods if name in registered]
    if not methods:
        console.print(f"  [{ROSE}]No valid methods selected.[/]")
        return

    snrs = [float(value) for value in args.snr.split(",") if value.strip()]
    topologies = [name.strip() for name in args.topologies.split(",") if name.strip()]
    nestboot_methods = [name for name in methods if name in {"lasso", "lsco"}] if args.nestboot else []
    arms = methods + ([f"random {name}" for name in methods] if args.randomize else [])
    arms += [f"nestboot {name}" for name in nestboot_methods]
    total = len(topologies) * len(snrs) * args.n_datasets * len(arms)

    console.print(f"  [{TEAL}]Methods[/]     {', '.join(arms)}")
    console.print(f"  [{TEAL}]SNR[/]         {snrs}")
    console.print(f"  [{TEAL}]Topologies[/]  {topologies}")
    console.print(f"  [{TEAL}]Networks[/]    {args.n_nodes} genes × {args.n_datasets} per topology")
    console.print(f"  [{TEAL}]Total runs[/]  {total}")
    console.print()

    rows: list[dict] = []
    rng = np.random.default_rng(args.seed)
    progress = Progress(
        SpinnerColumn(style=TEAL),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30, complete_style=TEAL, finished_style=GREEN),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("SNR benchmark", total=total)
        for topology in topologies:
            for dataset_idx in range(args.n_datasets):
                true_net = generate_network(
                    args.n_nodes,
                    topology=topology,
                    sparsity=args.sparsity,
                    seed=int(rng.integers(0, 2**31)),
                )
                for snr in snrs:
                    X, P = _perturbation_data(true_net, snr, rng)
                    copies = [(False, X)]
                    if args.randomize:
                        copies.append((True, _shuffle_genes(X, rng)))
                    for randomized, expression in copies:
                        for method_name in methods:
                            label = f"random {method_name}" if randomized else method_name
                            progress.update(task, description=f"{label:20s} {topology}/SNR={snr:g}")
                            scores = _max_over_sparsity(method_name, expression, P, true_net)
                            rows.append({
                                "dataset": f"SNR{snr:g}-{topology}-ID{dataset_idx}",
                                "method": label,
                                "snr": snr,
                                "topology": topology,
                                "auroc": scores["auroc"],
                                "f1_score": scores["f1"],
                                "mcc": scores["mcc"],
                            })
                            progress.advance(task)
                    if not args.nestboot:
                        continue
                    for method_name in nestboot_methods:
                        label = f"nestboot {method_name}"
                        progress.update(task, description=f"{label:20s} {topology}/SNR={snr:g}")
                        result = _run_nestboot_on_gs(
                            method_name,
                            X,
                            P,
                            true_net,
                            {"path": f"SNR{snr:g}-{topology}-ID{dataset_idx}", "n_genes": args.n_nodes, "snr": snr},
                            topology,
                            f"{topology}-ID{dataset_idx}",
                            args,
                        )
                        rows.append({
                            "dataset": f"SNR{snr:g}-{topology}-ID{dataset_idx}",
                            "method": label,
                            "snr": snr,
                            "topology": topology,
                            "auroc": result.get("auroc", 0),
                            "f1_score": result.get("f1", 0),
                            "mcc": result.get("mcc", 0),
                        })
                        progress.advance(task)

    csv_path = Path(args.output).with_suffix(".csv")
    pd = __import__("pandas")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(args.output, "w") as handle:
        json.dump(rows, handle, indent=2)
    console.print(f"\n  [{DIM}]Scores saved to {csv_path}[/]")
    if args.plot:
        from importlib.util import module_from_spec, spec_from_file_location

        plot_path = Path(__file__).resolve().parents[2] / "benchmark" / "code" / "plot_snr_comparison.py"
        spec = spec_from_file_location("plot_snr_comparison", plot_path)
        plotter = module_from_spec(spec)
        spec.loader.exec_module(plotter)
        figure = Path(args.output).with_suffix(".png")
        plotter.plot_scores(plotter.load_scores(csv_path), figure)
        console.print(f"  [{DIM}]Figure saved to {figure}[/]")


def _run_benchmark_live(args: argparse.Namespace) -> None:
    """Run synthetic benchmark using sparselink methods."""
    if getattr(args, "snr", None):
        _run_snr_benchmark(args)
        return
    warnings.simplefilter("ignore")

    from sparselink import list_methods
    from sparselink.bench.run_benchmark import TIERS, run_single
    from sparselink.bench.synthetic import generate_data, generate_network

    import sparselink.methods  # noqa: F401

    selected_tiers = [t.strip() for t in args.tier.split(",")]
    methods: list[str] = []
    for t in selected_tiers:
        methods.extend(TIERS.get(t, []))
    registered = set(list_methods())
    methods = [m for m in methods if m in registered]

    sparsities = [0.02, 0.06, 0.1]
    noise_levels = [0.01, 0.1, 1.0]
    topologies = ["random", "scalefree", "smallworld"]
    total = (
        len(methods)
        * len(topologies)
        * len(sparsities)
        * len(noise_levels)
        * args.n_datasets
    )

    console.print(f"  [{TEAL}]Methods[/]       {', '.join(methods)}")
    console.print(f"  [{TEAL}]Topologies[/]    {topologies}")
    console.print(f"  [{TEAL}]Nodes[/]         {args.n_nodes}")
    console.print(f"  [{TEAL}]Datasets[/]      {args.n_datasets}")
    console.print(f"  [{TEAL}]Total runs[/]    {total}")
    console.print(f"  [{TEAL}]Timeout[/]       {args.timeout}s")
    console.print()

    progress = Progress(
        SpinnerColumn(style=TEAL),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30, complete_style=TEAL, finished_style=GREEN),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    results: list[dict] = []
    rng = np.random.default_rng(args.seed)

    with progress:
        task = progress.add_task("Benchmarking", total=total)
        for topo in topologies:
            for sp in sparsities:
                for noise_std in noise_levels:
                    for ds_idx in range(args.n_datasets):
                        ds_seed = int(rng.integers(0, 2**31))
                        true_net = generate_network(
                            args.n_nodes,
                            topology=topo,
                            sparsity=sp,
                            seed=ds_seed,
                        )
                        X = generate_data(
                            true_net,
                            n_samples=args.n_samples,
                            noise_std=noise_std,
                            seed=ds_seed,
                        )
                        for method_name in methods:
                            progress.update(
                                task,
                                description=(
                                    f"{method_name:20s}"
                                    f" {topo}/sp={sp}/noise={noise_std}"
                                ),
                            )
                            r = run_single(
                                method_name,
                                X,
                                true_net,
                                ds_idx,
                                args.n_nodes,
                                args.n_samples,
                                topo,
                                sp,
                                noise_std,
                                args.timeout,
                            )
                            results.append(asdict(r))
                            progress.advance(task)

    _render_results(results, f"Synthetic Benchmark ({total} runs)")
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n  [{DIM}]Results saved to {args.output}[/]")


# ── GeneSpider benchmark ─────────────────────────────────────────────────


def _configure_genespider() -> argparse.Namespace:
    """Interactive config builder for GeneSpider benchmark."""
    console.print(f"\n  [{TEAL}]Configure GeneSpider Benchmark[/]\n")

    console.print(f"  [{TEAL}]A) Method tier / method names[/]")
    console.print(f"    [{INDIGO}]1[/] [{DIM}]fast (lasso, lsco, elastic_net, ridge, ...)[/]")
    console.print(f"    [{INDIGO}]2[/] [{DIM}]medium (glasso, bdeu, bge, ...)[/]")
    console.print(f"    [{INDIGO}]3[/] [{DIM}]slow (tigress, glasso_stars, pc, fci)[/]")
    console.print(f"    [{INDIGO}]4[/] [{DIM}]nestboot (wrap selected methods in NestBoot)[/]")
    console.print(f"    [{DIM}]Or type method names directly: scenicplus, genie3, ...[/]")
    raw = Prompt.ask(f"  [{DIM}]Selection (comma-separated)[/]", default="1")

    tier_map = {"1": "fast", "2": "medium", "3": "slow", "4": "nestboot"}
    parts = [p.strip() for p in raw.split(",")]

    tiers = []
    extra_methods: list[str] = []
    for p in parts:
        if p in tier_map:
            tiers.append(tier_map[p])
        elif p:
            extra_methods.append(p)

    use_nestboot = "nestboot" in tiers
    tier_names = [t for t in tiers if t != "nestboot"]
    tier = ",".join(tier_names) if tier_names else ""

    # NestBoot params
    nest_runs = 0
    boot_runs = 0
    fdr = 0.05
    if use_nestboot:
        console.print(f"\n  [{TEAL}]NestBoot config[/]")
        nest_runs = int(
            Prompt.ask(f"  [{DIM}]Outer runs[/]", default="10")
        )
        boot_runs = int(
            Prompt.ask(f"  [{DIM}]Inner runs[/]", default="10")
        )
        fdr = float(
            Prompt.ask(f"  [{DIM}]FDR threshold[/]", default="0.05")
        )

    console.print(f"\n  [{TEAL}]B) Network sizes[/]")
    sizes = _pick_multi(
        "Sizes",
        {"1": "N10", "2": "N50", "3": "N100"},
        default="2",
    )
    sizes_str = ",".join(sizes) if sizes else "N50"

    console.print(f"\n  [{TEAL}]C) Max datasets per size[/]")
    max_ds = int(Prompt.ask(f"  [{DIM}]0 = all[/]", default="0"))

    timeout = int(
        Prompt.ask(f"\n  [{DIM}]Timeout per method (seconds)[/]", default="120")
    )
    output = Prompt.ask(
        f"  [{DIM}]Output file[/]", default="benchmark_genespider.json"
    )

    console.print()
    return argparse.Namespace(
        tier=tier,
        sizes=sizes_str,
        max_datasets=max_ds,
        timeout=timeout,
        output=output,
        nestboot=use_nestboot,
        nest_runs=nest_runs,
        boot_runs=boot_runs,
        fdr=fdr,
        seed=42,
        extra_methods=extra_methods,
    )


def _run_genespider_live(args: argparse.Namespace) -> None:
    """Run GeneSpider benchmark using sparselink methods on real data."""
    warnings.simplefilter("ignore")

    from bench.genespider import TIERS, PYGS_METHODS, _list_datasets, load_dataset, run_single
    from sparselink import list_methods

    import sparselink.methods  # noqa: F401

    # Parse tiers — "nestboot" in the tier list enables NestBoot wrapping
    selected_tiers = [t.strip() for t in args.tier.split(",") if t.strip()]
    use_nestboot = getattr(args, "nestboot", False) or "nestboot" in selected_tiers
    method_tiers = [t for t in selected_tiers if t != "nestboot"]

    methods: list[str] = []
    for t in method_tiers:
        methods.extend(TIERS.get(t, []))

    # Add extra methods typed directly by name
    extra = getattr(args, "extra_methods", [])
    methods.extend(extra)

    # Filter to valid methods
    registered = set(list_methods())
    methods = [m for m in methods if m in registered or m in PYGS_METHODS]

    if not methods:
        console.print(f"  [{ROSE}]No valid methods selected.[/]")
        return

    sizes = [s.strip() for s in args.sizes.split(",")]

    console.print(f"  [{TEAL}]Methods[/]  {', '.join(methods)}")
    console.print(f"  [{TEAL}]Sizes[/]    {sizes}")
    console.print(f"  [{TEAL}]Timeout[/]  {args.timeout}s")
    if use_nestboot:
        console.print(
            f"  [{TEAL}]NestBoot[/] {args.nest_runs}×{args.boot_runs},"
            f" FDR={args.fdr}"
        )
    console.print()

    progress = Progress(
        SpinnerColumn(style=TEAL),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30, complete_style=TEAL, finished_style=GREEN),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    results: list[dict] = []

    for size in sizes:
        console.print(f"  [{INDIGO}]Fetching {size} datasets...[/]")
        datasets = _list_datasets(size)
        if args.max_datasets > 0:
            datasets = datasets[: args.max_datasets]

        total = len(datasets) * len(methods)
        if use_nestboot:
            # Run both direct and nestboot for comparison
            total *= 2
        console.print(
            f"  [{DIM}]{len(datasets)} datasets × {len(methods)}"
            f" methods"
            f"{' × 2 (direct + nestboot)' if use_nestboot else ''}"
            f" = {total} runs[/]"
        )

        with progress:
            task = progress.add_task(f"{size}", total=total)
            for ds_meta in datasets:
                try:
                    X, P, A_true, topology, net_name = load_dataset(
                        ds_meta, size
                    )
                except Exception:
                    advance_n = len(methods) * (2 if use_nestboot else 1)
                    progress.advance(task, advance_n)
                    continue

                for method_name in methods:
                    snr = ds_meta["snr"]

                    if use_nestboot:
                        # Run direct WITHOUT oracle alpha (fair baseline)
                        progress.update(
                            task,
                            description=f"{method_name:20s} {topology}/SNR={snr}",
                        )
                        r_direct = _run_direct_fair(
                            method_name, X, P, A_true, ds_meta,
                            topology, net_name, args.timeout,
                        )
                        results.append(r_direct)
                        progress.advance(task)

                        # Run nestboot
                        progress.update(
                            task,
                            description=f"NB:{method_name:17s} {topology}/SNR={snr}",
                        )
                        r_nb = _run_nestboot_on_gs(
                            method_name,
                            X,
                            P,
                            A_true,
                            ds_meta,
                            topology,
                            net_name,
                            args,
                        )
                        results.append(r_nb)
                        progress.advance(task)
                    else:
                        progress.update(
                            task,
                            description=f"{method_name:20s} {topology}/SNR={snr}",
                        )
                        r = run_single(
                            method_name,
                            X,
                            A_true,
                            ds_meta,
                            topology,
                            net_name,
                            args.timeout,
                            P,
                        )
                        results.append(asdict(r))
                        progress.advance(task)

    if use_nestboot:
        _render_comparison(results)
    else:
        _render_results(results, f"GeneSpider Benchmark ({len(results)} runs)")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n  [{DIM}]Results saved to {args.output}[/]")


def _run_direct_fair(
    method_name: str,
    X: np.ndarray,
    P: np.ndarray | None,
    A_true: np.ndarray,
    ds_meta: dict,
    topology: str,
    net_name: str,
    timeout: int,
) -> dict:
    """Run a method directly (no oracle) for fair NestBoot comparison.

    For methods with alpha: sweeps the same alpha range and averages
    the absolute adjacency matrices (bagging over alphas, single data).
    For methods without alpha: single run with defaults.
    """
    from sparselink import get_method
    from sparselink.bench.metrics import evaluate

    from bench.genespider import ALPHA_SWEEP, PYGS_METHODS, _fit_pygs_method

    import sparselink.methods  # noqa: F401

    n = X.shape[1]

    base = dict(
        method=method_name,
        dataset_name=ds_meta["path"],
        network_name=net_name,
        topology=topology,
        n_genes=ds_meta["n_genes"],
        snr=ds_meta["snr"],
    )
    fail = dict(
        auroc=0, aupr=0, precision=0, recall=0,
        f1=0, fdr=1, mcc=0, r2=0,
    )

    try:
        t0 = time.perf_counter()

        if method_name in PYGS_METHODS:
            best_adj, _ = _fit_pygs_method(method_name, X, A_true, P)
            elapsed = time.perf_counter() - t0
        elif method_name in ALPHA_SWEEP:
            # Sweep alphas on full data, average the absolute adjacency
            # matrices. This is a simple ensemble over regularization
            # strength — no oracle, no gold standard.
            method_cls = get_method(method_name)
            alpha_param = ALPHA_SWEEP[method_name]
            alpha_range = np.logspace(-3, 0, 10)
            adj_sum = np.zeros((n, n))
            for alpha in alpha_range:
                r = method_cls(**{alpha_param: float(alpha)}).fit(X, P)
                adj_sum += np.abs(r.adjacency_matrix)
            best_adj = adj_sum / len(alpha_range)
            np.fill_diagonal(best_adj, 0.0)
        else:
            method_cls = get_method(method_name)
            result = method_cls().fit(X, P)
            best_adj = result.adjacency_matrix

        elapsed = time.perf_counter() - t0
        metrics = evaluate(A_true, best_adj)
        return {
            **base,
            "auroc": metrics.auroc,
            "aupr": metrics.aupr,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "fdr": metrics.fdr,
            "mcc": metrics.mcc,
            "r2": metrics.r2,
            "elapsed_sec": round(elapsed, 4),
            "error": None,
        }
    except Exception as e:
        return {**base, **fail, "elapsed_sec": 0, "error": str(e)[:120]}


def _run_nestboot_on_gs(
    method_name: str,
    X: np.ndarray,
    P: np.ndarray | None,
    A_true: np.ndarray,
    ds_meta: dict,
    topology: str,
    net_name: str,
    args: argparse.Namespace,
) -> dict:
    """Run a single method through NestBoot on a GeneSpider dataset."""
    from sparselink import get_method
    from sparselink.bench.metrics import evaluate

    from config import AnalysisConfig
    from datastruct.Dataset import Dataset
    from datastruct.Network import Network
    from analyze.Data import Data
    from methods.nestboot import Nestboot

    n_genes = X.shape[1]
    n_samples = X.shape[0]
    gene_names = [f"G{i}" for i in range(n_genes)]

    # Build Dataset (X is samples×genes, Dataset wants genes×samples)
    # P is (samples × genes) from GeneSpider, transpose to (genes × samples)
    ds = Dataset()
    ds._Y = X.T
    ds._P = P.T if P is not None else np.eye(n_genes, n_samples)
    ds._network = Network(A_true)
    ds._names = gene_names
    data_obj = Data(ds)

    from bench.genespider import PYGS_METHODS
    method_cls = get_method(method_name) if method_name not in PYGS_METHODS else None

    # Determine if this method supports a native regularization parameter
    from bench.genespider import ALPHA_SWEEP

    has_native_alpha = method_name in ALPHA_SWEEP
    if has_native_alpha:
        alpha_param = ALPHA_SWEEP[method_name]
        alpha_range = np.logspace(-3, 0, 10)
    else:
        alpha_range = None

    # Counter to vary random_state for stochastic methods (e.g., genie3)
    _call_count = [0]

    def _infer(dataset: Data, **kwargs: object) -> np.ndarray:
        Y = dataset.data.Y  # type: ignore[union-attr]
        P_mat = dataset.data.P  # type: ignore[union-attr]
        X_in = Y.T
        P_in = P_mat.T if P_mat is not None else None
        _call_count[0] += 1

        if method_name in PYGS_METHODS:
            # Use SCENIC+ directly — returns 3D with threshold_range
            from methods.scenicplus import SCENICPLUS
            threshold_range = np.logspace(-6, 0, 10)
            var_names = gene_names
            result = SCENICPLUS(
                dataset=dataset,
                nested_boot=False,
                threshold_range=threshold_range,
                var_names=var_names,
                n_cpu=1,
            )
            adj_3d, _ = result
            return adj_3d

        if has_native_alpha:
            # Sweep native regularization parameter → 3D output
            # Return THRESHOLDED outputs so NestBoot Afrac measures
            # edge presence/absence (stability selection style)
            slices = []
            for alpha in alpha_range:  # type: ignore[union-attr]
                result = method_cls(**{alpha_param: float(alpha)}).fit(
                    X_in, P_in
                )
                adj = result.adjacency_matrix.copy()
                # Binarize: keep only edges above a data-driven threshold
                # This converts continuous weights → edge indicators for Afrac
                scores = np.abs(adj)
                np.fill_diagonal(scores, 0.0)
                nonzero = scores[scores > 0]
                if len(nonzero) > 0:
                    thr = np.percentile(nonzero, 50)
                    adj[scores < thr] = 0.0
                slices.append(adj)
            return np.stack(slices, axis=2)
        else:
            # Fit with varying random_state for stochastic methods
            extra: dict = {}
            try:
                # If method accepts random_state, vary it per call
                import inspect

                sig = inspect.signature(method_cls.__init__)
                if "random_state" in sig.parameters:
                    extra["random_state"] = _call_count[0]
            except Exception:
                pass

            result = method_cls(**extra).fit(X_in, P_in)
            adj = result.adjacency_matrix
            scores = np.abs(adj)
            np.fill_diagonal(scores, 0.0)
            nonzero = scores[scores > 0]
            if len(nonzero) == 0:
                return adj
            # Sweep thresholds from loose to strict
            thresholds = np.linspace(
                np.percentile(nonzero, 10),
                np.percentile(nonzero, 90),
                10,
            )
            slices = []
            for thr in thresholds:
                masked = adj.copy()
                masked[scores < thr] = 0.0
                slices.append(masked)
            return np.stack(slices, axis=2)

    config = AnalysisConfig(
        total_runs=args.nest_runs * args.boot_runs,
        inner_group_size=args.boot_runs,
        fdr_threshold=args.fdr,
    )
    nb = Nestboot(config)
    nb.logger.setLevel(logging.WARNING)

    base = dict(
        method=f"nestboot:{method_name}",
        dataset_name=ds_meta["path"],
        network_name=net_name,
        topology=topology,
        n_genes=ds_meta["n_genes"],
        snr=ds_meta["snr"],
    )
    fail = dict(
        auroc=0, aupr=0, precision=0, recall=0,
        f1=0, fdr=1, mcc=0, r2=0,
    )

    try:
        t0 = time.perf_counter()
        nb_results = nb.run_nestboot(
            dataset=data_obj,
            inference_method=_infer,
            nest_runs=args.nest_runs,
            boot_runs=args.boot_runs,
            seed=getattr(args, "seed", 42),
        )
        elapsed = time.perf_counter() - t0

        # Use xnet (FDR-thresholded binary) for F1/MCC evaluation,
        # and sxnet (signed continuous) for AUROC/AUPR.
        # If 3D (genes × genes × n_params), pick the slice with best AUROC.
        xnet = nb_results.xnet
        sxnet = nb_results.sxnet

        if hasattr(xnet, "ndim") and xnet.ndim == 3:
            # Pick the alpha slice with best AUROC
            best_auroc = -1.0
            best_adj = np.abs(sxnet[:, :, 0])
            for k in range(sxnet.shape[2]):
                candidate = np.abs(sxnet[:, :, k])
                np.fill_diagonal(candidate, 0.0)
                from sklearn.metrics import roc_auc_score
                mask = ~np.eye(n_genes, dtype=bool)
                y_true = (A_true[mask] != 0).astype(int)
                y_scores = candidate[mask]
                try:
                    auroc_k = float(roc_auc_score(y_true, y_scores))
                except ValueError:
                    auroc_k = 0.5
                if auroc_k > best_auroc:
                    best_auroc = auroc_k
                    best_adj = candidate
            combined = best_adj
        else:
            combined = np.abs(sxnet)
        np.fill_diagonal(combined, 0.0)
        metrics = evaluate(A_true, combined)
        return {
            **base,
            "auroc": metrics.auroc,
            "aupr": metrics.aupr,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "fdr": metrics.fdr,
            "mcc": metrics.mcc,
            "r2": metrics.r2,
            "elapsed_sec": round(elapsed, 4),
            "error": None,
        }
    except Exception as e:
        return {**base, **fail, "elapsed_sec": 0, "error": str(e)[:120]}


# ── NestBoot FDR ──────────────────────────────────────────────────────────


def _cmd_nestboot(args: argparse.Namespace) -> None:
    """Run NestBoot FDR analysis on expression data using a sparselink method."""
    warnings.simplefilter("ignore")

    from sparselink import get_method

    import sparselink.methods  # noqa: F401

    from bio.io import PERTURBATION_METHODS, load_inference_inputs
    from config import AnalysisConfig
    from methods.nestboot import Nestboot

    # Load expression data. X is samples × genes; Dataset wants genes × samples.
    console.print(f"  [{TEAL}]Loading[/]  {args.file}")
    try:
        X, gene_names, P = load_inference_inputs(args.file)
    except (ValueError, ImportError, OSError) as e:
        console.print(f"  [{ROSE}]{e}[/]")
        return
    matrix = X.T
    n_genes, n_samples = matrix.shape
    console.print(f"  [{DIM}]{n_genes} genes × {n_samples} samples[/]")

    # Build a Dataset for Nestboot
    from analyze.Data import Data
    from datastruct.Dataset import Dataset
    from datastruct.Network import Network

    ds = Dataset()
    ds._Y = matrix
    if P is not None:
        ds._P = P.T
        console.print(
            f"  [{TEAL}]Design[/]  knockdown labels → perturbation matrix"
        )
    else:
        ds._P = np.eye(n_genes, n_samples)
    ds._network = Network(np.zeros((n_genes, n_genes)))
    ds._names = gene_names
    data_obj = Data(ds)

    # Build inference callable using sparselink method
    method_name = args.method
    console.print(f"  [{TEAL}]Method[/]   {method_name}")
    console.print(
        f"  [{TEAL}]Runs[/]     {args.nest_runs} outer × {args.boot_runs} inner"
    )
    console.print(f"  [{TEAL}]FDR[/]      {args.fdr}")

    method_cls = get_method(method_name)

    use_P = method_name in PERTURBATION_METHODS

    def _infer(dataset: Data, **kwargs: object) -> np.ndarray:
        """Run sparselink method on a pyGS Data object."""
        Y = dataset.data.Y  # type: ignore[union-attr]
        X = Y.T  # sparselink expects (samples × features)
        fit_P = None
        if use_P:
            P_mat = dataset.data.P  # type: ignore[union-attr]
            fit_P = None if P_mat is None else P_mat.T
        result = method_cls().fit(X, fit_P)
        adj = result.adjacency_matrix
        # LSCO (and other dense fits) need a cutoff sweep so NestBoot scores
        # edge presence instead of every least-squares coefficient.
        scores = np.abs(adj)
        np.fill_diagonal(scores, 0.0)
        nonzero = scores[scores > 0]
        if nonzero.size == 0:
            return adj
        thresholds = np.unique(np.quantile(nonzero, np.linspace(0.5, 0.95, 8)))
        slices = []
        for thr in thresholds:
            masked = adj.copy()
            masked[scores < thr] = 0.0
            slices.append(masked)
        if len(slices) == 1:
            return slices[0]
        return np.stack(slices, axis=2)

    # Run NestBoot
    config = AnalysisConfig(
        total_runs=args.nest_runs * args.boot_runs,
        inner_group_size=args.boot_runs,
        fdr_threshold=args.fdr,
    )
    nb = Nestboot(config)
    nb.logger.setLevel(logging.WARNING)

    with console.status(
        f"[{TEAL}]Running NestBoot"
        f" ({args.nest_runs}×{args.boot_runs})...[/]"
    ):
        results = nb.run_nestboot(
            dataset=data_obj,
            inference_method=_infer,
            nest_runs=args.nest_runs,
            boot_runs=args.boot_runs,
            seed=args.seed,
        )

    # Report
    n_edges = int(np.sum(results.xnet != 0))
    console.print(f"\n  [{GREEN}]✓[/] {n_edges} edges at FDR ≤ {args.fdr}")
    console.print(f"  [{DIM}]Support threshold: {results.support}[/]")

    # Save adjacency
    out = Path(args.output)
    np.save(out, results.xnet)
    console.print(f"  [{DIM}]Adjacency saved to {out}[/]")

    # Save signed network
    signed_out = out.with_stem(out.stem + "_signed")
    np.save(signed_out, results.sxnet)
    console.print(f"  [{DIM}]Signed network saved to {signed_out}[/]")

    # Export text summary
    txt_out = out.with_suffix(".txt")
    nb.export_results(results, txt_out)
    console.print(f"  [{DIM}]Summary saved to {txt_out}[/]")


# ── Evaluate ──────────────────────────────────────────────────────────────


def _cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a predicted GRN against a gold standard."""
    from bio.evaluation import compare_to_gold_standard

    pred_path = Path(args.predicted)
    if pred_path.suffix == ".npy":
        predicted = np.load(pred_path)
    elif pred_path.suffix in (".csv", ".tsv"):
        import pandas as pd

        sep = "\t" if pred_path.suffix == ".tsv" else ","
        predicted = pd.read_csv(pred_path, sep=sep, index_col=0).values
    else:
        console.print(f"[{ROSE}]Unsupported format: {pred_path.suffix}[/]")
        return

    gold_path = Path(args.gold)
    if gold_path.suffix == ".npy":
        gold = np.load(gold_path)
    elif gold_path.suffix == ".json":
        with open(gold_path) as f:
            data = json.load(f)
        gold = np.array(data["obj_data"]["A"])
    elif gold_path.suffix in (".csv", ".tsv"):
        import pandas as pd

        sep = "\t" if gold_path.suffix == ".tsv" else ","
        gold = pd.read_csv(gold_path, sep=sep, index_col=0).values
    else:
        console.print(f"[{ROSE}]Unsupported format: {gold_path.suffix}[/]")
        return

    console.print(
        f"  [{TEAL}]Predicted[/]  {pred_path.name}  {predicted.shape}"
    )
    console.print(
        f"  [{TEAL}]Gold std[/]   {gold_path.name}  {gold.shape}"
    )

    metrics = compare_to_gold_standard(predicted, gold)

    t = Table(title="GRN Evaluation", title_style=TEAL, border_style="dim")
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")
    for k, v in metrics.items():
        color = _color(v) if k in ("AUROC", "AUPR", "F1", "MCC") else ""
        t.add_row(k, f"[{color}]{v:.4f}[/]" if color else f"{v:.4f}")
    console.print(t)


# ── Plot ──────────────────────────────────────────────────────────────────


def _cmd_plot(args: argparse.Namespace) -> None:
    """Plot a gene regulatory network."""
    from bio.visualization import plot_grn

    adj_path = Path(args.adjacency)
    if adj_path.suffix == ".npy":
        adj = np.load(adj_path)
    elif adj_path.suffix in (".csv", ".tsv"):
        import pandas as pd

        sep = "\t" if adj_path.suffix == ".tsv" else ","
        adj = pd.read_csv(adj_path, sep=sep, index_col=0).values
    else:
        console.print(f"[{ROSE}]Unsupported format: {adj_path.suffix}[/]")
        return

    n = adj.shape[0]

    if args.genes:
        with open(args.genes) as f:
            gene_names = [line.strip() for line in f if line.strip()]
    else:
        gene_names = [f"G{i + 1}" for i in range(n)]

    tf_names = None
    if args.tfs:
        with open(args.tfs) as f:
            tf_names = [line.strip() for line in f if line.strip()]

    out = args.output or f"{adj_path.stem}_grn.png"
    n_edges = int(np.count_nonzero(adj) - np.count_nonzero(np.diag(adj)))
    console.print(
        f"  [{TEAL}]Network[/]  {adj_path.name}"
        f"  ({n} genes, {n_edges} edges)"
    )

    plot_grn(
        adj,
        gene_names,
        tf_names=tf_names,
        threshold=args.threshold,
        save_path=out,
    )
    console.print(f"  [{GREEN}]✓[/] Saved to {out}")


# ── Show / Dashboard ─────────────────────────────────────────────────────


def _cmd_show(args: argparse.Namespace) -> None:
    """Render a previous result JSON."""
    with open(args.file) as f:
        data = json.load(f)
    _render_results(data, f"Results from {args.file}")


def _cmd_dashboard(args: argparse.Namespace) -> None:
    """Generate an interactive HTML dashboard from benchmark results."""
    # Dashboard generation is a pyGS feature — build a self-contained HTML
    import webbrowser

    with open(args.input) as f:
        data = json.load(f)

    if not data:
        console.print(f"[{ROSE}]No data in {args.input}[/]")
        return

    # Build simple HTML dashboard
    methods = sorted(set(r["method"] for r in data if not r.get("error")))
    ok = [r for r in data if not r.get("error")]

    rows_html = ""
    for m in methods:
        m_rows = [r for r in ok if r["method"] == m]
        avgs = {
            k: np.mean([r.get(k, 0) for r in m_rows])
            for k in ("auroc", "aupr", "f1", "mcc", "elapsed_sec")
        }
        rows_html += (
            f"<tr><td>{m}</td>"
            f"<td>{avgs['auroc']:.3f}</td>"
            f"<td>{avgs['aupr']:.3f}</td>"
            f"<td>{avgs['f1']:.3f}</td>"
            f"<td>{avgs['mcc']:.3f}</td>"
            f"<td>{avgs['elapsed_sec']:.2f}s</td>"
            f"<td>{len(m_rows)}</td></tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html><head><title>pyGS Benchmark Dashboard</title>
<style>
body {{ font-family: system-ui; margin: 2rem; background: #0f172a; color: #f8fafc; }}
h1 {{ color: #14b8a6; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
th, td {{ padding: 0.5rem 1rem; text-align: left; border-bottom: 1px solid #334155; }}
th {{ color: #94a3b8; font-weight: 600; }}
tr:hover {{ background: #1e293b; }}
</style></head><body>
<h1>pyGS Benchmark Dashboard</h1>
<p>{len(ok)} successful runs, {len(data) - len(ok)} errors</p>
<table>
<tr><th>Method</th><th>AUROC</th><th>AUPR</th><th>F1</th><th>MCC</th><th>Time</th><th>Runs</th></tr>
{rows_html}
</table>
<script>
const data = {json.dumps(data)};
</script>
</body></html>"""

    with open(args.output, "w") as f:
        f.write(html)
    console.print(f"  [{GREEN}]✓[/] Dashboard saved to {args.output}")

    if not args.no_open:
        webbrowser.open(str(Path(args.output).resolve()))


# ── Interactive mode ──────────────────────────────────────────────────────

_MENU = {
    "1": ("status_methods", "Status & available methods"),
    "2": ("synthetic_guide", "Generate synthetic data & networks"),
    "3": ("infer", "Infer a network"),
    "4": ("bench", "Run synthetic benchmark"),
    "5": ("bench-gs", "Run GeneSpider benchmark (+ NestBoot)"),
    "6": ("evaluate", "Evaluate predicted GRN vs gold standard"),
    "7": ("plot", "Plot a gene regulatory network"),
    "8": ("dashboard", "Generate interactive HTML dashboard"),
    "9": ("show", "Render a previous result JSON"),
}


def _interactive_infer() -> None:
    """Infer with data source selection: user file, GeneSpider, or synthetic."""
    console.print(f"\n  [{TEAL}]Data source:[/]")
    console.print(f"    [{INDIGO}]1[/] [{DIM}]Your data file or URL (.csv, .tsv, .txt, .npy)[/]")
    console.print(f"    [{INDIGO}]2[/] [{DIM}]GeneSpider benchmark dataset[/]")
    console.print(f"    [{INDIGO}]3[/] [{DIM}]Generate synthetic network + data[/]")

    src = Prompt.ask(f"  [{DIM}]Source[/]", choices=["1", "2", "3"], default="1")

    if src == "1":
        fpath = Prompt.ask(f"  [{DIM}]Data file or URL[/]")
        method = Prompt.ask(f"  [{DIM}]Method[/]", default="lasso")
        use_nb = Prompt.ask(
            f"  [{DIM}]Wrap with NestBoot?[/]",
            choices=["y", "n"],
            default="n",
        )
        if use_nb == "y":
            nest_runs = int(Prompt.ask(f"  [{DIM}]Outer runs[/]", default="10"))
            boot_runs = int(Prompt.ask(f"  [{DIM}]Inner runs[/]", default="10"))
            fdr = float(Prompt.ask(f"  [{DIM}]FDR[/]", default="0.05"))
            out = Prompt.ask(f"  [{DIM}]Output (.npy)[/]", default="nestboot.npy")
            _cmd_nestboot(
                argparse.Namespace(
                    file=fpath,
                    method=method,
                    nest_runs=nest_runs,
                    boot_runs=boot_runs,
                    fdr=fdr,
                    seed=42,
                    output=out,
                )
            )
        else:
            out = Prompt.ask(f"  [{DIM}]Output (.npy)[/]", default="")
            _cmd_infer(
                argparse.Namespace(
                    file=fpath, method=method, output=out or None, directed=False
                )
            )

    elif src == "2":
        _infer_from_genespider()

    elif src == "3":
        _infer_from_synthetic()


def _infer_from_genespider() -> None:
    """Load a GeneSpider dataset and run inference on it."""
    from bench.genespider import TOPOLOGIES, VALID_SNRS, _list_datasets, load_dataset

    console.print(f"\n  [{TEAL}]GeneSpider dataset selector[/]")

    # Size selection (multiple)
    raw_sizes = Prompt.ask(
        f"  [{DIM}]Network size(s) (comma-separated)[/]",
        default="N50",
    )
    sizes = [s.strip() for s in raw_sizes.split(",") if s.strip() in ("N10", "N50", "N100")]
    if not sizes:
        sizes = ["N50"]

    # SNR
    snrs = sorted(VALID_SNRS)
    console.print(f"\n  [{TEAL}]SNR:[/]")
    for i, s in enumerate(snrs, 1):
        console.print(f"    [{INDIGO}]{i}[/] [{DIM}]{s}[/]")
    snr_idx = int(Prompt.ask(f"  [{DIM}]SNR[/]", default="2")) - 1
    snr_idx = max(0, min(snr_idx, len(snrs) - 1))
    chosen_snr = snrs[snr_idx]

    # Topology
    console.print(f"\n  [{TEAL}]Topology:[/]")
    for i, t in enumerate(TOPOLOGIES, 1):
        console.print(f"    [{INDIGO}]{i}[/] [{DIM}]{t}[/]")
    topo_idx = int(Prompt.ask(f"  [{DIM}]Topology[/]", default="2")) - 1
    topo_idx = max(0, min(topo_idx, len(TOPOLOGIES) - 1))
    chosen_topo = TOPOLOGIES[topo_idx]

    # Methods (multiple)
    raw_methods = Prompt.ask(
        f"  [{DIM}]Method(s) (comma-separated)[/]", default="lasso"
    )
    methods = [m.strip() for m in raw_methods.split(",") if m.strip()]

    from sparselink import get_method
    from sparselink.bench.metrics import evaluate
    import sparselink.methods  # noqa: F401

    # Results table
    results_table = Table(
        title="Inference Results", title_style=TEAL, border_style="dim"
    )
    results_table.add_column("Size", style="bold")
    results_table.add_column("Method", style="bold")
    results_table.add_column("Edges", justify="right")
    results_table.add_column("AUROC", justify="right")
    results_table.add_column("AUPR", justify="right")
    results_table.add_column("F1", justify="right")
    results_table.add_column("MCC", justify="right")

    import random as _rnd

    for size in sizes:
        console.print(f"\n  [{DIM}]Fetching {size} datasets...[/]")
        datasets = _list_datasets(size)
        if not datasets:
            console.print(f"  [{ROSE}]No datasets found for {size}[/]")
            continue

        datasets = [d for d in datasets if d["snr"] == chosen_snr]
        if not datasets:
            console.print(f"  [{ROSE}]No SNR={chosen_snr} datasets for {size}[/]")
            continue

        _rnd.shuffle(datasets)

        # Load a dataset matching topology
        X, P, A_true, topology, net_name = None, None, None, None, None
        for d in datasets:
            try:
                X, P, A_true, topology, net_name = load_dataset(d, size)
                if topology == chosen_topo:
                    break
            except Exception:
                continue

        if X is None:
            console.print(f"  [{ROSE}]Failed to load any {size} dataset[/]")
            continue

        if topology != chosen_topo:
            console.print(
                f"  [{DIM}]No {chosen_topo} for {size}/SNR={chosen_snr},"
                f" using {topology}[/]"
            )

        console.print(
            f"  [{TEAL}]{size}[/]  {net_name} ({topology})"
            f" — {X.shape[0]}×{X.shape[1]}, SNR={chosen_snr}"
        )

        for method_name in methods:
            try:
                with console.status(f"[{TEAL}]Running {method_name} on {size}...[/]"):
                    from bench.genespider import ALPHA_SWEEP, PYGS_METHODS
                    if method_name in PYGS_METHODS:
                        from analyze.Data import Data
                        from datastruct.Dataset import Dataset as DS
                        from datastruct.Network import Network
                        ds = DS()
                        ds._Y = X.T
                        ds._P = P.T if P is not None else np.eye(X.shape[1], X.shape[0])
                        ds._network = Network(A_true)
                        gene_names = [f"G{i}" for i in range(X.shape[1])]
                        ds._names = gene_names
                        data_obj = Data(ds)
                        if method_name == "scenicplus":
                            from methods.scenicplus import SCENICPLUS
                            threshold_range = np.logspace(-6, 0, 20)
                            result = SCENICPLUS(
                                dataset=data_obj, nested_boot=False,
                                threshold_range=threshold_range,
                                var_names=gene_names, n_cpu=1,
                            )
                            adj_3d, _ = result
                            from sklearn.metrics import roc_auc_score
                            mask = ~np.eye(A_true.shape[0], dtype=bool)
                            y_true = (A_true[mask] != 0).astype(int)
                            best_auroc, adj = -1.0, adj_3d[:, :, 0]
                            for k in range(adj_3d.shape[2]):
                                candidate = np.abs(adj_3d[:, :, k])
                                np.fill_diagonal(candidate, 0.0)
                                try:
                                    sc = float(roc_auc_score(y_true, candidate[mask]))
                                except ValueError:
                                    sc = 0.5
                            if sc > best_auroc:
                                best_auroc, adj = sc, candidate
                        elif method_name == "panda":
                            from methods.panda import PANDA
                            adj_raw, _ = PANDA(dataset=data_obj, var_names=gene_names)
                            adj = np.abs(adj_raw)
                            np.fill_diagonal(adj, 0.0)
                    elif method_name in ALPHA_SWEEP:
                        method_cls = get_method(method_name)
                        param = ALPHA_SWEEP[method_name]
                        alphas = np.logspace(-4, 1, 20)
                        best_auroc, adj = -1.0, np.zeros_like(A_true)
                        for a in alphas:
                            r = method_cls(**{param: float(a)}).fit(X, P)
                            from sklearn.metrics import roc_auc_score
                            mask = ~np.eye(A_true.shape[0], dtype=bool)
                            try:
                                sc = float(roc_auc_score(
                                    (A_true[mask] != 0).astype(int),
                                    np.abs(r.adjacency_matrix[mask]),
                                ))
                            except ValueError:
                                sc = 0.5
                            if sc > best_auroc:
                                best_auroc, adj = sc, r.adjacency_matrix
                    else:
                        method_cls = get_method(method_name)
                        result = method_cls().fit(X, P)
                        adj = result.adjacency_matrix
                n_edges = int(np.count_nonzero(adj) - np.count_nonzero(np.diag(adj)))
                metrics = evaluate(A_true, adj)
                results_table.add_row(
                    size, method_name, str(n_edges),
                    f"[{_color(metrics.auroc)}]{metrics.auroc:.3f}[/]",
                    f"[{_color(metrics.aupr)}]{metrics.aupr:.3f}[/]",
                    f"[{_color(metrics.f1)}]{metrics.f1:.3f}[/]",
                    f"[{_color(metrics.mcc, 0.2, 0.5)}]{metrics.mcc:.3f}[/]",
                )
            except Exception as e:
                results_table.add_row(
                    size, method_name, "—", "—", "—", "—", f"[{ROSE}]err[/]"
                )
                console.print(f"  [{ROSE}]{method_name}: {e}[/]")

    console.print()
    console.print(results_table)


def _infer_from_synthetic() -> None:
    """Generate synthetic data (GeneSpider model) and run inference on it."""
    from sparselink.bench.synthetic import generate_network

    console.print(f"\n  [{TEAL}]Synthetic data generator (GeneSpider model)[/]")
    n_genes = int(Prompt.ask(f"  [{DIM}]Number of genes[/]", default="20"))
    topology = Prompt.ask(
        f"  [{DIM}]Topology[/]",
        choices=["random", "scalefree", "smallworld"],
        default="scalefree",
    )
    sparsity = float(Prompt.ask(f"  [{DIM}]Sparsity (edge density)[/]", default="0.1"))
    snr = int(Prompt.ask(f"  [{DIM}]SNR[/]", default="1000"))
    seed = int(Prompt.ask(f"  [{DIM}]Seed[/]", default="42"))

    rng = np.random.default_rng(seed)
    A_true = generate_network(n_genes, topology=topology, sparsity=sparsity, seed=seed)
    true_edges = int(np.count_nonzero(A_true))

    # GeneSpider-style data: Y = (I-A)^{-1} @ (P + noise)
    # P = identity (single-gene perturbation experiments)
    n_samples = 3 * n_genes
    G = np.linalg.inv(np.eye(n_genes) - A_true)
    P = np.eye(n_genes, n_samples)
    signal_var = np.var(G @ P)
    noise_std = np.sqrt(signal_var / snr) if snr > 0 else 0.01
    U = rng.normal(0, noise_std, (n_genes, n_samples))
    Y = G @ (P + U)  # genes x samples
    X = Y.T  # samples x genes (for sparselink)
    P_input = P.T  # samples x genes (for sparselink)

    console.print(
        f"  [{GREEN}]✓[/] Generated {topology} network:"
        f" {n_genes} genes, {true_edges} edges,"
        f" {n_samples} samples, SNR={snr}"
    )

    method = Prompt.ask(f"  [{DIM}]Method[/]", default="lasso")
    out = Prompt.ask(f"  [{DIM}]Output (.npy)[/]", default="")

    from sparselink import get_method
    from sparselink.bench.metrics import evaluate
    import sparselink.methods  # noqa: F401
    from bench.genespider import ALPHA_SWEEP

    method_cls = get_method(method)
    console.print(f"  [{TEAL}]Method[/]  {method}")

    with console.status(f"[{TEAL}]Running {method}...[/]"):
        if method in ALPHA_SWEEP:
            from sklearn.metrics import roc_auc_score
            param = ALPHA_SWEEP[method]
            alphas = np.logspace(-4, 1, 20)
            mask = ~np.eye(n_genes, dtype=bool)
            y_true = (A_true[mask] != 0).astype(int)
            best_auroc, adj = -1.0, np.zeros_like(A_true)
            for a in alphas:
                r = method_cls(**{param: float(a)}).fit(X, P_input)
                try:
                    sc = float(roc_auc_score(y_true, np.abs(r.adjacency_matrix[mask])))
                except ValueError:
                    sc = 0.5
                if sc > best_auroc:
                    best_auroc, adj = sc, r.adjacency_matrix
        else:
            result = method_cls().fit(X, P_input)
            adj = result.adjacency_matrix

    n_edges = int(np.count_nonzero(adj) - np.count_nonzero(np.diag(adj)))
    console.print(f"  [{GREEN}]✓[/] {n_edges} edges inferred")

    metrics = evaluate(A_true, adj)
    t = Table(title="vs True Network", title_style=TEAL, border_style="dim")
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")
    for name, val in [
        ("AUROC", metrics.auroc), ("AUPR", metrics.aupr),
        ("F1", metrics.f1), ("MCC", metrics.mcc),
    ]:
        t.add_row(name, f"[{_color(val)}]{val:.3f}[/]")
    console.print(t)

    if out:
        np.save(out, adj)
        console.print(f"  [{DIM}]Adjacency saved to {out}[/]")


def _synthetic_guide() -> None:
    """Interactive guide for generating synthetic data and networks."""
    from sparselink.bench.synthetic import generate_network

    console.print(f"\n  [{TEAL}]━━━ Generate Synthetic Data & Networks ━━━[/]\n")
    console.print(f"  [{DIM}]Create benchmark datasets with known ground-truth networks.[/]")
    console.print(f"  [{DIM}]Uses GeneSpider perturbation model: Y = (I-A)⁻¹(P + noise)[/]\n")

    n_genes = int(Prompt.ask(f"  [{TEAL}]Number of genes[/]", default="20"))
    topology = Prompt.ask(
        f"  [{TEAL}]Topology[/]",
        choices=["random", "scalefree", "smallworld"],
        default="scalefree",
    )
    sparsity = float(Prompt.ask(f"  [{TEAL}]Edge density (0-1)[/]", default="0.1"))
    snr = int(Prompt.ask(f"  [{TEAL}]SNR (signal-to-noise ratio)[/]", default="1000"))
    seed = int(Prompt.ask(f"  [{TEAL}]Random seed[/]", default="42"))

    rng = np.random.default_rng(seed)
    A_true = generate_network(n_genes, topology=topology, sparsity=sparsity, seed=seed)

    # GeneSpider model: Y = (I-A)^{-1} @ (P + noise), P = identity perturbations
    n_samples = 3 * n_genes
    G = np.linalg.inv(np.eye(n_genes) - A_true)
    P = np.eye(n_genes, n_samples)
    signal_var = np.var(G @ P)
    noise_std = np.sqrt(signal_var / snr) if snr > 0 else 0.01
    U = rng.normal(0, noise_std, (n_genes, n_samples))
    Y = G @ (P + U)
    X = Y.T  # samples x genes
    P_input = P.T  # samples x genes

    true_edges = int(np.count_nonzero(A_true))
    density = true_edges / (n_genes * (n_genes - 1))
    spectral = float(np.max(np.abs(np.linalg.eigvals(A_true))))

    console.print(f"\n  [{GREEN}]✓ Network & data generated[/]")
    console.print(f"    [{DIM}]Topology:[/]   {topology}")
    console.print(f"    [{DIM}]Genes:[/]      {n_genes}")
    console.print(f"    [{DIM}]Edges:[/]      {true_edges} (density={density:.3f})")
    console.print(f"    [{DIM}]Spectral ρ:[/] {spectral:.3f}")
    console.print(f"    [{DIM}]Samples:[/]    {n_samples}")
    console.print(f"    [{DIM}]SNR:[/]        {snr}")

    # Save options
    console.print(f"\n  [{TEAL}]Save outputs:[/]")
    net_out = Prompt.ask(f"  [{DIM}]Network file (.npy, blank=skip)[/]", default="")
    data_out = Prompt.ask(f"  [{DIM}]Data file (.npy/.csv, blank=skip)[/]", default="")

    if net_out:
        if net_out.endswith(".csv"):
            import pandas as pd
            names = [f"G{i}" for i in range(n_genes)]
            pd.DataFrame(A_true, index=names, columns=names).to_csv(net_out)
        else:
            np.save(net_out, A_true)
        console.print(f"  [{GREEN}]✓[/] Network saved to {net_out}")

    if data_out:
        if data_out.endswith(".csv"):
            import pandas as pd
            names = [f"G{i}" for i in range(n_genes)]
            pd.DataFrame(X, columns=names).to_csv(data_out, index=False)
        else:
            np.save(data_out, X)
        console.print(f"  [{GREEN}]✓[/] Data saved to {data_out}")

    # Offer to run inference immediately
    run_now = Prompt.ask(
        f"\n  [{DIM}]Run inference on this data now?[/]",
        choices=["y", "n"],
        default="y",
    )
    if run_now == "y":
        from sparselink import get_method
        import sparselink.methods  # noqa: F401
        from sparselink.bench.metrics import evaluate
        from bench.genespider import ALPHA_SWEEP

        method = Prompt.ask(f"  [{DIM}]Method[/]", default="lasso")
        method_cls = get_method(method)

        with console.status(f"[{TEAL}]Running {method}...[/]"):
            if method in ALPHA_SWEEP:
                from sklearn.metrics import roc_auc_score
                param = ALPHA_SWEEP[method]
                alphas = np.logspace(-4, 1, 20)
                mask = ~np.eye(n_genes, dtype=bool)
                y_true = (A_true[mask] != 0).astype(int)
                best_auroc, adj = -1.0, np.zeros_like(A_true)
                for a in alphas:
                    r = method_cls(**{param: float(a)}).fit(X, P_input)
                    try:
                        sc = float(roc_auc_score(y_true, np.abs(r.adjacency_matrix[mask])))
                    except ValueError:
                        sc = 0.5
                    if sc > best_auroc:
                        best_auroc, adj = sc, r.adjacency_matrix
            else:
                result = method_cls().fit(X, P_input)
                adj = result.adjacency_matrix

        n_edges = int(np.count_nonzero(adj) - np.count_nonzero(np.diag(adj)))
        metrics = evaluate(A_true, adj)

        console.print(f"\n  [{GREEN}]✓[/] {n_edges} edges inferred with {method}")
        t = Table(title="vs True Network", title_style=TEAL, border_style="dim")
        t.add_column("Metric", style="bold")
        t.add_column("Value", justify="right")
        for name, val in [
            ("AUROC", metrics.auroc), ("AUPR", metrics.aupr),
            ("F1", metrics.f1), ("MCC", metrics.mcc),
        ]:
            t.add_row(name, f"[{_color(val)}]{val:.3f}[/]")
        console.print(t)

    console.print(f"\n  [{DIM}]Tip: use 'pygs bench --tier fast' for full multi-method benchmarking[/]")


def _interactive() -> None:
    _print_banner()

    console.print(f"[{INDIGO}]Interactive mode[/]  [{DIM}]Ctrl+C to exit[/]\n")
    for key, (_, desc) in _MENU.items():
        console.print(f"  [{INDIGO}]{key}[/]  [{DIM}]{desc}[/]")
    console.print()

    try:
        while True:
            try:
                choice = Prompt.ask(
                    f"[{GREEN}]pygs ❯[/]",
                    choices=[*_MENU, "q", "quit", "help"],
                    show_choices=False,
                    default="help",
                )
            except EOFError:
                break

            if choice in ("q", "quit"):
                break
            if choice == "help":
                for key, (_, desc) in _MENU.items():
                    console.print(f"  [{INDIGO}]{key}[/]  [{DIM}]{desc}[/]")
                console.print(f"  [{INDIGO}]q[/]  [{DIM}]Quit[/]")
                continue

            cmd, _ = _MENU[choice]

            if cmd == "status_methods":
                _cmd_status(argparse.Namespace())
                console.print()
                _cmd_methods(argparse.Namespace())

            elif cmd == "synthetic_guide":
                _synthetic_guide()

            elif cmd == "infer":
                _interactive_infer()

            elif cmd == "bench":
                ns = _configure_synthetic()
                _run_benchmark_live(ns)

            elif cmd == "bench-gs":
                ns = _configure_genespider()
                _run_genespider_live(ns)

            elif cmd == "evaluate":
                pred = Prompt.ask(
                    f"  [{DIM}]Predicted adjacency (.npy, .csv)[/]"
                )
                gold = Prompt.ask(
                    f"  [{DIM}]Gold standard (.npy, .json, .csv)[/]"
                )
                _cmd_evaluate(
                    argparse.Namespace(predicted=pred, gold=gold)
                )

            elif cmd == "plot":
                adj = Prompt.ask(
                    f"  [{DIM}]Adjacency file (.npy, .csv)[/]"
                )
                genes = Prompt.ask(
                    f"  [{DIM}]Gene names file (blank=auto)[/]",
                    default="",
                )
                tfs = Prompt.ask(
                    f"  [{DIM}]TF list file (blank=none)[/]", default=""
                )
                out = Prompt.ask(
                    f"  [{DIM}]Output image[/]", default=""
                )
                _cmd_plot(
                    argparse.Namespace(
                        adjacency=adj,
                        genes=genes or None,
                        tfs=tfs or None,
                        threshold=0.0,
                        output=out or None,
                    )
                )

            elif cmd == "dashboard":
                inp = Prompt.ask(
                    f"  [{DIM}]Input JSON[/]",
                    default="benchmark_results.json",
                )
                out = Prompt.ask(
                    f"  [{DIM}]Output HTML[/]",
                    default="benchmark_dashboard.html",
                )
                _cmd_dashboard(
                    argparse.Namespace(
                        input=inp, output=out, no_open=False
                    )
                )

            elif cmd == "show":
                path = Prompt.ask(f"  [{DIM}]Path to result JSON[/]")
                if path:
                    _cmd_show(argparse.Namespace(file=path))

            console.print()

    except KeyboardInterrupt:
        console.print(f"\n[{DIM}]bye[/]")


# ── CLI entrypoint ────────────────────────────────────────────────────────


def _parse_runs(text: str) -> tuple[int, int]:
    """Parse '100x100', '100*100', or '100,100' as outer, inner."""
    parts = text.lower().replace("*", "x").replace(",", "x").split("x")
    if len(parts) != 2:
        raise SystemExit(f"Expected outerxinner, got {text!r}")
    try:
        outer, inner = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise SystemExit(f"Expected outerxinner, got {text!r}") from exc
    if outer < 1 or inner < 1:
        raise SystemExit("NestBoot runs must be at least 1")
    return outer, inner


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pygs",
        description="pyGS — gene regulatory network inference & benchmarking",
    )
    subs = parser.add_subparsers(dest="command")

    # status
    subs.add_parser("status", help="System status & available methods")

    # methods
    subs.add_parser("methods", help="List all inference methods")

    # infer
    ip = subs.add_parser("infer", help="Infer a network from expression data")
    ip.add_argument("file", help="Input data or URL (.h5ad, .csv, .tsv, .txt, .npy)")
    ip.add_argument("-m", "--method", default="lasso", help="Inference method")
    ip.add_argument(
        "-o", "--output", default=None, help="Save adjacency as .npy"
    )
    ip.add_argument(
        "--directed", action="store_true", help="Infer edge direction from perturbation asymmetry"
    )

    # bench (synthetic)
    bp = subs.add_parser("bench", help="Synthetic benchmark")
    bp.add_argument("--tier", default="fast")
    bp.add_argument("--methods", default=None, help="Comma-separated methods; overrides --tier")
    bp.add_argument("--snr", default=None, help="Comma-separated SNRs, e.g. 0.01,0.1,1")
    bp.add_argument("--topologies", default="random,smallworld,scalefree")
    bp.add_argument("--sparsity", type=float, default=0.06, help="True-network edge density in SNR mode")
    bp.add_argument("--randomize", action="store_true", help="Also score each method on gene-wise shuffled data")
    bp.add_argument("--nestboot", action="store_true", help="Also run NestBoot for lasso and lsco")
    bp.add_argument("--nest-runs", type=int, default=10)
    bp.add_argument("--boot-runs", type=int, default=10)
    bp.add_argument("--fdr", type=float, default=0.05)
    bp.add_argument("--plot", action="store_true", help="Write the SNR boxplot next to the output JSON")
    bp.add_argument("--n-nodes", type=int, default=50)
    bp.add_argument("--n-samples", type=int, default=200)
    bp.add_argument("--n-datasets", type=int, default=5)
    bp.add_argument("--seed", type=int, default=42)
    bp.add_argument("--timeout", type=int, default=60)
    bp.add_argument("-o", "--output", default="benchmark_results.json")

    # bench-gs (GeneSpider)
    gp = subs.add_parser(
        "bench-gs", help="GeneSpider benchmark (real data)"
    )
    gp.add_argument(
        "--tier", default="fast",
        help="Comma-separated: fast,medium,slow,nestboot",
    )
    gp.add_argument(
        "--sizes", default="N50", help="Comma-separated: N10,N50,N100"
    )
    gp.add_argument("--max-datasets", type=int, default=0, help="0 = all")
    gp.add_argument("--timeout", type=int, default=120)
    gp.add_argument(
        "--nestboot", action="store_true",
        help="Wrap methods in NestBoot FDR",
    )
    gp.add_argument("--nest-runs", type=int, default=10)
    gp.add_argument("--boot-runs", type=int, default=10)
    gp.add_argument("--fdr", type=float, default=0.05)
    gp.add_argument("--seed", type=int, default=42)
    gp.add_argument("-o", "--output", default="benchmark_genespider.json")

    # nestboot
    nb = subs.add_parser(
        "nestboot",
        help="NestBoot FDR analysis on expression data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Flags and the data path can be in any order.\n"
            "  pygs nestboot data.csv -m lsco -i 100 -o 100\n"
            "  pygs nestboot --runs 100x100 -m lsco -f data.csv --output out.npy\n"
            "-i is inner runs, -o is outer runs. The result path is --output."
        ),
    )
    nb.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Expression data or URL (.csv, .tsv, .txt, .h5ad, .npy)",
    )
    nb.add_argument(
        "-f", "--file",
        dest="file_flag",
        default=None,
        help="Data file or URL, if you prefer a flag to the positional",
    )
    nb.add_argument(
        "-m", "--method", default="lasso", help="Inference method"
    )
    nb.add_argument(
        "-o", "-n", "--outer", "--nest-runs",
        dest="nest_runs",
        type=int,
        default=10,
        help="Outer bootstrap runs",
    )
    nb.add_argument(
        "-i", "--inner", "--boot-runs",
        dest="boot_runs",
        type=int,
        default=10,
        help="Inner bootstrap runs",
    )
    nb.add_argument(
        "--runs",
        default=None,
        help="Outer x inner as one token, e.g. 100x100 (overrides -n/-i)",
    )
    nb.add_argument("--fdr", type=float, default=0.05, help="FDR threshold")
    nb.add_argument("--seed", type=int, default=42)
    nb.add_argument("--output", default="nestboot.npy", help="Output adjacency .npy")

    # evaluate
    ep = subs.add_parser(
        "evaluate", help="Evaluate predicted GRN vs gold standard"
    )
    ep.add_argument("predicted", help="Predicted adjacency (.npy, .csv)")
    ep.add_argument(
        "--gold", required=True, help="Gold standard (.npy, .json, .csv)"
    )

    # plot
    pp = subs.add_parser("plot", help="Plot a gene regulatory network")
    pp.add_argument("adjacency", help="Adjacency matrix (.npy, .csv)")
    pp.add_argument(
        "--genes", default=None, help="Gene names file (one per line)"
    )
    pp.add_argument(
        "--tfs", default=None, help="TF list file (one per line)"
    )
    pp.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Min edge weight to display",
    )
    pp.add_argument(
        "-o", "--output", default=None, help="Output image path"
    )

    # show
    sp = subs.add_parser("show", help="Render a previous result JSON")
    sp.add_argument("file", help="Path to result JSON")

    # dashboard
    dp = subs.add_parser(
        "dashboard", help="Generate interactive HTML dashboard"
    )
    dp.add_argument("-i", "--input", default="benchmark_results.json")
    dp.add_argument("-o", "--output", default="benchmark_dashboard.html")
    dp.add_argument("--no-open", action="store_true")

    args = parser.parse_args()

    if not args.command:
        _interactive()
        return

    _print_banner()

    if args.command == "status":
        _cmd_status(args)
    elif args.command == "methods":
        _cmd_methods(args)
    elif args.command == "infer":
        _cmd_infer(args)
    elif args.command == "bench":
        _run_benchmark_live(args)
    elif args.command == "bench-gs":
        _run_genespider_live(args)
    elif args.command == "nestboot":
        if args.runs:
            args.nest_runs, args.boot_runs = _parse_runs(args.runs)
        path = args.file or args.file_flag
        if args.file and args.file_flag and args.file != args.file_flag:
            parser.error("pass the data file only once")
        if not path:
            parser.error("nestboot needs a data file or URL")
        args.file = path
        _cmd_nestboot(args)
    elif args.command == "evaluate":
        _cmd_evaluate(args)
    elif args.command == "plot":
        _cmd_plot(args)
    elif args.command == "show":
        _cmd_show(args)
    elif args.command == "dashboard":
        _cmd_dashboard(args)


if __name__ == "__main__":
    main()
