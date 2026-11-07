#!/usr/bin/env python3
"""pygs — CLI for gene regulatory network inference & benchmarking.

Wraps sparselink inference methods with GeneSpider data, NestBoot FDR,
biology-specific preprocessing, and gold-standard evaluation.

Usage::

    pygs status                                  # system info
    pygs methods                                 # list inference methods
    pygs infer data.csv -m lasso                 # infer a GRN
    pygs infer https://example/ymatrix.csv -m lsco
    pygs bench --tier fast                       # sparselink synthetic benchmark
    uv run pygs bench --data generated --methods tigress,lasso,lsco \
        --snr 0.01,0.1,1,10,100 --n-nodes 50 --sparsity 0.06 --n-datasets 3 \
        --plot -o benchmark/plots/snr_simulated.json   # generated networks
    uv run pygs bench --data stored --tier fast --nestboot \
        --sizes N10 --max-networks 4 --plot -o gs_fast.png  # stored files, every SNR
    uv run pygs bench --data stored --methods tigress,lasso,lsco \
        --nestboot --nest-runs 10 --boot-runs 10 --randomize \
        --snr 0.01,1,100 --topologies random,smallworld,scalefree \
        --sizes N50 --n-datasets 3 \
        --plot -o benchmark/plots/snr_figure.json       # stored files at chosen SNRs
    uv run pygs bench --plot -o benchmark/plots/snr_figure.json   # redraw only
    pygs bench-gs ...                            # same as pygs bench --data stored
    pygs nestboot data.csv -m lsco --nest-runs 100 --boot-runs 100 --zeta=-4:0:41 --output out.npy
    pygs plot out.npy --zeta 0.063 --per-gene 3 -o edges.csv
    pygs nestboot --runs 100x100 -m lsco -f data.csv --output out.npy
    pygs evaluate pred.npy --gold gold.npy       # evaluate against gold standard
    pygs plot pred.npy --genes genes.txt         # plot a GRN
    pygs dashboard -i results.json               # HTML dashboard
    pygs show results.json                       # render result table
    pygs                                         # interactive; 10 = SNR figure
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
from typing import Any

import numpy as np

from rich import box
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


COMPARISON_METRICS: tuple[tuple[str, str, float, float], ...] = (
    ("auroc", "AUROC", 0.4, 0.7),
    ("f1", "F1", 0.4, 0.7),
    ("mcc", "MCC", 0.2, 0.5),
)


def _delta_cell(delta: float) -> str:
    if delta > 0.01:
        return f"[{GREEN}]▲{delta:+.3f}[/]"
    if delta < -0.01:
        return f"[{ROSE}]▼{delta:+.3f}[/]"
    return f"[{DIM}]={delta:+.3f}[/]"


def _comparison_table(
    direct: list[dict[str, Any]], nestboot: list[dict[str, Any]], title: str, title_style: str
) -> Table:
    """Direct and NestBoot means with deltas for each metric, one row per method.

    Every direct method is listed. A method without NestBoot rows shows
    "excluded" when it is in NESTBOOT_EXCLUDED and "—" otherwise.
    """
    from bench.genespider import NESTBOOT_EXCLUDED

    t = Table(
        title=title, title_style=title_style, border_style="dim",
        box=box.SIMPLE_HEAD, padding=(0, 0), pad_edge=False,
    )
    t.add_column("Method", style="bold", overflow="fold", min_width=10)
    for index, (_, label, _, _) in enumerate(COMPARISON_METRICS):
        t.add_column(f"{label}\ndir", justify="right", no_wrap=True, min_width=5)
        t.add_column(f"{label}\nNB", justify="right", no_wrap=True, min_width=8 if index == 0 else 5)
        t.add_column(f"Δ\n{label}", justify="right", no_wrap=True, min_width=7)
    t.add_column("Runs", justify="right", no_wrap=True, min_width=4)

    for method in sorted({r["method"] for r in direct}):
        d_rows = [r for r in direct if r["method"] == method]
        n_rows = [r for r in nestboot if r["method"] == f"nestboot:{method}"]
        missing = "excluded" if method in NESTBOOT_EXCLUDED else "—"
        cells = [method]
        for key, _, lo, hi in COMPARISON_METRICS:
            d_mean = float(np.mean([r.get(key, 0) for r in d_rows]))
            cells.append(f"[{_color(d_mean, lo, hi)}]{d_mean:.3f}[/]")
            if n_rows:
                n_mean = float(np.mean([r.get(key, 0) for r in n_rows]))
                cells.append(f"[{_color(n_mean, lo, hi)}]{n_mean:.3f}[/]")
                cells.append(_delta_cell(n_mean - d_mean))
            else:
                cells.extend([f"[{DIM}]{missing}[/]", f"[{DIM}]—[/]"])
                missing = "—"
        cells.append(f"{len(d_rows)}/{len(n_rows)}")
        t.add_row(*cells)
    return t


def _render_comparison(data: list[dict], level_key: str = "snr", level_label: str = "SNR") -> None:
    """Direct vs NestBoot tables, overall and per noise level, with Δ AUROC, Δ F1 and Δ MCC."""
    from bench.genespider import NESTBOOT_EXCLUDED

    ok = [r for r in data if not r.get("error")]
    if not ok:
        console.print("[dim]No results[/]")
        return

    direct = [r for r in ok if not r["method"].startswith("nestboot:")]
    nestboot = [r for r in ok if r["method"].startswith("nestboot:")]
    if level_key == "snr" and all(r.get("snr_l") is not None for r in ok):
        level_key, level_label = "snr_l", "SNR_L"

    console.print(_comparison_table(direct, nestboot, "Direct vs NestBoot", TEAL))
    console.print(f"  [{DIM}]dir = direct, NB = NestBoot, Δ = NB minus direct, Runs = direct/NB.[/]")
    for method in sorted({r["method"] for r in direct} & set(NESTBOOT_EXCLUDED)):
        console.print(f"  [{DIM}]{method}: not wrapped in NestBoot, {NESTBOOT_EXCLUDED[method]}[/]")

    levels = sorted({r.get(level_key, 0) for r in ok})
    if len(levels) > 1:
        for level in levels:
            d_sub = [r for r in direct if r.get(level_key) == level]
            n_sub = [r for r in nestboot if r.get(level_key) == level]
            if not d_sub:
                continue
            console.print()
            shown = f"{level:g}" if isinstance(level, float) else level
            console.print(_comparison_table(d_sub, n_sub, f"{level_label}={shown}", f"bold {ORANGE}"))


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


METHOD_TIERS: dict[str, str] = {"1": "fast", "2": "medium", "3": "slow", "4": "very_slow"}


def _pick_benchmark_methods() -> dict[str, Any]:
    """Tiers, extra methods and NestBoot settings, the same for every data source."""
    console.print(f"  [{TEAL}]A) Methods[/]")
    console.print(f"    [{INDIGO}]1[/] [{DIM}]fast (lasso, lsco, elastic_net, ridge, partial_correlation, clr, genie3, neighborhood_selection)[/]")
    console.print(f"    [{INDIGO}]2[/] [{DIM}]medium (glasso, bdeu, bge, granger_causality, transfer_entropy, pcmci)[/]")
    console.print(f"    [{INDIGO}]3[/] [{DIM}]slow (tigress, glasso_stars, pc, fci)[/]")
    console.print(f"    [{INDIGO}]4[/] [{DIM}]very_slow (notears, dag_gnn)[/]")
    console.print(f"    [{INDIGO}]5[/] [{DIM}]nestboot (also run each selected method through NestBoot)[/]")
    console.print(f"    [{INDIGO}]6[/] [{DIM}]tigress (scored directly; never wrapped in NestBoot)[/]")
    console.print(f"    [{DIM}]Or type method names directly: scenicplus, genie3, ...[/]")
    raw = Prompt.ask(f"  [{DIM}]Selection (comma-separated)[/]", default="1,5,6")

    tiers: list[str] = []
    extra_methods: list[str] = []
    use_nestboot = False
    for part in (p.strip() for p in raw.split(",")):
        if part in METHOD_TIERS:
            tiers.append(METHOD_TIERS[part])
        elif part == "5":
            use_nestboot = True
        elif part == "6":
            extra_methods.append("tigress")
        elif part:
            extra_methods.append(part)

    nest_runs, boot_runs, fdr = 10, 10, 0.05
    if use_nestboot:
        console.print(f"\n  [{TEAL}]NestBoot config[/]")
        nest_runs = int(Prompt.ask(f"  [{DIM}]Outer runs[/]", default="10"))
        boot_runs = int(Prompt.ask(f"  [{DIM}]Inner runs[/]", default="10"))
        fdr = float(Prompt.ask(f"  [{DIM}]FDR threshold[/]", default="0.05"))
    return {
        "tier": ",".join(tiers),
        "extra_methods": extra_methods,
        "nestboot": use_nestboot,
        "nest_runs": nest_runs,
        "boot_runs": boot_runs,
        "fdr": fdr,
    }


def _resolve_methods(tier: str, extra_methods: list[str]) -> list[str]:
    """Tier members plus named methods, registered ones only, without repeats."""
    from bench.genespider import PYGS_METHODS, TIERS
    from sparselink import list_methods

    import sparselink.methods  # noqa: F401

    names: list[str] = []
    for name in (t.strip() for t in tier.split(",")):
        names.extend(TIERS.get(name, []))
    names.extend(extra_methods)
    registered = set(list_methods())
    return [m for m in dict.fromkeys(names) if m in registered or m in PYGS_METHODS]


def _configure_synthetic() -> argparse.Namespace:
    """Interactive config builder for synthetic benchmark."""
    console.print(f"\n  [{TEAL}]Configure Synthetic Benchmark[/]\n")

    picked = _pick_benchmark_methods()

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
        **picked,
        n_nodes=n_nodes,
        n_samples=n_nodes * 4,
        n_datasets=n_datasets,
        seed=42,
        timeout=timeout,
        output=output,
    )


BENCHMARK_SOURCES: dict[str, tuple[str, str]] = {
    "1": (
        "generated",
        "New GeneSPIDER-style data: stable networks, knockdown design, any SNR_L"
        " (pygs bench --data generated)",
    ),
    "2": (
        "stored-nestboot",
        "Stored GeneSPIDER data (Bitbucket, cached in .gs_cache): whole networks"
        " at every stored SNR_L (pygs bench --data stored)",
    ),
    "3": (
        "stored-snr",
        "Stored GeneSPIDER data at chosen SNR_L levels (0.01, 1, 100), networks"
        " per topology (pygs bench --data stored --snr)",
    ),
    "4": (
        "sparselink",
        "sparselink synthetic data: no knockdown design, noise as a standard"
        " deviation, not comparable with 1-3 (pygs bench --data sparselink)",
    ),
}


def _run_benchmark_menu() -> None:
    """One benchmark entry; the data source decides which runner is used."""
    console.print(f"\n  [{TEAL}]Run benchmark: choose the data[/]")
    for key, (_, label) in BENCHMARK_SOURCES.items():
        console.print(f"    [{INDIGO}]{key}[/] [{DIM}]{label}[/]")
    choice = Prompt.ask(
        f"  [{DIM}]Data source[/]", choices=list(BENCHMARK_SOURCES), default="1"
    )
    source, _ = BENCHMARK_SOURCES[choice]
    if source == "generated":
        ns = _configure_snr_figure(simulate=True)
        if ns.redraw:
            _replot_snr_figure(ns)
        else:
            _run_snr_benchmark(ns, simulate=True)
    elif source == "stored-nestboot":
        _run_genespider_live(_configure_genespider())
    elif source == "stored-snr":
        ns = _configure_snr_figure()
        if ns.redraw:
            _replot_snr_figure(ns)
        else:
            _run_snr_benchmark(ns)
    else:
        _run_benchmark_live(_configure_synthetic())


def _gene_dataset(X: np.ndarray, P: np.ndarray | None):
    """Expression is samples×genes here. GeneSPIDER methods want genes×experiments."""
    from datastruct.Dataset import Dataset

    dataset = Dataset()
    dataset._Y = X.T
    dataset._P = None if P is None else P.T
    return dataset


def _scores_along_sparsity(method_name: str, X: np.ndarray, P: np.ndarray | None, true_net: np.ndarray) -> list[dict]:
    """One row per zeta, compared with the true network stored beside the file.

    LASSO and LSCO use the file's perturbation matrix. AUROC and AUPR are the
    threshold-free scores of that zeta. F1 and MCC are the best threshold of
    those same weights. Nothing is maximized across zetas here.
    """
    from sparselink.bench.metrics import evaluate

    zeta = np.logspace(-6, 0, 30)
    dataset = _gene_dataset(X, P)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method_name == "lasso":
            from methods.lasso import Lasso
            stack, used = Lasso(dataset, alpha_range=zeta)
        elif method_name == "lsco":
            from methods.lsco import LSCO
            stack, used = LSCO(dataset, threshold_range=zeta)
        elif method_name == "tigress":
            from methods.tigress import TIGRESS
            stack, used = TIGRESS(dataset)
        else:
            from sparselink import get_method
            from bench.genespider import ALPHA_SWEEP
            method_cls = get_method(method_name)
            param = ALPHA_SWEEP.get(method_name)
            settings = zeta if param is not None else [None]
            pieces = []
            for alpha in settings:
                kwargs = {param: float(alpha)} if param is not None else {}
                pieces.append(method_cls(**kwargs).fit(X, P).adjacency_matrix)
            stack = np.stack(pieces, axis=2)
            used = settings
    stack = np.asarray(stack, dtype=float)
    if stack.ndim == 2:
        stack = stack[:, :, None]
    used = np.atleast_1d(np.asarray(used, dtype=float))
    rows = []
    for k in range(stack.shape[2]):
        metrics = evaluate(true_net, stack[:, :, k])
        alpha = None if k >= len(used) else float(used[k])
        rows.append({
            "alpha": alpha,
            "auroc": metrics.auroc,
            "aupr": metrics.aupr,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "fdr": metrics.fdr,
            "mcc": metrics.mcc,
            "r2": metrics.r2,
        })
    return rows


def _run_snr_benchmark(args: argparse.Namespace, simulate: bool = False) -> None:
    """Infer against true networks and save every network.

    ``simulate`` generates the networks and expression. Otherwise the stored
    GeneSPIDER files are used.
    """
    from bench.snr_benchmark import run_snr_benchmark

    run_snr_benchmark(args, simulate=simulate)



def _snr_plotter():
    from importlib.util import module_from_spec, spec_from_file_location

    plot_path = Path(__file__).resolve().parents[2] / "benchmark" / "code" / "plot_snr_comparison.py"
    spec = spec_from_file_location("plot_snr_comparison", plot_path)
    plotter = module_from_spec(spec)
    spec.loader.exec_module(plotter)
    return plotter


def _write_snr_figure(table: Path, figure: Path) -> None:
    plotter = _snr_plotter()
    plotter.plot_scores(plotter.load_scores(table), figure)


def _saved_snr_table(output: Path) -> Path | None:
    """Prefer the checkpoint CSV. Fall back to the end-of-run JSON."""
    if output.suffix.lower() == ".csv":
        return output if output.exists() else None
    csv_path = output.with_suffix(".csv")
    if csv_path.exists():
        return csv_path
    json_path = output if output.suffix.lower() == ".json" else output.with_suffix(".json")
    if json_path.exists():
        return json_path
    return None


def _replot_snr_figure(args: argparse.Namespace) -> None:
    """Redraw the boxplot from a score table. Does not infer."""
    output = Path(args.output)
    table = _saved_snr_table(output)
    if table is None:
        console.print(
            f"  [{ROSE}]No score table for {output}.[/] "
            "Run the benchmark first, then redraw with the same -o path."
        )
        raise SystemExit(1)
    from bench.snr_benchmark import output_paths

    figure = output_paths(output)["figure"]
    _write_snr_figure(table, figure)
    console.print(f"  [{DIM}]Figure saved to {figure} from {table}[/]")


def _configure_snr_figure(simulate: bool = False) -> argparse.Namespace:
    """Interactive config for the SNR benchmark on stored or generated networks."""
    if simulate:
        console.print(f"\n  [{TEAL}]Benchmark on new GeneSPIDER-style data[/]\n")
        console.print(
            f"  [{DIM}]Stable networks with a single-gene knockdown design. "
            f"Noise is scaled so each requested SNR_L is hit exactly.[/]"
        )
    else:
        console.print(f"\n  [{TEAL}]Benchmark against GeneSPIDER true networks[/]\n")
        console.print(
            f"  [{DIM}]True networks come from the GeneSPIDER JSON files. "
            f"Only stored expression is used. The three N50 levels are "
            f"SNR_L 0.01, 1, and 100.[/]"
        )
    output = Prompt.ask(
        f"  [{DIM}]Output file[/]",
        default="benchmark/plots/snr_simulated.json" if simulate else "benchmark/plots/snr_figure.json",
    )
    if _saved_snr_table(Path(output)) is not None:
        redraw = Prompt.ask(
            f"  [{DIM}]Score table exists. Redraw the figure only?[/]",
            choices=["y", "n"],
            default="n",
        )
        if redraw == "y":
            return argparse.Namespace(output=output, redraw=True, plot=True)

    picked = _pick_benchmark_methods()
    methods = ",".join(_resolve_methods(picked["tier"], picked["extra_methods"]))
    console.print()
    snr_hint = "any positive SNR_L" if simulate else "0.01, 1 or 100"
    snr = Prompt.ask(f"  [{DIM}]SNR values ({snr_hint})[/]", default="0.01,1,100")
    n_nodes = int(Prompt.ask(f"  [{DIM}]Genes per network[/]", default="50"))
    sizes = f"N{n_nodes}"
    sparsity = 0.06
    n_samples = None
    if simulate:
        sparsity = float(Prompt.ask(f"  [{DIM}]True-network edge density[/]", default="0.06"))
        n_samples = int(Prompt.ask(f"  [{DIM}]Experiments per network[/]", default=str(3 * n_nodes)))
    n_datasets = int(
        Prompt.ask(f"  [{DIM}]Networks per topology[/]", default="3")
    )
    topologies = Prompt.ask(
        f"  [{DIM}]Topologies[/]",
        default="random,smallworld,scalefree",
    )
    randomize = Prompt.ask(
        f"  [{DIM}]Also score gene-wise shuffled expression?[/]",
        choices=["y", "n"],
        default="y",
    )
    plot = Prompt.ask(
        f"  [{DIM}]Write the boxplot when the run finishes?[/]",
        choices=["y", "n"],
        default="y",
    )
    console.print()
    return argparse.Namespace(
        methods=methods,
        snr=snr,
        sizes=sizes,
        topologies=topologies,
        n_datasets=n_datasets,
        n_nodes=n_nodes,
        sparsity=sparsity,
        n_samples=n_samples,
        randomize=randomize == "y",
        nestboot=picked["nestboot"],
        nest_runs=picked["nest_runs"],
        boot_runs=picked["boot_runs"],
        fdr=picked["fdr"],
        plot=plot == "y",
        output=output,
        seed=42,
        redraw=False,
        tier="",
    )


RUN_OPTIONS = (
    "data", "snr", "methods", "tier", "nestboot", "randomize", "max_datasets", "max_networks",
)
BENCH_DATA = ("stored", "generated", "sparselink")


def _cmd_bench(args: argparse.Namespace) -> None:
    """``pygs bench`` and its ``bench-gs`` alias (``--data stored``)."""
    data = (
        getattr(args, "data", None)
        or getattr(args, "data_alias", None)
        or ("generated" if getattr(args, "snr", None) else "sparselink")
    )
    if getattr(args, "output", None) is None:
        args.output = "benchmark_genespider.json" if data == "stored" else "benchmark_results.json"
    if getattr(args, "n_datasets", None) is None:
        args.n_datasets = 3 if data == "stored" else 5
    if _redraw_only(args):
        return
    if data == "sparselink" and getattr(args, "snr", None):
        console.print(
            f"  [{ROSE}]sparselink data has noise levels, not SNR_L.[/] "
            "Use --data generated or --data stored with --snr."
        )
        raise SystemExit(2)
    if data == "generated" and not getattr(args, "snr", None):
        console.print(f"  [{ROSE}]--data generated needs --snr, e.g. --snr 0.01,0.1,1[/]")
        raise SystemExit(2)
    if data == "stored":
        _run_genespider_live(args)
    else:
        _run_benchmark_live(args)


def _redraw_only(args: argparse.Namespace) -> bool:
    """``--plot`` with no run options redraws the saved score table at ``-o``."""
    if not getattr(args, "plot", False):
        return False
    if any(getattr(args, name, None) for name in RUN_OPTIONS):
        return False
    _replot_snr_figure(args)
    return True


def _run_benchmark_live(args: argparse.Namespace) -> None:
    """Run synthetic benchmark using sparselink methods."""
    if _redraw_only(args):
        return
    args.tier = getattr(args, "tier", None) or "fast"
    if getattr(args, "snr", None):
        _run_snr_benchmark(args, simulate=True)
        return
    warnings.simplefilter("ignore")

    from sparselink.bench.run_benchmark import run_single
    from sparselink.bench.synthetic import generate_data, generate_network

    import sparselink.methods  # noqa: F401

    from bench.genespider import NESTBOOT_EXCLUDED, PYGS_METHODS

    methods = [
        m for m in _resolve_methods(args.tier, list(getattr(args, "extra_methods", [])))
        if m not in PYGS_METHODS
    ]
    use_nestboot = bool(getattr(args, "nestboot", False))
    nb_methods = [m for m in methods if m not in NESTBOOT_EXCLUDED] if use_nestboot else []
    if not methods:
        console.print(f"  [{ROSE}]No valid methods selected.[/]")
        return

    sparsities = [0.02, 0.06, 0.1]
    noise_levels = [0.01, 0.1, 1.0]
    topologies = ["random", "scalefree", "smallworld"]
    total = (
        (len(methods) + len(nb_methods))
        * len(topologies)
        * len(sparsities)
        * len(noise_levels)
        * args.n_datasets
    )

    console.print(f"  [{TEAL}]Methods[/]       {', '.join(methods)}")
    if use_nestboot:
        console.print(
            f"  [{TEAL}]NestBoot[/]      {', '.join(nb_methods) or 'no methods'}"
            f" ({args.nest_runs}×{args.boot_runs}, FDR={args.fdr})"
        )
        for name in methods:
            if name in NESTBOOT_EXCLUDED:
                console.print(f"  [{DIM}]{name} is scored directly only: {NESTBOOT_EXCLUDED[name]}[/]")
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
                            n_samples=args.n_samples or 200,
                            noise_std=noise_std,
                            seed=ds_seed,
                        )
                        condition = {"sparsity": sp, "noise_std": noise_std}
                        ds_meta = {
                            "path": f"synthetic-{topo}-sp{sp}-noise{noise_std}-{ds_idx}",
                            "n_genes": args.n_nodes,
                            "snr": None,
                        }
                        for method_name in methods:
                            progress.update(
                                task,
                                description=(
                                    f"{method_name:20s}"
                                    f" {topo}/sp={sp}/noise={noise_std}"
                                ),
                            )
                            if use_nestboot:
                                r_direct = _run_direct_fair(
                                    method_name, X, None, true_net, ds_meta,
                                    topo, ds_meta["path"], args.timeout,
                                )
                                results.append({**r_direct, **condition})
                                progress.advance(task)
                                if method_name not in nb_methods:
                                    continue
                                progress.update(
                                    task,
                                    description=f"NB:{method_name:17s} {topo}/sp={sp}/noise={noise_std}",
                                )
                                r_nb = _run_nestboot_on_gs(
                                    method_name, X, None, true_net, ds_meta,
                                    topo, ds_meta["path"], args,
                                )
                                results.append({**r_nb, **condition})
                                progress.advance(task)
                                continue
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

    if use_nestboot:
        _render_comparison(results, level_key="noise_std", level_label="noise std")
    else:
        _render_results(results, f"Synthetic Benchmark ({total} runs)")
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n  [{DIM}]Results saved to {args.output}[/]")


# ── GeneSpider benchmark ─────────────────────────────────────────────────


def _configure_genespider() -> argparse.Namespace:
    """Interactive config builder for the NestBoot benchmark on GeneSPIDER data."""
    console.print(f"\n  [{TEAL}]Configure NestBoot Benchmark (GeneSPIDER data)[/]\n")

    picked = _pick_benchmark_methods()

    console.print(f"\n  [{TEAL}]B) Network sizes[/]")
    sizes = _pick_multi(
        "Sizes",
        {"1": "N10", "2": "N50", "3": "N100"},
        default="2",
    )
    sizes_str = ",".join(sizes) if sizes else "N50"

    console.print(f"\n  [{TEAL}]C) Networks per size, each scored at SNR_L 0.01, 1 and 100[/]")
    max_networks = int(Prompt.ask(f"  [{DIM}]0 = all[/]", default="5"))

    timeout = int(
        Prompt.ask(f"\n  [{DIM}]Timeout per method (seconds)[/]", default="120")
    )
    output = Prompt.ask(
        f"  [{DIM}]Output file[/]", default="benchmark_genespider.json"
    )
    figure = Prompt.ask(
        f"  [{DIM}]Write the boxplot when the run finishes?[/]",
        choices=["y", "n"],
        default="y",
    )

    console.print()
    return argparse.Namespace(
        **picked,
        sizes=sizes_str,
        max_datasets=0,
        max_networks=max_networks,
        timeout=timeout,
        output=output,
        seed=42,
        figure=figure == "y",
    )


def _run_genespider_live(args: argparse.Namespace) -> None:
    """Benchmark on stored GeneSPIDER data (``pygs bench --data stored``).

    Without ``--snr`` every stored file of the chosen networks is scored at
    its own SNR_L.
    """
    if _redraw_only(args):
        return
    args.tier = getattr(args, "tier", None) or "fast"
    warnings.simplefilter("ignore")
    _run_snr_benchmark(args)


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
        alpha_range = np.logspace(-6, 0, 30)
    else:
        alpha_range = None

    # Counter to vary random_state for stochastic methods (e.g., genie3)
    _call_count = [0]

    def _infer(dataset: Data, **kwargs: object) -> np.ndarray:
        Y = dataset.data.Y  # type: ignore[union-attr]
        P_mat = dataset.data.P  # type: ignore[union-attr]
        X_in = Y.T
        P_in = P_mat.T if P_mat is not None and P is not None else None
        _call_count[0] += 1

        if method_name in {"lasso", "lsco"} and P is not None:
            # Same zeta stack as the direct methods. NestBoot keeps one
            # support network per zeta; do not binarize before that.
            zeta = np.logspace(-6, 0, 30)
            if method_name == "lasso":
                from methods.lasso import Lasso
                adj, _ = Lasso(dataset, alpha_range=zeta)
            else:
                from methods.lsco import LSCO
                adj, _ = LSCO(dataset, threshold_range=zeta)
            return np.asarray(adj, dtype=float)

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

        # sxnet is signed support. A 3rd axis is one network per sparsity.
        # Keep every sparsity. The reported row is the sparsity with the best
        # F1; its AUROC is the threshold-free score of that same network,
        # not the sparsity with the best AUROC.
        sxnet = nb_results.sxnet
        if hasattr(sxnet, "ndim") and sxnet.ndim == 3:
            networks = [np.abs(sxnet[:, :, k]) for k in range(sxnet.shape[2])]
        else:
            networks = [np.abs(sxnet)]
        slices = []
        best = None
        for k, candidate in enumerate(networks):
            np.fill_diagonal(candidate, 0.0)
            metrics = evaluate(A_true, candidate)
            zeta = float(alpha_range[k]) if alpha_range is not None and k < len(alpha_range) else k
            row = {
                "slice": zeta,
                "auroc": metrics.auroc,
                "aupr": metrics.aupr,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "fdr": metrics.fdr,
                "mcc": metrics.mcc,
                "r2": metrics.r2,
            }
            slices.append(row)
            if best is None or row["f1"] > best["f1"]:
                best = row
        return {
            **base,
            "auroc": best["auroc"],
            "aupr": best["aupr"],
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
            "fdr": best["fdr"],
            "mcc": best["mcc"],
            "r2": best["r2"],
            "slices": slices,
            "elapsed_sec": round(elapsed, 4),
            "error": None,
        }
    except Exception as e:
        return {**base, **fail, "elapsed_sec": 0, "error": str(e)[:120]}


# ── NestBoot FDR ──────────────────────────────────────────────────────────



def _parse_zeta(text: str) -> np.ndarray:
    """``-4:0:41`` is logspace(-4, 0, 41). A comma list is used as written."""
    text = text.strip()
    if ":" in text:
        start, stop, count = text.split(":")
        return np.logspace(float(start), float(stop), int(count))
    return np.array([float(part) for part in text.split(",") if part.strip()])


def _save_nestboot_zeta(
    out: Path,
    results,
    genes: list[str],
    zeta: np.ndarray,
    Y,
    P,
    fdr: float,
    *,
    kd_z: np.ndarray | None = None,
    failed: np.ndarray | None = None,
    dropouts_filled: int = 0,
) -> None:
    """Write the signed stack, the support archive and the support table.

    ``pygs plot --zeta`` reads the archive; ``nestboot_report/plot_support.py``
    reads the archive and the table. ``kd_z`` is measured before failed
    knockdowns are removed, since a removed knockdown has no samples left.
    """
    import pandas as pd

    from bio.knockdown import knockdown_z

    def _stack(values) -> np.ndarray:
        arr = np.asarray(values)
        return arr[:, :, None] if arr.ndim == 2 else arr

    xnet = _stack(results.xnet)
    np.save(out, xnet)
    np.save(out.with_name(out.stem + "_signed.npy"), _stack(results.sxnet))
    cutoff = np.atleast_1d(np.asarray(results.support, dtype=float))
    fdr_curve = np.asarray(results.fdr_curve, dtype=float)
    zetas = np.asarray(zeta, dtype=float)
    if len(zetas) != xnet.shape[2]:
        zetas = np.full(xnet.shape[2], np.nan)
    if kd_z is not None:
        kz = np.asarray(kd_z, dtype=float)
    else:
        try:
            kz = knockdown_z(np.asarray(Y), np.asarray(P))
        except Exception:
            kz = np.full(len(genes), np.nan)
    if failed is None:
        failed = np.zeros(len(genes), dtype=bool)
    np.savez_compressed(
        out.with_name(out.stem + "_support.npz"),
        zetavec=zetas,
        support_cutoff=cutoff,
        fdr_at_cutoff=np.atleast_1d(np.asarray(results.fp_rate, dtype=float)),
        mean_support=_stack(results.mean_support),
        min_support=_stack(results.min_ab),
        run_histograms=results.run_histograms,
        fdr_curve=fdr_curve,
        fdr_reached=np.any(fdr_curve[:, :-1] <= fdr, axis=1),
        genes=np.asarray(genes),
        knockdown_z=kz,
        failed_knockdowns=np.asarray(failed, dtype=bool),
        dropouts_filled=int(dropouts_filled),
    )
    pd.DataFrame(_support_rows(results, zetas)).to_csv(
        out.with_name(out.stem + "_support.tsv"), sep="\t", index=False
    )


def _support_rows(results, zetas: np.ndarray) -> list[dict[str, float]]:
    """One row per sparsity and NB_FDR support bin.

    Links per outer run (mean, SD) for measured and shuffled data, the overlap
    of outer runs at the bin's lower edge, and the shuffled/measured area ratio
    of the tail starting at the bin.
    """
    from methods.nestboot import support_bins

    exact = np.asarray(results.run_histograms, dtype=float)
    n_outer, n_slices, n_counts, _ = exact.shape
    to_bin = support_bins(n_counts - 1, n_outer)
    hist = np.zeros((n_outer, n_slices, n_outer, 2))
    for count, b in enumerate(to_bin):
        hist[:, :, b] += exact[:, :, count]
    overlap = results.accumulated
    if not isinstance(overlap, list):
        overlap = [overlap]
    rows = []
    for k in range(n_slices):
        for b in range(n_outer):
            rows.append({
                "slice": k,
                "zeta": zetas[k],
                "bin": b + 1,
                "support_from": b / n_outer,
                "measured_links": hist[:, k, b, 0].mean(),
                "measured_sd": hist[:, k, b, 0].std(),
                "shuffled_links": hist[:, k, b, 1].mean(),
                "shuffled_sd": hist[:, k, b, 1].std(),
                "measured_overlap": overlap[k][b, 0],
                "shuffled_overlap": overlap[k][b, 1],
                "tail_area_ratio": results.fdr_curve[k, b],
            })
    return rows


def _nestboot_outputs(
    output: str, nodes: str | None = None
) -> tuple[Path, Path | None, Path | None]:
    """Adjacency path, and edge and node table paths when ``--output`` is a table.

    ``--output out.npy`` writes the arrays only. ``--output edges.tsv`` also
    writes the tables, with the arrays beside them as edges.npy,
    edges_signed.npy and so on. The node table is ``--nodes``, or
    edges_nodes.tsv.
    """
    from bio.nestboot_tables import nodes_path_for

    path = Path(output)
    if path.suffix.lower() not in (".tsv", ".csv"):
        if nodes:
            raise ValueError("--nodes needs --output to be an edge table (.tsv or .csv)")
        return path, None, None
    return path.with_suffix(".npy"), path, Path(nodes) if nodes else nodes_path_for(path)


def _cmd_nestboot(args: argparse.Namespace) -> None:
    """Run NestBoot FDR analysis on expression data using a sparselink method."""
    warnings.simplefilter("ignore")

    try:
        out, edges_path, nodes_path = _nestboot_outputs(args.output, getattr(args, "nodes", None))
    except ValueError as exc:
        console.print(f"  [{ROSE}]{exc}[/]")
        raise SystemExit(2)
    if edges_path is not None and not getattr(args, "zeta", None):
        console.print(f"  [{ROSE}]Edge and node tables need --zeta, e.g. --zeta=-4:0:41[/]")
        raise SystemExit(2)

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

    from bio.knockdown import drop_failed_knockdowns, fill_dropouts, knockdown_z

    design = None if P is None else P.T
    if design is not None:
        console.print(
            f"  [{TEAL}]Design[/]  knockdown labels → perturbation matrix"
        )
    dropout_below = getattr(args, "dropout_below", None)
    max_knockdown_z = getattr(args, "max_knockdown_z", None)
    n_dropouts = 0
    if dropout_below is not None:
        groups = design if design is not None else np.zeros_like(matrix)
        matrix, dropped = fill_dropouts(matrix, groups, dropout_below)
        n_dropouts = int(dropped.sum())
        console.print(
            f"  [{TEAL}]Dropouts[/] {n_dropouts} values < {dropout_below} "
            f"in {int(dropped.any(axis=1).sum())} genes filled from replicates"
        )
    kd_z = np.full(n_genes, np.nan) if design is None else knockdown_z(matrix, design)
    failed = np.zeros(n_genes, dtype=bool)
    if max_knockdown_z is not None:
        if design is None:
            console.print(f"  [{ROSE}]--max-knockdown-z needs a knockdown design[/]")
            raise SystemExit(2)
        matrix, design, failed = drop_failed_knockdowns(
            matrix, design, max_knockdown_z
        )
        console.print(
            f"  [{TEAL}]Knockdowns[/] {int(failed.sum())} with own-gene z > "
            f"{max_knockdown_z} removed ({matrix.shape[1]} samples left): "
            f"{', '.join(np.asarray(gene_names)[failed])}"
        )
    n_samples = matrix.shape[1]

    # Build a Dataset for Nestboot
    from analyze.Data import Data
    from datastruct.Dataset import Dataset
    from datastruct.Network import Network

    ds = Dataset()
    ds._Y = matrix
    if design is not None:
        ds._P = design
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
    zeta = _parse_zeta(args.zeta) if getattr(args, "zeta", None) else None
    if zeta is not None:
        console.print(
            f"  [{TEAL}]Zeta[/]      {len(zeta)} thresholds "
            f"from {zeta[0]:.3g} to {zeta[-1]:.3g}"
        )

    method_cls = get_method(method_name)

    use_P = method_name in PERTURBATION_METHODS
    # GeneSPIDER's NestBoot LSCO (rawZeta = 1) cuts |coefficient| <= zeta on Y
    # rescaled to [-1, 1] over the whole matrix, so a zeta grid means the same
    # sparsities on any data scale.
    rescale = zeta is not None and method_name == "lsco"
    y_lo, y_hi = float(matrix.min()), float(matrix.max())
    if rescale:
        console.print(f"  [{DIM}]LSCO cutoff |coef| ≤ zeta, with Y rescaled to [-1, 1][/]")

    def _infer(dataset: Data, **kwargs: object) -> np.ndarray:
        """Run sparselink method on a pyGS Data object."""
        Y = dataset.data.Y  # type: ignore[union-attr]
        if rescale:
            Y = 2 * (Y - y_lo) / (y_hi - y_lo) - 1
        X = Y.T  # sparselink expects (samples × features)
        fit_P = None
        if use_P:
            P_mat = dataset.data.P  # type: ignore[union-attr]
            fit_P = None if P_mat is None else P_mat.T
        result = method_cls().fit(X, fit_P)
        adj = result.adjacency_matrix
        # A dense fit is scored by which links pass each cutoff. --zeta supplies
        # those cutoffs; otherwise use a short quantile sweep.
        scores = np.abs(adj)
        np.fill_diagonal(scores, 0.0)
        nonzero = scores[scores > 0]
        if nonzero.size == 0:
            return adj
        thresholds = zeta if zeta is not None else np.unique(
            np.quantile(nonzero, np.linspace(0.5, 0.95, 8))
        )
        slices = []
        for thr in thresholds:
            masked = adj.copy()
            masked[scores <= thr] = 0.0
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
            keep_design=P is not None,
        )

    # Report
    n_edges = int(np.sum(results.xnet != 0))
    console.print(f"\n  [{GREEN}]✓[/] {n_edges} edges at FDR ≤ {args.fdr}")
    console.print(f"  [{DIM}]Support threshold: {results.support}[/]")

    # Save adjacency
    if zeta is not None:
        _save_nestboot_zeta(
            out, results, gene_names, zeta, matrix, ds._P, args.fdr,
            kd_z=kd_z, failed=failed, dropouts_filled=n_dropouts,
        )
        console.print(f"  [{DIM}]Adjacency saved to {out}[/]")
        console.print(f"  [{DIM}]Signed network saved to {out.with_name(out.stem + '_signed.npy')}[/]")
        console.print(f"  [{DIM}]Support saved to {out.with_name(out.stem + '_support.npz')}[/]")
        console.print(f"  [{DIM}]Support table saved to {out.with_name(out.stem + '_support.tsv')}[/]")
    else:
        np.save(out, results.xnet)
        console.print(f"  [{DIM}]Adjacency saved to {out}[/]")
        signed_out = out.with_stem(out.stem + "_signed")
        np.save(signed_out, results.sxnet)
        console.print(f"  [{DIM}]Signed network saved to {signed_out}[/]")

    # Export text summary
    txt_out = out.with_suffix(".txt")
    nb.export_results(results, txt_out)
    with open(txt_out, "a") as handle:
        handle.write(
            f"\nPreprocessing: dropouts below {dropout_below} filled: "
            f"{n_dropouts}; knockdowns removed (own-gene z > "
            f"{max_knockdown_z}): {int(failed.sum())} "
            f"[{', '.join(np.asarray(gene_names)[failed])}]; samples used: "
            f"{n_samples}\n"
        )
    console.print(f"  [{DIM}]Summary saved to {txt_out}[/]")

    if edges_path is not None:
        _write_nestboot_tables(out, edges_path, nodes_path, getattr(args, "table_zeta", None))


def _write_nestboot_tables(
    run: Path, edges_path: Path, nodes_path: Path, zeta: float | None
) -> None:
    """Edge and node tables at one sparsity, or at every FDR-controlled one."""
    from bio.nestboot_tables import nestboot_tables, nestboot_tables_all, write_table

    if zeta is None:
        edges, nodes = nestboot_tables_all(run)
        if edges.empty:
            console.print(f"  [{ORANGE}]No sparsity reached FDR ≤ cutoff; tables not written.[/]")
            return
        described = f"{edges['zeta'].nunique()} FDR-controlled sparsities"
    else:
        edges, nodes, info = nestboot_tables(run, zeta, per_gene=None)
        status = f"cutoff {info['cutoff']:.2f}" if info["fdr_reached"] else "not FDR-controlled"
        described = f"zeta {info['zeta']:.3g}, {status}"
    write_table(edges, edges_path)
    write_table(nodes, nodes_path)
    console.print(f"  [{DIM}]Edges ({len(edges)} rows, {described}) saved to {edges_path}[/]")
    console.print(f"  [{DIM}]Nodes saved to {nodes_path}[/]")


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



def _cmd_plot_nestboot(args: argparse.Namespace) -> None:
    """Write edge and node tables for one zeta of a NestBoot stack."""
    from bio.nestboot_tables import nestboot_tables, nodes_path_for, write_table

    run = Path(args.adjacency)
    per_gene = None if args.per_gene == 0 else args.per_gene
    try:
        edges, nodes, info = nestboot_tables(run, args.zeta, per_gene)
    except FileNotFoundError as exc:
        console.print(f"  [{ROSE}]{exc}[/]")
        raise SystemExit(1)

    suffix = "_3pergene" if per_gene else ""
    if args.output and Path(args.output).suffix.lower() in (".csv", ".tsv", ".txt"):
        edges_path = Path(args.output)
        nodes_path = nodes_path_for(edges_path)
    else:
        stem = run.with_name(f"{run.stem}_zeta{info['zeta']:.3g}{suffix}")
        edges_path = stem.with_name(stem.name + "_edges.csv")
        nodes_path = stem.with_name(stem.name + "_nodes.csv")
    write_table(edges, edges_path)
    write_table(nodes, nodes_path)
    status = f"cutoff {info['cutoff']:.2f}" if info["fdr_reached"] else "not FDR-controlled"
    console.print(
        f"  [{GREEN}]✓[/] zeta {info['zeta']:.3g}: {info['links']} links, "
        f"{len(nodes)} genes, {status}"
    )
    console.print(f"  [{DIM}]{edges_path}[/]")
    console.print(f"  [{DIM}]{nodes_path}[/]")

    if not args.graphistry:
        return
    try:
        import os
        import graphistry
    except ImportError:
        console.print("  [{ROSE}]graphistry is not installed. Tables were written.[/]")
        raise SystemExit(1)
    server = os.environ.get("GRAPHISTRY_SERVER", "hub.graphistry.com")
    if os.environ.get("GRAPHISTRY_PERSONAL_KEY_ID"):
        graphistry.register(
            api=3, server=server, protocol="https",
            personal_key_id=os.environ["GRAPHISTRY_PERSONAL_KEY_ID"],
            personal_key_secret=os.environ["GRAPHISTRY_PERSONAL_KEY_SECRET"],
        )
    elif os.environ.get("GRAPHISTRY_USERNAME"):
        import getpass
        username = os.environ["GRAPHISTRY_USERNAME"]
        password = os.environ.get("GRAPHISTRY_PASSWORD") or getpass.getpass(
            f"Graphistry password for {username}: "
        )
        graphistry.register(
            api=3, server=server, protocol="https", username=username, password=password
        )
    else:
        console.print(
            "  [{ROSE}]Set GRAPHISTRY_USERNAME or GRAPHISTRY_PERSONAL_KEY_ID. Tables were written.[/]"
        )
        raise SystemExit(1)
    url = (
        graphistry.edges(edges, "regulator", "target")
        .nodes(nodes, "gene")
        .bind(point_title="gene", edge_title="title")
        .encode_edge_color(
            "sign",
            categorical_mapping={"activation": "#2e7d32", "repression": "#c62828"},
        )
        .encode_point_color(
            "knockdown",
            categorical_mapping={"ok": "#1f5aa6", "failed (removed)": "#f39c12"},
        )
        .encode_point_size("out_degree")
        .plot(render="url")
    )
    console.print(f"  [{TEAL}]{url}[/]")


def _cmd_plot(args: argparse.Namespace) -> None:
    """Plot a gene regulatory network."""
    if args.zeta is not None:
        _cmd_plot_nestboot(args)
        return
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
    if any(str(r.get("method", "")).startswith("nestboot:") for r in data):
        _render_comparison(data)
    else:
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
    "4": ("bench", "Run benchmark (new, stored GeneSPIDER or sparselink data)"),
    "5": ("evaluate", "Evaluate predicted GRN vs gold standard"),
    "6": ("plot", "Plot a gene regulatory network"),
    "7": ("dashboard", "Generate interactive HTML dashboard"),
    "8": ("show", "Render a previous result JSON"),
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
                _run_benchmark_menu()

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


SKLEARN_DELAYED_WARNING = "`sklearn.utils.parallel.delayed` should be used with"


def _is_dropped_sklearn_warning(message: object, category: type) -> bool:
    """sklearn's ``delayed`` warning, or its ``ConvergenceWarning``.

    LASSO and elastic net reach their iteration limit at the smallest
    regularisation values of a sweep, once per bootstrap fit.
    """
    if SKLEARN_DELAYED_WARNING in str(message):
        return True
    return category.__name__ == "ConvergenceWarning" and category.__module__.startswith("sklearn")


def _drop_sklearn_warnings() -> None:
    """Never print the warnings ``_is_dropped_sklearn_warning`` matches.

    Each sklearn parallel task runs ``warnings.resetwarnings()``, which on
    Python < 3.14 clears the process-wide filter list. With NestBoot fitting
    in many threads, other fits then find no filters and print every
    warning, whatever filters the CLI set. ``showwarning`` is not reset, so
    the messages are dropped there.
    """
    show = warnings.showwarning

    def _show(message, category, filename, lineno, file=None, line=None):
        if _is_dropped_sklearn_warning(message, category):
            return
        show(message, category, filename, lineno, file, line)

    warnings.showwarning = _show


def main() -> None:
    _drop_sklearn_warnings()
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

    # bench (stored GeneSPIDER, generated GeneSPIDER-style, or sparselink synthetic data)
    bench_epilog = (
        "--data picks the data; every source but sparselink is scored the same way:\n"
        "every sparsity of each method is scored against the true network, the\n"
        "summary takes the best F1 and MCC and the AUROC at the best-F1\n"
        "sparsity, every inferred network is saved under <output>_runs/, and the\n"
        "direct vs NestBoot table is printed.\n"
        "\n"
        "  stored     Stored GeneSPIDER files at the pinned Bitbucket commits.\n"
        "             Without --snr, every stored file of the chosen networks at\n"
        "             its own SNR_L: --max-networks N takes N whole networks per\n"
        "             size, --max-datasets N the first N files, neither every\n"
        "             complete network. With --snr, those levels only, from\n"
        "             --n-datasets networks per topology of one size. N50 has SNR_L\n"
        "             0.01, 1 and 100 (filenames SNR10, SNR1000, SNR100000).\n"
        "  generated  New GeneSPIDER-style networks of --n-nodes genes, one of\n"
        "             --topologies, --sparsity edge density, stable with\n"
        "             cond(A) <= N/2, a single-gene knockdown design over\n"
        "             --n-samples experiments (default 3 per gene), and noise\n"
        "             scaled so each --snr value is hit exactly.\n"
        "  sparselink sparselink's synthetic benchmark at noise std 0.01, 0.1 and\n"
        "             1. It has no knockdown design.\n"
        "\n"
        "Without --data: generated with --snr, sparselink without.\n"
        "--nestboot (or tier nestboot) also runs every method except tigress\n"
        "through NestBoot; TIGRESS already bootstraps through stability\n"
        "selection, so its NestBoot column reads 'excluded'. --randomize also\n"
        "scores each method on gene-wise shuffled expression.\n"
        "\n"
        "  uv run pygs bench --data stored --tier fast --nestboot \\\n"
        "    --sizes N10 --max-networks 4 --plot -o benchmark/plots/gs_fast.png\n"
        "\n"
        "  uv run pygs bench --data generated \\\n"
        "    --methods tigress,lasso,lsco --nestboot --nest-runs 10 --boot-runs 10 \\\n"
        "    --snr 0.01,0.1,1 --topologies random,smallworld,scalefree \\\n"
        "    --n-nodes 10 --n-datasets 3 --plot -o Figure_3.png\n"
        "\n"
        "-o names the score table (.json), or the figure (.png, .pdf, .svg) with\n"
        "the tables beside it. Redraw the figure from a saved table, without\n"
        "running inference:\n"
        "\n"
        "  uv run pygs bench --plot -o Figure_3.png"
    )

    def _bench_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--data", choices=list(BENCH_DATA), default=None,
            help="stored, generated or sparselink (default: generated with --snr, else sparselink)",
        )
        parser.add_argument(
            "--tier", default=None,
            help="Comma-separated: fast,medium,slow,very_slow,nestboot (default fast)",
        )
        parser.add_argument("--methods", default=None, help="Comma-separated methods; overrides --tier")
        parser.add_argument(
            "--snr", default=None,
            help="Comma-separated SNR_L values: levels to generate, or stored levels to keep",
        )
        parser.add_argument("--topologies", default="random,smallworld,scalefree")
        parser.add_argument(
            "--n-datasets", type=int, default=None,
            help="Networks per topology (default 3 stored with --snr, 5 otherwise; 0 = all stored)",
        )
        parser.add_argument("--randomize", action="store_true", help="Also score each method on gene-wise shuffled data")
        parser.add_argument(
            "--nestboot", action="store_true", help="Also run NestBoot for every method except tigress"
        )
        parser.add_argument("--nest-runs", type=int, default=10)
        parser.add_argument("--boot-runs", type=int, default=10)
        parser.add_argument("--fdr", type=float, default=0.05)
        parser.add_argument(
            "--plot", action="store_true",
            help="Write the boxplot when the run finishes. With no run options, redraw it from the saved table at -o",
        )
        stored = parser.add_argument_group("stored data")
        stored.add_argument(
            "--sizes", default="N50", help="Comma-separated: N10,N50,N100. One size with --snr"
        )
        stored.add_argument(
            "--max-networks", type=int, default=0,
            help="Networks per size, each at every stored SNR; incomplete networks are skipped (0 = all)",
        )
        stored.add_argument(
            "--max-datasets", type=int, default=0,
            help="First N dataset files in listing order, ignoring networks; overridden by --max-networks",
        )
        generated = parser.add_argument_group("generated and sparselink data")
        generated.add_argument("--n-nodes", type=int, default=50)
        generated.add_argument(
            "--sparsity", type=float, default=0.06, help="True-network edge density (generated)"
        )
        generated.add_argument(
            "--n-samples", type=int, default=None,
            help="Experiments per dataset. Default 3 per gene for generated, 200 for sparselink",
        )
        generated.add_argument(
            "--timeout", type=int, default=60, help="Seconds per method (sparselink only)"
        )
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument(
            "-o", "--output", default=None,
            help="Score table (.json), or the figure (.png, .pdf, .svg) with the tables beside it",
        )

    bp = subs.add_parser(
        "bench",
        help="Benchmark against true networks: stored GeneSPIDER, generated, or sparselink data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=bench_epilog,
    )
    _bench_arguments(bp)

    gp = subs.add_parser(
        "bench-gs",
        help="Same as bench --data stored (kept for older commands)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Same as `pygs bench --data stored`; see `pygs bench --help`.",
    )
    _bench_arguments(gp)
    gp.set_defaults(data_alias="stored")

    # nestboot
    nb = subs.add_parser(
        "nestboot",
        help="NestBoot FDR analysis on expression data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Flags and the data path can be in any order.\n"
            "  pygs nestboot data.csv -m lsco -i 100 -o 100\n"
            "  pygs nestboot --runs 100x100 -m lsco -f data.csv --output out.npy\n"
            "  pygs nestboot data.tsv -m lsco --runs 100x100 --zeta=-4:0:41 \\\n"
            "      --output edges.tsv --nodes nodes.tsv\n"
            "-i is inner runs, -o is outer runs. The result path is --output.\n"
            "An edge table output writes edges and nodes at every FDR-controlled\n"
            "sparsity (zeta column), or at --table-zeta only."
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
    nb.add_argument(
        "--output",
        default="nestboot.npy",
        help="Adjacency .npy, or an edge table (.tsv or .csv) with the arrays saved "
        "beside it. Tables need --zeta",
    )
    nb.add_argument(
        "--nodes",
        default=None,
        help="Node table path when --output is an edge table (default <edges>_nodes.tsv)",
    )
    nb.add_argument(
        "--zeta",
        default=None,
        help="LSCO sparsity grid. -4:0:41 is logspace from 1e-4 to 1 with 41 steps. A comma list is used as written",
    )
    nb.add_argument(
        "--table-zeta",
        type=float,
        default=None,
        help="Write the tables for the sparsity nearest this zeta only. "
        "Default: every FDR-controlled sparsity, with a zeta column",
    )
    nb.add_argument(
        "--dropout-below",
        type=float,
        default=None,
        help="Treat values below this as dropouts; fill from replicates",
    )
    nb.add_argument(
        "--max-knockdown-z",
        type=float,
        default=None,
        help="Remove knockdowns whose own gene's robust z exceeds this "
        "(0 removes knockdowns that did not lower their gene)",
    )

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
        "-o", "--output", default=None,
        help="Output image, or a .csv path for the NestBoot edge table",
    )
    pp.add_argument(
        "--zeta", type=float, default=None,
        help="NestBoot sparsity. Writes edge and node tables from the .npy stack",
    )
    pp.add_argument(
        "--per-gene", type=float, default=3.0,
        help="With --zeta, keep the highest-support links nearest this density. 0 keeps the full FDR network",
    )
    pp.add_argument(
        "--graphistry", action="store_true",
        help="With --zeta, also upload the tables and print a Graphistry URL",
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
    elif args.command in ("bench", "bench-gs"):
        _cmd_bench(args)
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
