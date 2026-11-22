#!/usr/bin/env python3
"""pyGS TUI — graphical CLI for gene regulatory network benchmarking.

Extends sparselink-tui with GeneSpider dataset benchmarks.

Usage::

    pygs-tui                                # interactive mode
    pygs-tui bench --tier fast              # synthetic benchmark (sparselink)
    pygs-tui bench-gs --sizes N50           # GeneSpider benchmark
    pygs-tui show results.json
    pygs-tui dashboard -i results.json
    pygs-tui status
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

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

# Reuse sparselink TUI components
from sparselink.tui import (
    BOLD, DIM, GREEN, INDIGO, ORANGE, ROSE, TEAL, TEXT,
    _bar, _cmd_dashboard, _cmd_show, _cmd_status, _color,
    _configure_synthetic, _pick_multi, _render_results,
    _run_benchmark_live,
)

console = Console()

BANNER = [
    "            ▄▄▄▄   ▄▄▄▄ ",
    " ▄▄▄▄  █  █ █▀▀▀  █▀▀▀  ",
    " █▀ ▀█ █▄▄█ █ ▀█▀ ▀▀▀█  ",
    " █▀▀▀  ▀  █ ▀▄▄█▀ ▄▄▄█▀ ",
    " █     ▀  ▀               ",
]


def _print_banner() -> None:
    for line in BANNER:
        console.print(f"[{TEAL}]{line}[/]")
    console.print(f"  [{BOLD}]pyGS[/]  [{DIM}]gene regulatory network inference & benchmarking[/]")
    console.print()


# ── GeneSpider config builder ─────────────────────────────────────────────

def _configure_genespider() -> argparse.Namespace:
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
    extra_methods = []
    for p in parts:
        if p in tier_map:
            tiers.append(tier_map[p])
        else:
            extra_methods.append(p)

    tier = ",".join(tiers) if tiers else ""

    console.print(f"\n  [{TEAL}]B) Network sizes[/]")
    sizes = _pick_multi("Sizes", {
        "1": "N10", "2": "N50", "3": "N100",
    }, default="2")
    sizes_str = ",".join(sizes) if sizes else "N50"

    console.print(f"\n  [{TEAL}]C) Max datasets per size[/]")
    max_ds = int(Prompt.ask(f"  [{DIM}]0 = all[/]", default="0"))

    timeout = int(Prompt.ask(f"\n  [{DIM}]Timeout per method (seconds)[/]", default="120"))
    output = Prompt.ask(f"  [{DIM}]Output file[/]", default="benchmark_genespider.json")

    console.print()
    return argparse.Namespace(
        tier=tier, sizes=sizes_str, max_datasets=max_ds,
        timeout=timeout, output=output,
        extra_methods=extra_methods,
    )


# ── GeneSpider live runner ────────────────────────────────────────────────

def _run_genespider_live(args: argparse.Namespace) -> None:
    import warnings
    warnings.simplefilter("ignore")

    from bench.genespider import (
        TIERS, PYGS_METHODS, _list_datasets, load_dataset, run_single,
    )
    from sparselink import list_methods
    import sparselink.methods  # noqa: F401

    selected_tiers = [t.strip() for t in args.tier.split(",") if t.strip()]
    use_nestboot = "nestboot" in selected_tiers
    method_tiers = [t for t in selected_tiers if t != "nestboot"]

    methods: list[str] = []
    for t in method_tiers:
        methods.extend(TIERS.get(t, []))

    # Add any extra methods typed directly by name
    extra = getattr(args, "extra_methods", [])
    methods.extend(extra)

    # If nothing resolved, don't silently default to something
    if not methods:
        console.print(f"  [{ROSE}]No methods selected. Specify a tier or method name.[/]")
        return

    # Filter to registered sparselink methods + known pyGS methods
    registered = set(list_methods())
    methods = [m for m in methods if m in registered or m in PYGS_METHODS]

    if not methods:
        console.print(f"  [{ROSE}]No valid methods found after filtering.[/]")
        return

    console.print(f"  [{TEAL}]Methods[/]  {', '.join(methods)}")
    if use_nestboot:
        console.print(f"  [{TEAL}]NestBoot[/] enabled")

    sizes = [s.strip() for s in args.sizes.split(",")]

    console.print(f"  [{TEAL}]Methods[/]  {', '.join(methods)}")
    console.print(f"  [{TEAL}]Sizes[/]    {sizes}")
    console.print(f"  [{TEAL}]Timeout[/]  {args.timeout}s")
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
            datasets = datasets[:args.max_datasets]

        total = len(datasets) * len(methods)
        console.print(f"  [{DIM}]{len(datasets)} datasets × {len(methods)} methods = {total} runs[/]")

        with progress:
            task = progress.add_task(f"{size}", total=total)
            for ds_meta in datasets:
                try:
                    X, P, A_true, topology, net_name = load_dataset(ds_meta, size)
                except Exception:
                    progress.advance(task, len(methods))
                    continue
                for method_name in methods:
                    snr = ds_meta["snr"]
                    progress.update(task, description=f"{method_name:20s} {topology}/SNR={snr}")
                    r = run_single(method_name, X, A_true, ds_meta, topology, net_name, args.timeout, P)
                    results.append(asdict(r))
                    progress.advance(task)

    _render_results(results, f"GeneSpider Benchmark ({len(results)} runs)")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n  [{DIM}]Results saved to {args.output}[/]")


# ── pyGS status (extends sparselink status) ───────────────────────────────

def _cmd_pygs_status(args: argparse.Namespace) -> None:
    _cmd_status(args)
    # Add pyGS-specific info
    tree = Tree(f"[{TEAL}]pyGS[/]", guide_style="dim")

    cache = Path(".gs_cache")
    if cache.exists():
        n = len(list(cache.glob("*.json")))
        tree.add(f"[{GREEN}]✓[/] GeneSpider cache: {n} files")
    else:
        tree.add(f"[dim]○[/] No GeneSpider cache (auto-downloaded on first run)")

    # Check pyGS modules
    mods = tree.add("pyGS modules")
    for mod in ["analyze.Data", "datastruct.Network", "methods.lasso", "bench.genespider"]:
        try:
            __import__(mod)
            mods.add(f"[{GREEN}]✓[/] {mod}")
        except ImportError:
            mods.add(f"[dim]○[/] {mod}")

    console.print(tree)


# ── Interactive mode ──────────────────────────────────────────────────────

_MENU = {
    "1": ("status",    "Show system status & available methods"),
    "2": ("bench",     "Run synthetic benchmark (sparselink)"),
    "3": ("bench-gs",  "Run GeneSpider benchmark (saved data)"),
    "4": ("dashboard", "Generate interactive HTML dashboard"),
    "5": ("show",      "Render a previous result JSON"),
}


def _interactive() -> None:
    _print_banner()

    console.print(f"[{INDIGO}]Interactive mode[/]  [{DIM}]Ctrl+C to exit[/]\n")
    for key, (_, desc) in _MENU.items():
        console.print(f"  [{INDIGO}]{key}[/]  [{DIM}]{desc}[/]")
    console.print()

    try:
        while True:
            try:
                choice = Prompt.ask(f"[{GREEN}]pyGS ❯[/]",
                                    choices=[*_MENU, "q", "quit", "help"],
                                    show_choices=False, default="help")
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

            if cmd == "status":
                _cmd_pygs_status(argparse.Namespace())

            elif cmd == "bench":
                ns = _configure_synthetic()
                _run_benchmark_live(ns)

            elif cmd == "bench-gs":
                ns = _configure_genespider()
                _run_genespider_live(ns)

            elif cmd == "dashboard":
                inp = Prompt.ask(f"  [{DIM}]Input JSON[/]", default="benchmark_results.json")
                out = Prompt.ask(f"  [{DIM}]Output HTML[/]", default="benchmark_dashboard.html")
                _cmd_dashboard(argparse.Namespace(input=inp, output=out, no_open=False))

            elif cmd == "show":
                path = Prompt.ask(f"  [{DIM}]Path to result JSON[/]")
                if path:
                    _cmd_show(argparse.Namespace(file=path))

            console.print()

    except KeyboardInterrupt:
        console.print(f"\n[{DIM}]bye[/]")


# ── CLI entrypoint ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(prog="pygs-tui", description="pyGS graphical CLI")
    subs = parser.add_subparsers(dest="command")

    bp = subs.add_parser("bench", help="Run synthetic benchmark")
    bp.add_argument("--tier", default="fast")
    bp.add_argument("--n-genes", type=int, default=50)
    bp.add_argument("--n-samples", type=int, default=200)
    bp.add_argument("--n-datasets", type=int, default=5)
    bp.add_argument("--seed", type=int, default=42)
    bp.add_argument("--timeout", type=int, default=60)
    bp.add_argument("-o", "--output", default="benchmark_results.json")

    gp = subs.add_parser("bench-gs", help="Benchmark on saved GeneSPIDER data")
    gp.add_argument("--tier", default="fast")
    gp.add_argument("--sizes", default="N50")
    gp.add_argument("--max-datasets", type=int, default=0)
    gp.add_argument("--timeout", type=int, default=120)
    gp.add_argument("-o", "--output", default="benchmark_genespider.json")

    sp = subs.add_parser("show", help="Render previous results")
    sp.add_argument("file", help="Path to result JSON")

    dp = subs.add_parser("dashboard", help="Generate interactive HTML dashboard")
    dp.add_argument("-i", "--input", default="benchmark_results.json")
    dp.add_argument("-o", "--output", default="benchmark_dashboard.html")
    dp.add_argument("--no-open", action="store_true")

    subs.add_parser("status", help="Show system status")

    args = parser.parse_args()

    if not args.command:
        _interactive()
        return

    _print_banner()
    if args.command == "bench":
        _run_benchmark_live(args)
    elif args.command == "bench-gs":
        _run_genespider_live(args)
    elif args.command == "show":
        _cmd_show(args)
    elif args.command == "dashboard":
        _cmd_dashboard(args)
    elif args.command == "status":
        _cmd_pygs_status(args)


if __name__ == "__main__":
    main()
