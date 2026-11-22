"""Every exposed inference method, on one small fixed-seed network.

The case list is built from the benchmark tiers, the pyGS-only names, and
sparselink's registry. A method added to any of those is collected with no
edit here, which is how the Z-score method shows up once it is registered.
``RidgeCO`` is collected when ``methods.ridge`` defines it.
"""

from __future__ import annotations

import importlib
import time

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

SHORT_ZETA = np.logspace(-3, 0, 4)
SEED = 11
N_GENES = 10

# Tiers that are too heavy for a pull-request run. The scheduled and main
# jobs run them; see .github/workflows/ci.yml.
SLOW_TIERS = {"medium", "slow", "very_slow", "scenicplus"}
SLOW_NAMES = {"panda", "scenicplus", "tigress"}

# L1 penalties and hard cutoffs. A larger setting must not bring a link back.
# Ridge's L2 penalty shrinks coefficients without zeroing them, so a cutoff
# (RidgeCO) is what makes its sparsity move.
MONOTONE = {
    "lasso",
    "lsco",
    "elastic_net",
    "neighborhood_selection",
    "glasso",
    "RidgeCO",
    "zscore",
    "clr",
    "genie3",
    "tigress",
}

# Methods that should beat chance on this knockdown design. CLR's mutual
# information stays at chance here, and time-series / constraint methods
# (PC, FCI, Granger, PCMCI, transfer entropy) are a different experiment.
SANITY = {
    "lasso",
    "lsco",
    "elastic_net",
    "ridge",
    "neighborhood_selection",
    "RidgeCO",
    "zscore",
    "genie3",
    "partial_correlation",
    "glasso",
}

# Imported inside fit(). Skip cleanly when the extra is not installed.
OPTIONAL_MODULES = {
    "pc": "causallearn",
    "fci": "causallearn",
    "dag_gnn": "torch",
    "panda": "netZooPy",
}


def _catalog() -> tuple[list[str], dict[str, str]]:
    import sparselink.methods  # noqa: F401
    from sparselink import list_methods

    from bench.genespider import PYGS_METHODS, TIERS

    tier_of: dict[str, str] = {}
    names: list[str] = []
    for tier, members in TIERS.items():
        for name in members:
            tier_of.setdefault(name, tier)
            names.append(name)
    names.extend(sorted(PYGS_METHODS))
    names.extend(list_methods())
    try:
        ridge = importlib.import_module("methods.ridge")
    except ImportError:
        ridge = None
    if ridge is not None and callable(getattr(ridge, "RidgeCO", None)):
        names.append("RidgeCO")
        tier_of.setdefault("RidgeCO", "fast")
    return list(dict.fromkeys(names)), tier_of


METHODS, TIER_OF = _catalog()


def _params() -> list[object]:
    params = []
    for name in METHODS:
        marks = []
        if TIER_OF.get(name) in SLOW_TIERS or name in SLOW_NAMES:
            marks.append(pytest.mark.slow)
        params.append(pytest.param(name, marks=marks, id=name))
    return params


def test_registry_lists_the_core_methods():
    names = set(METHODS)
    for required in ("lasso", "lsco", "ridge", "elastic_net", "clr", "genie3"):
        assert required in names


@pytest.fixture(scope="module")
def easy_knockdown():
    """N10 random network, eight knockdown replicates, SNR_L 100."""
    from bench.snr_benchmark import (
        knockdown_design,
        simulate_expression,
        stable_network,
    )

    network = stable_network(N_GENES, "random", 0.2, SEED)
    design = knockdown_design(N_GENES, N_GENES * 8)
    expression = simulate_expression(
        network, design, 100.0, np.random.default_rng(SEED)
    )
    off = ~np.eye(N_GENES, dtype=bool)
    assert int(np.count_nonzero(network[off])) > 0
    assert int(np.count_nonzero(network[off] == 0)) > 0
    return network, expression, design


def _as_stack(stack: np.ndarray, used: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(stack, dtype=float)
    if values.ndim == 2:
        values = values[:, :, None]
    grid = np.atleast_1d(np.asarray(used, dtype=float))
    return values, grid


def _pygs_callable(name: str):
    module = importlib.import_module(f"methods.{name}")
    wanted = name.replace("_", "")
    for attr in dir(module):
        if attr.lower() == wanted:
            candidate = getattr(module, attr)
            if callable(candidate):
                return candidate
    raise AssertionError(f"{name} is exposed but methods.{name} has no {name} callable")


def _fit_pygs(name: str, expression: np.ndarray, design: np.ndarray):
    from bench.snr_benchmark import _gene_dataset

    dataset = _gene_dataset(expression, design)
    fn = _pygs_callable(name)
    try:
        stack, used = fn(dataset, threshold_range=SHORT_ZETA, n_cpu=1)
    except TypeError:
        stack, used = fn(dataset, threshold_range=SHORT_ZETA)
    if used is None:
        used = SHORT_ZETA if np.asarray(stack).ndim == 3 else np.array([np.nan])
    return _as_stack(stack, used)


def _fit(name: str, expression: np.ndarray, design: np.ndarray, monkeypatch):
    import bench.snr_benchmark as snr

    monkeypatch.setattr(snr, "ZETA", SHORT_ZETA)
    if name == "RidgeCO":
        from methods.ridge import RidgeCO

        dataset = snr._gene_dataset(expression, design)
        return _as_stack(*RidgeCO(dataset, threshold_range=SHORT_ZETA))
    try:
        return _as_stack(*snr.infer_stack(name, expression, design, SEED))
    except KeyError:
        return _fit_pygs(name, expression, design)


def _off_diagonal_counts(stack: np.ndarray) -> list[int]:
    off = ~np.eye(stack.shape[0], dtype=bool)
    return [int(np.count_nonzero(np.abs(stack[:, :, k])[off] > 1e-8)) for k in range(stack.shape[2])]


def _ranking(network: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    off = ~np.eye(network.shape[0], dtype=bool)
    truth = (network[off] != 0).astype(int)
    scores = np.abs(weights[off])
    prevalence = float(truth.mean())
    try:
        auroc = float(roc_auc_score(truth, scores))
        aupr = float(average_precision_score(truth, scores))
    except ValueError:
        auroc, aupr = 0.5, prevalence
    return auroc, aupr, prevalence


def _best_ranking(network: np.ndarray, stack: np.ndarray) -> tuple[float, float, float]:
    """Best zeta on the path. The least-regularized slice is not always it."""
    ranked = [_ranking(network, stack[:, :, k]) for k in range(stack.shape[2])]
    return max(ranked, key=lambda item: item[0])


@pytest.mark.parametrize("method_name", _params())
def test_method_stack_on_fixed_network(method_name, easy_knockdown, monkeypatch):
    module = OPTIONAL_MODULES.get(method_name)
    if module is not None:
        pytest.importorskip(module)

    network, expression, design = easy_knockdown
    started = time.perf_counter()
    try:
        stack, used = _fit(method_name, expression, design, monkeypatch)
    except ImportError as exc:
        pytest.skip(f"{method_name} is missing an optional dependency: {exc}")
    elapsed = time.perf_counter() - started

    assert stack.shape[:2] == (N_GENES, N_GENES), stack.shape
    assert stack.shape[2] == len(used), (stack.shape, used)
    assert stack.shape[2] >= 1
    assert np.isfinite(stack).all()

    if method_name in MONOTONE and stack.shape[2] > 1:
        counts = _off_diagonal_counts(stack)
        assert counts == sorted(counts, reverse=True), (
            f"{method_name} sparsity is not monotone across zeta: {counts}"
        )

    if method_name in SANITY:
        auroc, aupr, prevalence = _best_ranking(network, stack)
        assert auroc > 0.5, f"{method_name} AUROC {auroc:.3f} is not above chance"
        assert aupr > prevalence, (
            f"{method_name} AUPR {aupr:.3f} is not above the prevalence {prevalence:.3f}"
        )

    print(
        f"{method_name} {elapsed:.3f}s shape={tuple(stack.shape)} "
        f"links={_off_diagonal_counts(stack)}"
    )


def _knockdown_table(path, values, genes):
    import pandas as pd

    columns = [gene for gene in genes for _ in range(values.shape[1] // len(genes))]
    pd.DataFrame(values, index=genes, columns=columns).to_csv(path, sep="\t")


@pytest.mark.xfail(
    reason=(
        "An empty LASSO fit drops the zeta axis, so this stack is saved as "
        "(n, n, 1) instead of one slice per zeta. A mixed empty/non-empty "
        "run raises 'Inference shape (n, n, 1) != (n, n, n_zeta)'. Fixed in PR #5."
    ),
    strict=False,
)
def test_lasso_empty_network_keeps_one_slice_per_zeta(tmp_path):
    """The fit that crashed ``pygs nestboot -m lasso`` before PR #5.

    At the default penalty every off-diagonal weight is zero, including on
    the gene-shuffled null. The cutoff used to return that one matrix instead
    of one all-zero slice per zeta.
    """
    import argparse

    from bench import cli

    rng = np.random.default_rng(0)
    genes = ["G1", "G2", "G3", "G4"]
    columns = [gene for gene in genes for _ in range(2)]
    values = rng.normal(scale=1e-3, size=(4, 8))
    for column, gene in enumerate(columns):
        values[genes.index(gene), column] -= 1e-3
    data = tmp_path / "ymatrix.tsv"
    _knockdown_table(data, values, genes)
    out = tmp_path / "lasso.npy"

    cli._cmd_nestboot(
        argparse.Namespace(
            file=str(data),
            method="lasso",
            nest_runs=2,
            boot_runs=2,
            fdr=0.05,
            seed=0,
            zeta="-4:0:5",
            output=str(out),
        )
    )

    stack = np.load(out)
    assert stack.shape == (4, 4, 5)
    assert np.isfinite(stack).all()
