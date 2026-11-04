"""NestBoot benchmark (``pygs bench-gs``): TIGRESS is scored directly, never in NestBoot."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import bench.cli as cli
import bench.genespider as genespider


def _row(method: str) -> dict:
    return {
        "method": method, "dataset_name": "N50/x.json", "network_name": "net",
        "topology": "random", "n_genes": 3, "snr": 1000,
        "auroc": 0.8, "aupr": 0.5, "precision": 0.5, "recall": 0.5,
        "f1": 0.6, "fdr": 0.5, "mcc": 0.4, "r2": 0.0, "elapsed_sec": 0.1, "error": None,
    }


def _run(monkeypatch, tmp_path: Path) -> tuple[list[dict], list[str]]:
    nested: list[str] = []
    monkeypatch.setattr(
        genespider, "_list_datasets",
        lambda size: [
            {"path": f"N50/x-SNR{snr}.json", "snr": snr, "n_genes": 3, "network_id": "1"}
            for snr in (10, 1000, 100000)
        ],
    )
    monkeypatch.setattr(
        genespider, "load_dataset",
        lambda meta, size: (np.zeros((6, 3)), np.eye(6, 3), np.eye(3), "random", "net"),
    )
    monkeypatch.setattr(genespider, "stored_snr_l", lambda meta: 1.0)
    monkeypatch.setattr(cli, "_run_direct_fair", lambda name, *rest: _row(name))

    def fake_nestboot(name, *rest):
        nested.append(name)
        return _row(f"nestboot:{name}")

    monkeypatch.setattr(cli, "_run_nestboot_on_gs", fake_nestboot)
    output = tmp_path / "nb.json"
    cli._run_genespider_live(argparse.Namespace(
        tier="", methods="lasso,tigress", sizes="N50", max_datasets=0, timeout=10,
        output=str(output), nestboot=True, nest_runs=2, boot_runs=2, fdr=0.05, seed=1,
    ))
    return json.loads(output.read_text()), nested


def test_tigress_is_scored_directly_but_never_run_through_nestboot(monkeypatch, tmp_path):
    rows, nested = _run(monkeypatch, tmp_path)
    assert nested == ["lasso"] * 3
    assert sorted(row["method"] for row in rows) == sorted(["lasso", "nestboot:lasso", "tigress"] * 3)
    assert all(row["snr_l"] == 1.0 for row in rows)


def test_complete_networks_keeps_whole_networks_and_reports_the_rest(monkeypatch):
    def meta(network_id, snr):
        return {"path": f"N10/{network_id}-SNR{snr}.json", "snr": snr, "network_id": network_id}

    listed = [meta("a", s) for s in (10, 1000, 100000)]
    listed += [meta("b", s) for s in (10, 1000, 100000)]
    listed += [meta("c", 1000)]
    listed += [meta("d", s) for s in (10, 1000, 100000)]

    def load(m, size):
        if m["path"] == "N10/b-SNR10.json":
            raise FileNotFoundError("network missing")
        return (m["path"],)

    monkeypatch.setattr(genespider, "load_dataset", load)
    loaded, skipped = genespider.complete_networks(listed, "N10", 2)
    assert [m["network_id"] for m, _ in loaded] == ["a"] * 3 + ["d"] * 3
    assert skipped == ["IDb: network missing", "IDc: stored only at filename SNR [1000]"]


def test_comparison_table_marks_tigress_as_excluded(monkeypatch):
    rows = [_row("lasso"), _row("nestboot:lasso"), _row("tigress")]
    with cli.console.capture() as captured:
        cli._render_comparison(rows)
    text = captured.get()
    tigress_line = next(line for line in text.splitlines() if "tigress" in line and "excluded" in line)
    assert "0.800" in tigress_line
    assert "already bootstrapped" in text


def test_comparison_table_shows_deltas_for_auroc_f1_and_mcc():
    nested = {**_row("nestboot:lasso"), "auroc": 0.7, "f1": 0.65, "mcc": 0.55}
    with cli.console.capture() as captured:
        cli._render_comparison([_row("lasso"), nested])
    lasso_line = next(line for line in captured.get().splitlines() if line.lstrip().startswith("lasso"))
    assert "▼-0.100" in lasso_line
    assert "▲+0.050" in lasso_line
    assert "▲+0.150" in lasso_line


def test_picker_reads_tiers_nestboot_and_tigress(monkeypatch):
    answers = iter(["1,5,6", "3", "4", "0.1"])
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(answers))
    picked = cli._pick_benchmark_methods()
    assert picked == {
        "tier": "fast", "extra_methods": ["tigress"], "nestboot": True,
        "nest_runs": 3, "boot_runs": 4, "fdr": 0.1,
    }
    methods = cli._resolve_methods(picked["tier"], picked["extra_methods"])
    assert methods[:2] == ["lasso", "lsco"]
    assert methods[-1] == "tigress"


def test_nestboot_slices_make_dense_weights_sparse():
    from bench.snr_benchmark import nestboot_slices

    weights = np.arange(1.0, 17.0).reshape(4, 4)
    single = nestboot_slices(weights[:, :, None])
    assert single.shape == (4, 4, 10)
    counts = [np.count_nonzero(single[:, :, k]) for k in range(10)]
    assert counts == sorted(counts, reverse=True) and counts[-1] < counts[0] < 16
    swept = nestboot_slices(np.stack([weights, 2 * weights], axis=2))
    assert swept.shape == (4, 4, 2)
    assert all(np.count_nonzero(swept[:, :, k]) < 16 for k in range(2))


def test_comparison_rows_take_auroc_at_best_f1_and_best_mcc():
    from bench.snr_benchmark import comparison_rows

    base = {"dataset": "d", "topology": "random", "snr": 0.1}
    rows = [
        {**base, "method": "nestboot lsco", "auroc": 0.6, "f1_score": 0.3, "mcc": 0.5},
        {**base, "method": "nestboot lsco", "auroc": 0.9, "f1_score": 0.7, "mcc": 0.2},
    ]
    assert comparison_rows(rows) == [{
        "method": "nestboot:lsco", "dataset_name": "d", "topology": "random", "snr": 0.1,
        "auroc": 0.9, "f1": 0.7, "mcc": 0.5, "error": None,
    }]


@pytest.mark.parametrize(
    ("choice", "expected"),
    [("1", "snr:simulate"), ("2", "nestboot"), ("3", "snr:stored"), ("4", "sparselink")],
)
def test_benchmark_menu_routes_each_data_source(monkeypatch, choice, expected):
    calls: list[str] = []
    snr_args = argparse.Namespace(redraw=False)
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: choice)
    monkeypatch.setattr(cli, "_configure_snr_figure", lambda simulate=False: snr_args)
    monkeypatch.setattr(cli, "_configure_genespider", lambda: "gs")
    monkeypatch.setattr(cli, "_configure_synthetic", lambda: "syn")
    monkeypatch.setattr(
        cli, "_run_snr_benchmark",
        lambda args, simulate=False: calls.append("snr:simulate" if simulate else "snr:stored"),
    )
    monkeypatch.setattr(cli, "_run_genespider_live", lambda args: calls.append("nestboot"))
    monkeypatch.setattr(cli, "_run_benchmark_live", lambda args: calls.append("sparselink"))
    cli._run_benchmark_menu()
    assert calls == [expected]


def _plotter():
    path = Path(__file__).resolve().parents[1] / "benchmark" / "code" / "plot_snr_comparison.py"
    spec = importlib.util.spec_from_file_location("plot_snr_comparison", path)
    plotter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plotter)
    return plotter


def test_plot_splits_each_method_into_direct_and_nestboot(tmp_path):
    plotter = _plotter()
    import pandas as pd

    long = pd.DataFrame({"method": ["ridge", "nestboot ridge", "tigress"], "score": [0.5, 0.6, 0.4]})
    split = plotter.split_runs(long)
    assert list(split["base"]) == ["ridge", "ridge", "tigress"]
    assert list(split["run"]) == ["direct", "NestBoot", "direct"]

    rows = [
        {**_row(method), "snr_l": snr, "dataset_name": f"d{k}-{snr}", "auroc": 0.5 + 0.1 * k}
        for snr in (0.01, 1.0) for k in range(3)
        for method in ("ridge", "nestboot:ridge", "tigress")
    ]
    table = tmp_path / "rows.json"
    table.write_text(json.dumps(rows))
    figure = tmp_path / "rows.png"
    plotter.plot_scores(plotter.load_scores(table), figure)
    assert figure.stat().st_size > 0


def test_show_uses_the_comparison_table_for_nestboot_results(tmp_path):
    path = tmp_path / "nb.json"
    path.write_text(json.dumps([{**_row("lasso"), "snr_l": 1.0}, {**_row("nestboot:lasso"), "snr_l": 1.0}]))
    with cli.console.capture() as captured:
        cli._cmd_show(argparse.Namespace(file=str(path)))
    assert "Direct vs NestBoot" in captured.get()


def test_plotter_reads_bench_gs_rows(tmp_path):
    path = Path(__file__).resolve().parents[1] / "benchmark" / "code" / "plot_snr_comparison.py"
    spec = importlib.util.spec_from_file_location("plot_snr_comparison", path)
    plotter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plotter)
    rows = [{**_row("nestboot:lsco"), "snr_l": 0.01}, {**_row("tigress"), "snr_l": 0.01},
            {**_row("lsco"), "snr_l": 0.01, "error": "boom"}]
    table = tmp_path / "rows.json"
    table.write_text(json.dumps(rows))
    frame = plotter.load_scores(table)
    assert sorted(frame["method"]) == ["nestboot lsco", "tigress"]
    assert set(frame["snr"]) == {"0.01"}
