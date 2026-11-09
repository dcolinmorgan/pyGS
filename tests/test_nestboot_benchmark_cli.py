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


TRUE = np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0], [1.0, 0.0, -1.0]])
STACK = np.stack([TRUE, np.ones((3, 3))], axis=2)
ALPHA = np.array([0.1, 1.0])


def _run(monkeypatch, tmp_path: Path, **options) -> tuple[list[dict], list[str]]:
    import bench.snr_benchmark as snr_benchmark

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
        lambda meta, size: (np.zeros((6, 3)), np.eye(6, 3), TRUE, "random", "net"),
    )
    monkeypatch.setattr(
        genespider, "stored_snr_l", lambda meta: {10: 0.01, 1000: 1.0, 100000: 100.0}[meta["snr"]]
    )
    monkeypatch.setattr(snr_benchmark, "infer_stack", lambda method, Y, P, seed: (STACK, ALPHA))

    def fake_nestboot(method, Y, P, A, args):
        nested.append(method)
        return STACK, ALPHA

    monkeypatch.setattr(snr_benchmark, "nestboot_stack", fake_nestboot)
    args = dict(
        tier=None, methods="lasso,tigress", snr=None, sizes="N50",
        topologies="random,smallworld,scalefree", max_datasets=0, max_networks=0,
        output=str(tmp_path / "nb.json"), nestboot=True, nest_runs=2, boot_runs=2,
        fdr=0.05, seed=1, randomize=False, plot=False,
    )
    args.update(options)
    cli._run_genespider_live(argparse.Namespace(**args))
    table = Path(args["output"])
    if table.suffix == ".png":
        table = table.with_suffix(".json")
    return json.loads(table.read_text()), nested


def test_tigress_is_scored_directly_but_never_run_through_nestboot(monkeypatch, tmp_path):
    rows, nested = _run(monkeypatch, tmp_path)
    assert nested == ["lasso"] * 3
    # One row per sparsity: 2 sparsities x 3 stored files for each run.
    assert sorted(row["method"] for row in rows) == sorted(
        ["lasso", "nestboot lasso", "tigress"] * 6
    )


def test_bench_gs_without_snr_scores_every_stored_file_at_its_snr(monkeypatch, tmp_path):
    from bench.snr_benchmark import comparison_rows

    rows, _ = _run(monkeypatch, tmp_path, tier="fast,nestboot", nestboot=False, methods="lasso")
    assert sorted({row["snr"] for row in rows}) == [0.01, 1.0, 100.0]
    summary = comparison_rows(rows)
    assert sorted(row["method"] for row in summary) == sorted(["lasso", "nestboot:lasso"] * 3)


def test_image_output_keeps_the_tables_beside_the_figure(monkeypatch, tmp_path):
    from bench.snr_benchmark import output_paths

    paths = output_paths(tmp_path / "Figure_3.png")
    assert paths["figure"].name == "Figure_3.png"
    assert paths["json"].name == "Figure_3.json"
    assert paths["csv"].name == "Figure_3.csv"
    assert paths["runs"].name == "Figure_3_runs"
    assert output_paths("scores.csv")["json"].name == "scores.json"

    drawn = []
    monkeypatch.setattr(cli, "_write_snr_figure", lambda table, figure: drawn.append((table, figure)))
    rows, _ = _run(monkeypatch, tmp_path, output=str(tmp_path / "Figure_3.png"), plot=True)
    assert rows
    assert drawn == [(tmp_path / "Figure_3.csv", tmp_path / "Figure_3.png")]


def test_plot_alone_redraws_but_plot_with_run_options_runs(monkeypatch, tmp_path):
    redrawn = []
    monkeypatch.setattr(cli, "_replot_snr_figure", lambda args: redrawn.append(args.output))
    assert cli._redraw_only(argparse.Namespace(plot=True, output="x.json", tier=None))
    assert redrawn == ["x.json"]
    for option in ("snr", "methods", "tier", "nestboot", "max_datasets"):
        assert not cli._redraw_only(argparse.Namespace(plot=True, output="x.json", **{option: "1"}))


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


@pytest.mark.parametrize(
    ("argv", "runner", "output", "n_datasets"),
    [
        (["bench-gs", "--tier", "fast"], "stored", "benchmark_genespider.json", 3),
        (["bench", "--data", "stored", "--tier", "fast"], "stored", "benchmark_genespider.json", 3),
        (["bench", "--methods", "lasso", "--snr", "0.1"], "generated", "benchmark_results.json", 5),
        (["bench", "--tier", "fast"], "sparselink", "benchmark_results.json", 5),
    ],
)
def test_bench_and_bench_gs_share_one_dispatcher(monkeypatch, argv, runner, output, n_datasets):
    seen = []
    monkeypatch.setattr(cli, "_run_genespider_live", lambda args: seen.append(("stored", args)))
    monkeypatch.setattr(
        cli, "_run_benchmark_live",
        lambda args: seen.append(("generated" if args.snr else "sparselink", args)),
    )
    monkeypatch.setattr("sys.argv", ["pygs", *argv])
    cli.main()
    assert [name for name, _ in seen] == [runner]
    assert seen[0][1].output == output
    assert seen[0][1].n_datasets == n_datasets


@pytest.mark.parametrize(
    "argv",
    [
        ["bench", "--data", "sparselink", "--snr", "0.1"],
        ["bench", "--data", "generated", "--methods", "lasso"],
    ],
)
def test_bench_rejects_data_and_snr_that_do_not_fit(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["pygs", *argv])
    with pytest.raises(SystemExit):
        cli.main()


def test_sklearn_delayed_and_convergence_warnings_are_dropped_but_others_print(monkeypatch):
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    shown = []
    monkeypatch.setattr(warnings, "showwarning", lambda message, *rest: shown.append(str(message)))
    cli._drop_sklearn_warnings()
    for text, category in (
        (cli.SKLEARN_DELAYED_WARNING + " `sklearn.utils.parallel.Parallel`", UserWarning),
        ("Objective did not converge.", ConvergenceWarning),
        ("another warning", UserWarning),
    ):
        warnings.showwarning(category(text), category, "x.py", 1)
    assert shown == ["another warning"]


def _score_rows() -> list[dict]:
    rows = []
    for dataset, snr in (("d1", 0.01), ("d2", 1.0)):
        for method, f1 in (("lasso", 0.5), ("nestboot lasso", 0.6), ("tigress", 0.4)):
            for alpha, scale in ((0.1, 1.0), (0.01, 0.9)):
                rows.append({
                    "dataset": dataset, "network": "net", "method": method, "snr": snr,
                    "topology": "random", "source": "stored", "alpha": alpha, "auroc": 0.7,
                    "aupr": 0.5, "precision": 0.5, "recall": 0.5, "f1_score": f1 * scale,
                    "fdr": 0.5, "mcc": f1 * scale, "r2": 0.0, "adjacency_file": "x.npz",
                })
    return rows


@pytest.mark.parametrize("suffix", [".csv", ".json"])
def test_show_prints_comparison_from_a_score_table(tmp_path, monkeypatch, suffix):
    import pandas as pd

    table = tmp_path / f"scores{suffix}"
    if suffix == ".csv":
        pd.DataFrame(_score_rows()).to_csv(table, index=False)
    else:
        table.write_text(json.dumps(_score_rows()))
    printed = []
    monkeypatch.setattr(cli, "_render_comparison", lambda summary, **kw: printed.append(summary))
    cli._cmd_show(argparse.Namespace(file=str(table)))
    (summary,) = printed
    by_method = {(row["dataset_name"], row["method"]): row["f1"] for row in summary}
    assert by_method == {
        ("d1", "lasso"): 0.5, ("d1", "nestboot:lasso"): 0.6, ("d1", "tigress"): 0.4,
        ("d2", "lasso"): 0.5, ("d2", "nestboot:lasso"): 0.6, ("d2", "tigress"): 0.4,
    }


def test_plot_redraw_prints_the_tables_too(tmp_path, monkeypatch):
    import pandas as pd

    pd.DataFrame(_score_rows()).to_csv(tmp_path / "fig.csv", index=False)
    printed, drawn = [], []
    monkeypatch.setattr(cli, "_render_comparison", lambda summary, **kw: printed.append(summary))
    monkeypatch.setattr(cli, "_write_snr_figure", lambda table, figure: drawn.append((table, figure)))
    monkeypatch.setattr("sys.argv", ["pygs", "bench", "--plot", "-o", str(tmp_path / "fig.png")])
    cli.main()
    assert len(printed) == 1 and len(printed[0]) == 6
    assert drawn == [(tmp_path / "fig.csv", tmp_path / "fig.png")]
