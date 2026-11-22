"""CLI coverage for the interactive menu: methods, SNR_L, replicates, tiers."""

import argparse
import math
import sys

import numpy as np
import pytest

import bench.cli as cli
import bench.snr_benchmark as snr_benchmark
from bench.generate import generate_replicates
from bench.genespider import TIERS


def test_generate_hits_snr_l_instead_of_the_filename_scale():
    drawn = generate_replicates(
        n_genes=12,
        topology="random",
        sparsity=0.15,
        snr=1.0,
        seed=3,
        replicates=2,
        experiments_per_gene=3,
    )
    assert len(drawn) == 2
    assert not np.allclose(drawn[0]["A"], drawn[1]["A"])
    for item in drawn:
        assert math.isclose(item["snr_l"], 1.0, rel_tol=0.15)
        # Filename SNR1000 is this same level. The old menu printed 1000.
        assert item["snr_l"] < 10


def test_methods_flag_selects_those_methods_for_bench_and_bench_gs():
    args = argparse.Namespace(tier="fast", methods="lasso,lsco", extra_methods=[])
    assert snr_benchmark._methods(args) == ["lasso", "lsco"]
    sparselink = [
        name
        for name in cli._resolve_methods("fast", [], "lasso, lsco")
        if name not in {"scenicplus", "panda", "zscore"}
    ]
    assert sparselink == ["lasso", "lsco"]


def test_fast_tier_includes_zscore_and_help_lists_every_tier(capsys, monkeypatch):
    assert TIERS["fast"][-1] == "zscore"
    args = argparse.Namespace(tier="fast", methods=None, extra_methods=None)
    assert "zscore" in snr_benchmark._methods(args)

    monkeypatch.setattr(sys, "argv", ["pygs", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    for name, members in TIERS.items():
        assert f"{name}:" in help_text
        assert members[0] in help_text
    assert "zscore" in help_text

    monkeypatch.setattr(sys, "argv", ["pygs", "bench-gs", "--help"])
    with pytest.raises(SystemExit):
        cli.main()
    bench_gs = capsys.readouterr().out
    assert "saved" in bench_gs.lower()
    assert "real data" not in bench_gs.lower()
    assert "lasso" in bench_gs and "zscore" in bench_gs

    monkeypatch.setattr(sys, "argv", ["pygs", "generate", "--help"])
    with pytest.raises(SystemExit):
        cli.main()
    generate_help = capsys.readouterr().out
    assert "--replicates" in generate_help
    assert "independent" in generate_help
    assert "SNR1000" in generate_help


def test_generate_command_writes_each_replicate(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pygs",
            "generate",
            "--n-genes",
            "8",
            "--topology",
            "random",
            "--sparsity",
            "0.2",
            "--snr",
            "1",
            "--replicates",
            "2",
            "--seed",
            "1",
            "--network-out",
            str(tmp_path / "net.npy"),
            "--data-out",
            str(tmp_path / "expr.npy"),
            "--perturbation-out",
            str(tmp_path / "design.npy"),
        ],
    )
    cli.main()
    assert (tmp_path / "net_r1.npy").exists()
    assert (tmp_path / "net_r2.npy").exists()
    assert (tmp_path / "expr_r1.npy").exists()
    assert (tmp_path / "design_r2.npy").exists()
    assert not (tmp_path / "net.npy").exists()


def test_stored_benchmark_menu_asks_for_snr(monkeypatch):
    answers = iter(["1", "1", "0.01,100", "2", "15", "out.json", "n"])
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(answers))
    configured = cli._configure_genespider()
    assert configured.snr == "0.01,100"
    assert configured.sizes == "N10"
    assert configured.max_networks == 2


def test_sparselink_menu_asks_for_noise(monkeypatch):
    answers = iter(["1", "20", "2", "30", "out.json", "0.2,0.5"])
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(answers))
    configured = cli._configure_synthetic()
    assert configured.noise == "0.2,0.5"
    assert cli._parse_noise(configured.noise) == [0.2, 0.5]
    assert cli._parse_noise(None) == [0.01, 0.1, 1.0]


def test_synthetic_menu_asks_how_many_replicates(monkeypatch):
    answers = iter(["8", "random", "0.2", "1", "2", "1", "", "", "", "n"])
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(answers))
    with cli.console.capture() as captured:
        cli._synthetic_guide()
    text = captured.get()
    assert "Replicate 1" in text and "Replicate 2" in text
    assert "SNR_L=" in text
    assert "SNR_L=1000" not in text


def _fast_score_frame():
    import pandas as pd

    rng = np.random.default_rng(0)
    methods = [
        "lasso",
        "lsco",
        "elastic_net",
        "ridge",
        "partial_correlation",
        "clr",
        "genie3",
        "neighborhood_selection",
    ]
    rows = []
    for snr in (0.01, 1.0, 100.0):
        for dataset_index in range(3):
            dataset = f"net{dataset_index}"
            for method in methods:
                direct = float(np.clip(0.55 + rng.normal(0, 0.08), 0.05, 0.95))
                nested = float(np.clip(direct + 0.05 + rng.normal(0, 0.04), 0.05, 0.99))
                for name, score in ((method, direct), (f"nestboot {method}", nested)):
                    rows.append(
                        {
                            "dataset": dataset,
                            "method": name,
                            "snr": snr,
                            "auroc": score,
                            "f1": float(np.clip(score - 0.1, 0.02, 0.9)),
                            "mcc": float(np.clip(score - 0.2, -0.1, 0.8)),
                        }
                    )
    return pd.DataFrame(rows)


def _overlaps(left, right, pad: float = 1.0) -> bool:
    return not (
        left.x1 <= right.x0 + pad
        or right.x1 <= left.x0 + pad
        or left.y1 <= right.y0 + pad
        or right.y1 <= left.y0 + pad
    )


def test_fast_tier_figure_labels_do_not_overlap(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import importlib.util
    from pathlib import Path

    import matplotlib.pyplot as plt

    path = (
        Path(__file__).resolve().parents[1]
        / "benchmark"
        / "code"
        / "plot_snr_comparison.py"
    )
    spec = importlib.util.spec_from_file_location("plot_snr_comparison", path)
    plotter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plotter)

    frame = _fast_score_frame()
    frame["snr"] = frame["snr"].map(lambda value: f"{float(value):g}")
    fig = plotter.draw_split(frame, ["0.01", "1", "100"])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def boxes(gid: str):
        found = []
        for text in fig.findobj(plt.Text):
            if text.get_gid() == gid and text.get_text().strip() and text.get_visible():
                found.append(text.get_window_extent(renderer).expanded(1, 1))
        return found

    deltas = boxes("delta")
    ticks = boxes("xtick")
    titles = boxes("title")
    legend = boxes("legend")
    footnotes = boxes("footnote")
    assert len(deltas) == 8 * 3 * 3
    assert ticks and titles and legend and footnotes

    for index, box in enumerate(deltas):
        for other in deltas[index + 1 :]:
            assert not _overlaps(box, other)
        for title in titles:
            assert not _overlaps(box, title)
        for item in legend:
            assert not _overlaps(box, item)
    for index, box in enumerate(ticks):
        for other in ticks[index + 1 :]:
            assert not _overlaps(box, other, pad=0)
        for note in footnotes:
            assert not _overlaps(box, note)
    axes_boxes = [ax.get_window_extent(renderer) for ax in fig.axes]
    for item in legend:
        for panel in axes_boxes:
            assert not _overlaps(item, panel)
    for title in titles:
        for item in legend:
            assert not _overlaps(title, item)

    figure = tmp_path / "after.png"
    fig.savefig(figure, dpi=80)
    plt.close(fig)
    assert figure.stat().st_size > 0
