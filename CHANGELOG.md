# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `pygs` CLI with interactive mode and subcommands (`status`, `methods`, `infer`, `bench`, `bench-gs`, `nestboot`, `evaluate`, `plot`, `show`, `dashboard`)
- GeneSpider live benchmark runner with Rich progress bars in the `pygs` CLI
- NestBoot tier option in the interactive `bench-gs` wizard — select "nestboot" alongside method tiers to configure outer/inner runs and FDR threshold inline
- `pygs nestboot --zeta` writes `<output>_support.tsv` (links per outer run, outer-run overlap and shuffled/measured tail ratio for each sparsity and support bin), and `<output>_support.npz` now holds the per-run support histograms and the FDR curve, so `nestboot_report/plot_support.py` can draw support curves from a new run
- `pygs nestboot --dropout-below V` fills values below V from the same knockdown's replicates, and `--max-knockdown-z Z` removes knockdowns whose own gene's robust z is above Z (0 removes knockdowns that did not lower their gene). The support archive records the knockdown z, the removed knockdowns and the dropout count, and the `.txt` summary gets a preprocessing line. `nestboot_report/run_100x100.sh` runs on `main` again

### Fixed
- `pygs` no longer prints sklearn's "`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`" warning once per tree when NestBoot fits genie3 in many threads. On Python < 3.14 each sklearn task clears the process-wide warning filters, so the CLI's ignore filter could not hold; the message is now dropped when printed

### Changed
- Knockdown labels become -1 in P again, GeneSPIDER's convention for Y = -A⁻¹P, so LSCO returns a negative diagonal and activations as positive links. With +1, every link's sign was reversed, and NestBoot drew its bootstraps in a different order for the same seed
- `pygs nestboot -m lsco --zeta` cuts |coefficient| ≤ zeta on Y rescaled to [-1, 1] over the whole matrix, GeneSPIDER's rawZeta rule and the rule behind `nestboot_report`. Before, the cutoff was on unscaled Y, so the same zeta gave a different network
- `pygs bench` covers every benchmark: `--data stored` (stored GeneSPIDER files, what `bench-gs` ran), `--data generated` (new GeneSPIDER-style networks, needs `--snr`) or `--data sparselink` (sparselink synthetic data, no `--snr`). Without `--data`, `--snr` means generated and no `--snr` means sparselink, as before. Both commands take the same options. `pygs bench-gs` is kept as an alias for `pygs bench --data stored`, so older commands and the chapters' commands still run
- `pygs bench-gs` without `--snr` uses the same runner as `bench --snr` and `bench-gs --snr`: every stored file of the chosen networks is scored at its own SNR_L, every sparsity is scored (best F1 and MCC, AUROC at the best-F1 sparsity, for direct and NestBoot alike), every inferred network is saved, the direct vs NestBoot table is printed and `--plot` writes the figure. The old per-dataset rules (best AUROC over alphas without NestBoot; one alpha-averaged network as the direct score with NestBoot) are gone. `--sizes`, `--max-networks` and `--max-datasets` keep their meaning; `--timeout` is ignored
- `-o` on `bench` and `bench-gs` may name the figure (`.png`, `.pdf`, `.svg`): `-o Figure_3.png` writes the figure there and the score tables as `Figure_3.json` and `Figure_3.csv`. Before, the JSON table was written to the `-o` path and then overwritten by the figure
- `--plot` redraws a saved table only when no run options (`--snr`, `--methods`, `--tier`, `--nestboot`, `--randomize`, `--max-*`) are given; with them it runs the benchmark and then plots
- `pygs nestboot --output edges.tsv [--nodes nodes.tsv]` writes edge and node tables at every FDR-controlled sparsity (with `zeta` and `support_cutoff` columns), or at `--table-zeta` only, with the arrays beside them as `edges.npy` and so on. `pygs plot <run>.npy --zeta Z -o edges.tsv` writes tab-separated tables
- NestBoot follows GeneSPIDER's `NB_FDR` (Morgan et al. 2019): each outer run fits its own inner bootstraps of the measured data and of one gene-shuffled copy, support is pooled into one bin per outer run, the cutoff comes from the shuffled/measured tail area, and a link is kept only when its support reaches the cutoff in every outer run. With a knockdown design, bootstraps resample within each knockdown's replicates. NestBoot needs at least 2 outer runs
- Interactive items 4 (synthetic benchmark), 5 (GeneSpider benchmark) and 10 (benchmark against GeneSPIDER true networks) are merged into item 4, "Run benchmark", which first asks for the data: new GeneSPIDER-style data at any SNR_L, stored GeneSPIDER data with the direct vs NestBoot table, stored GeneSPIDER data at SNR_L 0.01, 1 or 100 with every network saved, or sparselink synthetic data. Evaluate, plot, dashboard and show are now items 5-8
- Every data source in "Run benchmark" uses the same method picker: tiers fast, medium, slow and very_slow, NestBoot, TIGRESS, or method names. The default is fast + NestBoot + TIGRESS. NestBoot runs for every selected method except TIGRESS, also on generated and sparselink synthetic data
- The direct vs NestBoot table shows direct, NestBoot and Δ for AUROC, F1 and MCC, overall and per noise level, and fits 80 columns. Every direct method is listed; TIGRESS reads "excluded". The SNR benchmark (`bench --snr`, `bench-gs --snr`) now prints this table at the end
- The stored direct vs NestBoot run can write the boxplot when it finishes
- `pygs show` prints the direct vs NestBoot table for results with NestBoot rows, with per-level tables labelled by SNR_L when the rows record it
- When the methods go beyond the chapter's TIGRESS, LASSO and LSCO set, the figure is a grid of split violins (metrics down, SNR_L across), with direct and NestBoot as the two halves of each method and every run drawn as a point. TIGRESS has only the direct half
- `bench-gs --max-networks N` (interactive default 5) scores whole networks, each at every stored SNR. A network with a missing SNR file or a file that fails to load is reported and replaced by the next one. Without `--max-networks` or `--max-datasets`, all complete networks are used. Files that fail to load are now reported instead of silently skipped
- TIGRESS is scored directly but never wrapped in NestBoot, because it already bootstraps through stability selection. Its NestBoot column in the comparison table reads "excluded". `bench-gs --methods` now also applies without `--snr`
- `bench-gs` rows record the stored SNR_L as `snr_l`, and `plot_snr_comparison.py` uses it for the SNR panels, so `pygs bench-gs --plot -o <json>` draws the boxplot from a NestBoot benchmark run
- The SNR benchmark on stored GeneSPIDER files moved from `bench --snr` to `bench-gs --snr`. A requested SNR with no stored file stops the run before any inference
- `bench --snr` generates new stable networks, knockdown designs and expression from `--n-nodes`, `--topologies`, `--sparsity`, `--n-datasets` and `--n-samples`, at any positive SNR_L. `bench` no longer takes `--sizes`
- Direct baseline in `bench-gs` now averages absolute adjacency matrices across the alpha sweep (bagging over regularization strengths) instead of selecting a single alpha by bootstrap stability

## [1.0.0] - 2026-04-11

### Added
- **sparselink** standalone package for domain-agnostic sparse network inference
- Unified `fit(X) -> InferenceResult` interface for all inference methods
- Method registry with `@registry.register` decorator pattern
- 20 inference methods: Lasso, Elastic Net, Ridge, LSCO, CLR, Partial Correlation,
  PCMCI, Granger Causality, Transfer Entropy, Graphical LASSO, GLASSO+StARS,
  Neighborhood Selection, GENIE3, TIGRESS, PC, FCI, NOTEARS, DAG-GNN, BDeu, BGe
- `sparselink.bench` benchmarking module with synthetic data, metrics, NestBoot, and CLI
- `pyGS.bio` subpackage for biology-specific preprocessing, wrappers, visualization, evaluation
- Full type annotations (PEP 561 py.typed) on all public APIs
- MkDocs documentation site with API reference, quickstart, and method comparison
- GitHub Actions CI/CD (lint, test matrix, PyPI publish, docs deploy)
- pytest-cov with 80% coverage gate on sparselink
- ruff linting and formatting (replaces flake8 + isort + black)
- pre-commit hooks configuration
- Deprecation policy with `DeprecationWarning` utilities

### Changed
- pyGS inference methods now delegate to sparselink (thin wrappers preserve API)
- Packaging migrated to PEP 621 with hatchling build backend
- Biology-heavy dependencies moved to optional extras (`[causal]`, `[deep]`)

### Deprecated
- `pyGS.LegacyNetworkBootstrap` — use `pyGS.NetworkBootstrap` instead
  (will be removed in 1.2.0)

[Unreleased]: https://github.com/dcolinmorgan/pyGS/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/dcolinmorgan/pyGS/releases/tag/v1.0.0
