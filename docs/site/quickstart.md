# Getting Started

## Installation

```bash
git clone --recurse-submodules https://github.com/dcolinmorgan/pyGS.git
cd pyGS

# With uv (recommended)
uv pip install -e ".[dev]" -e "sparselink/[dev]"

# Or with pip
pip install -e ".[dev]"
pip install -e "sparselink/[dev]"
```

Optional extras:

```bash
uv pip install -e "sparselink/[causal]"   # PC, FCI
uv pip install -e "sparselink/[deep]"     # DAG-GNN (torch)
```

Verify the install:

```bash
pygs status
```

---

## 1. Infer a network (Python)

All 20+ inference methods share the same interface: `get_method(name)(**params).fit(X)`.

```python
import numpy as np
from sparselink import get_method, list_methods

# See what's available
print(list_methods())
# ['lasso', 'elastic_net', 'lsco', 'clr', 'genie3', 'tigress',
#  'glasso', 'glasso_stars', 'pc', 'fci', 'notears', ...]

# Expression matrix: rows = samples, columns = genes
X = np.random.randn(100, 20)

# Infer
result = get_method("lasso")(alpha=0.01).fit(X)
print(result.adjacency_matrix.shape)  # (20, 20)
```

Or from the CLI:

```bash
pygs infer expression.csv -m lasso -o adjacency.npy
```

---

## 2. Benchmark on synthetic data

Generate a ground-truth network, simulate expression, infer, and evaluate — all in a few lines.

```python
from sparselink import get_method
from sparselink.bench import generate_network, generate_data, evaluate

# Ground truth: 50-gene scale-free network
A_true = generate_network(n_genes=50, topology="scalefree", sparsity=0.1, seed=42)

# Simulate expression data
X = generate_data(A_true, n_samples=200, noise_std=0.1, seed=42)

# Infer and evaluate
result = get_method("glasso")().fit(X)
metrics = evaluate(A_true, result.adjacency_matrix)
print(f"AUROC={metrics.auroc:.3f}  AUPR={metrics.aupr:.3f}  F1={metrics.f1:.3f}")
```

Run a full sweep from the CLI:

```bash
# Fast tier: lasso, elastic_net, lsco, ridge, clr, genie3, ...
pygs bench --tier fast --n-nodes 50 --timeout 60

# All tiers
pygs bench --tier fast,medium,slow -o results.json
pygs show results.json
```

---

## 3. Benchmark on GeneSpider data

GeneSpider provides saved perturbation-response datasets at controlled SNR levels
from the [Sonnhammer lab](https://bitbucket.org/sonnhammergrni/).
pyGS downloads and caches them automatically.

```python
from bench.genespider import load_dataset, _list_datasets, run_single
from dataclasses import asdict

# List available N50 datasets (SNR = 10, 1000, 100000)
datasets = _list_datasets("N50")
print(f"{len(datasets)} datasets available")

# Load one
ds = datasets[0]
X, P, A_true, topology, net_name = load_dataset(ds, "N50")
print(f"X={X.shape}, P={P.shape}, topology={topology}, SNR={ds['snr']}")

# Run a single method (with alpha sweep)
result = run_single("lasso", X, A_true, ds, topology, net_name, timeout=60, P=P)
print(f"AUROC={result.auroc:.3f}  F1={result.f1:.3f}  MCC={result.mcc:.3f}")
```

From the CLI:

```bash
# Fast methods on N50 datasets
pygs bench-gs --tier fast --sizes N50

# Specific sizes and tiers
pygs bench-gs --tier fast,medium --sizes N10,N50 --timeout 120

# Limit to 3 datasets per size (faster iteration)
pygs bench-gs --tier fast --sizes N50 --max-datasets 3
```

---

## 4. NestBoot FDR analysis

NestBoot wraps any inference method in a bootstrap loop to control false discovery rate.
Each bootstrap run infers a network; edges that appear consistently across bootstraps
(above an FDR-derived support threshold) are kept.

Methods with a regularization parameter (lasso, elastic_net, ridge, lsco, glasso)
are automatically swept across alpha values within each bootstrap.
Other methods (genie3, clr, etc.) use post-hoc threshold sweeping.

### From the CLI (GeneSpider benchmark + NestBoot comparison)

```bash
# Run direct AND NestBoot side-by-side, see the delta
pygs bench-gs --tier fast,nestboot --sizes N50 --nest-runs 10 --boot-runs 10

# Or with the flag
pygs bench-gs --tier fast --nestboot --sizes N50 --fdr 0.05
```

This produces a comparison table:

```
              Direct vs NestBoot
┃ Method      ┃ Direct ┃ NestBoot ┃ Δ AUROC  ┃
┡━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│ lasso       │  0.712 │    0.891 │ ▲ +0.179 │
│ elastic_net │  0.708 │    0.885 │ ▲ +0.177 │
│ clr         │  0.638 │    0.654 │ ▲ +0.016 │
│ lsco        │  0.941 │    0.944 │ = +0.003 │
```

Plus per-SNR breakdowns.

### From Python (on your own data)

```python
import numpy as np
from sparselink import get_method
from bio.preprocessing import load_expression_matrix
from methods.nestboot import Nestboot
from config import AnalysisConfig
from datastruct.Dataset import Dataset
from datastruct.Network import Network
from analyze.Data import Data

# Load expression data (genes × samples)
matrix, gene_names = load_expression_matrix("expression.csv")
n_genes, n_samples = matrix.shape

# Wrap in a Dataset
ds = Dataset()
ds._Y = matrix                                    # (genes × samples)
ds._P = np.eye(n_genes, n_samples)                # perturbation matrix
ds._network = Network(np.zeros((n_genes, n_genes)))
ds._names = gene_names
data = Data(ds)

# Define inference callback
method_cls = get_method("lasso")

def infer(dataset, **kwargs):
    Y = dataset.data.Y
    P = dataset.data.P
    # Sweep alphas → 3D output (genes × genes × n_alphas)
    alphas = np.logspace(-3, 0, 10)
    slices = [method_cls(alpha=float(a)).fit(Y.T, P.T).adjacency_matrix for a in alphas]
    return np.stack(slices, axis=2)

# Run NestBoot
config = AnalysisConfig(total_runs=100, inner_group_size=10, fdr_threshold=0.05)
nb = Nestboot(config)
results = nb.run_nestboot(data, infer, nest_runs=10, boot_runs=10, seed=42)

# Results
print(f"Edges found: {np.count_nonzero(results.xnet)}")
print(f"Support threshold: {results.support}")

# Save
np.save("nestboot_adj.npy", results.xnet)       # binary FDR-controlled
np.save("nestboot_signed.npy", results.sxnet)    # continuous signed scores
```

### From Python (on GeneSpider data)

```python
from bench.genespider import load_dataset, _list_datasets

datasets = _list_datasets("N50")
ds = [d for d in datasets if d["snr"] == 1000][0]
X, P, A_true, topology, net_name = load_dataset(ds, "N50")

# Build Dataset with real perturbation matrix
ds_obj = Dataset()
ds_obj._Y = X.T                                   # (genes × samples)
ds_obj._P = P.T                                   # (genes × samples)
ds_obj._network = Network(A_true)
ds_obj._names = [f"G{i}" for i in range(X.shape[1])]
data = Data(ds_obj)

# Infer callback passes P through to sparselink
def infer(dataset, **kwargs):
    Y = dataset.data.Y
    Pm = dataset.data.P
    alphas = np.logspace(-3, 0, 10)
    slices = [method_cls(alpha=float(a)).fit(Y.T, Pm.T).adjacency_matrix for a in alphas]
    return np.stack(slices, axis=2)

results = nb.run_nestboot(data, infer, nest_runs=10, boot_runs=10, seed=42)

# Evaluate against gold standard
from sparselink.bench.metrics import evaluate
metrics = evaluate(A_true, np.mean(np.abs(results.sxnet), axis=2))
print(f"AUROC={metrics.auroc:.3f}  F1={metrics.f1:.3f}")
```

---

## 5. Evaluate and visualize

### Evaluate against a gold standard

```python
from bio.evaluation import compare_to_gold_standard

metrics = compare_to_gold_standard(predicted_adj, gold_standard_adj)
print(metrics)
# {'AUROC': 0.87, 'AUPR': 0.65, 'F1': 0.72, 'MCC': 0.58, ...}
```

```bash
pygs evaluate predicted.npy --gold gold_standard.npy
```

### Plot a GRN

```python
from bio.visualization import plot_grn

plot_grn(
    adjacency,
    gene_names=["TP53", "MYC", "BRCA1", ...],
    tf_names=["TP53", "MYC"],       # highlighted in red
    threshold=0.1,                   # min edge weight
    save_path="network.png",
)
```

```bash
pygs plot adjacency.npy --genes genes.txt --tfs tfs.txt -o network.png
```

### Generate a dashboard

```bash
pygs bench-gs --tier fast --sizes N50 -o results.json
pygs dashboard -i results.json -o dashboard.html
```

---

## 6. Biology workflows

### Load expression data

```python
from bio.preprocessing import load_expression_matrix, filter_tf_targets

# Supports .csv, .tsv, .npy, .h5ad (scanpy)
matrix, gene_names = load_expression_matrix("expression.h5ad")

# Filter to known transcription factors
tf_names, tf_indices = filter_tf_targets(gene_names, tf_file="tfs.txt")
print(f"{len(tf_names)} TFs found in expression data")
```

### Format as regulon edge list

```python
from bio.preprocessing import format_regulons

edges_df = format_regulons(adjacency, gene_names, tf_names, threshold=0.05)
print(edges_df.head())
#       TF  target   weight
# 0   TP53    MDM2   0.8421
# 1   TP53   CDKN1   0.7103
# 2    MYC   CCND1   0.6892
```

---

## Method tiers

The `--tier` flag groups methods by speed:

| Tier | Methods | Typical time (50 genes) |
|------|---------|------------------------|
| fast | lasso, elastic_net, lsco, ridge, partial_correlation, clr, genie3, neighborhood_selection | < 1s each |
| medium | glasso, bdeu, bge, granger_causality, transfer_entropy, pcmci | 1–10s each |
| slow | tigress, glasso_stars, pc, fci | 10–60s each |
| nestboot | Wraps selected tiers in NestBoot FDR | 10–100× slower |

Combine tiers: `--tier fast,medium` or `--tier fast,nestboot`.
