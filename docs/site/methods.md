# Method Comparison

All methods implement the unified `InferenceMethod.fit(X) -> InferenceResult` interface.

## Summary Table

| Method | Registry Key | Category | Directed | Extra Deps | Best For |
|--------|-------------|----------|----------|------------|----------|
| Lasso | `lasso` | Regression | Yes | — | Sparse linear networks |
| Elastic Net | `elastic_net` | Regression | Yes | — | Correlated features |
| Ridge | `ridge` | Regression | Yes | — | Dense networks |
| LSCO | `lsco` | Regression | Yes | — | Hard-threshold sparse |
| TIGRESS | `tigress` | Stability selection | Yes | — | Robust feature selection |
| GENIE3 | `genie3` | Tree-based | Yes | — | Non-linear relationships |
| Partial Correlation | `partial_correlation` | Correlation | No | — | Undirected Gaussian |
| CLR | `clr` | Information theory | No | — | Mutual information + z-score |
| Z-score | `zscore` | Perturbation | Yes | — | DREAM/GeneSPIDER perturbation z-scores |
| Graphical Lasso | `graphical_lasso` | Graphical model | No | — | Sparse precision matrix |
| GLASSO+StARS | `glasso_stars` | Graphical model | No | — | Auto-tuned graphical lasso |
| Neighborhood Selection | `neighborhood_selection` | Graphical model | No | — | Meinshausen-Bühlmann |
| PCMCI | `pcmci` | Causal (time-series) | Yes | — | Lagged causal effects |
| Granger Causality | `granger` | Causal (time-series) | Yes | — | VAR-based causality |
| Transfer Entropy | `transfer_entropy` | Causal (time-series) | Yes | — | Non-linear temporal |
| PC | `pc` | Constraint-based | Partial | `[causal]` | Causal skeleton |
| FCI | `fci` | Constraint-based | Partial | `[causal]` | Latent confounders |
| NOTEARS | `notears` | Continuous optimization | Yes | — | Differentiable DAG |
| DAG-GNN | `dag_gnn` | Deep learning | Yes | `[deep]` | Non-linear DAG |
| BDeu | `bdeu` | Bayesian | Yes | — | Discrete/small data |
| BGe | `bge` | Bayesian | Yes | — | Continuous Gaussian |

## Category Details

### Regression Methods

Fit each gene as a function of all others. Fast, scalable, produce directed graphs.

```python
from sparselink import get_method
result = get_method("lasso")(alpha=0.1).fit(X)
```

### Information-Theoretic

Compute pairwise mutual information, then apply CLR normalization. Undirected output.

### Graphical Models

Estimate the precision matrix (inverse covariance). Zeros in the precision matrix indicate conditional independence.

### Causal Discovery

Constraint-based (PC, FCI) or score-based (BDeu, BGe) or continuous optimization (NOTEARS, DAG-GNN). Produce DAGs or PAGs.

### Time-Series Causal

Require temporal ordering. Data shape is still `(samples, features)` where samples are time points.

## Choosing a Method

- **Small p, want DAG**: NOTEARS or BDeu/BGe
- **Large p, sparse**: Lasso, TIGRESS, or Neighborhood Selection
- **Non-linear**: GENIE3 or DAG-GNN
- **Undirected suffices**: Graphical Lasso or Partial Correlation
- **Time-series data**: PCMCI or Granger
- **Latent confounders**: FCI
- **Robust selection**: TIGRESS or GLASSO+StARS
