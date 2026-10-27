"""Data loading utilities."""
import hashlib
import re
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# sparselink methods whose second fit() argument is a perturbation matrix,
# not a regression target. LSCO uses A = -P @ pinv(Y).
PERTURBATION_METHODS = frozenset({"lsco"})


def detect_format(path: str) -> str:
    suffix = Path(urllib.parse.urlparse(path).path).suffix.lower() if "://" in path else Path(path).suffix.lower()
    mapping = {".h5ad": "h5ad", ".csv": "csv", ".tsv": "tsv", ".npy": "npy", ".txt": "tsv"}
    if suffix not in mapping:
        raise ValueError(f"Unsupported format: {suffix or path}")
    return mapping[suffix]


def materialize(path: str) -> str:
    """Return a local path, downloading http(s) inputs into .gs_cache."""
    parsed = urllib.parse.urlparse(path)
    if parsed.scheme == "file":
        return urllib.request.url2pathname(parsed.path)
    if parsed.scheme not in ("http", "https"):
        return path
    cache = Path(".gs_cache") / "downloads"
    cache.mkdir(parents=True, exist_ok=True)
    suffix = Path(parsed.path).suffix or ".dat"
    dest = cache / f"{hashlib.sha256(path.encode()).hexdigest()[:16]}{suffix}"
    if not dest.exists() or dest.stat().st_size == 0:
        urllib.request.urlretrieve(path, dest)
    return str(dest)


def _sniff_sep(path: str) -> str:
    sample = Path(path).read_text(errors="replace")[:8192]
    return "\t" if sample.count("\t") > sample.count(",") else ","


def _base_label(name: object) -> str:
    """Strip pandas duplicate suffixes such as GENE.1."""
    return re.sub(r"\.\d+$", "", str(name))


def _perturbation_matrix(
    gene_names: list[str], column_labels: list[object]
) -> np.ndarray | None:
    """Build genes × experiments P when columns are knockdown labels."""
    index = {name: i for i, name in enumerate(gene_names)}
    bases = [_base_label(label) for label in column_labels]
    if not bases or sum(base in index for base in bases) < 0.8 * len(bases):
        return None
    design = np.zeros((len(gene_names), len(bases)), dtype=float)
    for j, base in enumerate(bases):
        i = index.get(base)
        if i is not None:
            design[i, j] = 1.0
    if not np.any(design):
        return None
    return design


def load_inference_inputs(
    path: str,
) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    """Load expression for inference.

    Returns ``(X, gene_names, P)`` where ``X`` and ``P`` are samples × genes.
    ``P`` is set when the table is a GeneSpider-style knockdown matrix:
    rows are genes and column labels name the perturbed gene (replicates may
    repeat that name). Otherwise rows are treated as samples and ``P`` is None.
    """
    local = materialize(path)
    fmt = detect_format(local)
    if fmt == "h5ad":
        try:
            import scanpy as sc
        except ImportError:
            raise ImportError("scanpy required for .h5ad: pip install scanpy")
        adata = sc.read_h5ad(local)
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        return np.asarray(X, dtype=float), list(adata.var_names), None
    if fmt == "npy":
        arr = np.load(local)
        return arr, [f"V{i}" for i in range(arr.shape[1])], None

    df = pd.read_csv(local, sep=_sniff_sep(local), index_col=0)
    if df.shape[1] == 0 or df.shape[0] == 0:
        raise ValueError(f"No expression values in {path}")
    values = df.apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError(f"Non-numeric expression values in {path}")
    gene_names = [str(name) for name in df.index]
    design = _perturbation_matrix(gene_names, list(df.columns))
    if design is not None:
        Y = values.to_numpy(dtype=float)
        return Y.T, gene_names, design.T
    return values.to_numpy(dtype=float), [str(name) for name in df.columns], None


def load_expression(path: str) -> tuple[np.ndarray, list[str]]:
    """Load expression matrix. Returns (samples x genes, gene_names)."""
    X, gene_names, _P = load_inference_inputs(path)
    return X, gene_names


def load_tf_list(path: str) -> list[str]:
    """Load TF names from text file (one per line)."""
    return Path(path).read_text().strip().splitlines()
