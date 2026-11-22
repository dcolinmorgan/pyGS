from typing import Any

from .clr import CLR
from .genie3 import GENIE3
from .lasso import Lasso
from .lsco import LSCO
from .tigress import TIGRESS
from .zscore import ZScore

try:
    from .scenicplus import SCENICPLUS
except ImportError:
    SCENICPLUS = None  # type: ignore[assignment]

try:
    from .panda import PANDA
except ImportError:
    PANDA = None  # type: ignore[assignment]


def run(
    method: Any,
    dataset: Any,
    nested_boot: bool = False,
    nest_runs: int = 50,
    boot_runs: int = 50,
    seed: int = 42,
    fdr: float = 0.05,
    **kwargs: Any,
) -> Any:
    """
    Unified runner for all inference methods.

    Args:
        method: The inference function (e.g., Lasso, GENIE3) or its name as string.
        dataset: The input dataset.
        nested_boot: Whether to run Nested Bootstrap FDR.
        nest_runs: Number of outer runs (if nested_boot=True).
        boot_runs: Number of inner runs (if nested_boot=True).
        seed: Random seed.
        fdr: False Discovery Rate threshold (if nested_boot=True).
        **kwargs: Arguments passed directly to the inference method.
    """
    # Resolve string method names to functions
    if isinstance(method, str):
        method_map = {
            "lasso": Lasso,
            "lsco": LSCO,
            "clr": CLR,
            "genie3": GENIE3,
            "tigress": TIGRESS,
            "zscore": ZScore,
        }
        if SCENICPLUS is not None:
            method_map["scenicplus"] = SCENICPLUS
        if PANDA is not None:
            method_map["panda"] = PANDA
        if method.lower() not in method_map:
            raise ValueError(
                f"Unknown method: {method}. Available: {list(method_map.keys())}"
            )
        method = method_map[method.lower()]

    if nested_boot:
        from .nestboot import Nestboot

        # Initialize Nestboot with configuration
        nb_config = {"fdr_threshold": fdr}
        nb = Nestboot(nb_config)

        return nb.run_nestboot(
            dataset=dataset,
            inference_method=method,
            nest_runs=nest_runs,
            boot_runs=boot_runs,
            seed=seed,
            method_kwargs=kwargs,
        )
    else:
        return method(dataset, **kwargs)


__all__ = ["Lasso", "LSCO", "CLR", "GENIE3", "TIGRESS", "ZScore", "SCENICPLUS", "PANDA", "run"]
