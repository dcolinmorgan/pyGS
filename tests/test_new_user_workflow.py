import os
import sys

import numpy as np
import pandas as pd

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import methods
from analyze.CompareModels import CompareModels
from datastruct.Dataset import Dataset
from datastruct.Experiment import Experiment
from datastruct.Network import Network
from datastruct.scalefree import scalefree


def test_new_user_workflow(tmp_path):
    """
    Simulates a new user workflow:
    1. Create synthetic network and data.
    2. Save data to file (CSV).
    3. Load data from file.
    4. Run inference methods.
    5. Compare results.
    """
    print("\n--- Starting New User Workflow Test ---")

    # 1. Generate Ground Truth
    print("Generating synthetic network...")
    N = 20
    avg_links = 2
    # scalefree returns an adjacency matrix
    A = scalefree(N, avg_links)
    net_true = Network(A)

    # 2. Generate Data
    print("Generating synthetic expression data...")
    exp = Experiment(net_true)
    exp.gaussian()
    ds_true = Dataset(exp)
    Y = ds_true.Y

    # Check data shape
    assert Y.shape[0] == N
    print(f"Data shape: {Y.shape}")

    gene_names = [f"Gene_{i}" for i in range(N)]

    # 3. Save to CSV (Simulating user saving their data)
    print("Saving data to CSV...")
    df = pd.DataFrame(Y, index=gene_names)
    # Transpose so samples are rows? Usually expression matrices are Genes x Samples or Samples x Genes.
    # pyGS seems to use Genes x Samples (N x M).
    # Let's save it as is.
    csv_path = tmp_path / "expression_data.csv"
    df.to_csv(csv_path)

    assert os.path.exists(csv_path)

    # 4. Load from CSV (Simulating user loading their data)
    print("Loading data from CSV...")
    df_loaded = pd.read_csv(csv_path, index_col=0)
    Y_loaded = df_loaded.values
    names_loaded = df_loaded.index.tolist()

    assert np.allclose(Y, Y_loaded)
    assert names_loaded == gene_names

    # Create Dataset from loaded data
    ds_loaded = Dataset()
    ds_loaded.Y = Y_loaded
    ds_loaded.P = ds_true.P
    ds_loaded.gene_names = names_loaded

    # 5. Inference
    print("Running inference methods...")

    # GENIE3
    print("Running GENIE3...")
    # GENIE3 returns (Afit, threshold_range)
    # Afit is 3D array (genes x genes x thresholds)
    Afit_genie3, _ = methods.GENIE3(ds_loaded)
    net_genie3 = Network(Afit_genie3)
    assert isinstance(net_genie3, Network)
    assert net_genie3.A.shape[0] == N
    assert net_genie3.A.shape[1] == N

    # CLR
    print("Running CLR...")
    Afit_clr, _ = methods.CLR(ds_loaded)
    net_clr = Network(Afit_clr)
    assert isinstance(net_clr, Network)
    assert net_clr.A.shape[0] == N
    assert net_clr.A.shape[1] == N

    # Lasso
    print("Running Lasso...")
    Afit_lasso, _ = methods.Lasso(ds_loaded)
    net_lasso = Network(Afit_lasso)
    assert isinstance(net_lasso, Network)

    # TIGRESS
    print("Running TIGRESS...")
    # TIGRESS might be slow
    Afit_tigress, _ = methods.TIGRESS(ds_loaded, n_bootstrap=2)
    net_tigress = Network(Afit_tigress)
    assert isinstance(net_tigress, Network)

    # 6. Comparison
    print("Comparing results...")

    # Compare GENIE3
    comp_genie3 = CompareModels(net_true, net_genie3)
    # Access properties directly
    auroc_genie3 = comp_genie3.AUROC
    print(f"GENIE3 AUROC: {auroc_genie3}")

    # Compare CLR
    comp_clr = CompareModels(net_true, net_clr)
    auroc_clr = comp_clr.AUROC
    print(f"CLR AUROC: {auroc_clr}")

    # Compare Lasso
    comp_lasso = CompareModels(net_true, net_lasso)
    auroc_lasso = comp_lasso.AUROC
    print(f"Lasso AUROC: {auroc_lasso}")

    # Compare TIGRESS
    comp_tigress = CompareModels(net_true, net_tigress)
    auroc_tigress = comp_tigress.AUROC
    print(f"TIGRESS AUROC: {auroc_tigress}")

    # Basic sanity check: AUROC should be calculable (not NaN)
    assert not np.isnan(auroc_genie3).any()
    assert not np.isnan(auroc_clr).any()
    assert not np.isnan(auroc_lasso).any()
    assert not np.isnan(auroc_tigress).any()

    print("--- New User Workflow Test Completed Successfully ---")

def test_data_creation_details():
    """
    More detailed test on data creation to ensure it behaves as expected for a user.
    """
    N = 10
    A = np.eye(N) # Simple identity network
    net = Network(A)
    exp = Experiment(net)
    exp.gaussian()
    ds = Dataset(exp)

    assert ds.Y is not None
    assert ds.Y.shape[0] == N
    # Default samples might be N or something else, let's check
    print(f"Samples generated: {ds.Y.shape[1]}")

    # Check noise
    assert ds.E is not None
    assert ds.E.shape == ds.Y.shape

