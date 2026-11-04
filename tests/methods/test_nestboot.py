import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.methods.nestboot import Nestboot, NetworkResults, AnalysisConfig
from src.methods.lsco import LSCO
from src.datastruct.Dataset import Dataset
from src.datastruct.Network import Network
from src.analyze.Data import Data
from src.analyze.CompareModels import CompareModels
import logging

class TestNestboot:
    @pytest.fixture
    def synthetic_data(self):
        np.random.seed(42)
        Y = np.random.rand(5, 10)
        P = np.random.rand(5, 10)
        
        # Use a simple class instead of MagicMock to avoid spec issues with private attributes
        class MockDataset:
            def __init__(self):
                self.Y = Y
                self.P = P
                self.N = 5
                self.M = 10
                self._network = None
                self._names = [f"Gene_{i:02d}" for i in range(5)]
                self._E = None
                self._lambda = None
                self._dataset_name = "Test"
                self.data = None # To simulate it not being a Data wrapper itself
        
        dataset = MockDataset()
        
        # Mock Data wrapper
        data = MagicMock(spec=Data)
        data.data = dataset
        
        return data

    @pytest.fixture
    def real_data(self):
        dataset = Data.from_json_url(
            'https://bitbucket.org/sonnhammergrni/gs-datasets/raw/d2047430263f5ffe473525c74b4318f723c23b0e/N50/Tjarnberg-ID252384-D20151111-N50-E150-SNR100000-IDY252384.json'
        )
        true_net = Network.from_json_url(
            'https://bitbucket.org/sonnhammergrni/gs-networks/raw/0b3a66e67d776eadaa5d68667ad9c8fbac12ef85/random/N50/Tjarnberg-D20150910-random-N50-L158-ID252384.json'
        )
        return dataset, true_net

    def test_nestboot_real_data_auroc(self, real_data):
        """Test Nestboot on real data to ensure AUROC > 0.5"""
        dataset, true_net = real_data
        
        # Verify node alignment
        data_genes = dataset.data.gene_names
        net_genes = true_net.names
        
        if data_genes != net_genes:
            print("Note: Dataset and Network gene order mismatch. Aligning Network to Dataset.")
            # Create mapping
            # We need to permute true_net.A to match data_genes order
            if set(data_genes) != set(net_genes):
                print(f"Gene sets differ! Data: {len(data_genes)}, Net: {len(net_genes)}")
                # If sets differ, we can't easily compare standard AUROC without intersection
                # But for this known dataset, they should matches set-wise.
            
            # Reorder true_net
            name_to_idx = {name: i for i, name in enumerate(net_genes)}
            new_indices = []
            valid_mask = []
            
            for i, name in enumerate(data_genes):
                if name in name_to_idx:
                    new_indices.append(name_to_idx[name])
                    valid_mask.append(True)
                else:
                    valid_mask.append(False) # Should not happen for GS benchmark
            
            # Permute A
            # A is (N, N)
            A = true_net.A
            # Select rows/cols
            # If dataset has genes not in net, we assume 0 links?
            # If net has genes not in dataset, they are ignored.
            
            new_A = A[new_indices, :][:, new_indices]
            true_net = Network(new_A, list(data_genes))

        # Use LSCO with threshold range
        zetavec = np.logspace(-6, 0, 10) # Smaller range for speed
        
        nb = Nestboot()
        
        # Run with fewer iterations for speed in test
        results = nb.run_nestboot(
            dataset=dataset,
            inference_method=LSCO,
            method_params={'threshold_range': zetavec},
            nest_runs=5, 
            boot_runs=5,
            seed=42
        )
        
        # Compare against true network(aligned)
        M4 = CompareModels(true_net, results.sxnet)
        
        # Check max AUROC
        max_auroc = np.max(M4.AUROC)
        
        # Should be significantly better than random (0.5)
        # Typically > 0.7 or 0.8 for this dataset
        assert max_auroc > 0.6, f"AUROC {max_auroc} is too low, expected > 0.6"
        
        # Also check other metrics to ensure we are getting something
        assert np.max(M4.F1) > 0.0, "F1 score is 0"

    def test_nestboot_init(self):
        nb = Nestboot()
        assert isinstance(nb.config, AnalysisConfig)
        
        config = AnalysisConfig(total_runs=10)
        nb = Nestboot(config)
        assert nb.config.total_runs == 10
        
        nb = Nestboot({'total_runs': 20})
        assert nb.config.total_runs == 20

    def test_compute_assign_frac(self):
        nb = Nestboot()
        
        # Create synthetic dataframe
        data = {
            'gene_i': ['G1', 'G1', 'G2'],
            'gene_j': ['G2', 'G2', 'G3'],
            'run': [0, 1, 0],
            'link_value': [0.5, 0.6, 0.7]
        }
        df = pd.DataFrame(data)
        
        results = nb.compute_assign_frac(df, total_runs=2, inner_group_size=1)
        
        assert 'Afrac' in results.columns
        # G1-G2 appears in 2 runs out of 2 -> Afrac = 1.0
        # G2-G3 appears in 1 run out of 2 -> Afrac = 0.5
        
        g1g2 = results[(results['gene_i'] == 'G1') & (results['gene_j'] == 'G2')]
        assert g1g2['Afrac'].values[0] == 1.0
        
        g2g3 = results[(results['gene_i'] == 'G2') & (results['gene_j'] == 'G3')]
        assert g2g3['Afrac'].values[0] == 0.5

    def test_nestboot_lasso_compatibility(self, synthetic_data):
        """Test Nestboot works with Lasso and legacy parameter names."""
        from src.methods.lasso import Lasso
        
        zetavec = np.logspace(-6, 0, 5)
        nb = Nestboot()
        
        # Should not raise TypeError: Lasso() got an unexpected keyword argument 'threshold_range'
        results = nb.run_nestboot(
            dataset=synthetic_data,
            inference_method=Lasso,
            method_params={'threshold_range': zetavec},
            nest_runs=2,
            boot_runs=2,
            seed=42
        )
        assert isinstance(results, NetworkResults)
        assert len(results.sxnet.shape) in [2, 3]

    def test_run_nestboot_mock(self, synthetic_data):
        nb = Nestboot({'fdr_threshold': 0.1})
        
        # Mock inference method
        # Returns a random adjacency matrix
        def mock_inference(dataset, **kwargs):
            return np.random.rand(5, 5)
            
        # Run nestboot with small number of runs
        # We need to patch Data and Dataset imports inside run_nestboot if they are imported there
        # But looking at the code, they are imported inside run_nestboot
        
        # We can mock the imports using patch.dict or just rely on the fact that we are passing objects
        # The code does: from datastruct.Dataset import Dataset
        # So we need to make sure that import works or is mocked if we want to control the object creation
        
        # However, since we are running in the same environment, the real imports should work.
        # The issue is that run_nestboot creates NEW Dataset objects.
        # We need to ensure those new objects work with our mock inference.
        
        # Our mock inference just takes whatever is passed and returns a matrix.
        # So it should be fine as long as the Dataset creation doesn't fail.
        
        results = nb.run_nestboot(
            dataset=synthetic_data,
            inference_method=mock_inference,
            nest_runs=2,
            boot_runs=2,
            seed=42
        )
        
        assert isinstance(results, NetworkResults)
        # The result xnet size depends on the number of genes found in the bootstrap data
        # Since we use "Gene_00" etc, and we have 5 genes, it should be 5x5 if all genes are found
        # But if some genes have no links in any bootstrap, they might be missing from the merged dataframe?
        # Actually compute_assign_frac groups by gene_i, gene_j.
        # And _compute_network_metrics creates xnet from merged dataframe.
        # The xnet returned is a numpy array.
        # Wait, NetworkResults.xnet is a numpy array.
        # But how is it constructed?
        # xnet = (merged['Afrac_norm'] >= support_threshold).astype(float)
        # This is a Series/Array, not a matrix.
        # Ah, looking at the code:
        # xnet=xnet.values
        # So it returns a 1D array of edges?
        # Let's check the code again.
        
        # In nestboot.py:
        # xnet = (merged['Afrac_norm'] >= support_threshold).astype(float)
        # ...
        # return NetworkResults(xnet=xnet.values, ...)
        
        # Yes, it seems to return arrays corresponding to the rows in the merged dataframe (edges).
        # It does NOT return a square adjacency matrix.
        
        assert isinstance(results.xnet, np.ndarray)


def _genespider_nb_fdr(cm, cs, n_inner, fdr):
    """GeneSPIDER's NB_FDR, FDRcutoff and networks for one zeta, written
    from +Methods/@NestBoot/NestBoot.m. cm, cs: support counts per outer run,
    shape (R, N, N). Tail area ratios use exact fractions, so ties at fdr
    do not depend on rounding."""
    from fractions import Fraction

    R, N, _ = cm.shape
    off = ~np.eye(N, dtype=bool)
    edges = np.arange(R + 1) / R
    counts = np.stack(
        [np.histogram(c[:, off].ravel() / n_inner, bins=edges)[0] for c in (cm, cs)], 1
    )
    L = R
    nlks = 0
    for ni in range(1, L):
        tail = [[Fraction(int(x)) for x in counts[L - ni - 1 :, i]] for i in (0, 1)]
        area = [sum(t) - (t[0] + t[-1]) / 2 for t in tail]
        if area[0] != 0 and area[1] / area[0] <= Fraction(fdr).limit_denominator(1000):
            nlks += 1
    orig_index = L - nlks
    cutoff = orig_index / R  # orig_index/100 in GeneSPIDER, run with R = 100
    xnet = np.all(cm / n_inner >= cutoff, axis=0) & off
    return orig_index, cutoff, xnet


@pytest.mark.parametrize("seed", range(20))
def test_nested_fdr_matches_genespider(seed):
    rng = np.random.default_rng(seed)
    R, Q, N = int(rng.integers(2, 12)), int(rng.integers(2, 25)), int(rng.integers(3, 10))
    if seed % 2:
        Q = R
    pm = rng.beta(rng.uniform(0.3, 3), rng.uniform(0.3, 3), size=(N, N))
    ps = rng.beta(0.5, rng.uniform(1, 8), size=(N, N))
    cm = rng.binomial(Q, np.broadcast_to(pm, (R, N, N))).astype(np.int64)
    cs = rng.binomial(Q, np.broadcast_to(ps, (R, N, N))).astype(np.int64)
    d = np.arange(N)
    cm[:, d, d] = 0
    cs[:, d, d] = 0
    off = ~np.eye(N, dtype=bool)
    fdr = float(rng.choice([0.05, 0.1, 0.2, 0.5]))

    run_hist = np.zeros((R, 1, Q + 1, 2), dtype=np.int64)
    for r in range(R):
        run_hist[r, 0, :, 0] = np.bincount(cm[r][off], minlength=Q + 1)
        run_hist[r, 0, :, 1] = np.bincount(cs[r][off], minlength=Q + 1)
    lo = np.stack([cm.min(0), cs.min(0)])[..., None]
    hi = np.stack([cm.max(0), cs.max(0)])[..., None]
    total = cm.sum(0)[..., None]
    res = Nestboot()._nested_fdr(run_hist, lo, hi, total, total, R, Q, fdr)

    orig_index, cutoff, xnet = _genespider_nb_fdr(cm, cs, Q, fdr)
    assert res.orig_index == orig_index
    assert np.isclose(res.support, cutoff)
    assert np.array_equal(res.xnet != 0, xnet)
