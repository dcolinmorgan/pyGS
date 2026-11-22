import sys
from unittest.mock import MagicMock, patch

# Mock scanpy before importing src.methods
sys.modules['scanpy'] = MagicMock()


import numpy as np
import pandas as pd
import pytest

from src.datastruct.Dataset import Dataset
from src.methods import run


@pytest.fixture
def mock_dataset():
    """Create a dummy dataset for testing."""
    dataset = Dataset()
    # 10 genes, 5 samples
    dataset._Y = np.random.rand(10, 5)
    dataset._P = np.random.rand(10, 5) # Add P matrix with matching dimensions
    dataset._names = [f"Gene_{i}" for i in range(10)]
    return dataset

@patch('src.methods.scenicplus.subprocess.run')
@patch('src.methods.scenicplus.os.path.exists') # Mock os.path.exists
@patch('src.methods.scenicplus.yaml.safe_load')
@patch('src.methods.scenicplus.yaml.dump')
def test_scenicplus_wrapper_success(mock_yaml_dump, mock_yaml_load, mock_exists, mock_subprocess, mock_dataset, tmp_path):
    """Test SCENICPLUS wrapper success path with mocking."""

    # Mock subprocess.run to return success
    mock_subprocess.return_value = MagicMock(returncode=0, stdout="Success")

    # Mock yaml load
    mock_yaml_load.return_value = {
        'input_data': {},
        'params_general': {'output_dir': 'results/run_{run_id}'}
    }

    # Mock os.path.exists to return True for config, snakefile, and results
    def side_effect(path):
        if "eRegulons" in str(path):
            return True
        if "config.yaml" in str(path):
            return True
        if "Snakefile" in str(path):
            return True
        return False # Default to False for others if any

    mock_exists.side_effect = side_effect

    with patch('src.methods.scenicplus.pd.read_csv') as mock_read_csv:
        # Mock the results dataframe
        mock_df = pd.DataFrame({
            'TF': ['Gene_0', 'Gene_1'],
            'Gene': ['Gene_1', 'Gene_2'],
            'importance': [0.5, 0.8]
        })
        mock_read_csv.return_value = mock_df

        workflow = tmp_path / "scenic_workflow"
        (workflow / "config").mkdir(parents=True)
        (workflow / "Snakefile").write_text("# stub\n")
        (workflow / "config" / "config.yaml").write_text("input_data: {}\n")
        work = tmp_path / "work"
        work.mkdir()

        # The direct Python path is the default. This test covers Snakemake.
        adj, _thresholds = run(
            "scenicplus",
            mock_dataset,
            cisTopic_obj_fname="dummy.pkl",
            use_snakemake=True,
            scenic_workflow_dir=str(workflow),
            work_dir=str(work),
        )

        # Verify write_h5ad was called (input preparation)
        # Since scanpy is mocked via sys.modules, we can check the mock
        import src.methods.scenicplus as sp
        # sp.sc is the MagicMock for scanpy
        # sp.sc.AnnData is the class/constructor
        # sp.sc.AnnData.return_value is the instance returned
        assert sp.sc.AnnData.return_value.write_h5ad.called

        # Verify yaml dump was called (config generation)
        assert mock_yaml_dump.called
        # Check that the dumped config contains the overrides
        dumped_config = mock_yaml_dump.call_args[0][0]
        assert 'input_data' in dumped_config
        assert 'GEX_anndata_fname' in dumped_config['input_data']
        assert 'cisTopic_obj_fname' in dumped_config['input_data']
        assert dumped_config['input_data']['cisTopic_obj_fname'].endswith('dummy.pkl')

        # Verify subprocess was called
        assert mock_subprocess.called
        args, _ = mock_subprocess.call_args
        cmd = args[0]
        assert "snakemake" in cmd
        assert "all" in cmd
        # Verify it uses the config.yaml (was run_config.yaml in test but code uses config.yaml)
        assert any("config.yaml" in str(arg) for arg in cmd)
        # Verify result shape
        assert adj.shape == (3, 3)
        assert adj[0, 1] == 0.5
        assert adj[1, 2] == 0.8

@patch('src.methods.scenicplus.subprocess.run')
def test_scenicplus_wrapper_failure(mock_subprocess, mock_dataset, tmp_path):
    """Test SCENICPLUS wrapper failure handling."""

    # Mock subprocess.run to return failure
    mock_subprocess.return_value = MagicMock(returncode=1, stdout="Error", stderr="Failed")
    workflow = tmp_path / "scenic_workflow"
    (workflow / "config").mkdir(parents=True)
    (workflow / "Snakefile").write_text("# stub\n")
    (workflow / "config" / "config.yaml").write_text(
        "input_data: {}\nparams_general:\n  output_dir: results/run_{run_id}\n"
    )
    work = tmp_path / "work"
    work.mkdir()

    with pytest.raises(RuntimeError, match="Snakemake failed"):
        run(
            "scenicplus",
            mock_dataset,
            cisTopic_obj_fname="dummy.pkl",
            use_snakemake=True,
            scenic_workflow_dir=str(workflow),
            work_dir=str(work),
        )

def test_unified_run_lasso(mock_dataset):
    """Test unified run with Lasso (mocked or real if fast)."""
    # Lasso is fast enough to run real, but let's mock to avoid dependency issues if any
    # We must patch 'src.methods.Lasso' because 'run' uses the imported symbol in src.methods
    with patch('src.methods.Lasso') as mock_lasso:
        mock_lasso.return_value = (np.zeros((10, 10)), np.array([0.1]))

        run('lasso', mock_dataset, alpha_range=np.array([0.5]))

        mock_lasso.assert_called_once()
        # Check args
        args, kwargs = mock_lasso.call_args
        assert args[0] == mock_dataset
        assert np.array_equal(kwargs['alpha_range'], np.array([0.5]))

@patch('src.methods.nestboot.Nestboot.run_nestboot')
def test_unified_run_nested_boot(mock_run_nestboot, mock_dataset):
    """Test unified run with nested_boot=True."""

    run('lasso', mock_dataset, nested_boot=True, nest_runs=10, boot_runs=5, fdr=0.1, alpha_range=np.array([0.5]))

    mock_run_nestboot.assert_called_once()
    _, kwargs = mock_run_nestboot.call_args

    assert kwargs['dataset'] == mock_dataset
    assert kwargs['nest_runs'] == 10
    assert kwargs['boot_runs'] == 5
    # Check that method_kwargs contains the alpha parameter
    assert np.array_equal(kwargs['method_kwargs']['alpha_range'], np.array([0.5]))

    # Check that the inference method passed is Lasso
    from src.methods.lasso import Lasso
    assert kwargs['inference_method'] == Lasso

