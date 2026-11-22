"""
Conftest for pytest.

This file adds the 'src' directory to sys.path so that the package can be imported
as if it were installed and provides common fixtures for testing.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

# Add the src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Mock scanpy globally to avoid dependency issues
sys.modules['scanpy'] = Mock()


# Configure logging for tests
logging.getLogger().setLevel(logging.WARNING)  # Suppress info logs during testing

# Configure pytest markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "web: marks tests that require internet connection"
    )

# Test fixtures
@pytest.fixture
def sample_adjacency_matrix():
    """Fixture providing a sample adjacency matrix."""
    return np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ])

@pytest.fixture
def sample_gene_expression_data():
    """Fixture providing sample gene expression data."""
    return pd.DataFrame({
        'gene_i': ['GENE_A', 'GENE_B', 'GENE_C', 'GENE_A', 'GENE_B'],
        'gene_j': ['GENE_B', 'GENE_C', 'GENE_A', 'GENE_C', 'GENE_A'],
        'link_value': [0.85, 0.72, 0.63, 0.45, 0.91],
        'run': [0, 0, 0, 1, 1],
        'p_value': [0.001, 0.005, 0.020, 0.080, 0.002]
    })

@pytest.fixture
def temp_output_dir():
    """Fixture providing a temporary output directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)
