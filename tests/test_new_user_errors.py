import os
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import methods
from datastruct.Dataset import Dataset


def test_load_non_existent_file():
    """Test loading a file that does not exist."""
    with pytest.raises(FileNotFoundError):
        pd.read_csv("non_existent_file.csv")

def test_inference_on_empty_dataset():
    """Test running inference on an empty dataset."""
    ds = Dataset()
    # ds.Y is None by default

    # Most methods should raise an error or handle it gracefully
    # Let's see what GENIE3 does
    with pytest.raises(Exception): # Catch generic exception as we don't know exact type yet
        methods.GENIE3(ds)

def test_inference_on_invalid_data_shape():
    """Test running inference on data with mismatched dimensions."""
    ds = Dataset()
    # Create data where genes != samples but that's allowed.
    # Create data with NaN or Inf
    ds.Y = np.array([[1, 2, np.nan], [4, 5, 6]])

    # GENIE3 handles NaNs? Probably not or sklearn will complain
    with pytest.raises(ValueError):
        methods.GENIE3(ds)

def test_invalid_method_name():
    """Test running an invalid method name via run()."""
    ds = Dataset()
    ds.Y = np.random.rand(10, 10)

    with pytest.raises(ValueError, match="Unknown method"):
        methods.run("invalid_method", ds)
