"""
Automated tests for AutoML Studio.
Validates configurations, data utilities, and UI helper logic.
"""
import pytest
import importlib.util
import os
import sys
from pathlib import Path

STUDIO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.fspath(STUDIO_ROOT))
import pandas as pd
import numpy as np


def _load_from_path(module_name: str, file_path: Path):
    """Load a module by file path, bypassing the ambiguous `src` package name.

    Both MLine/ and `sagemaker based/` have a top-level `src/` package, so a plain
    `from src...` can resolve to the wrong project when the full suite runs together
    (whichever `src` was imported first wins in sys.modules).
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_config = _load_from_path("automl_studio_config", STUDIO_ROOT / "config.py")
_data_utils = _load_from_path("automl_studio_data_utils", STUDIO_ROOT / "src" / "data_utils.py")
_ui_utils = _load_from_path("automl_studio_ui_utils", STUDIO_ROOT / "src" / "ui_utils.py")

MODELS_DIR = _config.MODELS_DIR
DATA_DIR = _config.DATA_DIR
APP_NAME = _config.APP_NAME
SUPPORTED_EXTENSIONS = _config.SUPPORTED_EXTENSIONS
detect_column_type = _data_utils.detect_column_type
detect_problem_type = _data_utils.detect_problem_type
profile_dataset = _data_utils.profile_dataset
load_global_css = _ui_utils.load_global_css

def test_config_paths():
    """Verify core directories exist or were created."""
    assert Path(MODELS_DIR).exists()
    assert Path(DATA_DIR).exists()
    assert APP_NAME == "AutoML Studio"

def test_file_extensions():
    """Verify supported file formats."""
    assert "csv" in SUPPORTED_EXTENSIONS
    assert "parquet" in SUPPORTED_EXTENSIONS

def test_column_type_detection():
    """Test semantic type detection logic."""
    df = pd.DataFrame({
        "num": [1, 2, 3, 4, 5],
        "bin": [0, 1, 0, 1, 0],
        "cat": ["A", "B", "A", "C", "A"],
        "dt": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
    })
    
    assert detect_column_type(df["num"]) == "Numeric"
    assert detect_column_type(df["bin"]) == "Binary"
    assert detect_column_type(df["cat"]) == "Categorical"
    # Datetime detection depends on sample parsing
    assert detect_column_type(df["dt"]) == "Datetime"

def test_problem_type_detection():
    """Test auto-discovery of ML problem category."""
    df_reg = pd.DataFrame({"target": np.random.rand(100)})
    df_bin = pd.DataFrame({"target": [0, 1, 0, 1, 0] * 20})
    df_multi = pd.DataFrame({"target": ["A", "B", "C", "A", "B"] * 20})
    
    assert detect_problem_type(df_reg, "target") == "regression"
    assert detect_problem_type(df_bin, "target") == "binary"
    assert detect_problem_type(df_multi, "target") == "multiclass"

def test_profiling_logic():
    """Verify dataset profiling generates expected metrics."""
    df = pd.DataFrame({
        "A": [1, 2, np.nan, 4, 5],
        "B": ["X", "Y", "X", "Y", "X"]
    })
    profile = profile_dataset(df)
    
    assert "A" in profile
    assert profile["A"]["missing_count"] == 1
    assert profile["A"]["type"] == "Numeric"
    assert "B" in profile
    assert profile["B"]["unique_count"] == 2

def test_ui_utils_load():
    """Verify CSS utility doesn't crash (smoke test)."""
    # This won't render in pytest but shouldn't raise exceptions if imported correctly
    load_global_css()
    assert True
