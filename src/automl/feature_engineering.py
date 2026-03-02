"""
Feature Engineering module — transformers and auto-selection logic.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PowerTransformer, QuantileTransformer, LabelEncoder, OneHotEncoder
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from typing import List, Dict, Optional, Tuple


# ── Transformer catalogue ──────────────────────────────────────────────────────

NUMERIC_TRANSFORMERS = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler(),
    "PowerTransformer (Yeo-Johnson)": PowerTransformer(method="yeo-johnson"),
    "QuantileTransformer": QuantileTransformer(output_distribution="normal", random_state=42),
    "None": "passthrough",
}

FEATURE_SELECTIONS = {
    "SelectKBest": None,   # built dynamically
    "PCA": None,           # built dynamically
    "None": "passthrough",
}


def detect_column_types(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    """Return (numeric_cols, categorical_cols) excluding target."""
    feature_df = df.drop(columns=[target])
    numeric = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric, categorical


def choose_transformers_for_data(df: pd.DataFrame, numeric_cols: List[str]) -> List[str]:
    """Auto-select a subset of transformer names based on data statistics."""
    if not numeric_cols:
        return ["StandardScaler"]

    skewness = df[numeric_cols].skew().abs().mean()
    chosen = ["StandardScaler"]

    if skewness > 1.0:
        chosen.append("PowerTransformer (Yeo-Johnson)")
    if len(numeric_cols) > 5:
        chosen.append("MinMaxScaler")
    if skewness > 0.5:
        chosen.append("QuantileTransformer")

    return list(set(chosen))


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    numeric_transformer_name: str = "StandardScaler",
) -> ColumnTransformer:
    """Build a ColumnTransformer for numeric + categorical features."""
    transformer_obj = NUMERIC_TRANSFORMERS.get(numeric_transformer_name, StandardScaler())

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", transformer_obj),
    ])

    if categorical_cols:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ])
    else:
        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, numeric_cols),
        ])

    return preprocessor


def build_feature_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    numeric_transformer_name: str,
    use_pca: bool = False,
    pca_components: Optional[int] = None,
    use_select_k: bool = False,
    k_best: int = 10,
    task_type: str = "classification",
) -> List:
    """Return list of (name, step) tuples ready for sklearn Pipeline."""
    steps = [
        ("preprocessor", build_preprocessor(numeric_cols, categorical_cols, numeric_transformer_name))
    ]

    if use_select_k:
        score_func = f_classif if task_type == "classification" else f_regression
        actual_k = min(k_best, len(numeric_cols) + len(categorical_cols))
        steps.append(("feature_selection", SelectKBest(score_func=score_func, k=actual_k)))

    if use_pca:
        n = pca_components or max(2, (len(numeric_cols) // 2))
        steps.append(("pca", PCA(n_components=n, random_state=42)))

    return steps
