"""
Data profiler — generates summary statistics for the uploaded dataset.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple


def profile_dataset(df: pd.DataFrame, target: Optional[str] = None) -> Dict[str, Any]:
    """Return a comprehensive profile dict for a DataFrame."""
    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    missing = df.isnull().sum()
    missing_pct = (missing / n_rows * 100).round(2)

    profile = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_numeric": len(numeric_cols),
        "n_categorical": len(categorical_cols),
        "missing_total": int(missing.sum()),
        "missing_pct": float((missing.sum() / (n_rows * n_cols) * 100).round(2)),
        "duplicated_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "columns": _col_profiles(df, numeric_cols, categorical_cols, missing_pct),
    }

    if target and target in df.columns:
        profile["target"] = _target_profile(df[target])

    return profile


def _col_profiles(df, numeric_cols, categorical_cols, missing_pct) -> list:
    profiles = []
    for col in df.columns:
        p = {
            "name": col,
            "dtype": str(df[col].dtype),
            "missing_pct": float(missing_pct[col]),
            "n_unique": int(df[col].nunique()),
        }
        if col in numeric_cols:
            p.update({
                "type": "numeric",
                "mean": round(float(df[col].mean()), 4),
                "std": round(float(df[col].std()), 4),
                "min": round(float(df[col].min()), 4),
                "max": round(float(df[col].max()), 4),
                "skew": round(float(df[col].skew()), 4),
            })
        else:
            top = df[col].value_counts().head(3).to_dict()
            p.update({
                "type": "categorical",
                "top_values": {str(k): int(v) for k, v in top.items()},
            })
        profiles.append(p)
    return profiles


def _target_profile(series: pd.Series) -> Dict:
    if series.dtype == object or str(series.dtype) == "category":
        vc = series.value_counts()
        return {
            "type": "categorical",
            "n_classes": int(series.nunique()),
            "class_distribution": {str(k): int(v) for k, v in vc.items()},
        }
    else:
        return {
            "type": "numeric",
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "skew": round(float(series.skew()), 4),
        }


def infer_task_type(series: pd.Series, max_classes: int = 20) -> str:
    """Heuristically determine if regression or classification."""
    if series.dtype == object or str(series.dtype) == "category":
        return "classification"
    if series.nunique() <= max_classes:
        return "classification"
    return "regression"


def get_numeric_and_categorical(df: pd.DataFrame, target: str) -> Tuple[list, list]:
    feature_df = df.drop(columns=[target])
    numeric = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric, categorical
