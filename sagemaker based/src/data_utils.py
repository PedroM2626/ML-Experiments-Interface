"""
Data utilities: loading, profiling, type detection, and transformations.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ── Loading ────────────────────────────────────────────────────────────────────

def load_dataset(file_path: str | Path | io.IOBase, filename: str = "") -> pd.DataFrame:
    """Load a dataset from a file path or uploaded file object."""
    fname = filename or (str(file_path) if isinstance(file_path, (str, Path)) else "")
    ext = Path(fname).suffix.lower()

    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    elif ext == ".parquet":
        return pd.read_parquet(file_path)
    else:
        # Try CSV as default
        return pd.read_csv(file_path)


# ── Type Detection ─────────────────────────────────────────────────────────────

def detect_column_type(series: pd.Series) -> str:
    """
    Detect the semantic type of a DataFrame column.
    Returns: 'Numeric', 'Binary', 'Categorical', 'Datetime', 'Text', 'ID'
    """
    if series.dtype == "bool":
        return "Binary"

    # Try datetime
    if series.dtype == "object":
        try:
            import warnings
            sample = series.dropna().head(50).astype(str)
            # Skip if values look purely numeric
            if not sample.str.match(r'^\d+(\.\d+)?$').all():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().sum() > len(sample) * 0.7:
                    return "Datetime"
        except Exception:
            pass

    if pd.api.types.is_datetime64_any_dtype(series):
        return "Datetime"

    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique()
        if n_unique == 2:
            return "Binary"
        return "Numeric"

    if series.dtype == "object":
        n_unique = series.nunique()
        n_total = len(series.dropna())
        if n_total == 0:
            return "Categorical"
        uniqueness_ratio = n_unique / n_total
        # High uniqueness → likely ID or free text
        if uniqueness_ratio > 0.9 and n_total > 50:
            avg_len = series.dropna().astype(str).str.len().mean()
            if avg_len > 30:
                return "Text"
            return "ID"
        return "Categorical"

    return "Categorical"


# ── Profiling ──────────────────────────────────────────────────────────────────

def profile_dataset(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Return a profile dict for each column with stats and distribution sample.
    """
    profile = {}
    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        col_type = detect_column_type(series)
        n_missing = series.isna().sum()
        pct_missing = round(n_missing / n_rows * 100, 2) if n_rows > 0 else 0.0

        # Mismatched: only relevant for numeric cols with coercion errors
        n_mismatched = 0
        if col_type in ("Numeric", "Binary"):
            coerced = pd.to_numeric(series, errors="coerce")
            n_mismatched = coerced.isna().sum() - n_missing

        n_unique = series.nunique()

        col_info: dict[str, Any] = {
            "type": col_type,
            "missing_count": int(n_missing),
            "missing_pct": pct_missing,
            "mismatched_count": int(max(n_mismatched, 0)),
            "mismatched_pct": round(max(n_mismatched, 0) / n_rows * 100, 2) if n_rows > 0 else 0.0,
            "unique_count": int(n_unique),
        }

        # Distribution data
        clean = series.dropna()
        if col_type in ("Numeric",):
            col_info["distribution"] = "histogram"
            col_info["min"] = float(clean.min()) if len(clean) > 0 else 0
            col_info["max"] = float(clean.max()) if len(clean) > 0 else 0
            col_info["mean"] = float(clean.mean()) if len(clean) > 0 else 0
            col_info["median"] = float(clean.median()) if len(clean) > 0 else 0
            col_info["sample"] = clean.sample(min(500, len(clean)), random_state=42).tolist()
        elif col_type in ("Binary", "Categorical"):
            col_info["distribution"] = "bar"
            vc = clean.value_counts()
            col_info["value_counts"] = vc.head(20).to_dict()
            col_info["n_categories"] = int(n_unique)
        elif col_type == "Datetime":
            col_info["distribution"] = "datetime"
            parsed = pd.to_datetime(clean, errors="coerce").dropna()
            col_info["min"] = str(parsed.min()) if len(parsed) > 0 else ""
            col_info["max"] = str(parsed.max()) if len(parsed) > 0 else ""
        else:
            col_info["distribution"] = "text"

        profile[col] = col_info

    return profile


# ── Problem Type Detection ──────────────────────────────────────────────────────

def detect_problem_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Heuristically determine problem type from target column.
    Returns: 'binary', 'multiclass', 'regression'
    """
    series = df[target_col].dropna()
    col_type = detect_column_type(series)

    if col_type in ("Binary",):
        return "binary"
    if col_type in ("Categorical",):
        n_unique = series.nunique()
        if n_unique == 2:
            return "binary"
        return "multiclass"
    if col_type in ("Numeric",):
        n_unique = series.nunique()
        if n_unique == 2:
            return "binary"
        if n_unique <= 20:
            return "multiclass"
        return "regression"
    return "binary"


# ── Transformations ─────────────────────────────────────────────────────────────

def extract_datetime_features(
    df: pd.DataFrame,
    col: str,
    features: list[str],
) -> pd.DataFrame:
    """Extract year/month/day/hour/weekday from a datetime column."""
    df = df.copy()
    dt_series = pd.to_datetime(df[col], errors="coerce")

    feature_map = {
        "Year": dt_series.dt.year,
        "Month": dt_series.dt.month,
        "Day": dt_series.dt.day,
        "Hour": dt_series.dt.hour,
        "Weekday": dt_series.dt.weekday,
        "Quarter": dt_series.dt.quarter,
    }
    for feat in features:
        if feat in feature_map:
            df[f"{col}_{feat}"] = feature_map[feat]
    return df


def drop_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Drop specified columns from DataFrame."""
    return df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")


def filter_rows(df: pd.DataFrame, col: str, operator: str, value: Any) -> pd.DataFrame:
    """Apply a simple row filter."""
    try:
        if operator == ">":
            return df[df[col] > float(value)]
        elif operator == "<":
            return df[df[col] < float(value)]
        elif operator == "==":
            return df[df[col].astype(str) == str(value)]
        elif operator == "!=":
            return df[df[col].astype(str) != str(value)]
        elif operator == ">=":
            return df[df[col] >= float(value)]
        elif operator == "<=":
            return df[df[col] <= float(value)]
        elif operator == "contains":
            return df[df[col].astype(str).str.contains(str(value), na=False)]
    except Exception:
        pass
    return df


def apply_transforms(df: pd.DataFrame, recipe: list[dict]) -> pd.DataFrame:
    """
    Apply a list of transform steps from the model recipe.
    Each step: {"type": "drop_column"|"extract_datetime"|"filter_rows", ...params}
    """
    for step in recipe:
        t = step.get("type")
        if t == "drop_column":
            df = drop_columns(df, [step["col"]])
        elif t == "extract_datetime":
            df = extract_datetime_features(df, step["col"], step["features"])
        elif t == "filter_rows":
            df = filter_rows(df, step["col"], step["operator"], step["value"])
    return df


# ── Dataset Summary ─────────────────────────────────────────────────────────────

def dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Return top-level summary of the dataset."""
    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "n_cells": len(df) * len(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
