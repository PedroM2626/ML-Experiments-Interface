"""
AutoML Engine – orchestrates AutoGluon training runs with FLAML as fallback.
Handles quick build, standard build, feature importance, predictions, and leaderboard.
"""
from __future__ import annotations

import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import MODELS_DIR, QUICK_BUILD_TIME


# ── AutoGluon Helpers ──────────────────────────────────────────────────────────

def _get_autogluon_predictor():
    """Lazy import of AutoGluon to avoid startup delays."""
    try:
        from autogluon.tabular import TabularPredictor
        return TabularPredictor
    except Exception:
        return None


def _get_flaml_automl():
    """Lazy import of FLAML."""
    try:
        from flaml import AutoML
        return AutoML
    except Exception:
        return None


def _format_error(e: Exception) -> str:
    """Return a human-readable error string, never empty."""
    msg = str(e)
    if not msg:
        msg = repr(e)
    if not msg:
        msg = f"Unexpected error: {type(e).__name__}"
    return msg


# ── Preview Build ──────────────────────────────────────────────────────────────

def preview_model(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
) -> dict[str, Any]:
    """
    Fast model preview using AutoGluon with a very short time limit (30s).
    Returns estimated accuracy and rough feature importance.
    """
    TabularPredictor = _get_autogluon_predictor()
    if TabularPredictor is None:
        # Try FLAML preview
        return _flaml_preview(df, target_col, problem_type)

    model_path = str(MODELS_DIR / "_preview_tmp")
    shutil.rmtree(model_path, ignore_errors=True)
    try:
        predictor = TabularPredictor(
            label=target_col,
            problem_type=problem_type,
            path=model_path,
            verbosity=0,
        ).fit(
            df,
            time_limit=30,
            presets="medium_quality",
            excluded_model_types=["KNN", "NN_TORCH"],
        )

        leaderboard = predictor.leaderboard(silent=True)
        score = float(leaderboard["score_val"].iloc[0]) if len(leaderboard) > 0 else 0.0

        fi_dict: dict[str, float] = {}
        try:
            fi = predictor.feature_importance(df, silent=True)
            fi_dict = fi["importance"].to_dict() if "importance" in fi.columns else fi.iloc[:, 0].to_dict()
        except Exception:
            pass

        # Normalize importance scores to [0, 1]
        if fi_dict:
            max_fi = max(abs(v) for v in fi_dict.values())
            if max_fi > 0:
                fi_dict = {k: v / max_fi for k, v in fi_dict.items()}

        return {
            "estimated_score": abs(score),
            "feature_importance": fi_dict,
            "leaderboard": leaderboard.to_dict("records") if len(leaderboard) > 0 else [],
        }
    except Exception as e:
        return {"error": _format_error(e), "traceback": traceback.format_exc()}
    finally:
        shutil.rmtree(model_path, ignore_errors=True)


def _flaml_preview(df: pd.DataFrame, target_col: str, problem_type: str) -> dict[str, Any]:
    """Lightweight preview using FLAML with a 30-second budget."""
    AutoML = _get_flaml_automl()
    if AutoML is None:
        return {"error": "No AutoML backend found. Install autogluon or flaml."}
    try:
        from flaml import AutoML as FL

        flaml_task = "classification" if problem_type in ("binary", "multiclass") else "regression"
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_enc = pd.get_dummies(X, drop_first=True)

        automl = FL()
        automl.fit(X_enc, y, task=flaml_task, time_budget=20, verbose=0)

        fi_dict: dict[str, float] = {}
        try:
            if hasattr(automl.model, "estimator") and hasattr(automl.model.estimator, "feature_importances_"):
                importances = automl.model.estimator.feature_importances_
                max_fi = max(importances) if max(importances) > 0 else 1
                fi_dict = {col: float(imp / max_fi) for col, imp in zip(X_enc.columns, importances)}
        except Exception:
            pass

        return {
            "estimated_score": max(0.0, 1.0 - float(automl.best_loss)),
            "feature_importance": fi_dict,
            "leaderboard": [],
        }
    except Exception as e:
        return {"error": _format_error(e), "traceback": traceback.format_exc()}


# ── Main Training ──────────────────────────────────────────────────────────────

def train_model(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
    model_name: str,
    build_type: str = "quick",
    time_limit: int | None = None,
    excluded_features: list[str] | None = None,
    presets: str | None = None,
) -> dict[str, Any]:
    """
    Train an AutoGluon model. Falls back to FLAML if AutoGluon unavailable.

    Returns:
        dict with keys: predictor_path, leaderboard, feature_importance,
                        best_model, fit_time, train_columns, error (on failure)
    """
    # Resolve time limit
    if time_limit is None:
        if build_type == "quick":
            time_limit = QUICK_BUILD_TIME
        # else None = no limit (standard build runs to completion)

    # Drop excluded features from training data
    train_df = df.copy()
    if excluded_features:
        cols_to_drop = [c for c in excluded_features if c in train_df.columns and c != target_col]
        train_df = train_df.drop(columns=cols_to_drop, errors="ignore")

    # Store the final set of feature columns the model will use
    train_columns = [c for c in train_df.columns if c != target_col]

    # ── AutoGluon ──────────────────────────────────────────────────────────────
    TabularPredictor = _get_autogluon_predictor()
    if TabularPredictor is not None:
        result = _train_autogluon(
            train_df, target_col, problem_type,
            model_name, build_type, time_limit, presets,
        )
        if "error" not in result:
            result["train_columns"] = train_columns
        return result

    # ── FLAML fallback ─────────────────────────────────────────────────────────
    AutoML = _get_flaml_automl()
    if AutoML is not None:
        result = _train_flaml(
            train_df, target_col, problem_type,
            model_name, build_type, time_limit,
        )
        if "error" not in result:
            result["train_columns"] = train_columns
        return result

    return {"error": "No AutoML backend found. Please install autogluon or flaml."}


def _train_autogluon(
    df, target_col, problem_type, model_name, build_type, time_limit, presets
) -> dict[str, Any]:
    try:
        from autogluon.tabular import TabularPredictor
    except Exception as e:
        return {"error": f"AutoGluon import failed: {_format_error(e)}"}

    model_path = str(MODELS_DIR / model_name.replace(" ", "_"))
    preset = presets or ("medium_quality" if build_type == "quick" else "best_quality")

    # ── Clean up any existing model path to avoid AG-0009 conflict ────────────
    shutil.rmtree(model_path, ignore_errors=True)

    try:
        t0 = time.time()
        predictor = TabularPredictor(
            label=target_col,
            problem_type=problem_type,
            path=model_path,
            verbosity=1,
        ).fit(
            df,
            time_limit=time_limit,
            presets=preset,
        )
        fit_time = time.time() - t0

        leaderboard = predictor.leaderboard(silent=True)
        leaderboard_records = leaderboard.to_dict("records") if len(leaderboard) > 0 else []

        # Feature importance
        fi_dict: dict[str, float] = {}
        try:
            fi = predictor.feature_importance(df, silent=True)
            fi_dict = fi["importance"].to_dict() if "importance" in fi.columns else fi.iloc[:, 0].to_dict()
            max_fi = max(abs(v) for v in fi_dict.values()) if fi_dict else 1
            if max_fi > 0:
                fi_dict = {k: v / max_fi for k, v in fi_dict.items()}
        except Exception:
            pass

        best_model = str(leaderboard["model"].iloc[0]) if len(leaderboard) > 0 else "unknown"
        best_score = float(leaderboard["score_val"].iloc[0]) if len(leaderboard) > 0 else 0.0

        metadata = {
            "model_name": model_name,
            "problem_type": problem_type,
            "target_col": target_col,
            "build_type": build_type,
            "time_limit": time_limit,
            "preset": preset,
            "fit_time": fit_time,
            "best_model": best_model,
            "best_score": best_score,
            "backend": "autogluon",
            "n_rows": len(df),
            "n_features": len(df.columns) - 1,
        }
        meta_path = Path(model_path) / "automl_metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return {
            "predictor_path": model_path,
            "predictor": predictor,
            "leaderboard": leaderboard_records,
            "feature_importance": fi_dict,
            "best_model": best_model,
            "best_score": best_score,
            "fit_time": fit_time,
            "backend": "autogluon",
            "metadata": metadata,
        }
    except Exception as e:
        tb = traceback.format_exc()
        err_msg = _format_error(e)
        return {"error": err_msg, "traceback": tb}


def _train_flaml(
    df, target_col, problem_type, model_name, build_type, time_limit
) -> dict[str, Any]:
    try:
        from flaml import AutoML
        import pickle
    except Exception as e:
        return {"error": f"FLAML import failed: {_format_error(e)}"}

    flaml_task_map = {
        "binary": "classification",
        "multiclass": "classification",
        "regression": "regression",
    }

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # One-hot encode categoricals; store column order for consistent prediction
    X_enc = pd.get_dummies(X, drop_first=True)
    encoded_columns = X_enc.columns.tolist()  # ← store for prediction alignment

    automl = AutoML()
    t0 = time.time()
    try:
        automl.fit(
            X_enc, y,
            task=flaml_task_map.get(problem_type, "classification"),
            time_budget=time_limit or 120,
            verbose=1,
        )
        fit_time = time.time() - t0

        model_path = str(MODELS_DIR / model_name.replace(" ", "_"))
        shutil.rmtree(model_path, ignore_errors=True)
        Path(model_path).mkdir(parents=True, exist_ok=True)

        model_file = Path(model_path) / "flaml_model.pkl"
        # Persist both model AND encoded column order
        with open(model_file, "wb") as f:
            pickle.dump({"automl": automl, "encoded_columns": encoded_columns}, f)

        fi_dict: dict[str, float] = {}
        try:
            if hasattr(automl.model, "estimator") and hasattr(automl.model.estimator, "feature_importances_"):
                importances = automl.model.estimator.feature_importances_
                max_fi = max(importances) if max(importances) > 0 else 1
                fi_dict = {col: float(imp / max_fi) for col, imp in zip(encoded_columns, importances)}
        except Exception:
            pass

        metadata = {
            "model_name": model_name,
            "problem_type": problem_type,
            "target_col": target_col,
            "build_type": build_type,
            "best_model": str(automl.best_estimator),
            "best_score": float(automl.best_loss),
            "fit_time": fit_time,
            "backend": "flaml",
            "n_rows": len(df),
            "n_features": len(X.columns),
        }

        with open(Path(model_path) / "automl_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return {
            "predictor_path": model_path,
            "predictor": automl,
            "encoded_columns": encoded_columns,   # ← used in predict()
            "leaderboard": [],
            "feature_importance": fi_dict,
            "best_model": str(automl.best_estimator),
            "best_score": float(automl.best_loss),
            "fit_time": fit_time,
            "backend": "flaml",
            "metadata": metadata,
        }
    except Exception as e:
        tb = traceback.format_exc()
        return {"error": _format_error(e), "traceback": tb}


# ── Predictions ────────────────────────────────────────────────────────────────

def predict(
    predictor,
    df: pd.DataFrame,
    problem_type: str,
    encoded_columns: list[str] | None = None,
) -> dict[str, Any]:
    """
    Generate predictions from a loaded predictor.
    Returns dict with 'predictions' and optionally 'probabilities'.

    For FLAML models, pass `encoded_columns` (from train result) so that
    the prediction features are aligned to the training schema.
    """
    try:
        backend = _detect_backend(predictor)

        if backend == "autogluon":
            preds = predictor.predict(df)
            result: dict[str, Any] = {"predictions": preds}
            if problem_type in ("binary", "multiclass"):
                try:
                    probas = predictor.predict_proba(df)
                    result["probabilities"] = probas
                except Exception:
                    pass
            return result

        else:
            # FLAML: align feature columns to training schema
            target_columns = encoded_columns
            X = pd.get_dummies(df, drop_first=True)
            if target_columns:
                # Add missing columns as 0, drop extra columns, reorder
                for col in target_columns:
                    if col not in X.columns:
                        X[col] = 0
                X = X[target_columns]
            preds = predictor.predict(X)
            return {"predictions": pd.Series(preds)}

    except Exception as e:
        return {"error": _format_error(e), "traceback": traceback.format_exc()}


def _detect_backend(predictor) -> str:
    """Detect if predictor is AutoGluon or FLAML."""
    cls_name = type(predictor).__name__
    if "TabularPredictor" in cls_name:
        return "autogluon"
    return "flaml"


# ── Load / List ────────────────────────────────────────────────────────────────

def load_predictor(model_path: str):
    """Load a saved predictor from disk (AutoGluon or FLAML)."""
    TabularPredictor = _get_autogluon_predictor()
    if TabularPredictor is not None:
        try:
            return TabularPredictor.load(model_path)
        except Exception:
            pass
    # Try FLAML pickle
    import pickle
    flaml_file = Path(model_path) / "flaml_model.pkl"
    if flaml_file.exists():
        with open(flaml_file, "rb") as f:
            data = pickle.load(f)
            # Support both old (raw AutoML) and new (dict) pickle formats
            if isinstance(data, dict):
                return data.get("automl")
            return data
    return None


def get_saved_models() -> list[dict[str, Any]]:
    """List all saved models with their metadata."""
    models = []
    if not MODELS_DIR.exists():
        return models
    for model_dir in MODELS_DIR.iterdir():
        if model_dir.is_dir() and not model_dir.name.startswith("_"):
            meta_file = model_dir / "automl_metadata.json"
            if meta_file.exists():
                try:
                    with open(meta_file) as f:
                        meta = json.load(f)
                    meta["path"] = str(model_dir)
                    models.append(meta)
                except Exception:
                    pass
    return models
