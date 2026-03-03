"""
Stacked Ensemble — combines top-N AutoML pipelines into a meta-learner.
Mirrors the approach used by AutoAI (IBM) and AutoGluon.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")

ENSEMBLE_PIPELINE_ID = "P_Ens"
ENSEMBLE_COLOR = "#FBBF24"  # gold


def _select_top_pipelines(results: List[Dict], n_top: int = 3) -> List[Dict]:
    """Return top-N results by primary_metric_cv (descending absolute value)."""
    sorted_results = sorted(
        [r for r in results if r.get("pipeline") is not None],
        key=lambda r: abs(r.get("primary_metric_cv", 0)),
        reverse=True,
    )
    return sorted_results[:n_top]


def build_stacking_ensemble(
    results: List[Dict],
    X_train: Any,
    y_train: Any,
    X_holdout: Any,
    y_holdout: Any,
    task_type: str,
    optimization_metric: str,
    n_folds: int = 3,
    n_top: int = 3,
) -> Optional[Dict]:
    """
    Build a StackingClassifier or StackingRegressor from the top-N pipelines.
    Returns a result dict in the same shape as PipelineResult.asdict() or None on failure.
    """
    top = _select_top_pipelines(results, n_top=n_top)
    if len(top) < 2:
        return None  # Need at least 2 base learners

    t0 = time.time()

    # Build named estimators list
    estimators = [
        (r["pipeline_id"], r["pipeline"])
        for r in top
    ]

    try:
        if task_type in ("classification",):
            meta = LogisticRegression(max_iter=500, C=1.0)
            stacker = StackingClassifier(
                estimators=estimators,
                final_estimator=meta,
                cv=min(n_folds, 3),
                passthrough=False,
                n_jobs=1,
            )
            cv_splitter = StratifiedKFold(n_splits=min(n_folds, 3), shuffle=True, random_state=42)
            scoring = _get_scoring(task_type, optimization_metric)
            cv_res = cross_validate(stacker, X_train, y_train,
                                    cv=cv_splitter, scoring=scoring,
                                    return_train_score=False, n_jobs=1)
        else:  # regression / time_series
            meta = Ridge(alpha=1.0)
            stacker = StackingRegressor(
                estimators=estimators,
                final_estimator=meta,
                cv=min(n_folds, 3),
                passthrough=False,
                n_jobs=1,
            )
            cv_splitter = KFold(n_splits=min(n_folds, 3), shuffle=True, random_state=42)
            scoring = _get_scoring(task_type, optimization_metric)
            cv_res = cross_validate(stacker, X_train, y_train,
                                    cv=cv_splitter, scoring=scoring,
                                    return_train_score=False, n_jobs=1)

        # Fit on full train set
        stacker.fit(X_train, y_train)

        # CV scores
        cv_scores = _extract_cv_scores(cv_res, task_type)

        # Holdout
        holdout_scores = _evaluate_holdout(stacker, X_holdout, y_holdout, task_type)

        # Primary metric
        primary_cv = _get_primary(cv_scores, task_type, optimization_metric)
        primary_holdout = _get_primary(holdout_scores, task_type, optimization_metric)

        build_time = time.time() - t0

        return {
            "pipeline_id": ENSEMBLE_PIPELINE_ID,
            "algorithm": f"StackedEnsemble({'+'.join(r['pipeline_id'] for r in top)})",
            "transformer": "stacking",
            "use_pca": False,
            "use_select_k": False,
            "hyperparams": {"n_top": len(top), "meta_learner": type(meta).__name__},
            "cv_scores": cv_scores,
            "holdout_scores": holdout_scores,
            "primary_metric_cv": primary_cv,
            "primary_metric_holdout": primary_holdout,
            "build_time": build_time,
            "pipeline": stacker,
            "feature_importance": [],  # Stacking doesn't expose single FI
            "color": ENSEMBLE_COLOR,
        }

    except Exception as e:
        warnings.warn(f"[Ensemble] Stacking failed: {e}")
        return None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_scoring(task_type: str, opt_metric: str):
    _MAP = {
        "classification": {
            "roc_auc": "roc_auc",
            "f1": "f1_weighted",
            "accuracy": "accuracy",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
        },
        "regression": {
            "r2": "r2",
            "neg_rmse": "neg_root_mean_squared_error",
            "neg_mae": "neg_mean_absolute_error",
        },
    }
    task_map = _MAP.get(task_type, _MAP["classification"])
    return task_map.get(opt_metric, list(task_map.values())[0])


def _extract_cv_scores(cv_res, task_type: str) -> Dict:
    scores = {}
    for key, vals in cv_res.items():
        if key.startswith("test_"):
            metric = key[5:]
            val = float(np.mean(np.abs(vals)))
            scores[metric] = val
    return scores


def _evaluate_holdout(model, X, y, task_type: str) -> Dict:
    from sklearn.metrics import (
        roc_auc_score, f1_score, accuracy_score,
        r2_score, mean_squared_error, mean_absolute_error,
    )
    scores = {}
    try:
        y_pred = model.predict(X)
        if task_type == "classification":
            scores["accuracy"] = float(accuracy_score(y, y_pred))
            scores["f1"] = float(f1_score(y, y_pred, average="weighted", zero_division=0))
            try:
                if hasattr(model, "predict_proba"):
                    yp = model.predict_proba(X)
                    if yp.shape[1] == 2:
                        scores["roc_auc"] = float(roc_auc_score(y, yp[:, 1]))
                    else:
                        scores["roc_auc"] = float(roc_auc_score(y, yp, multi_class="ovr", average="weighted"))
            except Exception:
                pass
        else:
            scores["r2"] = float(r2_score(y, y_pred))
            scores["neg_rmse"] = float(-np.sqrt(mean_squared_error(y, y_pred)))
            scores["neg_mae"] = float(-mean_absolute_error(y, y_pred))
    except Exception:
        pass
    return scores


def _get_primary(scores: Dict, task_type: str, opt_metric: str) -> float:
    if opt_metric in scores:
        return scores[opt_metric]
    # Fallback
    if task_type == "classification":
        for k in ("roc_auc", "f1", "accuracy"):
            if k in scores:
                return scores[k]
    else:
        for k in ("r2", "neg_rmse", "neg_mae"):
            if k in scores:
                return scores[k]
    return 0.0
