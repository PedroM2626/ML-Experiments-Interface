"""
Hyperparameter optimization using Optuna TPE sampler.
"""

import optuna
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
)
from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso, ElasticNet
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False
    CatBoostClassifier = None
    CatBoostRegressor = None
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Algorithm registries ───────────────────────────────────────────────────────

CLASSIFIERS = {
    "GradientBoosting": GradientBoostingClassifier,
    "RandomForest": RandomForestClassifier,
    "ExtraTrees": ExtraTreesClassifier,
    "XGBoost": XGBClassifier,
    "LightGBM": LGBMClassifier,
    "LogisticRegression": LogisticRegression,
    "KNeighbors": KNeighborsClassifier,
}

REGRESSORS = {
    "GradientBoosting": GradientBoostingRegressor,
    "RandomForest": RandomForestRegressor,
    "ExtraTrees": ExtraTreesRegressor,
    "XGBoost": XGBRegressor,
    "LightGBM": LGBMRegressor,
    "Ridge": Ridge,
    "KNeighbors": KNeighborsRegressor,
    "ARIMA": None, # Handled specially or imported from time_series
}

if _CATBOOST_AVAILABLE:
    CLASSIFIERS["CatBoost"] = CatBoostClassifier
    REGRESSORS["CatBoost"] = CatBoostRegressor

ALGORITHM_COLORS = {
    "GradientBoosting": "#8B5CF6",   # purple
    "RandomForest": "#3B82F6",       # blue
    "ExtraTrees": "#06B6D4",         # cyan
    "XGBoost": "#10B981",            # emerald
    "LightGBM": "#F59E0B",          # amber
    "CatBoost": "#EF4444",           # red
    "LogisticRegression": "#6366F1", # indigo
    "Ridge": "#6366F1",
    "KNeighbors": "#EC4899",         # pink
}


def get_algorithm_registry(task_type: str) -> Dict:
    return CLASSIFIERS if task_type == "classification" else REGRESSORS


def _suggest_params_tree(trial: optuna.Trial, prefix: str = "") -> Dict:
    return {
        f"{prefix}n_estimators": trial.suggest_int(f"{prefix}n_estimators", 50, 400, step=50),
        f"{prefix}max_depth": trial.suggest_int(f"{prefix}max_depth", 3, 12),
        f"{prefix}min_samples_split": trial.suggest_int(f"{prefix}min_samples_split", 2, 20),
        f"{prefix}min_samples_leaf": trial.suggest_int(f"{prefix}min_samples_leaf", 1, 10),
        f"{prefix}max_features": trial.suggest_categorical(f"{prefix}max_features", ["sqrt", "log2", None]),
    }


def get_search_space(algo_name: str, trial: optuna.Trial) -> Dict:
    """Return hyperparameter suggestions for each algorithm."""
    if algo_name in ("RandomForest", "ExtraTrees"):
        return _suggest_params_tree(trial)

    elif algo_name == "GradientBoosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        }

    elif algo_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    elif algo_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

    elif algo_name == "CatBoost":
        return {
            "iterations": trial.suggest_int("iterations", 50, 300, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-4, 10.0, log=True),
        }

    elif algo_name == "LogisticRegression":
        return {
            "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 500, step=100),
            "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
        }

    elif algo_name == "Ridge":
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True),
        }

    elif algo_name == "KNeighbors":
        return {
            "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan"]),
        }

    elif algo_name == "ARIMA":
        return {
            "p": trial.suggest_int("p", 0, 5),
            "d": trial.suggest_int("d", 0, 2),
            "q": trial.suggest_int("q", 0, 5),
            "s": trial.suggest_categorical("s", [0, 7, 12, 24]),
        }

    return {}


def get_default_params(algo_name: str, task_type: str) -> Dict:
    """Return sensible default params before HPO runs."""
    defaults = {
        "RandomForest": {"n_estimators": 100, "max_depth": 6, "random_state": 42},
        "ExtraTrees": {"n_estimators": 100, "max_depth": 6, "random_state": 42},
        "GradientBoosting": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 4, "random_state": 42},
        "XGBoost": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5, "random_state": 42,
                    "eval_metric": "logloss" if task_type == "classification" else "rmse",
                    "verbosity": 0},
        "LightGBM": {"n_estimators": 100, "learning_rate": 0.1, "verbose": -1, "random_state": 42},
        "CatBoost": {"iterations": 100, "learning_rate": 0.1, "verbose": 0, "random_state": 42},
        "LogisticRegression": {"C": 1.0, "max_iter": 300, "random_state": 42},
        "Ridge": {"alpha": 1.0},
        "KNeighbors": {"n_neighbors": 5},
        "ARIMA": {"p": 1, "d": 0, "q": 1, "s": 0},
    }
    return defaults.get(algo_name, {})


def instantiate_model(algo_name: str, task_type: str, params: Dict = None):
    """Instantiate an algorithm with given params."""
    if algo_name == "ARIMA":
        from .time_series import ArimaForecaster
        registry = {"ARIMA": ArimaForecaster}
    else:
        registry = get_algorithm_registry(task_type)
    
    cls = registry.get(algo_name)
    if cls is None:
        # Fallback
        if task_type == "classification":
            from sklearn.linear_model import LogisticRegression
            cls = LogisticRegression
        else:
            from sklearn.linear_model import Ridge
            cls = Ridge
    if params is None:
        params = get_default_params(algo_name, task_type)
    try:
        return cls(**params)
    except TypeError:
        # Strip unknown kwargs
        import inspect
        valid = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
        clean = {k: v for k, v in params.items() if k in valid}
        return cls(**clean)
