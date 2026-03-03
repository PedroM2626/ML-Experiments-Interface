"""
Time Series module — feature engineering, walk-forward CV, and TS-specific
algorithms for AutoML Studio forecasting experiments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
import warnings
warnings.filterwarnings("ignore")


# ── Metrics ───────────────────────────────────────────────────────────────────

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of time steps where prediction direction matches actual direction."""
    if len(y_true) < 2:
        return np.nan
    actual_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return float(np.mean(actual_dir == pred_dir))


TS_METRICS = {
    "rmse": "RMSE",
    "mae": "MAE",
    "mape": "MAPE (%)",
    "directional_accuracy": "Dir. Accuracy",
    "r2": "R²",
}

TS_OPTIMIZATION_METRICS = {
    "RMSE": "rmse",
    "MAE": "mae",
    "MAPE (%)": "mape",
    "R²": "r2",
}


# ── Date Feature Engineering ──────────────────────────────────────────────────

def extract_date_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add temporal features derived from a date column."""
    df = df.copy()
    try:
        dt = pd.to_datetime(df[date_col])
    except Exception:
        return df

    df["_ts_year"] = dt.dt.year
    df["_ts_month"] = dt.dt.month
    df["_ts_day"] = dt.dt.day
    df["_ts_dayofweek"] = dt.dt.dayofweek
    df["_ts_dayofyear"] = dt.dt.dayofyear
    df["_ts_quarter"] = dt.dt.quarter
    df["_ts_weekofyear"] = dt.dt.isocalendar().week.astype(int)
    df["_ts_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    # Cyclical encoding for month and dayofweek (avoids discontinuity)
    df["_ts_month_sin"] = np.sin(2 * np.pi * df["_ts_month"] / 12)
    df["_ts_month_cos"] = np.cos(2 * np.pi * df["_ts_month"] / 12)
    df["_ts_dow_sin"] = np.sin(2 * np.pi * df["_ts_dayofweek"] / 7)
    df["_ts_dow_cos"] = np.cos(2 * np.pi * df["_ts_dayofweek"] / 7)

    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: List[int],
) -> pd.DataFrame:
    """Add lagged values of the target column."""
    df = df.copy()
    for lag in lags:
        df[f"_lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: List[int],
) -> pd.DataFrame:
    """Add rolling statistics of the target column."""
    df = df.copy()
    for w in windows:
        df[f"_roll_mean_{w}"] = df[target_col].shift(1).rolling(window=w).mean()
        df[f"_roll_std_{w}"] = df[target_col].shift(1).rolling(window=w).std()
        df[f"_roll_min_{w}"] = df[target_col].shift(1).rolling(window=w).min()
        df[f"_roll_max_{w}"] = df[target_col].shift(1).rolling(window=w).max()
    return df


def add_diff_features(df: pd.DataFrame, target_col: str, orders: List[int] = [1]) -> pd.DataFrame:
    """Add differenced series (removes trend)."""
    df = df.copy()
    for order in orders:
        df[f"_diff_{order}"] = df[target_col].diff(order)
    return df


def engineer_time_series_features(
    df: pd.DataFrame,
    target_col: str,
    date_col: Optional[str] = None,
    lags: Optional[List[int]] = None,
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Full time series feature engineering pipeline."""
    # Auto-choose lags and windows based on dataset size
    n = len(df)
    if lags is None:
        if n >= 365:
            lags = [1, 2, 3, 7, 14, 28, 90, 365]
        elif n >= 52:
            lags = [1, 2, 3, 4, 8, 13, 26, 52]
        elif n >= 12:
            lags = [1, 2, 3, 6, 12]
        else:
            lags = [1, 2, 3]

    if windows is None:
        windows = [3, 7, 14] if n >= 14 else [2, 3]

    if date_col and date_col in df.columns:
        df = extract_date_features(df, date_col)

    df = add_lag_features(df, target_col, lags)
    df = add_rolling_features(df, target_col, windows)
    df = add_diff_features(df, target_col, orders=[1])

    # Drop rows with NaN caused by lags/rolling (max lag rows)
    max_lag = max(lags) if lags else 1
    df = df.dropna().reset_index(drop=True)

    return df


# ── Walk-Forward Cross-Validation ─────────────────────────────────────────────

def run_ts_cross_validation(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    optimization_metric: str = "rmse",
) -> Dict[str, float]:
    """Walk-forward (time series) cross-validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_rmse, all_mae, all_mape, all_da, all_r2 = [], [], [], [], []

    for train_idx, val_idx in tscv.split(X):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

        try:
            pipeline.fit(X_train_cv, y_train_cv)
            y_pred = pipeline.predict(X_val_cv)

            rmse_val = float(np.sqrt(mean_squared_error(y_val_cv, y_pred)))
            mae_val = float(mean_absolute_error(y_val_cv, y_pred))
            mape_val = mape(y_val_cv.values, y_pred)
            da_val = directional_accuracy(y_val_cv.values, y_pred)

            from sklearn.metrics import r2_score
            r2_val = float(r2_score(y_val_cv, y_pred))

            all_rmse.append(rmse_val)
            all_mae.append(mae_val)
            if not np.isnan(mape_val):
                all_mape.append(mape_val)
            if not np.isnan(da_val):
                all_da.append(da_val)
            all_r2.append(r2_val)
        except Exception:
            continue

    def _mean(lst):
        return float(np.mean([x for x in lst if not np.isnan(x)])) if lst else np.nan

    return {
        "rmse": round(_mean(all_rmse), 6),
        "mae": round(_mean(all_mae), 6),
        "mape": round(_mean(all_mape), 6) if all_mape else np.nan,
        "directional_accuracy": round(_mean(all_da), 6) if all_da else np.nan,
        "r2": round(_mean(all_r2), 6),
    }


def run_ts_expanding_window_cv(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    initial_train_size: float = 0.5,
    step_size: float = 0.1,
    optimization_metric: str = "rmse",
) -> Dict[str, float]:
    """Expanding window CV: more robust for long time series."""
    n = len(X)
    start = int(n * initial_train_size)
    step = int(n * step_size) or 1

    all_rmse, all_mae, all_mape, all_r2 = [], [], [], []

    for end in range(start, n - 1, step):
        # Validation window is next 'step' points
        val_end = min(end + step, n)
        X_train_cv, X_val_cv = X.iloc[:end], X.iloc[end:val_end]
        y_train_cv, y_val_cv = y.iloc[:end], y.iloc[end:val_end]

        if len(y_val_cv) == 0:
            break

        try:
            pipeline.fit(X_train_cv, y_train_cv)
            y_pred = pipeline.predict(X_val_cv)

            all_rmse.append(float(np.sqrt(mean_squared_error(y_val_cv, y_pred))))
            all_mae.append(float(mean_absolute_error(y_val_cv, y_pred)))
            all_mape.append(mape(y_val_cv.values, y_pred))

            from sklearn.metrics import r2_score
            all_r2.append(float(r2_score(y_val_cv, y_pred)))
        except Exception:
            continue

    def _mean(lst):
        return float(np.mean([x for x in lst if not np.isnan(x)])) if lst else np.nan

    return {
        "rmse": round(_mean(all_rmse), 6),
        "mae": round(_mean(all_mae), 6),
        "mape": round(_mean(all_mape), 6),
        "r2": round(_mean(all_r2), 6),
    }


# ── ARIMA Wrapper ─────────────────────────────────────────────────────────────

class ArimaForecaster(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible wrapper for statsmodels ARIMA."""
    def __init__(self, p=1, d=0, q=1, seasonal_p=0, seasonal_d=0, seasonal_q=0, s=0):
        self.p = p
        self.d = d
        self.q = q
        self.seasonal_p = seasonal_p
        self.seasonal_d = seasonal_d
        self.seasonal_q = seasonal_q
        self.s = s
        self.model_res_ = None

    def fit(self, X, y):
        from statsmodels.tsa.arima.model import ARIMA
        order = (int(self.p), int(self.d), int(self.q))
        seasonal_order = (int(self.seasonal_p), int(self.seasonal_d), int(self.seasonal_q), int(self.s)) if self.s > 0 else (0,0,0,0)

        # X is ignored in univariate ARIMA if not exogenous
        model = ARIMA(y, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        self.model_res_ = model.fit()
        return self

    def predict(self, X):
        if self.model_res_ is None:
            return np.zeros(len(X))
        # Returns out-of-sample forecast
        return self.model_res_.forecast(steps=len(X)).values


def get_ts_algorithm_registry():
    """Returns forecasting-compatible regressors."""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
    from sklearn.linear_model import Ridge, ElasticNet
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    registry = {
        "XGBoost": XGBRegressor,
        "LightGBM": LGBMRegressor,
        "GradientBoosting": GradientBoostingRegressor,
        "RandomForest": RandomForestRegressor,
        "ExtraTrees": ExtraTreesRegressor,
        "Ridge": Ridge,
        "ElasticNet": ElasticNet,
        "ARIMA": ArimaForecaster,
    }
    return registry


def instantiate_ts_model(algo_name: str, params: Optional[Dict] = None):
    registry = get_ts_algorithm_registry()
    cls = registry.get(algo_name)
    if cls is None:
        from sklearn.linear_model import Ridge
        return Ridge()
    params = params or {}
    try:
        import inspect
        valid = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
        accepted = {k: v for k, v in params.items() if k in valid}
        return cls(**accepted)
    except Exception:
        return cls()


def get_ts_primary_metric(scores: Dict[str, float], metric_key: str) -> float:
    """For TS, lower RMSE/MAE/MAPE = better; higher R² = better."""
    val = scores.get(metric_key, float("nan"))
    if np.isnan(val):
        return 0.0
    if metric_key in ("rmse", "mae", "mape"):
        return -val
    return val


TS_ALGORITHM_COLORS = {
    "XGBoost": "#F97316",
    "LightGBM": "#10B981",
    "GradientBoosting": "#8B5CF6",
    "RandomForest": "#3B82F6",
    "ExtraTrees": "#06B6D4",
    "Ridge": "#F59E0B",
    "ElasticNet": "#EC4899",
    "ARIMA": "#8B5CF6",
}


# ── Forecast Generation ───────────────────────────────────────────────────────

def generate_forecast(
    best_pipeline,
    X_history: pd.DataFrame,
    y_history: pd.Series,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Multi-step ahead forecast using proxy.
    """
    best_pipeline.fit(X_history, y_history)
    last_preds = best_pipeline.predict(X_history.tail(horizon))

    y_pred_all = best_pipeline.predict(X_history)
    residuals = y_history.values - y_pred_all
    std = np.std(residuals)

    lower = last_preds - 1.96 * std
    upper = last_preds + 1.96 * std

    return last_preds, lower, upper
