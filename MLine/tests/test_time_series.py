"""
Tests for time series modules — feature engineering, walk-forward CV,
TS metrics, and ExperimentManager.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def ts_df():
    """Monthly time series — 60 rows."""
    dates = pd.date_range("2019-01-01", periods=60, freq="MS").strftime("%Y-%m-%d")
    trend = np.linspace(100, 200, 60)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(60) / 12)
    noise = np.random.default_rng(42).normal(0, 5, 60)
    df = pd.DataFrame({
        "Date": dates,
        "Value": (trend + seasonal + noise).round(2),
    })
    return df


@pytest.fixture
def daily_ts_df():
    """Daily TS — 120 rows."""
    rng = np.random.default_rng(7)
    vals = 50 + np.arange(120) * 0.3 + rng.normal(0, 3, 120)
    df = pd.DataFrame({
        "ds": pd.date_range("2023-01-01", periods=120, freq="D").strftime("%Y-%m-%d"),
        "y": vals.round(2),
    })
    return df


# ── Time series feature engineering ───────────────────────────────────────────

class TestTSFeatureEngineering:

    def test_extract_date_features(self, ts_df):
        from src.automl.time_series import extract_date_features
        result = extract_date_features(ts_df, "Date")
        # Should have added several _ts_* columns
        ts_cols = [c for c in result.columns if c.startswith("_ts_")]
        assert len(ts_cols) >= 6, f"Expected at least 6 date features, got {len(ts_cols)}: {ts_cols}"
        assert "_ts_month" in result.columns
        assert "_ts_dayofweek" in result.columns
        assert "_ts_month_sin" in result.columns
        assert "_ts_is_weekend" in result.columns

    def test_add_lag_features(self, ts_df):
        from src.automl.time_series import add_lag_features
        result = add_lag_features(ts_df, "Value", lags=[1, 3, 6])
        assert "_lag_1" in result.columns
        assert "_lag_3" in result.columns
        assert "_lag_6" in result.columns
        # Lag 1 should shift by 1
        assert pd.isna(result["_lag_1"].iloc[0])

    def test_add_rolling_features(self, ts_df):
        from src.automl.time_series import add_rolling_features
        result = add_rolling_features(ts_df, "Value", windows=[3, 7])
        assert "_roll_mean_3" in result.columns
        assert "_roll_std_7" in result.columns
        # Rolling window with shift(1) → first row should be NaN
        assert pd.isna(result["_roll_mean_3"].iloc[0])

    def test_full_pipeline(self, ts_df):
        from src.automl.time_series import engineer_time_series_features
        # Use explicit smaller lags to ensure enough rows survive after NaN drop
        result = engineer_time_series_features(
            ts_df, "Value", date_col="Date",
            lags=[1, 2, 3], windows=[2, 3]
        )
        assert "Value" in result.columns
        # With lags=[1,2,3] and window=3, max lag is 3 → 60-3 = 57 rows should survive
        assert len(result) > 30, f"Expected >30 rows after NaN drop, got {len(result)}"
        assert len(result) < len(ts_df), "Should have fewer rows (NaN rows dropped)"
        feature_cols = [c for c in result.columns if c != "Value" and c != "Date"]
        assert len(feature_cols) >= 5, f"Expected ≥5 features, got {len(feature_cols)}"

    def test_no_future_leakage(self, ts_df):
        """Lag features must not include the target's current value."""
        from src.automl.time_series import add_lag_features
        result = add_lag_features(ts_df, "Value", lags=[1])
        # lag_1.iloc[i] should equal Value.iloc[i-1]
        for i in range(1, min(5, len(result))):
            assert result["_lag_1"].iloc[i] == result["Value"].iloc[i - 1]


# ── TS Metrics ─────────────────────────────────────────────────────────────────

class TestTSMetrics:

    def test_mape_perfect(self):
        from src.automl.time_series import mape
        y = np.array([100, 200, 300, 400])
        assert mape(y, y) == pytest.approx(0.0)

    def test_mape_50_percent(self):
        from src.automl.time_series import mape
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([150.0, 300.0])
        assert mape(y_true, y_pred) == pytest.approx(50.0)

    def test_mape_zero_denominator(self):
        from src.automl.time_series import mape
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 1.0])
        # Should return NaN, not raise
        result = mape(y_true, y_pred)
        assert np.isnan(result)

    def test_directional_accuracy_perfect(self):
        from src.automl.time_series import directional_accuracy
        y = np.array([1, 2, 3, 4, 5])
        assert directional_accuracy(y, y) == pytest.approx(1.0)

    def test_directional_accuracy_opposite(self):
        from src.automl.time_series import directional_accuracy
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([5, 4, 3, 2, 1])
        assert directional_accuracy(y_true, y_pred) == pytest.approx(0.0)

    def test_directional_accuracy_short(self):
        from src.automl.time_series import directional_accuracy
        y = np.array([1.0])
        result = directional_accuracy(y, y)
        assert np.isnan(result)


# ── Walk-Forward CV ────────────────────────────────────────────────────────────

class TestTSCrossValidation:

    def test_run_ts_cv(self, ts_df):
        from src.automl.time_series import engineer_time_series_features, run_ts_cross_validation
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline

        df_feat = engineer_time_series_features(ts_df, "Value", date_col="Date")
        X = df_feat.drop(columns=["Value", "Date"], errors="ignore")
        y = df_feat["Value"]

        pipeline = Pipeline([("model", Ridge())])
        scores = run_ts_cross_validation(pipeline, X, y, n_splits=3)

        assert "rmse" in scores
        assert "mae" in scores
        assert "r2" in scores
        assert scores["rmse"] >= 0, "RMSE should be non-negative"

    def test_ts_primary_metric_rmse(self):
        from src.automl.time_series import get_ts_primary_metric
        scores = {"rmse": 5.0, "mae": 3.0}
        # Lower RMSE = better, returned negated
        val = get_ts_primary_metric(scores, "rmse")
        assert val == pytest.approx(-5.0)

    def test_ts_primary_metric_r2(self):
        from src.automl.time_series import get_ts_primary_metric
        scores = {"r2": 0.95}
        val = get_ts_primary_metric(scores, "r2")
        assert val == pytest.approx(0.95)


# ── TS Algorithm Registry ──────────────────────────────────────────────────────

class TestTSAlgorithmRegistry:

    def test_registry_has_expected_algos(self):
        from src.automl.time_series import get_ts_algorithm_registry
        reg = get_ts_algorithm_registry()
        for algo in ["XGBoost", "LightGBM", "Ridge", "RandomForest"]:
            assert algo in reg, f"{algo} not in TS registry"

    def test_instantiate_ts_model(self):
        from src.automl.time_series import instantiate_ts_model
        model = instantiate_ts_model("Ridge", {"alpha": 1.0})
        assert model is not None
        assert hasattr(model, "fit")

    def test_instantiate_unknown_model(self):
        from src.automl.time_series import instantiate_ts_model
        # Should fall back to Ridge
        model = instantiate_ts_model("NonExistentAlgo")
        assert model is not None
        assert hasattr(model, "fit")


# ── TS Engine Smoke Test ───────────────────────────────────────────────────────

class TestTSEngine:

    def test_ts_engine_smoke(self, ts_df):
        """
        Smoke test: TSEngineConfig + TimeSeriesEngine should complete
        within 120 seconds on the small monthly fixture.
        """
        import queue
        from src.automl.ts_engine import TimeSeriesEngine, TSEngineConfig
        from src.automl.engine import EventType

        config = TSEngineConfig(
            task_type="time_series",
            target_column="Value",
            date_column="Date",
            horizon=6,
            freq="M",
            n_splits=2,
            max_algorithms=2,
            n_estimators_per_algo=1,
            hpo_trials=3,
            optimization_metric="rmse",
        )

        engine = TimeSeriesEngine(config)
        q = queue.Queue()
        thread = engine.run_async(ts_df, q)
        thread.join(timeout=120)

        events = []
        while not q.empty():
            events.append(q.get_nowait())

        event_types = {e.get("type") for e in events}
        assert EventType.STAGE in event_types, "Should have emitted STAGE events"
        # Should have at least one pipeline done or a DONE event
        assert (EventType.PIPELINE_DONE in event_types or EventType.DONE in event_types), \
            f"Expected PIPELINE_DONE or DONE, got: {event_types}"
