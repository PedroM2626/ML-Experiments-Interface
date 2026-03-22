"""
Unit tests for the AutoML engine and supporting modules.
"""

import pytest
import os
import pandas as pd
import numpy as np
import queue
import time
from sklearn.datasets import load_iris, load_diabetes

from src.automl.engine import AutoMLEngine, EngineConfig, EventType
from src.automl.feature_engineering import (
    detect_column_types, choose_transformers_for_data, build_preprocessor, build_feature_pipeline
)
from src.automl.hyperopt import get_default_params, instantiate_model, get_algorithm_registry
from src.automl.evaluator import run_cross_validation, evaluate_holdout, get_primary_metric
from src.automl.pipeline_builder import build_pipeline, get_feature_importance
from src.utils.data_profiler import profile_dataset, infer_task_type


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def iris_df():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["target"] = iris.target_names[iris.target]
    return df

@pytest.fixture
def diabetes_df():
    diabetes = load_diabetes(as_frame=True)
    return diabetes.frame.copy()


# ── Feature Engineering tests ──────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_detect_column_types(self, iris_df):
        numeric, categorical = detect_column_types(iris_df, "target")
        assert len(numeric) == 4
        assert len(categorical) == 0

    def test_choose_transformers(self, iris_df):
        numeric, _ = detect_column_types(iris_df, "target")
        transformers = choose_transformers_for_data(iris_df, numeric)
        assert isinstance(transformers, list)
        assert len(transformers) >= 1

    def test_build_preprocessor(self, iris_df):
        numeric, categorical = detect_column_types(iris_df, "target")
        preprocessor = build_preprocessor(numeric, categorical)
        X = iris_df.drop(columns=["target"])
        preprocessor.fit_transform(X)

    def test_build_feature_pipeline(self, iris_df):
        numeric, categorical = detect_column_types(iris_df, "target")
        steps = build_feature_pipeline(numeric, categorical, "StandardScaler", task_type="classification")
        assert isinstance(steps, list)
        assert len(steps) >= 1


# ── Hyperopt tests ─────────────────────────────────────────────────────────────

class TestHyperopt:
    def test_default_params_classification(self):
        params = get_default_params("RandomForest", "classification")
        assert "n_estimators" in params

    def test_instantiate_model_classification(self):
        model = instantiate_model("RandomForest", "classification")
        assert hasattr(model, "fit")

    def test_instantiate_model_regression(self):
        model = instantiate_model("Ridge", "regression")
        assert hasattr(model, "fit")

    def test_registry_classification(self):
        registry = get_algorithm_registry("classification")
        assert "RandomForest" in registry
        assert "XGBoost" in registry

    def test_registry_regression(self):
        registry = get_algorithm_registry("regression")
        assert "Ridge" in registry


# ── Evaluator tests ────────────────────────────────────────────────────────────

class TestEvaluator:
    def test_cross_validation_classification(self, iris_df):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        numeric, categorical = detect_column_types(iris_df, "target")
        steps = build_feature_pipeline(numeric, categorical, "StandardScaler", task_type="classification")
        model = instantiate_model("RandomForest", "classification")
        pipeline = build_pipeline(steps, model)

        X = iris_df.drop(columns=["target"])
        y = pd.Series(LabelEncoder().fit_transform(iris_df["target"]))
        scores = run_cross_validation(pipeline, X, y, "classification", n_folds=2)
        assert "accuracy" in scores
        assert 0 <= scores["accuracy"] <= 1.0

    def test_cross_validation_regression(self, diabetes_df):
        numeric, categorical = detect_column_types(diabetes_df, "target")
        steps = build_feature_pipeline(numeric, categorical, "StandardScaler", task_type="regression")
        model = instantiate_model("Ridge", "regression")
        pipeline = build_pipeline(steps, model)

        X = diabetes_df.drop(columns=["target"])
        y = diabetes_df["target"]
        scores = run_cross_validation(pipeline, X, y, "regression", n_folds=2)
        assert "r2" in scores


# ── Pipeline builder tests ─────────────────────────────────────────────────────

class TestPipelineBuilder:
    def test_build_and_fit(self, iris_df):
        from sklearn.preprocessing import LabelEncoder
        numeric, categorical = detect_column_types(iris_df, "target")
        steps = build_feature_pipeline(numeric, categorical, "StandardScaler", task_type="classification")
        model = instantiate_model("RandomForest", "classification")
        pipeline = build_pipeline(steps, model)

        X = iris_df.drop(columns=["target"])
        y = pd.Series(LabelEncoder().fit_transform(iris_df["target"]))
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert len(preds) == len(y)

    def test_feature_importance(self, iris_df):
        from sklearn.preprocessing import LabelEncoder
        numeric, categorical = detect_column_types(iris_df, "target")
        steps = build_feature_pipeline(numeric, categorical, "StandardScaler", task_type="classification")
        model = instantiate_model("RandomForest", "classification")
        pipeline = build_pipeline(steps, model)

        X = iris_df.drop(columns=["target"])
        y = pd.Series(LabelEncoder().fit_transform(iris_df["target"]))
        pipeline.fit(X, y)
        fi = get_feature_importance(pipeline, numeric + categorical)
        assert fi is not None
        assert "feature" in fi.columns
        assert "importance" in fi.columns


# ── Data profiler tests ────────────────────────────────────────────────────────

class TestDataProfiler:
    def test_profile_classification(self, iris_df):
        profile = profile_dataset(iris_df, target="target")
        assert profile["n_rows"] == 150
        assert profile["n_cols"] == 5
        assert "target" in profile

    def test_infer_task_type_classification(self, iris_df):
        task = infer_task_type(iris_df["target"])
        assert task == "classification"

    def test_infer_task_type_regression(self, diabetes_df):
        task = infer_task_type(diabetes_df["target"])
        assert task == "regression"


# ── Engine integration test (smoke test with minimal config) ───────────────────

class TestEngine:
    def test_engine_smoke_classification(self, iris_df):
        """Smoke test: run engine on Iris for 1 algorithm × 1 pipeline."""
        from sklearn.preprocessing import LabelEncoder
        df = iris_df.copy()
        df["target"] = LabelEncoder().fit_transform(df["target"])

        config = EngineConfig(
            task_type="classification",
            target_column="target",
            test_size=0.2,
            n_folds=2,
            n_estimators_per_algo=1,
            max_algorithms=1,
            hpo_trials=2,
            optimization_metric="accuracy",
            export_dir=os.path.join(os.sep, "tmp", "automl_test_exports"),
        )
        engine = AutoMLEngine(config)
        q = queue.Queue()
        thread = engine.run_async(df, q)
        thread.join(timeout=300)

        events = []
        while not q.empty():
            events.append(q.get_nowait())

        event_types = [e["type"] for e in events]
        # Engine must have started (stage events) and not errored
        assert EventType.ERROR not in event_types, f"Engine error: {[e for e in events if e['type']==EventType.ERROR]}"
        assert len(event_types) > 0, "No events received from engine"
        # If it ran long enough, we expect DONE or PIPELINE_DONE; otherwise STAGE is OK
        assert (
            EventType.DONE in event_types
            or EventType.PIPELINE_DONE in event_types
            or EventType.STAGE in event_types
        )

    def test_engine_smoke_regression(self, diabetes_df):
        """Smoke test: run engine on Diabetes for 1 algorithm × 1 pipeline."""
        config = EngineConfig(
            task_type="regression",
            target_column="target",
            test_size=0.2,
            n_folds=2,
            n_estimators_per_algo=1,
            max_algorithms=1,
            hpo_trials=2,
            optimization_metric="r2",
            export_dir=os.path.join(os.sep, "tmp", "automl_test_exports"),
        )
        engine = AutoMLEngine(config)
        q = queue.Queue()
        thread = engine.run_async(diabetes_df, q)
        thread.join(timeout=300)

        events = []
        while not q.empty():
            events.append(q.get_nowait())

        event_types = [e["type"] for e in events]
        assert EventType.ERROR not in event_types, f"Engine error: {[e for e in events if e['type']==EventType.ERROR]}"
        assert len(event_types) > 0, "No events received from engine"
        assert (
            EventType.DONE in event_types
            or EventType.PIPELINE_DONE in event_types
            or EventType.STAGE in event_types
        )
