"""
Time Series Engine — wraps the AutoML engine loop for forecasting tasks,
using walk-forward CV, TS feature engineering, and TS-specific algorithms.
"""

import queue
import time
import threading
import traceback
import numpy as np
import pandas as pd
import optuna
import joblib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .time_series import (
    engineer_time_series_features,
    run_ts_cross_validation,
    get_ts_primary_metric,
    get_ts_algorithm_registry,
    instantiate_ts_model,
    TS_ALGORITHM_COLORS,
    TS_OPTIMIZATION_METRICS,
    generate_forecast,
)
from .pipeline_builder import build_pipeline, get_feature_importance

import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── EventType (shared with main engine) ───────────────────────────────────────
from .engine import EventType, _evt


# ── TS Engine Config ──────────────────────────────────────────────────────────

@dataclass
class TSEngineConfig:
    task_type: str = "time_series"
    target_column: str = ""
    date_column: str = ""
    horizon: int = 12
    freq: str = "M"               # M=monthly, D=daily, W=weekly, H=hourly
    n_splits: int = 5             # walk-forward CV folds
    max_algorithms: int = 4
    n_estimators_per_algo: int = 2
    hpo_trials: int = 8
    optimization_metric: str = "rmse"
    random_state: int = 42
    export_dir: str = "exports"
    algorithms_to_include: Optional[List[str]] = None


# ── Time Series AutoML Engine ──────────────────────────────────────────────────

class TimeSeriesEngine:
    """
    Orchestrates time series AutoML:
    1. Date feature extraction + lag/rolling engineering
    2. Walk-forward CV evaluation
    3. HPO with Optuna (time-aware)
    4. Forecast generation for best pipeline
    """

    BASE_STAGES = [
        "Read dataset",
        "Feature engineering",
        "Split walk-forward folds",
        "Model selection",
        "HPO & evaluation",
    ]

    def __init__(self, config: TSEngineConfig):
        self.config = config
        self.results: List[Dict] = []
        self._stop_event = threading.Event()
        self.forecast_df: Optional[pd.DataFrame] = None  # best pipeline forecast

    def run_async(self, df: pd.DataFrame, event_queue: queue.Queue) -> threading.Thread:
        t = threading.Thread(target=self._run, args=(df, event_queue), daemon=True)
        t.start()
        return t

    def stop(self):
        self._stop_event.set()

    def _emit(self, q, evt):
        q.put(evt)

    def _log(self, q, msg: str):
        self._emit(q, _evt(EventType.LOG, message=msg))

    def _run(self, df: pd.DataFrame, q: queue.Queue):
        cfg = self.config
        try:
            # Stage 1 — read dataset
            self._emit(q, _evt(EventType.STAGE, stage="Read dataset", stage_idx=0))
            self._log(q, f"📂 Time Series dataset: {df.shape[0]} rows × {df.shape[1]} columns")

            # Sort by date if date_column exists
            if cfg.date_column and cfg.date_column in df.columns:
                df = df.copy()
                df[cfg.date_column] = pd.to_datetime(df[cfg.date_column], errors="coerce")
                df = df.sort_values(cfg.date_column).reset_index(drop=True)
                self._log(q, f"📅 Sorted by {cfg.date_column} ({df[cfg.date_column].min()} → {df[cfg.date_column].max()})")
            time.sleep(0.2)

            # Stage 2 — Feature engineering
            self._emit(q, _evt(EventType.STAGE, stage="Feature engineering", stage_idx=1))
            df_feat = engineer_time_series_features(
                df,
                target_col=cfg.target_column,
                date_col=cfg.date_column if cfg.date_column in df.columns else None,
            )
            feature_cols = [c for c in df_feat.columns
                           if c != cfg.target_column and c != cfg.date_column]
            X = df_feat[feature_cols]
            y = df_feat[cfg.target_column]
            self._log(q, f"🔧 Engineered {len(feature_cols)} features from {df.shape[1]-1} original")
            self._log(q, f"📊 After lag dropping: {len(df_feat)} rows (removed {len(df)-len(df_feat)} NaN rows)")
            time.sleep(0.3)

            # Stage 3 — Walk-forward split info
            self._emit(q, _evt(EventType.STAGE, stage="Split walk-forward folds", stage_idx=2))
            fold_size = len(df_feat) // (cfg.n_splits + 1)
            self._log(q, f"🔄 Walk-forward CV: {cfg.n_splits} folds, ~{fold_size} rows/fold")
            time.sleep(0.2)

            # Stage 4 — Model selection
            self._emit(q, _evt(EventType.STAGE, stage="Model selection", stage_idx=3))
            full_registry = get_ts_algorithm_registry()
            if cfg.algorithms_to_include:
                registry = {k: v for k, v in full_registry.items() if k in cfg.algorithms_to_include}
                if not registry:
                    registry = full_registry
            else:
                registry = full_registry
                
            selected_algos = list(registry.keys())[: cfg.max_algorithms]
            self._log(q, f"🤖 TS Algorithms: {selected_algos}")
            time.sleep(0.2)

            # Stage 5 — HPO + training loop
            self._emit(q, _evt(EventType.STAGE, stage="HPO & evaluation", stage_idx=4))
            pipeline_counter = 1
            self.results = []

            for algo_name in selected_algos:
                if self._stop_event.is_set():
                    break

                for variant_idx in range(cfg.n_estimators_per_algo):
                    if self._stop_event.is_set():
                        break

                    pid = f"P{pipeline_counter}"
                    pipeline_counter += 1
                    t_start = time.time()
                    color = TS_ALGORITHM_COLORS.get(algo_name, "#8B5CF6")

                    use_scaler = variant_idx == 1  # second variant uses StandardScaler

                    self._emit(q, _evt(
                        EventType.PIPELINE_START,
                        pipeline_id=pid,
                        algorithm=algo_name,
                        transformer="StandardScaler" if use_scaler else "PassThrough",
                        color=color,
                    ))
                    self._log(q, f"▶ [{pid}] {algo_name} | Scaler={use_scaler}")

                    try:
                        result = self._train_ts_pipeline(
                            pid, algo_name, use_scaler, X, y, feature_cols, q
                        )
                        dt = round(time.time() - t_start, 1)
                        result["build_time"] = dt
                        result["status"] = "completed"
                        self.results.append(result)

                        self._emit(q, _evt(
                            EventType.PIPELINE_DONE,
                            pipeline_id=pid,
                            algorithm=algo_name,
                            result=result,
                            color=color,
                        ))
                        self._log(q, (
                            f"✅ [{pid}] Done in {dt}s | "
                            f"CV {cfg.optimization_metric.upper()}: {result['primary_metric_cv']:.4f}"
                        ))

                    except Exception as exc:
                        self._emit(q, _evt(EventType.PIPELINE_FAILED, pipeline_id=pid, error=str(exc)))
                        self._log(q, f"❌ [{pid}] Failed: {exc}")
                        continue

            # Best model forecast
            if self.results:
                self.results.sort(key=lambda r: r.get("primary_metric_cv", float("-inf")), reverse=True)
                best = self.results[0]
                os.makedirs(cfg.export_dir, exist_ok=True)
                model_path = os.path.join(cfg.export_dir, "best_ts_model.pkl")
                if best.get("_pipeline"):
                    joblib.dump(best["_pipeline"], model_path)
                self._log(q, f"🏆 Best: {best['pipeline_id']} ({best['algorithm']}) → {model_path}")

                # Generate forecast data for plotting
                try:
                    preds, lower, upper = generate_forecast(
                        best["_pipeline"], X, y, horizon=cfg.horizon
                    )
                    horizon_idx = list(range(len(y) - cfg.horizon, len(y)))
                    self.forecast_df = pd.DataFrame({
                        "index": horizon_idx,
                        "actual": y.iloc[-cfg.horizon:].values,
                        "forecast": preds,
                        "lower": lower,
                        "upper": upper,
                    })
                except Exception:
                    pass

            self._emit(q, _evt(
                EventType.DONE,
                results=[{k: v for k, v in r.items() if k != "_pipeline"} for r in self.results],
                best={k: v for k, v in self.results[0].items() if k != "_pipeline"} if self.results else None,
                forecast_df=self.forecast_df.to_dict("records") if self.forecast_df is not None else None,
            ))
            self._log(q, f"🎉 TS Experiment complete! {len(self.results)} pipelines.")

        except Exception as exc:
            self._emit(q, _evt(EventType.ERROR, error=str(exc), traceback=traceback.format_exc()))

    def _train_ts_pipeline(
        self,
        pid: str,
        algo_name: str,
        use_scaler: bool,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: List[str],
        q: queue.Queue,
    ) -> Dict:
        cfg = self.config

        # Build pipeline
        steps = []
        if use_scaler:
            steps.append(("scaler", StandardScaler()))

        # HPO
        self._log(q, f"  [{pid}] HPO walk-forward ({cfg.hpo_trials} trials)…")
        best_params, best_score = self._run_ts_hpo(pid, algo_name, use_scaler, X, y, q)
        self._emit(q, _evt(EventType.HPO_PROGRESS, pipeline_id=pid,
                           trial=cfg.hpo_trials, total=cfg.hpo_trials,
                           best_score=round(best_score, 4) if not np.isnan(best_score) else 0))

        # Final model
        model = instantiate_ts_model(algo_name, best_params)
        pipeline = build_pipeline(steps, model)
        pipeline.fit(X, y)

        # Evaluate
        cv_scores = run_ts_cross_validation(pipeline, X, y, n_splits=cfg.n_splits,
                                            optimization_metric=cfg.optimization_metric)
        primary_cv = get_ts_primary_metric(cv_scores, cfg.optimization_metric)

        # Feature importance
        fi = get_feature_importance(pipeline, feature_cols)

        result = {
            "pipeline_id": pid,
            "algorithm": algo_name,
            "transformer": "StandardScaler" if use_scaler else "PassThrough",
            "use_pca": False,
            "use_select_k": False,
            "hyperparams": best_params,
            "cv_scores": cv_scores,
            "holdout_scores": cv_scores,  # same for TS (walk-forward already covers it)
            "primary_metric_cv": primary_cv,
            "primary_metric_holdout": primary_cv,
            "build_time": 0.0,
            "feature_importance": fi.to_dict("records") if fi is not None else [],
            "_pipeline": pipeline,
        }
        return result

    def _run_ts_hpo(self, pid, algo_name, use_scaler, X, y, q):
        cfg = self.config
        best_score = float("-inf")
        best_params = {}

        from .hyperopt import get_search_space, get_default_params

        def objective(trial):
            params = get_search_space(algo_name, trial)
            if not params:
                return float("-inf")
            try:
                steps = [("scaler", StandardScaler())] if use_scaler else []
                model = instantiate_ts_model(algo_name, {**get_default_params(algo_name, "regression"), **params})
                pipeline = build_pipeline(steps, model)
                scores = run_ts_cross_validation(pipeline, X, y, n_splits=2)
                score = get_ts_primary_metric(scores, cfg.optimization_metric)
                return score if not np.isnan(score) else float("-inf")
            except Exception:
                return float("-inf")

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=cfg.random_state))
        for t_num in range(cfg.hpo_trials):
            if self._stop_event.is_set():
                break
            trial = study.ask()
            try:
                val = objective(trial)
                study.tell(trial, val)
                if val > best_score:
                    best_score = val
                    from .hyperopt import get_default_params
                    best_params = {**get_default_params(algo_name, "regression"), **trial.params}
            except Exception:
                continue

        return best_params, best_score
