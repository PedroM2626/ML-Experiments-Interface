"""
AutoML Engine — main orchestrator that runs all pipelines in sequence,
emitting real-time updates via a queue for the Streamlit UI.
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
from typing import Dict, List, Optional, Any, Callable
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .feature_engineering import (
    detect_column_types,
    choose_transformers_for_data,
    build_feature_pipeline,
    NUMERIC_TRANSFORMERS,
)
from .hyperopt import (
    get_algorithm_registry,
    get_search_space,
    get_default_params,
    instantiate_model,
    ALGORITHM_COLORS,
)
from .evaluator import (
    run_cross_validation,
    evaluate_holdout,
    get_primary_metric,
    OPTIMIZATION_METRICS,
)
from .pipeline_builder import build_pipeline, get_feature_names_after_preprocessor, get_feature_importance
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    pipeline_id: str          # e.g. "P1"
    algorithm: str
    transformer: str
    use_pca: bool
    use_select_k: bool
    hyperparams: Dict
    cv_scores: Dict[str, float]
    holdout_scores: Dict[str, float]
    primary_metric_cv: float
    primary_metric_holdout: float
    build_time: float         # seconds
    pipeline: Any = field(default=None, repr=False)
    feature_importance: Optional[pd.DataFrame] = None
    status: str = "completed"  # completed | failed | running


@dataclass
class EngineConfig:
    task_type: str = "classification"
    target_column: str = ""
    test_size: float = 0.1
    n_folds: int = 3
    n_estimators_per_algo: int = 2     # how many pipelines per algo
    max_algorithms: int = 5
    hpo_trials: int = 10
    optimization_metric: str = "roc_auc"   # key in evaluator metrics
    random_state: int = 42
    export_dir: str = "exports"


# ── Status event types ─────────────────────────────────────────────────────────

class EventType:
    STAGE = "stage"
    PIPELINE_START = "pipeline_start"
    HPO_PROGRESS = "hpo_progress"
    PIPELINE_DONE = "pipeline_done"
    PIPELINE_FAILED = "pipeline_failed"
    LOG = "log"
    DONE = "done"
    ERROR = "error"


def _evt(etype: str, **kwargs) -> Dict:
    return {"type": etype, "ts": time.time(), **kwargs}


# ── Main engine ────────────────────────────────────────────────────────────────

class AutoMLEngine:
    """
    Orchestrates the full AutoML training loop.

    Usage:
        engine = AutoMLEngine(config)
        q = queue.Queue()
        thread = engine.run_async(df, q)
        # consume q in main thread for real-time updates
    """

    BASE_STAGES = [
        "Read dataset",
        "Split holdout data",
        "Read training data",
        "Preprocessing",
        "Model selection",
    ]

    def __init__(self, config: EngineConfig):
        self.config = config
        self.results: List[PipelineResult] = []
        self._stop_event = threading.Event()

    # ── Public API ─────────────────────────────────────────────────────────────

    def run_async(self, df: pd.DataFrame, event_queue: queue.Queue) -> threading.Thread:
        """Start training in a background thread, publishing events to the queue."""
        t = threading.Thread(target=self._run, args=(df, event_queue), daemon=True)
        t.start()
        return t

    def stop(self):
        self._stop_event.set()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _emit(self, q: queue.Queue, evt: Dict):
        q.put(evt)

    def _log(self, q: queue.Queue, msg: str):
        self._emit(q, _evt(EventType.LOG, message=msg))

    def _run(self, df: pd.DataFrame, q: queue.Queue):
        cfg = self.config
        try:
            # Stage 1 — read dataset
            self._emit(q, _evt(EventType.STAGE, stage="Read dataset", stage_idx=0))
            self._log(q, f"📂 Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
            time.sleep(0.3)

            # Stage 2 — split holdout
            self._emit(q, _evt(EventType.STAGE, stage="Split holdout data", stage_idx=1))
            X = df.drop(columns=[cfg.target_column])
            y = df[cfg.target_column]
            # Encode target if classification and not numeric
            if cfg.task_type == "classification" and y.dtype == object:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y), name=y.name)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=cfg.test_size,
                random_state=cfg.random_state,
                stratify=y if cfg.task_type == "classification" else None,
            )
            self._log(q, f"✂️  Train: {len(X_train)} | Holdout: {len(X_test)} (stratified={cfg.task_type=='classification'})")
            time.sleep(0.3)

            # Stage 3 — read training data
            self._emit(q, _evt(EventType.STAGE, stage="Read training data", stage_idx=2))
            self._log(q, f"📊 Features: {X_train.shape[1]} | Target: {cfg.target_column}")
            time.sleep(0.3)

            # Stage 4 — preprocessing analysis
            self._emit(q, _evt(EventType.STAGE, stage="Preprocessing", stage_idx=3))
            # X_train already has target removed — detect types directly
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
            chosen_transformers = choose_transformers_for_data(X_train, numeric_cols)
            self._log(q, f"🔧 Numeric cols: {len(numeric_cols)} | Categorical: {len(categorical_cols)}")
            self._log(q, f"🔧 Auto-selected transformers: {chosen_transformers}")
            time.sleep(0.4)

            # Stage 5 — model selection
            self._emit(q, _evt(EventType.STAGE, stage="Model selection", stage_idx=4))
            registry = get_algorithm_registry(cfg.task_type)
            all_algos = list(registry.keys())
            # Pick top N algorithms based on config
            selected_algos = all_algos[: cfg.max_algorithms]
            self._log(q, f"🤖 Algorithms selected: {selected_algos}")
            time.sleep(0.3)

            # ── Pipeline training loop ──────────────────────────────────────
            pipeline_counter = 1
            self.results = []

            for algo_name in selected_algos:
                if self._stop_event.is_set():
                    break

                # Each algorithm spawns N pipelines (with different transformers/FE)
                fe_variants = self._get_fe_variants(chosen_transformers)

                for variant_idx, fe_cfg in enumerate(fe_variants[: cfg.n_estimators_per_algo]):
                    if self._stop_event.is_set():
                        break

                    pid = f"P{pipeline_counter}"
                    pipeline_counter += 1
                    t_start = time.time()

                    self._emit(q, _evt(
                        EventType.PIPELINE_START,
                        pipeline_id=pid,
                        algorithm=algo_name,
                        transformer=fe_cfg["transformer"],
                        color=ALGORITHM_COLORS.get(algo_name, "#8B5CF6"),
                    ))
                    self._log(q, f"▶ [{pid}] Starting {algo_name} | FE: {fe_cfg['transformer']}")

                    try:
                        result = self._train_pipeline(
                            pid, algo_name, fe_cfg,
                            X_train, y_train, X_test, y_test,
                            numeric_cols, categorical_cols, q
                        )
                        dt = round(time.time() - t_start, 1)
                        result.build_time = dt
                        self.results.append(result)

                        self._emit(q, _evt(
                            EventType.PIPELINE_DONE,
                            pipeline_id=pid,
                            algorithm=algo_name,
                            result=self._result_to_dict(result),
                            color=ALGORITHM_COLORS.get(algo_name, "#8B5CF6"),
                        ))
                        self._log(q, (
                            f"✅ [{pid}] Done in {dt}s | "
                            f"CV {cfg.optimization_metric}: {result.primary_metric_cv:.4f} | "
                            f"Holdout: {result.primary_metric_holdout:.4f}"
                        ))

                    except Exception as exc:
                        self._emit(q, _evt(EventType.PIPELINE_FAILED, pipeline_id=pid, error=str(exc)))
                        self._log(q, f"❌ [{pid}] Failed: {exc}")
                        continue

            # ── Sort results & export best model ──────────────────────────
            if self.results:
                self.results.sort(key=lambda r: r.primary_metric_cv, reverse=True)
                best = self.results[0]
                os.makedirs(cfg.export_dir, exist_ok=True)
                model_path = os.path.join(cfg.export_dir, "best_model.pkl")
                joblib.dump(best.pipeline, model_path)
                self._log(q, f"🏆 Best pipeline: {best.pipeline_id} ({best.algorithm}) — saving to {model_path}")

            self._emit(q, _evt(
                EventType.DONE,
                results=[self._result_to_dict(r) for r in self.results],
                best=self._result_to_dict(self.results[0]) if self.results else None,
            ))
            self._log(q, f"🎉 Experiment complete! {len(self.results)} pipelines generated.")

        except Exception as exc:
            self._emit(q, _evt(EventType.ERROR, error=str(exc), traceback=traceback.format_exc()))
            self._log(q, f"💥 Engine error: {exc}")

    def _get_fe_variants(self, transformers: List[str]) -> List[Dict]:
        """Return list of FE config dicts to generate variant pipelines."""
        variants = []
        for t in transformers:
            variants.append({"transformer": t, "use_pca": False, "use_select_k": False})
        # Add a PCA variant
        variants.append({"transformer": transformers[0], "use_pca": True, "use_select_k": False})
        return variants

    def _train_pipeline(
        self,
        pid: str,
        algo_name: str,
        fe_cfg: Dict,
        X_train, y_train,
        X_test, y_test,
        numeric_cols, categorical_cols,
        q: queue.Queue,
    ) -> PipelineResult:
        cfg = self.config

        # Phase A: baseline with default params
        fe_steps = build_feature_pipeline(
            numeric_cols, categorical_cols,
            fe_cfg["transformer"],
            use_pca=fe_cfg.get("use_pca", False),
            use_select_k=fe_cfg.get("use_select_k", False),
            task_type=cfg.task_type,
        )
        default_params = get_default_params(algo_name, cfg.task_type)
        base_model = instantiate_model(algo_name, cfg.task_type, default_params)
        base_pipeline = build_pipeline(fe_steps, base_model)

        self._log(q, f"  [{pid}] Training baseline {algo_name}...")
        base_pipeline.fit(X_train, y_train)

        # Phase B: HPO with Optuna
        self._log(q, f"  [{pid}] Running HPO ({cfg.hpo_trials} trials)...")
        best_params, best_score = self._run_hpo(
            pid, algo_name, fe_steps, X_train, y_train, q
        )

        # Phase C: final pipeline with best params
        self._log(q, f"  [{pid}] Training final pipeline with best params...")
        final_model = instantiate_model(algo_name, cfg.task_type, best_params)
        final_pipeline = build_pipeline(fe_steps, final_model)
        final_pipeline.fit(X_train, y_train)

        # Phase D: evaluate
        self._log(q, f"  [{pid}] Running {cfg.n_folds}-fold cross-validation...")
        cv_scores = run_cross_validation(final_pipeline, X_train, y_train, cfg.task_type, cfg.n_folds)
        holdout_scores = evaluate_holdout(final_pipeline, X_test, y_test, cfg.task_type)

        primary_cv = get_primary_metric(cv_scores, cfg.task_type, cfg.optimization_metric)
        primary_hd = get_primary_metric(holdout_scores, cfg.task_type, cfg.optimization_metric)

        # Feature importance
        feature_names = get_feature_names_after_preprocessor(final_pipeline, numeric_cols, categorical_cols)
        fi = get_feature_importance(final_pipeline, feature_names)

        return PipelineResult(
            pipeline_id=pid,
            algorithm=algo_name,
            transformer=fe_cfg["transformer"],
            use_pca=fe_cfg.get("use_pca", False),
            use_select_k=fe_cfg.get("use_select_k", False),
            hyperparams=best_params,
            cv_scores=cv_scores,
            holdout_scores=holdout_scores,
            primary_metric_cv=primary_cv,
            primary_metric_holdout=primary_hd,
            build_time=0.0,
            pipeline=final_pipeline,
            feature_importance=fi,
        )

    def _run_hpo(
        self,
        pid: str,
        algo_name: str,
        fe_steps: List,
        X_train, y_train,
        q: queue.Queue,
    ):
        cfg = self.config
        best_score = -np.inf
        best_params = get_default_params(algo_name, cfg.task_type)

        def objective(trial: optuna.Trial) -> float:
            params = get_search_space(algo_name, trial)
            if not params:
                return -np.inf
            try:
                full_params = {**get_default_params(algo_name, cfg.task_type), **params}
                model = instantiate_model(algo_name, cfg.task_type, full_params)
                pipeline = build_pipeline(fe_steps, model)
                cv_scores = run_cross_validation(pipeline, X_train, y_train, cfg.task_type, n_folds=2)
                score = get_primary_metric(cv_scores, cfg.task_type, cfg.optimization_metric)
                return score if not np.isnan(score) else -np.inf
            except Exception:
                return -np.inf

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

        for t_num in range(cfg.hpo_trials):
            if self._stop_event.is_set():
                break
            trial = study.ask()
            try:
                value = objective(trial)
                study.tell(trial, value)
                if value > best_score:
                    best_score = value
                    best_params = {**get_default_params(algo_name, cfg.task_type), **trial.params}
                if (t_num + 1) % max(1, cfg.hpo_trials // 3) == 0:
                    self._emit(q, _evt(
                        EventType.HPO_PROGRESS,
                        pipeline_id=pid,
                        trial=t_num + 1,
                        total=cfg.hpo_trials,
                        best_score=round(best_score, 4),
                    ))
            except Exception:
                continue

        return best_params, best_score

    @staticmethod
    def _result_to_dict(r: Optional[PipelineResult]) -> Optional[Dict]:
        if r is None:
            return None
        return {
            "pipeline_id": r.pipeline_id,
            "algorithm": r.algorithm,
            "transformer": r.transformer,
            "use_pca": r.use_pca,
            "use_select_k": r.use_select_k,
            "hyperparams": r.hyperparams,
            "cv_scores": r.cv_scores,
            "holdout_scores": r.holdout_scores,
            "primary_metric_cv": r.primary_metric_cv,
            "primary_metric_holdout": r.primary_metric_holdout,
            "build_time": r.build_time,
            "feature_importance": r.feature_importance.to_dict("records") if r.feature_importance is not None else [],
            "status": r.status,
        }
