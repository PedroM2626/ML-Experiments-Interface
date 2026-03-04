"""
NLP Engine — text classification via TF-IDF + classic ML algorithms.
Emits the same event types as AutoMLEngine so the UI works seamlessly.
"""

import queue
import time
import threading
import traceback
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
import optuna
import warnings

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.db.experiment_store import (
    upsert_experiment, update_experiment_status, save_pipeline_result
)


# ── NLP Algorithm registry ────────────────────────────────────────────────────

NLP_CLASSIFIERS = {
    "LogisticRegression": LogisticRegression,
    "LinearSVC":         LinearSVC,
    "ComplementNB":      ComplementNB,
    "SGDClassifier":     SGDClassifier,
    "MLP":               MLPClassifier,
    "XGBoost":           XGBClassifier,
}

NLP_ALGORITHM_COLORS = {
    "LogisticRegression": "#6366F1",
    "LinearSVC":         "#3B82F6",
    "ComplementNB":      "#F59E0B",
    "SGDClassifier":     "#10B981",
    "MLP":               "#8B5CF6",
    "XGBoost":           "#EF4444",
}

NLP_DEFAULTS = {
    "LogisticRegression": {"C": 1.0, "max_iter": 500, "random_state": 42},
    "LinearSVC":         {"C": 1.0, "max_iter": 1000},
    "ComplementNB":      {"alpha": 0.1},
    "SGDClassifier":     {"loss": "modified_huber", "max_iter": 200, "random_state": 42},
    "MLP":               {
        "hidden_layer_sizes": (256, 128),
        "activation": "relu",
        "alpha": 1e-4,
        "max_iter": 200,
        "early_stopping": True,
        "random_state": 42,
    },
    "XGBoost":           {"n_estimators": 100, "learning_rate": 0.1, "verbosity": 0, "random_state": 42},
}


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class NLPConfig:
    text_column: str = "text"
    target_column: str = "label"
    max_features: int = 20_000
    ngram_range: tuple = (1, 2)
    sublinear_tf: bool = True
    max_algorithms: int = 5
    hpo_trials: int = 8
    optimization_metric: str = "f1"
    random_state: int = 42
    export_dir: str = "exports"
    algorithms_to_include: Optional[List[str]] = None
    # Filled by engine / experiment manager
    experiment_id: str = ""
    name: str = "NLP Experiment"
    dataset_name: str = "dataset"


# ── Event types (same as AutoMLEngine) ────────────────────────────────────────

class EventType:
    STAGE           = "stage"
    PIPELINE_START  = "pipeline_start"
    HPO_PROGRESS    = "hpo_progress"
    PIPELINE_DONE   = "pipeline_done"
    PIPELINE_FAILED = "pipeline_failed"
    LOG             = "log"
    DONE            = "done"
    ERROR           = "error"


def _evt(etype: str, **kwargs) -> Dict:
    return {"type": etype, "ts": time.time(), **kwargs}


# ── NLP Engine ────────────────────────────────────────────────────────────────

class NLPEngine:
    """
    Orchestrates TF-IDF + classifier AutoML for text classification.
    Emits the same events as AutoMLEngine for seamless UI integration.
    """

    STAGES = [
        "Load text data",
        "Split holdout",
        "TF-IDF Vectorization",
        "Model selection",
    ]

    def __init__(self, config: NLPConfig):
        self.config = config
        self.results: List[Dict] = []
        self._stop_event = threading.Event()

    def run_async(self, df: pd.DataFrame, event_queue: queue.Queue) -> threading.Thread:
        t = threading.Thread(target=self._run, args=(df, event_queue), daemon=True)
        t.start()
        return t

    def stop(self):
        self._stop_event.set()

    def _emit(self, q: queue.Queue, evt: Dict):
        q.put(evt)

    def _log(self, q: queue.Queue, msg: str):
        self._emit(q, _evt(EventType.LOG, message=msg))

    def _run(self, df: pd.DataFrame, q: queue.Queue):
        cfg = self.config
        exp_id = cfg.experiment_id or str(int(time.time()))

        try:
            # Init DB
            exp_data = {
                "id": exp_id,
                "name": cfg.name,
                "status": "running",
                "task_type": "text_classification",
                "dataset_name": cfg.dataset_name,
                "target_column": cfg.target_column,
                "optimization_metric": cfg.optimization_metric,
                "config": cfg,
            }
            upsert_experiment(exp_data)

            # Stage 1 — load
            self._emit(q, _evt(EventType.STAGE, stage="Load text data", stage_idx=0))
            self._log(q, f"📂 Text dataset: {df.shape[0]} rows × {df.shape[1]} cols")
            time.sleep(0.2)

            texts = df[cfg.text_column].fillna("").astype(str)
            labels = df[cfg.target_column]
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(labels), name=labels.name)
            self._log(q, f"🏷️  Classes: {list(le.classes_)} ({len(le.classes_)} total)")

            # Stage 2 — split
            self._emit(q, _evt(EventType.STAGE, stage="Split holdout", stage_idx=1))
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    texts, y, test_size=0.15, random_state=cfg.random_state, stratify=y
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    texts, y, test_size=0.15, random_state=cfg.random_state
                )
            self._log(q, f"✂️  Train: {len(X_train)} | Holdout: {len(X_test)}")
            time.sleep(0.2)

            # Stage 3 — TF-IDF
            self._emit(q, _evt(EventType.STAGE, stage="TF-IDF Vectorization", stage_idx=2))
            tfidf = TfidfVectorizer(
                max_features=cfg.max_features,
                ngram_range=cfg.ngram_range,
                sublinear_tf=cfg.sublinear_tf,
                strip_accents="unicode",
                analyzer="word",
                token_pattern=r"\w{1,}",
            )
            X_train_vec = tfidf.fit_transform(X_train)
            X_test_vec = tfidf.transform(X_test)
            self._log(q, f"📝 TF-IDF vocabulary: {len(tfidf.vocabulary_):,} terms | n-grams: {cfg.ngram_range}")
            time.sleep(0.2)

            # Stage 4 — model selection
            self._emit(q, _evt(EventType.STAGE, stage="Model selection", stage_idx=3))
            registry = NLP_CLASSIFIERS.copy()
            if cfg.algorithms_to_include:
                registry = {k: v for k, v in registry.items() if k in cfg.algorithms_to_include}
            selected = list(registry.keys())[:cfg.max_algorithms]
            self._log(q, f"🤖 Algorithms: {selected}")
            time.sleep(0.2)

            # ── Training loop ─────────────────────────────────────────────────
            pipeline_counter = 1
            self.results = []

            for algo_name in selected:
                if self._stop_event.is_set():
                    break

                pid = f"P{pipeline_counter}"
                pipeline_counter += 1
                t_start = time.time()

                self._emit(q, _evt(
                    EventType.PIPELINE_START,
                    pipeline_id=pid,
                    algorithm=algo_name,
                    transformer="TF-IDF",
                    color=NLP_ALGORITHM_COLORS.get(algo_name, "#8B5CF6"),
                ))
                self._log(q, f"▶ [{pid}] {algo_name}")

                try:
                    result = self._train_pipeline(
                        pid, algo_name, X_train_vec, y_train, X_test_vec, y_test, tfidf, q
                    )
                    result["build_time"] = round(time.time() - t_start, 1)
                    self.results.append(result)

                    self._emit(q, _evt(
                        EventType.PIPELINE_DONE,
                        pipeline_id=pid,
                        algorithm=algo_name,
                        result=result,
                        color=NLP_ALGORITHM_COLORS.get(algo_name, "#8B5CF6"),
                    ))
                    self._log(q, (
                        f"✅ [{pid}] Done in {result['build_time']}s | "
                        f"F1: {result['primary_metric_cv']:.4f}"
                    ))
                    save_pipeline_result(exp_id, result)
                    update_experiment_status(exp_id, "running", n_pipelines_done=len(self.results))

                except Exception as exc:
                    self._emit(q, _evt(EventType.PIPELINE_FAILED, pipeline_id=pid, error=str(exc)))
                    self._log(q, f"❌ [{pid}] Failed: {exc}")
                    continue

            # ── Export best model ─────────────────────────────────────────────
            if self.results:
                best = max(self.results, key=lambda r: r.get("primary_metric_cv", 0))
                import os, joblib
                os.makedirs(os.path.join(cfg.export_dir), exist_ok=True)
                model_path = os.path.join(cfg.export_dir, "best_model.pkl")
                pipeline_obj = best.get("_pipeline_obj")
                if pipeline_obj:
                    joblib.dump({"tfidf": tfidf, "classifier": pipeline_obj, "classes": le.classes_}, model_path)
                    self._log(q, f"🏆 Best: {best['algorithm']} | F1={best['primary_metric_cv']:.4f}")

            update_experiment_status(
                exp_id, "completed",
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                best_result=max(self.results, key=lambda r: r.get("primary_metric_cv", 0)) if self.results else None
            )

            self._emit(q, _evt(
                EventType.DONE,
                results=self.results,
                best=max(self.results, key=lambda r: r.get("primary_metric_cv", 0)) if self.results else None,
            ))
            self._log(q, f"🎉 NLP experiment complete! {len(self.results)} pipelines evaluated.")

        except Exception as exc:
            update_experiment_status(exp_id, "failed", error=str(exc))
            self._emit(q, _evt(EventType.ERROR, error=str(exc), traceback=traceback.format_exc()))
            self._log(q, f"💥 NLP Engine error: {exc}")

    def _train_pipeline(
        self, pid, algo_name, X_train, y_train, X_test, y_test, tfidf, q
    ) -> Dict:
        cfg = self.config
        defaults = NLP_DEFAULTS.get(algo_name, {}).copy()

        cls = NLP_CLASSIFIERS[algo_name]

        # LinearSVC needs CalibratedClassifierCV for predict_proba
        if algo_name == "LinearSVC":
            base = cls(**defaults)
            model = CalibratedClassifierCV(base, cv=3)
        else:
            try:
                model = cls(**defaults)
            except TypeError:
                model = cls()

        # HPO (lightweight — 2-fold CV for speed)
        best_params = defaults.copy()
        best_score = -np.inf

        if algo_name in ("LogisticRegression", "ComplementNB", "SGDClassifier"):
            def objective(trial):
                if algo_name == "LogisticRegression":
                    params = {"C": trial.suggest_float("C", 0.01, 100.0, log=True)}
                elif algo_name == "ComplementNB":
                    params = {"alpha": trial.suggest_float("alpha", 1e-3, 5.0, log=True)}
                else:
                    params = {"alpha": trial.suggest_float("alpha", 1e-5, 1.0, log=True)}
                try:
                    m = cls(**{**defaults, **params})
                    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
                    sc = cross_validate(m, X_train, y_train, cv=cv, scoring="f1_weighted")
                    return float(np.nanmean(sc["test_score"]))
                except Exception:
                    return -np.inf

            study = optuna.create_study(direction="maximize")
            for i in range(cfg.hpo_trials):
                if self._stop_event.is_set():
                    break
                trial = study.ask()
                try:
                    val = objective(trial)
                    study.tell(trial, val)
                    if val > best_score:
                        best_score = val
                        best_params = {**defaults, **trial.params}
                except Exception:
                    continue

            self._emit(q, _evt(EventType.HPO_PROGRESS, pipeline_id=pid,
                                trial=cfg.hpo_trials, total=cfg.hpo_trials,
                                best_score=round(best_score, 4)))

        # Final model
        if algo_name == "LinearSVC":
            final_base = cls(**{k: v for k, v in best_params.items()
                                if k in ("C", "max_iter")})
            final_model = CalibratedClassifierCV(final_base, cv=3)
        else:
            try:
                final_model = cls(**best_params)
            except TypeError:
                final_model = cls()

        final_model.fit(X_train, y_train)

        # CV metrics
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_sc = cross_validate(final_model, X_train, y_train, cv=cv, scoring="f1_weighted",
                               error_score=np.nan)
        f1_cv = float(np.nanmean(cv_sc["test_score"]))

        # Holdout
        y_pred = final_model.predict(X_test)
        f1_hd  = round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 6)
        acc_hd = round(accuracy_score(y_test, y_pred), 6)
        prec   = round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 6)
        rec    = round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 6)

        result = {
            "pipeline_id": pid,
            "algorithm": algo_name,
            "transformer": f"TF-IDF({cfg.max_features})",
            "use_pca": False,
            "use_select_k": False,
            "hyperparams": best_params,
            "cv_scores": {"f1": round(f1_cv, 6)},
            "holdout_scores": {"f1": f1_hd, "accuracy": acc_hd, "precision": prec, "recall": rec},
            "primary_metric_cv": round(f1_cv, 6),
            "primary_metric_holdout": f1_hd,
            "build_time": 0.0,
            "feature_importance": [],
            "calibration_data": None,
            "learning_curve_data": None,
            "residuals_data": None,
            "status": "completed",
            "_pipeline_obj": final_model,   # not serialized to JSON
        }
        return result


def get_nlp_algorithm_registry() -> Dict:
    return NLP_CLASSIFIERS.copy()
