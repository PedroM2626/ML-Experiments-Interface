import os
import re
import time
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional, Any

# MLOps
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix

# Models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict, StratifiedKFold



import random

warnings.filterwarnings("ignore")

try:
    import mlflow  # type: ignore
except Exception:
    class _DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    class _DummyMlflow:
        def set_tracking_uri(self, *_args, **_kwargs):
            return None

        def set_experiment(self, *_args, **_kwargs):
            return None

        def start_run(self, *_args, **_kwargs):
            return _DummyRun()

        def log_params(self, *_args, **_kwargs):
            return None

        def log_param(self, *_args, **_kwargs):
            return None

        def log_artifact(self, *_args, **_kwargs):
            return None

        def log_metric(self, *_args, **_kwargs):
            return None

    mlflow = _DummyMlflow()

try:
    import dagshub  # type: ignore
except Exception:
    dagshub = None

def set_seed(seed):
     """Ensures 100% reproducibility across runs."""
     random.seed(seed)
     np.random.seed(seed)
     os.environ['PYTHONHASHSEED'] = str(seed)
     # Note: Sklearn models use np.random but some underlying libs might need more



# ─── Configuration ─────────────────────────────────────────────────────────────
load_dotenv()

# Parameters (Default values, can be overridden)
SEED = 2007
NUM_LAYERS = 12  
CV_FOLDS = 3
EXP_NAME = "Ensemble_Pyramid_Flexible"
PATIENCE = 3  # Layers without improvement before stopping


# Dynamic Variation Parameters
MIN_MODELS_PER_LAYER = 2
MAX_MODELS_PER_LAYER = 6 # Max models to pick per layer
EPSILON_RL = 0.2         # RL exploration rate
OPTIM_METRIC = "f1"      # Metric to optimize: f1, accuracy
TFIDF_MAX = 50000
TFIDF_NGRAMS = (1, 2)
JITTER = True            # Random hyperparameter variation
STRATEGY = "dense"       # "dense", "residual", "simple"
PYRAMID_LAYER_TYPE = "heterogeneous"  # "heterogeneous", "homogeneous"
HOMOGENEOUS_MODEL = "lr"
HOMOGENEOUS_VARIANTS = 4



# ─── Setup Setup Tracking ─────────────────────────────────────────────────────
def setup_tracking():
    use_dagshub = (os.getenv("DAGSHUB_TOKEN") is not None or os.getenv("USE_DAGSHUB") == "True") and dagshub is not None
    
    if use_dagshub:
        try:
            print("[INFO] Initializing Dagshub...", flush=True)
            dagshub.init(repo_owner="PedroM2626", repo_name="experiments")
            mlflow.set_tracking_uri("https://dagshub.com/PedroM2626/experiments.mlflow")
            print("[INFO] Dagshub initialized.", flush=True)
        except Exception as e:
            print(f"[ERROR] Could not initialize Dagshub: {e}", flush=True)
            mlflow.set_tracking_uri("file:./mlruns")
    else:
        print("[INFO] Using local file-based MLflow tracking (file:./mlruns)", flush=True)
        mlflow.set_tracking_uri("file:./mlruns")
    
    try:
        print(f"[INFO] Setting experiment: {EXP_NAME}", flush=True)
        mlflow.set_experiment(EXP_NAME)
        print(f"[INFO] Experiment set to: {EXP_NAME}", flush=True)
    except Exception as e:
        print(f"[ERROR] Could not set experiment: {e}", flush=True)

# ─── Data Preparation ──────────────────────────────────────────────────────────
def clean_tweet(text: str) -> str:
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-z0-9\s!?.,\'\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_data(subsample_train=0, subsample_val=0):
    base_path = Path("experiments/senti-pred-variations/logistic-senti-pred/data/raw")
    cols = ['tweet_id', 'entity', 'sentiment', 'text']
    train_file = base_path / "twitter_training.csv"
    val_file = base_path / "twitter_validation.csv"
    
    if not train_file.exists():
        base_path = Path(__file__).parent / "senti-pred-variations/logistic-senti-pred/data/raw"
        train_file = base_path / "twitter_training.csv"
        val_file = base_path / "twitter_validation.csv"

    train_df = pd.read_csv(train_file, names=cols, header=None)
    val_df = pd.read_csv(val_file, names=cols, header=None)
    valid_sentiments = ['Positive', 'Negative', 'Neutral', 'Irrelevant']
    
    # Pre-cleaning to ensure valid samples
    for df in (train_df, val_df):
        df['clean_text'] = df['text'].apply(clean_tweet)
        
    train_df = train_df[(train_df['clean_text'].str.len() > 0) & (train_df['sentiment'].isin(valid_sentiments))]
    val_df = val_df[(val_df['clean_text'].str.len() > 0) & (val_df['sentiment'].isin(valid_sentiments))]

    if subsample_train > 0 and subsample_train < len(train_df):
        train_df = train_df.sample(n=subsample_train, random_state=SEED)
    if subsample_val > 0 and subsample_val < len(val_df):
        val_df = val_df.sample(n=subsample_val, random_state=SEED)
        
    return train_df, val_df

# ─── Model Factories ─────────────────────────────────────────────────────────
def get_model(model_type, seed=SEED, jitter=False, n_jobs=2, variant_scale=1.0):
    # Bagging Wrapper
    is_bagging = model_type.startswith("bag_")
    base_type = model_type.replace("bag_", "") if is_bagging else model_type

    # Hyperparameter Jitter (Variability)
    j_c = np.random.uniform(0.5, 20.0) if jitter else 1.0
    j_tree = np.random.randint(50, 200) if jitter else 100
    j_alpha = np.random.uniform(0.01, 1.0) if jitter else 1.0

    if base_type == "lr":
        m = LogisticRegression(C=max(0.01, (11.0 * j_c if jitter else 11.0) * variant_scale), max_iter=1000, random_state=seed, n_jobs=n_jobs)
    elif base_type == "svc":
        m = CalibratedClassifierCV(LinearSVC(C=max(0.01, (19.0 * j_c if jitter else 19.0) * variant_scale), random_state=seed), cv=2, n_jobs=n_jobs)
    elif base_type == "nb":
        m = MultinomialNB(alpha=max(1e-4, (0.1 * j_alpha if jitter else 0.1) * variant_scale))
    elif base_type == "rf":
        m = RandomForestClassifier(n_estimators=max(20, int(j_tree * variant_scale)), random_state=seed, n_jobs=n_jobs)
    elif base_type == "et":
        m = ExtraTreesClassifier(n_estimators=max(20, int(j_tree * variant_scale)), random_state=seed, n_jobs=n_jobs)
    elif base_type == "ridge":
        m = CalibratedClassifierCV(RidgeClassifier(alpha=max(0.01, (1.0 * j_alpha if jitter else 1.0) * variant_scale)), cv=2, n_jobs=n_jobs)
    else:
        m = LogisticRegression(random_state=seed, n_jobs=n_jobs)
    
    if is_bagging:
        return BaggingClassifier(m, n_estimators=10, random_state=seed, n_jobs=n_jobs)
    return m


def _to_jsonable(v):
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, (np.ndarray,)):
        return v.tolist()
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _extract_model_hyperparams(model):
    if hasattr(model, "get_params"):
        try:
            params = model.get_params(deep=False)
            allowed = {
                "C", "alpha", "max_iter", "n_estimators", "random_state", "n_jobs",
                "cv", "penalty", "solver", "fit_intercept", "class_weight",
                "max_depth", "min_samples_split", "min_samples_leaf"
            }
            out = {}
            for k, val in params.items():
                if k in allowed:
                    out[k] = _to_jsonable(val)
            return out
        except Exception:
            return {}

    keys = ["n_models", "n_classes", "n_bags", "bag_fraction", "seed"]
    out = {}
    for k in keys:
        if hasattr(model, k):
            out[k] = _to_jsonable(getattr(model, k))
    return out



# ─── RL Meta-Learner ─────────────────────────────────────────────────────────
class RLMetaLearner:
    """
    Reinforcement Learning agent to optimize model selection across runs.
    Uses an epsilon-greedy approach with persistent memory.
    Enhanced with NAS integration for architecture optimization.
    """
    def __init__(self, knowledge_path="pyramid_rl_knowledge.json", epsilon=0.2, metric="f1", nas_controller=None):
        self.path = Path(knowledge_path)
        self.epsilon = epsilon
        self.metric = metric
        self.knowledge = self._load_knowledge()
        self.nas_controller = nas_controller

    def _load_knowledge(self):
        if self.path.exists():
            try:
                with open(self.path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"runs": 0, "layer_stats": {}}


    def _save_knowledge(self):
        with open(self.path, 'w') as f:
            json.dump(self.knowledge, f, indent=4)

    def suggest_models(self, layer_idx, available_models, min_p=MIN_MODELS_PER_LAYER, max_p=MAX_MODELS_PER_LAYER):
        """Suggests models using RL + NAS hybrid approach."""
        l_str = str(layer_idx)
        
        # Try NAS first if available
        if self.nas_controller and np.random.random() < 0.7:  # 70% NAS preference
            nas_models = self.nas_controller.suggest_models_for_layer(layer_idx, available_models)
            if nas_models and min_p <= len(nas_models) <= max_p:
                print(f"  [NAS] Selected set ({len(nas_models)} models): {nas_models}", flush=True)
                return nas_models
        
        # Fallback to RL epsilon-greedy
        n_to_pick = np.random.randint(min_p, min(max_p, len(available_models)) + 1)
        
        # Explore: Epsilon-greedy random selection
        if np.random.random() < self.epsilon or l_str not in self.knowledge["layer_stats"]:
            choices = np.random.choice(available_models, n_to_pick, replace=False)
            return [str(m) for m in choices] 
        
        stats = self.knowledge["layer_stats"][l_str]
        # Exploit: Pick top-N performers with a bit of noise (Thompson Sampling-lite)
        m_metric = f"avg_{self.metric}"
        sorted_models = sorted(
            [m for m in available_models if m in stats],
            key=lambda x: (stats[x].get(m_metric, 0) + np.random.normal(0, 0.01)), 
            reverse=True
        )

        
        # Take the top N, but ensure we return exactly n_to_pick if possible
        final_list = [str(m) for m in sorted_models[:n_to_pick]]
        
        # Fallback if too few models were known
        if len(final_list) < n_to_pick:
            remaining = [m for m in available_models if m not in final_list]
            extra = np.random.choice(remaining, n_to_pick - len(final_list), replace=False)
            final_list.extend([str(m) for m in extra])
            
        return final_list


    def update_knowledge(self, results):
        self.knowledge["runs"] += 1
        for res in results:
            l_str = str(res["layer"])
            m_base = re.sub(r"_L\d+$", "", res["model"])
            m_base = re.sub(r"__v\d+$", "", m_base)
            val_f1 = res["f1"]
            val_acc = res["accuracy"]
            val_prec = res.get("precision", 0.0)
            val_rec = res.get("recall", 0.0)
            
            if l_str not in self.knowledge["layer_stats"]:
                self.knowledge["layer_stats"][l_str] = {}
            if m_base not in self.knowledge["layer_stats"][l_str]:
                self.knowledge["layer_stats"][l_str][m_base] = {
                    "count": 0, 
                    "avg_f1": 0.0, 
                    "avg_accuracy": 0.0,
                    "avg_precision": 0.0,
                    "avg_recall": 0.0
                }
            
            s = self.knowledge["layer_stats"][l_str][m_base]
            s.setdefault("count", 0)
            s.setdefault("avg_f1", 0.0)
            s.setdefault("avg_accuracy", 0.0)
            s.setdefault("avg_precision", 0.0)
            s.setdefault("avg_recall", 0.0)
            s["count"] += 1
            # Update all metrics
            s["avg_f1"] = s["avg_f1"] + (val_f1 - s["avg_f1"]) / s["count"]
            s["avg_accuracy"] = s["avg_accuracy"] + (val_acc - s["avg_accuracy"]) / s["count"]
            s["avg_precision"] = s["avg_precision"] + (val_prec - s["avg_precision"]) / s["count"]
            s["avg_recall"] = s["avg_recall"] + (val_rec - s["avg_recall"]) / s["count"]

        self._save_knowledge()
        print(f"[RL] Meta-Knowledge updated. Total runs: {self.knowledge['runs']}", flush=True)


# ─── Neural Architecture Search (NAS) Controller ────────────────────────────────
class NASController:
    """
    Neural Architecture Search controller for optimizing ensemble architectures.
    Uses evolutionary strategies and Bayesian optimization principles.
    """
    def __init__(self, population_size=10, generations=5, mutation_rate=0.1, 
                 crossover_rate=0.7, knowledge_path="nas_knowledge.json", metric="f1"):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.metric = metric
        self.path = Path(knowledge_path)
        self.knowledge = self._load_knowledge()
        self.current_generation = 0
        self.population = []
        
    def _load_knowledge(self):
        if self.path.exists():
            try:
                with open(self.path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"generations": 0, "architectures": [], "best_architecture": None}
    
    def _save_knowledge(self):
        with open(self.path, 'w') as f:
            json.dump(self.knowledge, f, indent=4)
    
    def _generate_random_architecture(self, max_layers=8, max_models_per_layer=6):
        """Generate a random ensemble architecture."""
        num_layers = np.random.randint(2, max_layers + 1)
        architecture = {
            "layers": num_layers,
            "models_per_layer": [],
            "strategies": [],
            "model_types": []
        }
        
        base_models = ["lr", "svc", "nb", "ridge", "rf", "et"]
        ensemble_models = ["bag_lr", "bag_svc", "bag_nb"]
        strategies = ["dense", "residual", "simple"]
        
        for layer in range(num_layers):
            # Layer 1 only base models, subsequent layers can have ensembles
            current_pool = base_models if layer == 0 else (base_models + ensemble_models)
            n_models = np.random.randint(2, min(max_models_per_layer, len(current_pool)) + 1)
            layer_models = np.random.choice(current_pool, n_models, replace=False).tolist()
            strategy = np.random.choice(strategies)
            
            architecture["models_per_layer"].append(n_models)
            architecture["model_types"].append(layer_models)
            architecture["strategies"].append(strategy)
        
        return architecture
    
    def _evaluate_architecture_fitness(self, architecture, performance_metrics):
        """Evaluate fitness of an architecture based on performance metrics."""
        base_fitness = performance_metrics.get(self.metric, 0.0)
        
        # Complexity penalty - favor simpler architectures
        complexity_penalty = 0.01 * sum(architecture["models_per_layer"])
        
        # Diversity bonus - reward different strategies
        strategy_diversity = len(set(architecture["strategies"])) / len(architecture["strategies"])
        diversity_bonus = 0.05 * strategy_diversity
        
        fitness = base_fitness - complexity_penalty + diversity_bonus
        return max(0.0, fitness)
    
    def _mutate_architecture(self, architecture):
        """Apply mutation to an architecture."""
        mutated = architecture.copy()
        mutated["models_per_layer"] = mutated["models_per_layer"].copy()
        mutated["model_types"] = [layer.copy() for layer in mutated["model_types"]]
        mutated["strategies"] = mutated["strategies"].copy()
        
        if np.random.random() < self.mutation_rate:
            # Mutate number of layers
            if np.random.random() < 0.3 and len(mutated["models_per_layer"]) > 2:
                # Remove layer
                idx = np.random.randint(len(mutated["models_per_layer"]))
                mutated["models_per_layer"].pop(idx)
                mutated["model_types"].pop(idx)
                mutated["strategies"].pop(idx)
                mutated["layers"] -= 1
            elif len(mutated["models_per_layer"]) < 8:
                # Add layer
                base_models = ["lr", "svc", "nb", "ridge", "rf", "et"]
                ensemble_models = ["bag_lr", "bag_svc", "bag_nb"]
                strategies = ["dense", "residual", "simple"]
                
                # Logic for new layer position
                new_layer_idx = len(mutated["models_per_layer"]) # added at the end
                current_pool = base_models if new_layer_idx == 0 else (base_models + ensemble_models)
                
                n_models = np.random.randint(2, 5)
                layer_models = np.random.choice(current_pool, n_models, replace=False).tolist()
                strategy = np.random.choice(strategies)
                
                mutated["models_per_layer"].append(n_models)
                mutated["model_types"].append(layer_models)
                mutated["strategies"].append(strategy)
                mutated["layers"] += 1
        
        if np.random.random() < self.mutation_rate:
            # Mutate models in a random layer
            if mutated["models_per_layer"]:
                layer_idx = np.random.randint(len(mutated["models_per_layer"]))
                base_models = ["lr", "svc", "nb", "ridge", "rf", "et"]
                ensemble_models = ["bag_lr", "bag_svc", "bag_nb"]
                current_pool = base_models if layer_idx == 0 else (base_models + ensemble_models)
                
                n_models = np.random.randint(2, 5)
                mutated["model_types"][layer_idx] = np.random.choice(current_pool, n_models, replace=False).tolist()
                mutated["models_per_layer"][layer_idx] = n_models
        
        if np.random.random() < self.mutation_rate:
            # Mutate strategy of a random layer
            if mutated["strategies"]:
                layer_idx = np.random.randint(len(mutated["strategies"]))
                strategies = ["dense", "residual", "simple"]
                mutated["strategies"][layer_idx] = np.random.choice(strategies)
        
        return mutated
    
    def _crossover_architectures(self, parent1, parent2):
        """Perform crossover between two parent architectures."""
        if np.random.random() > self.crossover_rate:
            return parent1.copy()
        
        child = {
            "layers": 0,
            "models_per_layer": [],
            "strategies": [],
            "model_types": []
        }
        
        # Take layers from parents with some crossover logic
        max_layers = min(parent1["layers"], parent2["layers"])
        for i in range(max_layers):
            if np.random.random() < 0.5:
                child["models_per_layer"].append(parent1["models_per_layer"][i])
                child["model_types"].append(parent1["model_types"][i].copy())
                child["strategies"].append(parent1["strategies"][i])
            else:
                child["models_per_layer"].append(parent2["models_per_layer"][i])
                child["model_types"].append(parent2["model_types"][i].copy())
                child["strategies"].append(parent2["strategies"][i])
            child["layers"] += 1
        
        # Ensure at least 2 layers
        if child["layers"] < 2:
            return parent1.copy()
        
        return child
    
    def initialize_population(self):
        """Initialize the NAS population with random architectures."""
        self.population = []
        for _ in range(self.population_size):
            arch = self._generate_random_architecture()
            self.population.append({
                "architecture": arch,
                "fitness": 0.0,
                "performance": {}
            })
    
    def evolve_population(self, performance_data):
        """Evolve the population based on performance feedback."""
        if not self.population:
            self.initialize_population()
        
        # Update fitness based on performance data
        for i, individual in enumerate(self.population):
            if i < len(performance_data):
                individual["performance"] = performance_data[i]
                individual["fitness"] = self._evaluate_architecture_fitness(
                    individual["architecture"], performance_data[i]
                )
        
        # Sort by fitness
        self.population.sort(key=lambda x: x["fitness"], reverse=True)
        
        # Selection - keep top performers
        elite_size = max(2, self.population_size // 3)
        new_population = self.population[:elite_size].copy()
        
        # Generate offspring through crossover and mutation
        while len(new_population) < self.population_size:
            if len(self.population) >= 2:
                parent1, parent2 = np.random.choice(self.population[:elite_size*2], 2, replace=False)
                child_arch = self._crossover_architectures(parent1["architecture"], parent2["architecture"])
                child_arch = self._mutate_architecture(child_arch)
                
                new_population.append({
                    "architecture": child_arch,
                    "fitness": 0.0,
                    "performance": {}
                })
            else:
                # Fallback to random generation
                arch = self._generate_random_architecture()
                new_population.append({
                    "architecture": arch,
                    "fitness": 0.0,
                    "performance": {}
                })
        
        self.population = new_population
        self.current_generation += 1
        
        # Update knowledge base
        self.knowledge["generations"] = self.current_generation
        if self.population:
            best = self.population[0]
            self.knowledge["best_architecture"] = {
                "architecture": best["architecture"],
                "fitness": best["fitness"],
                "performance": best["performance"],
                "generation": self.current_generation
            }
        self._save_knowledge()
        
        print(f"[NAS] Generation {self.current_generation} completed. Best fitness: {self.population[0]['fitness']:.4f}", flush=True)
    
    def get_best_architecture(self):
        """Get the best architecture found so far."""
        if not self.population:
            return None
        return self.population[0]["architecture"]
    
    def suggest_models_for_layer(self, layer_idx, available_models, architecture=None):
        """Suggest models for a specific layer based on NAS optimization."""
        if architecture is None:
            architecture = self.get_best_architecture()
        
        if architecture is None or layer_idx >= len(architecture["model_types"]):
            # Fallback to random selection
            n_to_pick = np.random.randint(MIN_MODELS_PER_LAYER, 
                                        min(MAX_MODELS_PER_LAYER, len(available_models)) + 1)
            return np.random.choice(available_models, n_to_pick, replace=False).tolist()
        
        # Use NAS-optimized architecture
        layer_models = architecture["model_types"][layer_idx]
        valid_models = [m for m in layer_models if m in available_models]
        
        if not valid_models:
            # Fallback
            n_to_pick = architecture["models_per_layer"][layer_idx]
            return np.random.choice(available_models, min(n_to_pick, len(available_models)), replace=False).tolist()
        
        return valid_models
    
    def get_strategy_for_layer(self, layer_idx, default_strategy="dense"):
        """Get the connection strategy for a specific layer."""
        architecture = self.get_best_architecture()
        if architecture is None or layer_idx >= len(architecture["strategies"]):
            return default_strategy
        return architecture["strategies"][layer_idx]


# ─── Pre-fitted Voting Ensembles (module-level for pickle compatibility) ─────
class _PreFittedSoftVoting:
    """Soft voting across pre-fitted estimators (averages predict_proba)."""
    def __init__(self, estimators):
        self.estimators_ = [m for _, m in estimators]
    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.estimators_], axis=0)
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class _PreFittedHardVoting:
    """Hard (majority) voting across pre-fitted estimators."""
    def __init__(self, estimators, n_classes):
        self.estimators_ = [m for _, m in estimators]
        self.n_classes_ = n_classes
    def predict(self, X):
        preds = np.column_stack([m.predict(X) for m in self.estimators_])
        return np.apply_along_axis(
            lambda row: np.bincount(row, minlength=self.n_classes_).argmax(),
            axis=1, arr=preds
        )
    def predict_proba(self, X):
        preds = self.predict(X)
        out = np.zeros((len(preds), self.n_classes_))
        out[np.arange(len(preds)), preds] = 1.0
        return out


class _PreviousLayerSoftVoting:
    def __init__(self, n_models, n_classes):
        self.n_models = n_models
        self.n_classes = n_classes

    def fit(self, X, y=None):
        return self

    def _reshape(self, X):
        return np.asarray(X).reshape(X.shape[0], self.n_models, self.n_classes)

    def predict_proba(self, X):
        return self._reshape(X).mean(axis=1)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class _PreviousLayerBaggingVoting:
    def __init__(self, n_models, n_classes, n_bags=8, bag_fraction=0.7, seed=SEED):
        self.n_models = n_models
        self.n_classes = n_classes
        self.n_bags = n_bags
        self.bag_fraction = bag_fraction
        self.seed = seed
        self.bag_indices = []

    def fit(self, X, y=None):
        rng = np.random.RandomState(self.seed)
        bag_size = max(1, int(np.ceil(self.n_models * self.bag_fraction)))
        self.bag_indices = [
            rng.choice(self.n_models, size=bag_size, replace=True)
            for _ in range(self.n_bags)
        ]
        return self

    def _reshape(self, X):
        return np.asarray(X).reshape(X.shape[0], self.n_models, self.n_classes)

    def predict_proba(self, X):
        probs = self._reshape(X)
        bag_preds = [probs[:, idxs, :].mean(axis=1) for idxs in self.bag_indices]
        return np.mean(bag_preds, axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# ─── Pyramid Logic ────────────────────────────────────────────────────────────
class PyramidEnsemble:
    def __init__(self, num_layers=3, seed=SEED, meta_learner=None, patience=PATIENCE, 
                 metric=OPTIM_METRIC, jitter=JITTER, strategy=STRATEGY,
                 min_models=MIN_MODELS_PER_LAYER, max_models=MAX_MODELS_PER_LAYER, 
                 nas_controller=None, n_jobs=2,
                 layer_type=PYRAMID_LAYER_TYPE, homogeneous_model=HOMOGENEOUS_MODEL,
                 homogeneous_variants=HOMOGENEOUS_VARIANTS):
        self.num_layers = num_layers
        self.seed = seed
        self.meta_learner = meta_learner
        self.patience = patience
        self.metric = metric
        self.jitter = jitter
        self.strategy = strategy
        self.min_models = min_models
        self.max_models = max_models
        self.nas_controller = nas_controller
        self.n_jobs = n_jobs
        self.layer_type = layer_type
        self.homogeneous_model = homogeneous_model
        self.homogeneous_variants = homogeneous_variants
        self.layers = []
        self.layer_meta_models = []
        self.results = []
        self.training_metadata = {"run": {}, "layers": [], "models": []}
        self.best_model = None
        self.best_score = 0
        self.no_improve_layers = 0 # Early stopping counter


        
    def _evaluate_and_log(self, model, X, y, layer_idx, name, start_time):
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, average="weighted")
        precision = precision_score(y, preds, average="weighted", zero_division=0)
        recall = recall_score(y, preds, average="weighted", zero_division=0)
        
        # Calculate AUC if possible
        auc = 0.0
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X)
                if probs.shape[1] > 2:
                    auc = roc_auc_score(y, probs, multi_class="ovr", average="weighted")
                else:
                    auc = roc_auc_score(y, probs[:, 1])
            except:
                pass
        
        duration = time.time() - start_time
        
        metrics = {
            "layer": layer_idx, 
            "model": name, 
            "accuracy": acc, 
            "f1": f1, 
            "precision": precision,
            "recall": recall,
            "auc": auc,
            "duration": duration
        }
        
        self.results.append(metrics)
        
        # Log to MLflow if a run is active
        try:
            if mlflow.active_run():
                # Use a prefix to avoid collisions with other models in the same run
                prefix = f"L{layer_idx}_{name}"
                mlflow.log_metrics({
                    f"{prefix}_f1": f1,
                    f"{prefix}_accuracy": acc,
                    f"{prefix}_precision": precision,
                    f"{prefix}_recall": recall,
                    f"{prefix}_auc": auc,
                    f"{prefix}_duration": duration
                })
        except:
            pass

        score = f1 if self.metric == "f1" else acc
        status = "NEW BEST!" if score > self.best_score else "       "
        print(f"  [{status}] Layer {layer_idx:02} | {name:<20} | F1: {f1:.4f} | Acc: {acc:.4f} | P: {precision:.4f} | R: {recall:.4f} | AUC: {auc:.4f} | {duration:>5.1f}s", flush=True)
        
        if score > self.best_score:
            self.best_score = score
            self.best_model = model
        return score


    def train(self, X_train, y_train, X_val, y_val, progress_callback=None):
        current_X_train, current_X_val = X_train, X_val
        all_oof_preds, all_val_preds = [], []
        prev_layer_train_stack, prev_layer_val_stack = None, None
        prev_layer_model_names = []
        self.layer_meta_models = []
        self.training_metadata = {
            "run": {
                "num_layers": self.num_layers,
                "seed": self.seed,
                "metric": self.metric,
                "strategy": self.strategy,
                "layer_type": self.layer_type,
                "homogeneous_model": self.homogeneous_model,
                "homogeneous_variants": self.homogeneous_variants,
                "min_models": self.min_models,
                "max_models": self.max_models,
                "cv_folds": CV_FOLDS,
                "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
                "val_shape": [int(X_val.shape[0]), int(X_val.shape[1])]
            },
            "layers": [],
            "models": []
        }
        t_global_start = time.time()
        
        print(f"\n{'='*85}\n{'STARTING ENSEMBLE PYRAMID TRAINING':^85}\n{'='*85}", flush=True)
        print(f"Dataset: {X_train.shape} | Layers: {self.num_layers} | RL: {'Active' if self.meta_learner else 'Off'}", flush=True)
        print(f"{'-'*85}", flush=True)

        for l in range(1, self.num_layers + 1):
            t_layer_start = time.time()
            print(f"\n[LAYER {l:02}] Training...", flush=True)
            if callable(progress_callback):
                progress_callback({
                    "event": "layer_start",
                    "layer": l,
                    "num_layers": self.num_layers,
                    "results": list(self.results),
                    "best_score": self.best_score,
                })
            
            # Base pool: Layer 1 should only have simple base models to build the foundation
            # Subsequent layers only use ensembles built from previous-layer outputs
            avail_base = ["lr", "svc", "nb", "ridge", "rf", "et"]
            avail_meta = ["stack_prev", "bag_prev", "vote_prev"]
            
            pool = avail_base if l == 1 else avail_meta
            effective_min = max(1, min(self.min_models, len(pool)))
            effective_max = max(effective_min, min(self.max_models, len(pool)))

            if l == 1 and self.layer_type == "homogeneous":
                n_variants = max(1, min(self.homogeneous_variants, self.max_models))
                models_to_run = [f"{self.homogeneous_model}__v{i+1}" for i in range(n_variants)]
            else:
                # The heart of variability: RL Meta-Learner + NAS decides WHAT and HOW MANY
                models_to_run = self.meta_learner.suggest_models(
                    l,
                    pool,
                    min_p=effective_min,
                    max_p=effective_max,
                ) if self.meta_learner else pool
            
            # NAS-optimized strategy selection
            current_strategy = self.strategy
            if self.nas_controller:
                nas_strategy = self.nas_controller.get_strategy_for_layer(l, self.strategy)
                current_strategy = nas_strategy
                if nas_strategy != self.strategy:
                    print(f"  [NAS] Strategy override: {self.strategy} -> {nas_strategy}", flush=True)
            
            print(f"  [HYBRID] Selected set ({len(models_to_run)} models): {models_to_run}", flush=True)
            layer_metadata = {
                "layer": l,
                "selected_models": list(models_to_run),
                "strategy": current_strategy,
                "layer_type": self.layer_type if l == 1 else "meta_ensemble",
                "source_previous_layer_models": list(prev_layer_model_names),
                "input_features": int(layer_train_input.shape[1]) if l > 1 and hasattr(prev_layer_train_stack, "shape") else (int(current_X_train.shape[1]) if hasattr(current_X_train, "shape") else None)
            }
            self.training_metadata["layers"].append(layer_metadata)
            if callable(progress_callback):
                progress_callback({
                    "event": "layer_config",
                    "layer": l,
                    "num_layers": self.num_layers,
                    "strategy": current_strategy,
                    "models_to_run": list(models_to_run),
                    "results": list(self.results),
                    "best_score": self.best_score,
                })


            layer_models, layer_oof, layer_val = [], [], []
            layer_transformers = []
            layer_feature_model_names = []
            score_before_layer = self.best_score  # snapshot before any model in this layer updates it
            for model_idx, m_type in enumerate(models_to_run):
                t0 = time.time()
                model_name = f"{m_type}_L{l}"
                raw_type = re.sub(r"__v\d+$", "", m_type)
                variant_scale = 1.0 + ((model_idx % 5) - 2) * 0.12
                model_seed = self.seed + l + (model_idx * 37)

                if l == 1:
                    layer_train_input, layer_val_input = current_X_train, current_X_val
                    model = get_model(raw_type, model_seed, jitter=self.jitter, n_jobs=self.n_jobs, variant_scale=variant_scale)
                    model.fit(layer_train_input, y_train)
                else:
                    if prev_layer_train_stack is None or prev_layer_val_stack is None:
                        raise ValueError("Camada anterior indisponível para montar ensembles hierárquicos.")
                    layer_train_input, layer_val_input = prev_layer_train_stack, prev_layer_val_stack
                    n_classes = len(np.unique(y_train))
                    n_prev_models = max(1, len(prev_layer_model_names))
                    if m_type == "stack_prev":
                        model = LogisticRegression(max_iter=1000, random_state=model_seed, n_jobs=self.n_jobs)
                    elif m_type == "bag_prev":
                        model = _PreviousLayerBaggingVoting(
                            n_models=n_prev_models,
                            n_classes=n_classes,
                            seed=model_seed
                        )
                    elif m_type == "vote_prev":
                        model = _PreviousLayerSoftVoting(n_models=n_prev_models, n_classes=n_classes)
                    else:
                        raise ValueError(f"Tipo de modelo inválido para camada meta: {m_type}")
                    model.fit(layer_train_input, y_train)

                self._evaluate_and_log(model, layer_val_input, y_val, l, model_name, t0)
                cv_info = {
                    "method": "stratified_kfold" if (m_type == "stack_prev" or l == 1) else "none",
                    "n_splits": CV_FOLDS if (m_type == "stack_prev" or l == 1) else None,
                    "shuffle": True if (m_type == "stack_prev" or l == 1) else None,
                    "random_state": model_seed if (m_type == "stack_prev" or l == 1) else None
                }
                model_metadata = {
                    "layer": l,
                    "model": model_name,
                    "model_type": m_type,
                    "model_family": raw_type,
                    "seed": int(model_seed),
                    "variant_scale": float(variant_scale),
                    "hyperparameters": _extract_model_hyperparams(model),
                    "cv_info": cv_info,
                    "split_info": {
                        "train_rows": int(layer_train_input.shape[0]) if hasattr(layer_train_input, "shape") else None,
                        "val_rows": int(layer_val_input.shape[0]) if hasattr(layer_val_input, "shape") else None,
                        "train_features": int(layer_train_input.shape[1]) if hasattr(layer_train_input, "shape") else None,
                        "val_features": int(layer_val_input.shape[1]) if hasattr(layer_val_input, "shape") else None
                    },
                    "previous_layer_models": list(prev_layer_model_names) if l > 1 else []
                }
                self.training_metadata["models"].append(model_metadata)
                self.results[-1].update({
                    "model_type": m_type,
                    "model_family": raw_type,
                    "seed": int(model_seed),
                    "variant_scale": float(variant_scale),
                    "cv_method": cv_info["method"],
                    "cv_n_splits": cv_info["n_splits"],
                    "cv_random_state": cv_info["random_state"],
                    "hyperparameters": model_metadata["hyperparameters"],
                    "train_rows": model_metadata["split_info"]["train_rows"],
                    "val_rows": model_metadata["split_info"]["val_rows"],
                    "train_features": model_metadata["split_info"]["train_features"],
                    "val_features": model_metadata["split_info"]["val_features"]
                })
                if callable(progress_callback):
                    progress_callback({
                        "event": "model_trained",
                        "layer": l,
                        "num_layers": self.num_layers,
                        "model_name": model_name,
                        "strategy": current_strategy,
                        "models_to_run": list(models_to_run),
                        "latest_result": dict(self.results[-1]),
                        "results": list(self.results),
                        "best_score": self.best_score,
                    })

                layer_models.append((model_name, model))

                if l < self.num_layers:
                    layer_transformers.append(model)
                    layer_feature_model_names.append(model_name)
                    if m_type == "stack_prev":
                        cv_splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=model_seed)
                        oof = cross_val_predict(model, layer_train_input, y_train, cv=cv_splitter, method="predict_proba", n_jobs=self.n_jobs)
                        vp = model.predict_proba(layer_val_input)
                    elif hasattr(model, "predict_proba") and l == 1:
                        cv_splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=model_seed)
                        oof = cross_val_predict(model, layer_train_input, y_train, cv=cv_splitter, method="predict_proba", n_jobs=self.n_jobs)
                        vp = model.predict_proba(layer_val_input)
                    elif hasattr(model, "predict_proba"):
                        oof = model.predict_proba(layer_train_input)
                        vp = model.predict_proba(layer_val_input)
                    else:
                        oof_idx = model.predict(layer_train_input)
                        vp_idx = model.predict(layer_val_input)
                        n_c = len(np.unique(y_train))
                        oof = np.eye(n_c)[oof_idx]
                        vp = np.eye(n_c)[vp_idx]
                    layer_oof.append(oof)
                    layer_val.append(vp)

            self.layers.append(layer_models)

            
            # Early Stopping Check — compare against score snapshot taken before this layer
            layer_best_score = max([(res["f1"] if self.metric == "f1" else res["accuracy"]) for res in self.results if res["layer"] == l])
            if layer_best_score > score_before_layer + 1e-5:
                self.no_improve_layers = 0
            else:
                self.no_improve_layers += 1
                if self.no_improve_layers >= self.patience:
                    print(f"\n[EARLY STOPPING] No improvement for {self.patience} layers. Stopping at Layer {l}.", flush=True)
                    mlflow.log_param("early_stop_layer", l)
                    break

            if l < self.num_layers:
                self.layer_meta_models.append(layer_transformers)
                prev_layer_train_stack = np.hstack(layer_oof)
                prev_layer_val_stack = np.hstack(layer_val)
                prev_layer_model_names = list(layer_feature_model_names)
                all_oof_preds.append(prev_layer_train_stack)
                all_val_preds.append(prev_layer_val_stack)
                
                # Dynamic Strategy (use NAS-optimized strategy for this layer)
                if current_strategy == "dense":
                    current_X_train, current_X_val = np.hstack(all_oof_preds), np.hstack(all_val_preds)
                elif current_strategy == "residual":
                    # Only last layer's OOF (no accumulation)
                    current_X_train, current_X_val = np.hstack(layer_oof), np.hstack(layer_val)
                else: # "simple"
                    current_X_train, current_X_val = np.hstack(layer_oof), np.hstack(layer_val)
                    
                print(f"  [INFO] Layer {l} Done ({time.time()-t_layer_start:.1f}s). New Feats: {current_X_train.shape[1] if hasattr(current_X_train, 'shape') else len(current_X_train)}", flush=True)
                if callable(progress_callback):
                    progress_callback({
                        "event": "layer_done",
                        "layer": l,
                        "num_layers": self.num_layers,
                        "strategy": current_strategy,
                        "results": list(self.results),
                        "best_score": self.best_score,
                        "n_features": int(current_X_train.shape[1]) if hasattr(current_X_train, "shape") else None,
                    })


        print(f"\n{'='*85}\nCOMPLETE | Time: {(time.time()-t_global_start)/60:.1f} min\n{'='*85}", flush=True)
        self.training_metadata["run"]["elapsed_seconds"] = float(time.time() - t_global_start)
        self.training_metadata["run"]["best_score"] = float(self.best_score)
        if self.meta_learner: self.meta_learner.update_knowledge(self.results)
        if callable(progress_callback):
            progress_callback({
                "event": "training_done",
                "num_layers": self.num_layers,
                "results": list(self.results),
                "best_score": self.best_score,
                "elapsed_seconds": time.time() - t_global_start,
            })

    def predict(self, X):
        """Transform X through the layers to match the best model's required input."""
        if self.best_model is None:
            raise ValueError("Pyramid not trained or no best model found.")

        def _model_output_as_proba(model, features):
            if hasattr(model, "predict_proba"):
                return model.predict_proba(features)
            raw = np.asarray(model.predict(features), dtype=int)
            n_classes = int(raw.max()) + 1 if raw.size > 0 else 1
            return np.eye(n_classes)[raw]
        
        # Determine the layer of the best model
        best_layer_idx = 1
        found = False
        for l_idx, l_models in enumerate(self.layers):
            for m_name, m_obj in l_models:
                if m_obj == self.best_model:
                    best_layer_idx = l_idx + 1
                    found = True
                    break
            if found:
                break
        
        if best_layer_idx == 1:
            return self.best_model.predict(X)
        
        # Build meta-features chain using only models that generated next-layer features during training.
        current_X = X
        all_meta_preds = []
        
        for l in range(1, best_layer_idx):
            if l - 1 >= len(self.layer_meta_models):
                raise ValueError("Missing layer meta-models for inference. Retrain the pyramid before predicting.")

            layer_meta = [_model_output_as_proba(m_obj, current_X) for m_obj in self.layer_meta_models[l - 1]]
            layer_stack = np.hstack(layer_meta)

            if self.strategy == "dense":
                all_meta_preds.append(layer_stack)
                current_X = np.hstack(all_meta_preds)
            elif self.strategy == "residual":
                current_X = layer_stack
            else:  # "simple"
                current_X = layer_stack
            
        return self.best_model.predict(current_X)



# ─── Execution ───────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Highly Customizable Flexible Ensemble Pyramid with RL + NAS")
    parser.add_argument("--layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--min_models", type=int, default=MIN_MODELS_PER_LAYER)
    parser.add_argument("--max_models", type=int, default=MAX_MODELS_PER_LAYER)
    parser.add_argument("--epsilon", type=float, default=EPSILON_RL)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--metric", type=str, choices=["f1", "accuracy"], default=OPTIM_METRIC)
    parser.add_argument("--tfidf_max", type=int, default=TFIDF_MAX)
    parser.add_argument("--tfidf_ngrams", type=int, default=2, help="Max ngrams (1 or 2)")
    parser.add_argument("--jitter", type=bool, default=JITTER)
    parser.add_argument("--strategy", type=str, choices=["dense", "residual", "simple"], default=STRATEGY)
    parser.add_argument("--layer_type", type=str, choices=["heterogeneous", "homogeneous"], default=PYRAMID_LAYER_TYPE)
    parser.add_argument("--homogeneous_model", type=str, choices=["lr", "svc", "nb", "ridge", "rf", "et"], default=HOMOGENEOUS_MODEL)
    parser.add_argument("--homogeneous_variants", type=int, default=HOMOGENEOUS_VARIANTS)
    parser.add_argument("--subsample_train", type=int, default=0, help="Number of training samples (0=all)")
    parser.add_argument("--subsample_val", type=int, default=0, help="Number of validation samples (0=all)")
    parser.add_argument("--n_jobs", type=int, default=2, help="Parallel jobs for models and CV")
    
    # NAS Parameters
    parser.add_argument("--use_nas", type=bool, default=False, help="Enable Neural Architecture Search")
    parser.add_argument("--nas_population", type=int, default=10, help="NAS population size")
    parser.add_argument("--nas_generations", type=int, default=5, help="NAS generations")
    parser.add_argument("--nas_mutation", type=float, default=0.1, help="NAS mutation rate")
    parser.add_argument("--nas_crossover", type=float, default=0.7, help="NAS crossover rate")
    
    args = parser.parse_args()


    set_seed(args.seed)
    setup_tracking()

    
    current_layers = args.layers
    current_seed = args.seed

    print(f"\n[CONFIG] Layers: {current_layers} | Seed: {current_seed} | RL Epsilon: {args.epsilon}")
    print(f"[CONFIG] Diversity: {args.min_models}-{args.max_models} models per layer")
    print(f"[CONFIG] Layer Type: {args.layer_type}")
    if args.layer_type == "homogeneous":
        print(f"[CONFIG] Homogeneous Model: {args.homogeneous_model} | Variants: {args.homogeneous_variants}")
    print(f"[CONFIG] Parallelism: n_jobs={args.n_jobs}")
    if args.subsample_train > 0: print(f"[CONFIG] Subsampling: train={args.subsample_train}, val={args.subsample_val}")
    print(f"[CONFIG] NAS: {'Enabled' if args.use_nas else 'Disabled'}")
    if args.use_nas:
        print(f"[CONFIG] NAS: Pop={args.nas_population}, Gen={args.nas_generations}, Mut={args.nas_mutation}, Cross={args.nas_crossover}")
    print()

    # Initialize NAS Controller if enabled
    nas_controller = None
    if args.use_nas:
        nas_controller = NASController(
            population_size=args.nas_population,
            generations=args.nas_generations,
            mutation_rate=args.nas_mutation,
            crossover_rate=args.nas_crossover,
            metric=args.metric
        )
        nas_controller.initialize_population()
        print("[NAS] Population initialized")

    with mlflow.start_run(run_name=f"Pyramid_RL_NAS_L{current_layers}_Var" if args.use_nas else f"Pyramid_RL_L{current_layers}_Var"):
        mlflow.log_params({
            "num_layers": current_layers, 
            "seed": current_seed,
            "epsilon": args.epsilon,
            "min_models": args.min_models,
            "max_models": args.max_models,
            "metric": args.metric,
            "jitter": args.jitter,
            "strategy": args.strategy,
            "layer_type": args.layer_type,
            "homogeneous_model": args.homogeneous_model,
            "homogeneous_variants": args.homogeneous_variants,
            "tfidf_max": args.tfidf_max,
            "use_nas": args.use_nas,
            "nas_population": args.nas_population,
            "nas_generations": args.nas_generations,
            "nas_mutation": args.nas_mutation,
            "nas_crossover": args.nas_crossover,
            "n_jobs": args.n_jobs
        })


        
        train_df, val_df = load_data(subsample_train=args.subsample_train, subsample_val=args.subsample_val)
        
        status_msg = f"Training on {len(train_df)} rows, validating on {len(val_df)} rows"
        print(f"[INFO] {status_msg}")

        le = LabelEncoder()
        y_train = le.fit_transform(train_df['sentiment'])
        y_val = le.transform(val_df['sentiment'])
        
        tfidf = TfidfVectorizer(max_features=args.tfidf_max, ngram_range=(1, args.tfidf_ngrams), stop_words='english', min_df=2)
        X_train = tfidf.fit_transform(train_df['clean_text'].fillna(""))
        X_val = tfidf.transform(val_df['clean_text'].fillna(""))
        
        rl_learner = RLMetaLearner(epsilon=args.epsilon, metric=args.metric, nas_controller=nas_controller)
        pyramid = PyramidEnsemble(num_layers=current_layers, seed=current_seed, 
                                 meta_learner=rl_learner, patience=args.patience,
                                 metric=args.metric, jitter=args.jitter,
                                 strategy=args.strategy,
                                 min_models=args.min_models, max_models=args.max_models,
                                 nas_controller=nas_controller,
                                 n_jobs=args.n_jobs,
                                 layer_type=args.layer_type,
                                 homogeneous_model=args.homogeneous_model,
                                 homogeneous_variants=args.homogeneous_variants)
        pyramid.train(X_train, y_train, X_val, y_val)



        
        # Artifacts
        pd.DataFrame(pyramid.results).to_csv("pyramid_results.csv", index=False)
        mlflow.log_artifact("pyramid_results.csv")
        run_details = {
            "run_config": {
                "num_layers": current_layers,
                "seed": current_seed,
                "epsilon": args.epsilon,
                "min_models": args.min_models,
                "max_models": args.max_models,
                "metric": args.metric,
                "jitter": args.jitter,
                "strategy": args.strategy,
                "layer_type": args.layer_type,
                "homogeneous_model": args.homogeneous_model,
                "homogeneous_variants": args.homogeneous_variants,
                "cv_folds": CV_FOLDS,
                "tfidf_max": args.tfidf_max,
                "tfidf_ngrams": args.tfidf_ngrams,
                "n_jobs": args.n_jobs
            },
            "dataset_profile": {
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "train_class_distribution": train_df["sentiment"].value_counts().to_dict(),
                "val_class_distribution": val_df["sentiment"].value_counts().to_dict()
            },
            "training_metadata": pyramid.training_metadata,
            "results_detailed": pyramid.results
        }
        with open("pyramid_run_details.json", "w", encoding="utf-8") as f:
            json.dump(run_details, f, indent=2)
        mlflow.log_artifact("pyramid_run_details.json")
        if rl_learner.path.exists(): mlflow.log_artifact(str(rl_learner.path))
        if nas_controller and nas_controller.path.exists(): mlflow.log_artifact(str(nas_controller.path))
        
        if pyramid.best_model:
            joblib.dump(pyramid.best_model, "best_pyramid_model.pkl")
            mlflow.log_artifact("best_pyramid_model.pkl")

            # Log best model metrics so runs are comparable in MLflow UI
            best_res = max(pyramid.results, key=lambda r: r["f1"] if pyramid.metric == "f1" else r["accuracy"])
            mlflow.log_params({"best_model": best_res["model"], "best_layer": best_res["layer"]})
            try:
                mlflow.log_metric("best_f1", best_res["f1"])
                mlflow.log_metric("best_accuracy", best_res["accuracy"])
            except Exception:
                pass  # Dummy mlflow silently ignores metrics

            # Use the pyramid's predict method which handles feature transformation
            y_pred = pyramid.predict(X_val)
            cm = confusion_matrix(y_val, y_pred)
            plt.figure(figsize=(10,7))

            sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
            plt.savefig("pyramid_cm.png")
            mlflow.log_artifact("pyramid_cm.png")
            
        print("\nRun Completed!")

if __name__ == "__main__":
    main()
