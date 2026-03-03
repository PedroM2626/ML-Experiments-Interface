"""
SHAP-based explainability for AutoML Studio.
Computes global (beeswarm) and local (waterfall) explanations for any pipeline.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Any

import warnings
warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def _get_estimator(pipeline):
    """Extract the final real estimator from a sklearn Pipeline."""
    if hasattr(pipeline, "steps"):
        return pipeline.steps[-1][1]
    if hasattr(pipeline, "final_estimator"):
        return pipeline.final_estimator
    return pipeline


def _get_preprocessor(pipeline):
    """Extract the preprocessor (all steps except last) from a sklearn Pipeline."""
    if hasattr(pipeline, "steps") and len(pipeline.steps) > 1:
        from sklearn.pipeline import Pipeline as SKPipeline
        return SKPipeline(pipeline.steps[:-1])
    return None


def _is_tree_model(estimator) -> bool:
    tree_types = (
        "XGBClassifier", "XGBRegressor",
        "LGBMClassifier", "LGBMRegressor",
        "GradientBoostingClassifier", "GradientBoostingRegressor",
        "RandomForestClassifier", "RandomForestRegressor",
        "ExtraTreesClassifier", "ExtraTreesRegressor",
        "DecisionTreeClassifier", "DecisionTreeRegressor",
    )
    return type(estimator).__name__ in tree_types


def _is_linear_model(estimator) -> bool:
    linear_types = (
        "LogisticRegression", "Ridge", "Lasso", "ElasticNet",
        "LinearRegression", "LinearSVC",
    )
    return type(estimator).__name__ in linear_types


def compute_shap_values(
    pipeline,
    X_sample: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    max_samples: int = 200,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[List[str]]]:
    """
    Compute SHAP values for a fitted pipeline.
    Returns: (shap_values, expected_value, feature_names_out)
    """
    if not SHAP_AVAILABLE:
        return None, None, None

    try:
        # Sample to keep computation fast
        if len(X_sample) > max_samples:
            idx = np.random.default_rng(42).choice(len(X_sample), max_samples, replace=False)
            X_s = X_sample.iloc[idx].reset_index(drop=True)
        else:
            X_s = X_sample.reset_index(drop=True)

        estimator = _get_estimator(pipeline)
        preprocessor = _get_preprocessor(pipeline)

        # Transform data through preprocessor
        if preprocessor is not None:
            try:
                X_transformed = preprocessor.transform(X_s)
                if hasattr(X_transformed, "toarray"):
                    X_transformed = X_transformed.toarray()
                X_transformed = np.array(X_transformed, dtype=float)
            except Exception:
                X_transformed = X_s.values.astype(float)
        else:
            X_transformed = X_s.values.astype(float)

        # Feature names after transform
        feat_names = feature_names
        if feat_names is None:
            feat_names = [f"f{i}" for i in range(X_transformed.shape[1])]

        # Choose explainer
        if _is_tree_model(estimator):
            explainer = shap.TreeExplainer(estimator)
            sv = explainer.shap_values(X_transformed)
            expected = float(np.mean(explainer.expected_value)) if isinstance(
                explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value)
            # For multi-class, take class 1 or mean
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
        elif _is_linear_model(estimator):
            explainer = shap.LinearExplainer(estimator, X_transformed)
            sv = explainer.shap_values(X_transformed)
            expected = float(explainer.expected_value) if not isinstance(
                explainer.expected_value, (list, np.ndarray)) else float(np.mean(explainer.expected_value))
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
        else:
            # Kernel explainer — slower, use tiny sample
            X_bg = shap.sample(X_transformed, min(50, len(X_transformed)))
            explainer = shap.KernelExplainer(estimator.predict, X_bg)
            sv = explainer.shap_values(X_transformed[:min(30, len(X_transformed))], nsamples=50)
            expected = float(explainer.expected_value)

        return np.array(sv), expected, list(feat_names)

    except Exception as e:
        warnings.warn(f"[SHAP] compute_shap_values failed: {e}")
        return None, None, None


def compute_local_explanation(
    pipeline,
    X_row: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[List[str]]]:
    """
    Compute SHAP values for a single row (local explanation).
    Returns: (shap_values_1d, base_value, feature_names)
    """
    sv, expected, feat_names = compute_shap_values(
        pipeline, X_row, feature_names=feature_names, max_samples=50)
    if sv is None:
        return None, None, None
    # Return just first row
    return sv[0] if sv.ndim > 1 else sv, expected, feat_names


def get_shap_feature_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Global feature importance from SHAP values.
    Returns DataFrame with columns ['feature', 'mean_abs_shap'].
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    df = df.sort_values("mean_abs_shap", ascending=False).head(top_n)
    return df.reset_index(drop=True)
