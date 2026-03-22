"""
Pipeline builder — assembles sklearn Pipelines from preprocessor + model steps.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from typing import List, Tuple, Dict, Any, Optional


def build_pipeline(preprocessor_steps: List[Tuple], model) -> Pipeline:
    """Build a full sklearn Pipeline from preprocessor steps + final estimator."""
    steps = list(preprocessor_steps) + [("model", model)]
    return Pipeline(steps)


def get_feature_names_after_preprocessor(pipeline: Pipeline, numeric_cols: List[str], categorical_cols: List[str]) -> List[str]:
    """Attempt to extract feature names from the preprocessor step."""
    try:
        preprocessor = pipeline.named_steps.get("preprocessor")
        if preprocessor is None:
            return numeric_cols + categorical_cols
        feature_names = preprocessor.get_feature_names_out()
        # Clean up prefix (num__, cat__)
        cleaned = [f.replace("num__", "").replace("cat__", "").split("__")[-1] for f in feature_names]
        return cleaned
    except Exception:
        return numeric_cols + categorical_cols


def get_feature_importance(pipeline: Pipeline, feature_names: List[str]) -> Optional[pd.DataFrame]:
    """Extract feature importances from the final model in the pipeline."""
    model = pipeline.named_steps.get("model")
    if model is None:
        return None

    importances = None
    cols = feature_names

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            importances = np.abs(coef).mean(axis=0)
        else:
            importances = np.abs(coef)

    if importances is None:
        return None

    # Align lengths
    min_len = min(len(importances), len(cols))
    importances = importances[:min_len]
    cols = cols[:min_len]

    df_imp = pd.DataFrame({"feature": cols, "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)
    return df_imp
