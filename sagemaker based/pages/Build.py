"""
AutoML Studio – Page: Build
Full model building workflow: Select → Build → Analyze → Predict
This is the core page of the system, inspired by SageMaker Canvas.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import time
import pandas as pd
import numpy as np
import streamlit as st

from config import PROBLEM_TYPE_LABELS, APP_ICON
from src.data_utils import detect_problem_type, profile_dataset, dataset_summary, apply_transforms
from src.automl_engine import preview_model, train_model, predict, get_saved_models
from src.ml_utils import (
    compute_metrics, get_confusion_matrix, get_roc_curve,
    get_precision_recall, format_metric_display, primary_metric, primary_metric_label,
)
from src.mlflow_utils import setup_mlflow, start_run, log_params, log_metrics, log_model_artifact, end_run, log_figure
from src.visualization import (
    feature_importance_chart, confusion_matrix_chart, roc_curve_chart,
    precision_recall_chart, residuals_chart, leaderboard_chart,
    target_distribution_chart, column_distribution_chart,
)
from src.ui_utils import load_global_css

st.set_page_config(page_title="Build – AutoML Studio", page_icon="🔨", layout="wide")

# ── CSS ────────────────────────────────────────────────────────────────────────
load_global_css()

# ── Step Progress Bar ──────────────────────────────────────────────────────────
STEPS = ["1. Select", "2. Build", "3. Analyze", "4. Predict"]

def render_step_bar(current: int):
    items = ""
    for i, label in enumerate(STEPS):
        if i < current:
            cls = "done"
            color = "#A78BFA"
            bg = "rgba(124,58,237,0.1)"
        elif i == current:
            cls = "active"
            color = "white"
            bg = "linear-gradient(135deg,#7C3AED,#4F46E5)"
        else:
            cls = ""
            color = "#475569"
            bg = "transparent"
        items += f'<div style="flex:1;text-align:center;padding:0.6rem 0;border-radius:8px;font-weight:500;font-size:0.85rem;color:{color};background:{bg};">{label}</div>'
    st.markdown(
        f'<div style="display:flex;background:#1E293B;border:1px solid #334155;border-radius:10px;padding:4px;gap:4px;margin-bottom:1.5rem;">{items}</div>',
        unsafe_allow_html=True,
    )

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="font-size:1.8rem;font-weight:800;color:#F1F5F9;margin-bottom:0.25rem;">🔨 Build Model</h1>
<p style="color:#94A3B8;font-size:0.9rem;margin-top:0;">Train an AutoML model on your dataset — no code required.</p>
""", unsafe_allow_html=True)

# ── State Helpers ──────────────────────────────────────────────────────────────
def get_build_state() -> dict:
    if "build_state" not in st.session_state:
        st.session_state["build_state"] = {}
    return st.session_state["build_state"]

def set_build_state(key, val):
    get_build_state()[key] = val

bs = get_build_state()

# Determine current step
if "current_model" in st.session_state and st.session_state["current_model"]:
    model_name = st.session_state["current_model"]
    model_result = st.session_state.get("models", {}).get(model_name)
    if model_result and "error" not in model_result:
        bs.setdefault("step", 2)
    else:
        bs.setdefault("step", 0)
else:
    bs.setdefault("step", 0)

current_step = bs.get("step", 0)
render_step_bar(current_step)

datasets = st.session_state.get("datasets", {})

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 – SELECT DATASET
# ══════════════════════════════════════════════════════════════════════════════
if current_step == 0:
    st.markdown("### Select a Dataset")

    if not datasets:
        st.warning("No datasets available. Go to **📊 Datasets** to upload one first.")
        col_btn, _ = st.columns([1, 4])
        with col_btn:
            if st.button("← Go to Datasets", key="go_datasets_btn"):
                try:
                    st.switch_page("pages/Datasets.py")
                except Exception:
                    st.info("Please click '📊 Datasets' in the left sidebar.")
        st.stop()

    # Model name
    model_name_input = st.text_input(
        "Model name",
        value=bs.get("model_name", f"Model_{int(time.time()) % 10000}"),
        max_chars=32,
        help="Use only letters, numbers, and underscores.",
    )

    # Problem type selection cards
    st.markdown("**Problem type**")
    st.markdown("<p style='color:#94A3B8;font-size:0.82rem;margin-bottom:1rem;'>Select the type of problem you want the model to solve.</p>", unsafe_allow_html=True)

    prob_options = [
        ("🧮", "Predictive Analysis",
         "Build models using tabular datasets to predict values or categories.",
         "tabular"),
        ("⏱️", "Time Series Forecasting",
         "Predict future values based on historical time-ordered data.",
         "timeseries"),
    ]

    selected_prob = bs.get("problem_category", "tabular")
    prob_cols = st.columns(len(prob_options))
    for col, (icon, title, desc, key) in zip(prob_cols, prob_options):
        with col:
            is_sel = selected_prob == key
            border = "2px solid #7C3AED" if is_sel else "1px solid #334155"
            bg = "rgba(124,58,237,0.1)" if is_sel else "#1E293B"
            st.markdown(f"""
            <div style="background:{bg};border:{border};border-radius:12px;padding:1.25rem;text-align:center;transition:all 0.2s;cursor:pointer;">
                <div style="font-size:2rem;margin-bottom:0.5rem;">{icon}</div>
                <div style="font-weight:600;color:#F1F5F9;font-size:0.9rem;">{title}</div>
                <div style="color:#64748B;font-size:0.75rem;margin-top:0.5rem;">{desc}</div>
            </div>""", unsafe_allow_html=True)
            if st.button(f"Select {title}", key=f"prob_sel_{key}"):
                set_build_state("problem_category", key)
                st.rerun()

    st.divider()

    # Dataset selector
    ds_name = st.selectbox(
        "Select dataset",
        list(datasets.keys()),
        index=list(datasets.keys()).index(st.session_state.get("current_dataset") or list(datasets.keys())[0]),
    )

    df = datasets[ds_name]
    st.markdown(f"**Preview** — {len(df):,} rows × {len(df.columns)} columns")
    st.dataframe(df.head(20), use_container_width=True, height=250)

    if st.button("Next: Configure Model →", key="step0_next"):
        if not model_name_input.strip():
            st.error("Please enter a model name.")
        else:
            set_build_state("model_name", model_name_input.strip())
            set_build_state("dataset_name", ds_name)
            set_build_state("step", 1)
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 – BUILD CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
elif current_step == 1:
    ds_name = bs.get("dataset_name")
    df = datasets.get(ds_name)
    model_name = bs.get("model_name", "My Model")

    if df is None:
        st.error("Dataset not found. Please go back.")
        if st.button("← Back"):
            set_build_state("step", 0)
            st.rerun()
        st.stop()

    profile = st.session_state.get("dataset_profiles", {}).get(ds_name, profile_dataset(df))

    st.markdown(f"### Configure: **{model_name}**")

    # Left column: target + model type | Right: actions
    left, right = st.columns([3, 1.5])

    with left:
        # Target column selector
        st.markdown("**Select a column to predict**")
        st.markdown("<p style='color:#94A3B8;font-size:0.82rem;'>Choose the target column. The model will predict values for this column.</p>", unsafe_allow_html=True)

        target_col = st.selectbox(
            "Target column",
            df.columns.tolist(),
            index=df.columns.tolist().index(bs.get("target_col")) if bs.get("target_col") in df.columns else 0,
            key="target_col_select",
        )
        set_build_state("target_col", target_col)

        # Value distribution
        fig_dist = target_distribution_chart(df[target_col], target_col)
        st.plotly_chart(fig_dist, use_container_width=True, key="target_dist_chart")

    with right:
        # Auto-detected problem type
        auto_pt = detect_problem_type(df, target_col)
        pt_label = PROBLEM_TYPE_LABELS.get(auto_pt, auto_pt.title())
        st.markdown("**Model type**")
        st.markdown("<p style='color:#94A3B8;font-size:0.82rem;'>AutoML Studio automatically recommends the appropriate model type for your analysis.</p>", unsafe_allow_html=True)

        pt_icons = {"binary": "⚖️", "multiclass": "🏷️", "regression": "📈"}
        st.markdown(f"""
        <div style="background:rgba(124,58,237,0.1);border:1px solid #7C3AED;border-radius:10px;padding:1rem;margin-bottom:0.75rem;">
            <div style="font-size:1.5rem;margin-bottom:0.5rem;">{pt_icons.get(auto_pt, '🤖')}</div>
            <div style="font-weight:600;color:#A78BFA;font-size:0.9rem;">{pt_label}</div>
        </div>""", unsafe_allow_html=True)

        override_pt = st.selectbox("Override type", ["Auto-detect"] + list(PROBLEM_TYPE_LABELS.values()), key="pt_override")
        if override_pt != "Auto-detect":
            inv_map = {v: k for k, v in PROBLEM_TYPE_LABELS.items()}
            set_build_state("problem_type", inv_map.get(override_pt, auto_pt))
        else:
            set_build_state("problem_type", auto_pt)

        # Build type
        st.markdown("**Build mode**")
        build_type = st.radio(
            "", ["Quick Build", "Standard Build"],
            index=0 if bs.get("build_type", "quick") == "quick" else 1,
            key="build_type_radio",
        )
        set_build_state("build_type", "quick" if build_type == "Quick Build" else "standard")

        # Time limit
        use_time_limit = st.checkbox("Custom time limit", value=bs.get("use_time_limit", False))
        if use_time_limit:
            time_limit_val = st.number_input("Seconds", min_value=30, max_value=86400,
                                              value=bs.get("time_limit_val", 120), step=30)
            set_build_state("time_limit_val", time_limit_val)
            set_build_state("use_time_limit", True)
        else:
            set_build_state("use_time_limit", False)

    st.divider()

    # Feature selection
    st.markdown("### Features")
    st.markdown("<p style='color:#94A3B8;font-size:0.82rem;'>Select which columns to include as features. Uncheck columns to exclude them.</p>", unsafe_allow_html=True)

    feature_cols = [c for c in df.columns if c != target_col]
    excluded = bs.get("excluded_features", [])

    TYPE_ICONS = {"Numeric": "🔢", "Binary": "⚖️", "Categorical": "🏷️", "Datetime": "📅", "Text": "📝", "ID": "🆔"}
    TYPE_COLORS = {"Numeric": "#60A5FA", "Binary": "#A78BFA", "Categorical": "#34D399",
                   "Datetime": "#FBBF24", "Text": "#FB923C", "ID": "#94A3B8"}

    # Header
    hc = st.columns([0.5, 2.5, 1.5, 1.5, 1.5, 2])
    for h, lbl in zip(hc, ["✓", "Column", "Type", "Missing", "Unique", "Distribution"]):
        h.markdown(f"<div style='color:#64748B;font-size:0.7rem;text-transform:uppercase;font-weight:600;'>{lbl}</div>", unsafe_allow_html=True)

    new_excluded = []
    for col_name in feature_cols:
        col_info = profile.get(col_name, {})
        col_type = col_info.get("type", "Categorical")
        icon = TYPE_ICONS.get(col_type, "•")
        color = TYPE_COLORS.get(col_type, "#94A3B8")

        rc = st.columns([0.5, 2.5, 1.5, 1.5, 1.5, 2])
        with rc[0]:
            checked = st.checkbox("", value=col_name not in excluded, key=f"feat_{col_name}", label_visibility="collapsed")
        with rc[1]:
            st.markdown(f"<div style='padding:0.4rem 0;color:#F1F5F9;font-size:0.85rem;'>{col_name}</div>", unsafe_allow_html=True)
        with rc[2]:
            st.markdown(f"<div style='padding:0.4rem 0;'><span style='border:1px solid {color};color:{color};padding:2px 8px;border-radius:12px;font-size:0.72rem;'>{icon} {col_type}</span></div>", unsafe_allow_html=True)
        with rc[3]:
            pct = col_info.get("missing_pct", 0)
            miss_color = "#F87171" if pct > 10 else "#FBBF24" if pct > 0 else "#34D399"
            st.markdown(f"<div style='padding:0.4rem 0;color:{miss_color};font-size:0.85rem;'>{pct:.1f}%</div>", unsafe_allow_html=True)
        with rc[4]:
            st.markdown(f"<div style='padding:0.4rem 0;color:#94A3B8;font-size:0.85rem;'>{col_info.get('unique_count', '—')}</div>", unsafe_allow_html=True)
        with rc[5]:
            mini = column_distribution_chart(df[col_name], col_name, col_type)
            mini.update_layout(height=60, margin=dict(l=0, r=0, t=0, b=0), title="",
                               xaxis=dict(showticklabels=False, showgrid=False),
                               yaxis=dict(showticklabels=False, showgrid=False))
            st.plotly_chart(mini, use_container_width=True, key=f"build_mini_{col_name}")

        if not checked:
            new_excluded.append(col_name)
        st.markdown("<hr style='border-color:#1E293B;margin:0;'>", unsafe_allow_html=True)

    set_build_state("excluded_features", new_excluded)

    st.divider()

    # Apply transforms from recipe
    recipe = st.session_state.get("transform_recipe", {}).get(ds_name, [])
    if recipe:
        st.info(f"📋 {len(recipe)} transform(s) from the dataset recipe will be applied before training.")

    # Action buttons
    btn_col1, btn_col2, btn_col3, _ = st.columns([1, 1.5, 1.5, 3])
    with btn_col1:
        if st.button("← Back", key="step1_back"):
            set_build_state("step", 0)
            st.rerun()

    with btn_col2:
        if st.button("👁️ Preview Model", key="step1_preview"):
            with st.spinner("Running quick 30-second preview..."):
                preview_df = apply_transforms(df.copy(), recipe) if recipe else df.copy()
                result = preview_model(preview_df, target_col, bs["problem_type"])
            if "error" in result:
                st.error(f"Preview failed: {result['error']}")
            else:
                score = result.get("estimated_score", 0)
                fi = result.get("feature_importance", {})
                st.success(f"**Estimated accuracy: {score:.1%}**")
                if fi:
                    fig_fi = feature_importance_chart(fi)
                    st.plotly_chart(fig_fi, use_container_width=True, key="preview_fi")

    with btn_col3:
        build_label = "⚡ Quick Build" if bs.get("build_type") == "quick" else "🏆 Standard Build"
        if st.button(build_label, key="step1_build"):
            model_name = bs.get("model_name", "My Model")
            problem_type = bs.get("problem_type", "binary")
            build_type_val = bs.get("build_type", "quick")
            excluded_features = bs.get("excluded_features", [])
            time_limit = bs.get("time_limit_val") if bs.get("use_time_limit") else None

            # Apply transforms
            full_df = apply_transforms(df.copy(), recipe) if recipe else df.copy()
            train_df = full_df.copy()  # train_model will drop excluded_features internally

            # MLflow run
            mlflow_run = None
            try:
                mlflow_run = start_run(model_name, problem_type, build_type_val)
                log_params({
                    "model_name": model_name,
                    "problem_type": problem_type,
                    "build_type": build_type_val,
                    "time_limit": str(time_limit),
                    "target_col": target_col,
                    "n_features": len(df.columns) - len(excluded_features) - 1,
                    "n_rows": len(train_df),
                    "excluded_features": str(excluded_features),
                })
            except Exception as e:
                st.warning(f"MLflow tracking unavailable: {e}")

            progress = st.progress(0, text="Initializing AutoML...")
            status_text = st.empty()

            try:
                progress.progress(10, text="Preparing data...")
                status_text.info(f"🚀 Training **{model_name}** — {build_label}...")

                result = train_model(
                    df=train_df,
                    target_col=target_col,
                    problem_type=problem_type,
                    model_name=model_name,
                    build_type=build_type_val,
                    time_limit=time_limit,
                    excluded_features=excluded_features,
                )

                progress.progress(80, text="Evaluating models...")

                if "error" in result:
                    err_msg = result['error'] or "Unknown error occurred during training."
                    st.error(f"Training failed: {err_msg}")
                    if result.get("traceback"):
                        with st.expander("🔍 Full error traceback"):
                            st.code(result["traceback"], language="python")
                    if mlflow_run:
                        end_run()
                else:
                    # Build the evaluation dataframe (with excluded features removed, matching training)
                    predictor = result.get("predictor")
                    train_columns = result.get("train_columns", [])
                    encoded_columns = result.get("encoded_columns")  # FLAML only
                    metrics = {}
                    if predictor:
                        try:
                            # Use only the columns the model was trained on
                            if train_columns:
                                eval_df = full_df[[c for c in train_columns + [target_col] if c in full_df.columns]]
                            else:
                                eval_df = full_df
                            pred_result = predict(predictor, eval_df, problem_type, encoded_columns)
                            if "error" not in pred_result:
                                preds = pred_result["predictions"]
                                y_true = full_df[target_col]
                                y_proba = None
                                if problem_type in ("binary",) and pred_result.get("probabilities") is not None:
                                    y_proba = pred_result["probabilities"]
                                metrics = compute_metrics(y_true, preds, problem_type, y_proba)
                                result["eval_metrics"] = metrics
                                result["train_predictions"] = preds
                                result["y_true"] = y_true
                                if y_proba is not None:
                                    result["y_proba"] = y_proba
                            else:
                                st.warning(f"Post-training evaluation skipped: {pred_result['error']}")
                        except Exception as e:
                            st.warning(f"Metric computation skipped: {e}")

                    if mlflow_run:
                        try:
                            log_metrics(metrics)
                            if result.get("predictor_path"):
                                log_model_artifact(result["predictor_path"])
                            # Log feature importance chart
                            if result.get("feature_importance"):
                                fig_fi_log = feature_importance_chart(result["feature_importance"])
                                log_figure(fig_fi_log, "feature_importance")
                            end_run()
                        except Exception:
                            pass

                    # Store in session
                    if "models" not in st.session_state:
                        st.session_state["models"] = {}
                    st.session_state["models"][model_name] = result
                    st.session_state["current_model"] = model_name

                    progress.progress(100, text="Training complete!")
                    status_text.success(f"✅ **{model_name}** trained successfully!")
                    time.sleep(0.5)

                    set_build_state("step", 2)
                    st.rerun()

            except Exception as e:
                import traceback as _tb
                progress.progress(0)
                err_msg = str(e) or repr(e) or "Unexpected error during training."
                st.error(f"Unexpected error: {err_msg}")
                with st.expander("🔍 Full error traceback"):
                    st.code(_tb.format_exc(), language="python")
                if mlflow_run:
                    try:
                        end_run()
                    except Exception:
                        pass

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 – ANALYZE
# ══════════════════════════════════════════════════════════════════════════════
elif current_step == 2:
    model_name = st.session_state.get("current_model") or bs.get("model_name")
    model_result = st.session_state.get("models", {}).get(model_name)

    if not model_result:
        st.error("No model found. Please train a model first.")
        if st.button("← Back to Build"):
            set_build_state("step", 1)
            st.rerun()
        st.stop()

    meta = model_result.get("metadata", {})
    problem_type = meta.get("problem_type", bs.get("problem_type", "binary"))
    target_col = meta.get("target_col", bs.get("target_col", ""))
    metrics = model_result.get("eval_metrics", {})
    fi = model_result.get("feature_importance", {})
    leaderboard_records = model_result.get("leaderboard", [])

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(f"### {model_name}")

    # Status bar
    st.markdown(f"""
    <div style="display:flex;gap:1rem;margin-bottom:1rem;flex-wrap:wrap;">
        <span style="color:#64748B;font-size:0.8rem;">Problem: <span style="color:#A78BFA;">{PROBLEM_TYPE_LABELS.get(problem_type,'—')}</span></span>
        <span style="color:#64748B;font-size:0.8rem;">Target: <span style="color:#60A5FA;">{target_col}</span></span>
        <span style="color:#64748B;font-size:0.8rem;">Backend: <span style="color:#34D399;">{meta.get('backend','AutoGluon').upper()}</span></span>
        <span style="color:#64748B;font-size:0.8rem;">Fit time: <span style="color:#FBBF24;">{model_result.get('fit_time',0):.1f}s</span></span>
    </div>
    """, unsafe_allow_html=True)

    # Primary metric hero
    pm = primary_metric(problem_type)
    pm_label = primary_metric_label(problem_type)
    pm_val = metrics.get(pm, abs(model_result.get("best_score", 0)))
    pm_display = format_metric_display(pm, pm_val, problem_type)

    # Action bar
    action_col, _ = st.columns([1, 3])
    with action_col:
        if st.button("🔮 Make Predictions →", key="analyze_to_predict"):
            set_build_state("step", 3)
            st.rerun()

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(124,58,237,0.1),rgba(37,99,235,0.1));border:1px solid rgba(124,58,237,0.3);border-radius:16px;padding:2rem;margin-bottom:1.5rem;">
        <div style="display:flex;align-items:center;gap:2rem;flex-wrap:wrap;">
            <div>
                <div style="font-size:4rem;font-weight:800;background:linear-gradient(135deg,#A78BFA,#60A5FA);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1;">{pm_display}</div>
                <div style="color:#94A3B8;font-size:0.9rem;margin-top:0.5rem;">The model predicts <b style="color:#F1F5F9">{target_col}</b> with <b style="color:#A78BFA">{pm_display}</b> {pm_label}.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metric cards
    if metrics:
        metric_cols = st.columns(len(metrics))
        for col, (mname, mval) in zip(metric_cols, metrics.items()):
            col.metric(mname.upper().replace("_", " "), format_metric_display(mname, mval, problem_type))

    st.divider()

    # Sub-tabs
    tab_overview, tab_scoring, tab_leaderboard = st.tabs(["📊 Overview", "🎯 Scoring", "🏆 Leaderboard"])

    # ── Overview ──────────────────────────────────────────────────────────────
    with tab_overview:
        if fi:
            st.markdown("#### Column Impact")
            fig_fi = feature_importance_chart(fi)
            st.plotly_chart(fig_fi, use_container_width=True, key="analyze_fi")
        else:
            st.info("Feature importance not available for this model.")

    # ── Scoring ───────────────────────────────────────────────────────────────
    with tab_scoring:
        y_true = model_result.get("y_true")
        y_pred = model_result.get("train_predictions")
        y_proba = model_result.get("y_proba")

        if y_true is not None and y_pred is not None:
            if problem_type in ("binary", "multiclass"):
                col_cm, col_roc = st.columns(2)
                with col_cm:
                    try:
                        cm, labels = get_confusion_matrix(y_true, y_pred)
                        fig_cm = confusion_matrix_chart(cm, labels)
                        st.plotly_chart(fig_cm, use_container_width=True, key="analyze_cm")
                    except Exception as e:
                        st.info(f"Confusion matrix: {e}")

                with col_roc:
                    if problem_type == "binary" and y_proba is not None:
                        try:
                            y_p = np.array(y_proba)
                            if y_p.ndim == 2:
                                y_p = y_p[:, 1]
                            fpr, tpr, auc = get_roc_curve(y_true, y_p)
                            fig_roc = roc_curve_chart(fpr, tpr, auc)
                            st.plotly_chart(fig_roc, use_container_width=True, key="analyze_roc")

                            prec, rec, ap = get_precision_recall(y_true, y_p)
                            fig_pr = precision_recall_chart(prec, rec, ap)
                            st.plotly_chart(fig_pr, use_container_width=True, key="analyze_pr")
                        except Exception as e:
                            st.info(f"ROC curve: {e}")

            elif problem_type == "regression":
                try:
                    fig_res = residuals_chart(y_true, y_pred)
                    st.plotly_chart(fig_res, use_container_width=True, key="analyze_residuals")
                except Exception as e:
                    st.info(f"Residuals chart: {e}")
        else:
            st.info("Training data predictions not available for detailed scoring.")

    # ── Leaderboard ───────────────────────────────────────────────────────────
    with tab_leaderboard:
        if leaderboard_records:
            lb_df = pd.DataFrame(leaderboard_records)
            # Display human-readable columns
            display_cols = ["model", "score_val", "pred_time_val", "fit_time"]
            display_cols = [c for c in display_cols if c in lb_df.columns]
            if display_cols:
                st.dataframe(lb_df[display_cols].rename(columns={
                    "model": "Algorithm",
                    "score_val": "Validation Score",
                    "pred_time_val": "Predict Time (s)",
                    "fit_time": "Fit Time (s)",
                }).round(4), use_container_width=True, hide_index=True)

                # Chart
                metric_col = "score_val" if "score_val" in lb_df.columns else lb_df.columns[1]
                lb_chart_df = lb_df[["model", metric_col]].copy() if "model" in lb_df.columns else lb_df
                lb_chart_df.columns = ["model", metric_col] if len(lb_chart_df.columns) == 2 else lb_chart_df.columns
                fig_lb = leaderboard_chart(lb_chart_df, metric_col)
                st.plotly_chart(fig_lb, use_container_width=True, key="analyze_lb")
        else:
            st.info("Leaderboard not available. This may be a FLAML model.")

    st.divider()
    # Navigation
    nb1, nb2, _ = st.columns([1, 1, 4])
    with nb1:
        if st.button("← Retrain", key="analyze_retrain"):
            set_build_state("step", 1)
            st.rerun()
    with nb2:
        if st.button("🔮 Predict →", key="analyze_predict"):
            set_build_state("step", 3)
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 – PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif current_step == 3:
    model_name = st.session_state.get("current_model") or bs.get("model_name")
    model_result = st.session_state.get("models", {}).get(model_name)

    if not model_result or "error" in model_result:
        st.error("No trained model available. Please train a model first.")
        if st.button("← Back to Build"):
            set_build_state("step", 1)
            st.rerun()
        st.stop()

    predictor = model_result.get("predictor")
    meta = model_result.get("metadata", {})
    problem_type = meta.get("problem_type", "binary")
    target_col = meta.get("target_col", "")
    ds_name = bs.get("dataset_name")
    train_df = datasets.get(ds_name, pd.DataFrame())

    if predictor is None:
        st.error("Predictor not loaded in session. Try retraining the model.")
        if st.button("← Back"):
            set_build_state("step", 1)
            st.rerun()
        st.stop()

    st.markdown(f"### 🔮 Predict — *{model_name}*")

    tab_single, tab_batch = st.tabs(["🔍 Single Prediction", "📦 Batch Prediction"])

    # ── Single Prediction ──────────────────────────────────────────────────────
    with tab_single:
        st.markdown("#### Enter feature values to get a prediction")
        st.markdown("<p style='color:#94A3B8;font-size:0.85rem;'>Fill in the fields below with your input values.</p>", unsafe_allow_html=True)

        feature_cols = [c for c in train_df.columns if c != target_col and
                        c not in bs.get("excluded_features", [])]

        input_data = {}
        profile = st.session_state.get("dataset_profiles", {}).get(ds_name, {})

        # Render input fields dynamically per data type
        ncols = 3
        rows = [feature_cols[i:i+ncols] for i in range(0, len(feature_cols), ncols)]
        for row_cols in rows:
            form_cols = st.columns(ncols)
            for fc, col_name in zip(form_cols, row_cols):
                col_info = profile.get(col_name, {})
                col_type = col_info.get("type", "Categorical")
                with fc:
                    if col_type == "Numeric":
                        mean_val = col_info.get("mean", 0)
                        input_data[col_name] = st.number_input(col_name, value=float(mean_val), key=f"single_{col_name}")
                    elif col_type == "Binary":
                        vc = col_info.get("value_counts", {})
                        opts = list(vc.keys()) if vc else ["0", "1"]
                        input_data[col_name] = st.selectbox(col_name, opts, key=f"single_{col_name}")
                    elif col_type == "Categorical":
                        vc = col_info.get("value_counts", {})
                        opts = list(vc.keys()) if vc else [""]
                        input_data[col_name] = st.selectbox(col_name, opts, key=f"single_{col_name}")
                    elif col_type == "Datetime":
                        input_data[col_name] = st.text_input(col_name, placeholder="YYYY-MM-DD", key=f"single_{col_name}")
                    else:
                        input_data[col_name] = st.text_input(col_name, key=f"single_{col_name}")

        if st.button("🔮 Generate Prediction", key="single_pred_btn"):
            with st.spinner("Generating prediction..."):
                try:
                    input_df = pd.DataFrame([input_data])
                    encoded_columns = model_result.get("encoded_columns")
                    result = predict(predictor, input_df, problem_type, encoded_columns)
                    if "error" in result:
                        st.error(f"Prediction error: {result['error']}")
                    else:
                        pred_val = result["predictions"].iloc[0]
                        probas = result.get("probabilities")

                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,rgba(124,58,237,0.15),rgba(37,99,235,0.15));
                            border:1px solid rgba(124,58,237,0.4);border-radius:16px;padding:2rem;margin-top:1rem;text-align:center;">
                            <div style="color:#94A3B8;font-size:0.85rem;margin-bottom:0.5rem;">Predicted {target_col}</div>
                            <div style="font-size:3rem;font-weight:800;background:linear-gradient(135deg,#A78BFA,#60A5FA);
                                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">{pred_val}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        if probas is not None:
                            prob_df = pd.DataFrame(probas).iloc[0]
                            st.markdown("**Class Probabilities:**")
                            for class_label, prob in prob_df.items():
                                st.markdown(f"""
                                <div style="margin-bottom:0.5rem;">
                                    <div style="display:flex;justify-content:space-between;color:#F1F5F9;font-size:0.85rem;margin-bottom:3px;">
                                        <span>{class_label}</span><span>{prob:.1%}</span>
                                    </div>
                                    <div style="background:#1E293B;border-radius:4px;height:8px;">
                                        <div style="background:linear-gradient(90deg,#7C3AED,#2563EB);height:8px;border-radius:4px;width:{prob*100:.1f}%;"></div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    # ── Batch Prediction ───────────────────────────────────────────────────────
    with tab_batch:
        st.markdown("#### Upload a dataset to generate batch predictions")
        st.markdown("<p style='color:#94A3B8;font-size:0.85rem;'>Upload a CSV file with the same columns as the training data (without the target column).</p>", unsafe_allow_html=True)

        batch_file = st.file_uploader("Upload batch data (CSV)", type=["csv", "xlsx", "parquet"], key="batch_upload")

        if batch_file:
            from src.data_utils import load_dataset
            batch_df = load_dataset(batch_file, filename=batch_file.name)
            st.markdown(f"**Preview** — {len(batch_df):,} rows × {len(batch_df.columns)} cols")
            st.dataframe(batch_df.head(10), use_container_width=True, height=200)

            if st.button("🚀 Generate Batch Predictions", key="batch_pred_btn"):
                with st.spinner(f"Generating predictions for {len(batch_df):,} rows..."):
                    try:
                        encoded_columns = model_result.get("encoded_columns")
                        result = predict(predictor, batch_df, problem_type, encoded_columns)
                        if "error" in result:
                            st.error(f"Batch prediction error: {result['error']}")
                        else:
                            output_df = batch_df.copy()
                            output_df[f"predicted_{target_col}"] = result["predictions"].values

                            if result.get("probabilities") is not None:
                                probas_df = pd.DataFrame(result["probabilities"])
                                probas_df.columns = [f"prob_{c}" for c in probas_df.columns]
                                output_df = pd.concat([output_df, probas_df.reset_index(drop=True)], axis=1)

                            st.success(f"✅ Predictions generated for {len(output_df):,} rows!")
                            st.dataframe(output_df.head(50), use_container_width=True, height=300)

                            # Download
                            csv_bytes = output_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "⬇️ Download Predictions (CSV)",
                                data=csv_bytes,
                                file_name=f"{model_name}_predictions.csv",
                                mime="text/csv",
                            )

                            # Store in session for history
                            if "prediction_history" not in st.session_state:
                                st.session_state["prediction_history"] = []
                            st.session_state["prediction_history"].append({
                                "model": model_name,
                                "type": "batch",
                                "rows": len(output_df),
                                "df": output_df,
                            })

                    except Exception as e:
                        st.error(f"Batch prediction failed: {e}")

    # Navigation footer
    st.divider()
    if st.button("← Back to Analysis", key="predict_back"):
        set_build_state("step", 2)
        st.rerun()
