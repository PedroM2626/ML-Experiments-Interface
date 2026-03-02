"""
Upload page — dataset upload, profiling, and training configuration.
"""

import streamlit as st
import pandas as pd
import io
import os
from src.utils import state
from src.utils.data_profiler import profile_dataset, infer_task_type
from src.automl.evaluator import OPTIMIZATION_METRICS
from src.automl.hyperopt import CLASSIFIERS, REGRESSORS


def render():
    state.init_state()

    # Hero Section
    st.markdown("""
    <div style="text-align:center;padding:40px 0 20px 0;">
        <div style="display:inline-flex;align-items:center;gap:14px;margin-bottom:12px;">
            <span style="font-size:48px;">🤖</span>
            <div style="text-align:left;">
                <h1 style="margin:0;font-size:34px;font-weight:900;background:linear-gradient(135deg,#8B5CF6,#06B6D4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">AutoML Studio</h1>
                <p style="margin:0;color:#64748b;font-size:14px;">Automated Machine Learning · Watson AutoAI style</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_config = st.columns([1, 1], gap="large")

    # ── Upload section ──────────────────────────────────────────────────────────
    with col_upload:
        st.markdown("### 📁 Dataset")
        uploaded = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            key="file_uploader",
            label_visibility="collapsed",
        )

        df = None
        if uploaded is not None:
            try:
                if uploaded.name.endswith(".csv"):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded)
                state.set("df", df)
                state.set("dataset_name", uploaded.name)
                st.success(f"✅ Loaded **{uploaded.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")
            except Exception as e:
                st.error(f"Failed to read file: {e}")

        elif state.get("df") is not None:
            df = state.get("df")

        # Sample dataset shortcuts
        st.markdown("<p style='color:#64748b;font-size:12px;margin-top:8px;'>Or try a sample dataset:</p>", unsafe_allow_html=True)
        sample_cols = st.columns(3)
        if sample_cols[0].button("🌸 Iris", use_container_width=True):
            from sklearn.datasets import load_iris
            iris = load_iris(as_frame=True)
            df = iris.frame
            df["target"] = iris.target_names[iris.target]
            state.set("df", df)
            state.set("dataset_name", "iris.csv")
            state.set("target_column", "target")
            st.rerun()

        if sample_cols[1].button("🚢 Titanic", use_container_width=True):
            url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
            try:
                df = pd.read_csv(url)
                state.set("df", df)
                state.set("dataset_name", "titanic.csv")
                state.set("target_column", "Survived")
                st.rerun()
            except Exception:
                st.warning("Could not load Titanic dataset (check internet).")

        if sample_cols[2].button("🏠 Boston", use_container_width=True):
            from sklearn.datasets import fetch_california_housing
            housing = fetch_california_housing(as_frame=True)
            df = housing.frame
            state.set("df", df)
            state.set("dataset_name", "california_housing.csv")
            state.set("target_column", "MedHouseVal")
            st.rerun()

        # Dataset preview & profile
        if df is not None:
            with st.expander("📊 Dataset Preview & Profile", expanded=True):
                st.dataframe(df.head(6), use_container_width=True)
                _render_profile_badges(df, state.get("target_column"))

    # ── Config section ─────────────────────────────────────────────────────────
    with col_config:
        st.markdown("### ⚙️ Experiment Settings")
        df = state.get("df")

        if df is None:
            st.info("👈 Upload a dataset first to configure the experiment.")
            return

        # Target column
        target = st.selectbox(
            "🎯 Prediction column (target)",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(state.get("target_column")) if state.get("target_column") in df.columns else 0,
        )
        state.set("target_column", target)

        # Auto-infer task type
        inferred_task = infer_task_type(df[target])
        task_type = st.radio(
            "🔰 Task type",
            options=["classification", "regression"],
            index=0 if inferred_task == "classification" else 1,
            horizontal=True,
        )
        state.set("task_type", task_type)

        st.divider()

        # Optimization metric
        metric_opts = list(OPTIMIZATION_METRICS[task_type].keys())
        opt_metric_label = st.selectbox("📈 Optimization metric", metric_opts)
        opt_metric_key = OPTIMIZATION_METRICS[task_type][opt_metric_label]
        state.set("optimization_metric", opt_metric_key)

        col_a, col_b = st.columns(2)
        with col_a:
            test_size = st.slider("📦 Holdout %", 5, 30, int(state.get("test_size", 0.1) * 100), step=5)
            state.set("test_size", test_size / 100)
            n_folds = st.slider("🔄 CV Folds", 2, 10, state.get("n_folds", 3))
            state.set("n_folds", n_folds)

        with col_b:
            max_algos = st.slider("🤖 Max algorithms", 2, 8, state.get("max_algorithms", 4))
            state.set("max_algorithms", max_algos)
            hpo_trials = st.slider("🔬 HPO trials", 5, 30, state.get("hpo_trials", 10))
            state.set("hpo_trials", hpo_trials)

        st.divider()

        # MLflow settings
        with st.expander("📦 MLflow Tracking"):
            mlflow_uri = st.text_input(
                "Tracking URI",
                value=state.get("mlflow_tracking_uri", "mlruns"),
            )
            state.set("mlflow_tracking_uri", mlflow_uri)
            mlflow_exp = st.text_input(
                "Experiment Name",
                value=state.get("mlflow_experiment_name", "AutoML_Experiment"),
            )
            state.set("mlflow_experiment_name", mlflow_exp)

        st.divider()

        # Launch button
        can_train = df is not None and target is not None
        if st.button(
            "🚀 Start AutoML Training",
            use_container_width=True,
            disabled=not can_train,
            type="primary",
        ):
            state.reset_training_state()
            state.set("current_page", "training")
            st.rerun()


def _render_profile_badges(df: pd.DataFrame, target: str = None):
    """Render compact profile badges."""
    n_rows, n_cols = df.shape
    missing_pct = round(df.isnull().sum().sum() / (n_rows * n_cols) * 100, 1)
    numeric = df.select_dtypes(include="number").shape[1]
    categorical = n_cols - numeric

    badges = [
        ("📏 Rows", f"{n_rows:,}"),
        ("📋 Columns", f"{n_cols}"),
        ("🔢 Numeric", f"{numeric}"),
        ("🔤 Categorical", f"{categorical}"),
        ("❓ Missing", f"{missing_pct}%"),
    ]
    if target and target in df.columns:
        n_unique = df[target].nunique()
        badges.append(("🎯 Classes", str(n_unique)))

    badge_html = "".join([
        f'<div style="background:#0f1929;border:1px solid #1e293b;border-radius:8px;padding:8px 14px;text-align:center;">'
        f'<div style="font-size:18px;">{icon}</div>'
        f'<div style="font-weight:700;font-size:15px;color:#e2e8f0;">{val}</div>'
        f'<div style="font-size:10px;color:#64748b;">{label.split()[-1]}</div>'
        f'</div>'
        for icon_label, val in [(b[0].split()[0], b[1]) for b in badges]
        for icon, label in [(icon_label, [b[0] for b in badges if b[0].startswith(icon_label)][0])]
    ])

    # Simpler approach: columns
    cols = st.columns(len(badges))
    for col, (label, val) in zip(cols, badges):
        icon, lbl = label.split(" ", 1)
        col.markdown(
            f"<div style='text-align:center;background:#0f1929;border:1px solid #1e293b;border-radius:8px;padding:8px 4px;'>"
            f"<div style='font-size:20px;'>{icon}</div>"
            f"<div style='font-weight:700;color:#e2e8f0;font-size:14px;'>{val}</div>"
            f"<div style='font-size:10px;color:#64748b;'>{lbl}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
