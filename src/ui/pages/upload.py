"""
Upload page — dataset upload, profiling, and training configuration.
"""

import streamlit as st
import pandas as pd
import numpy as np
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
                <p style="margin:0;color:#64748b;font-size:14px;">Automated Machine Learning · Azure ML · SageMaker · Vertex AI · WatsonX style</p>
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
        sample_cols = st.columns(4)
        if sample_cols[0].button("🌸 Iris", use_container_width=True):
            from sklearn.datasets import load_iris
            iris = load_iris(as_frame=True)
            df = iris.frame.copy()
            df["target"] = iris.target_names[iris.target]
            state.set("df", df); state.set("dataset_name", "iris.csv")
            state.set("target_column", "target"); state.set("task_type", "classification")
            st.rerun()

        if sample_cols[1].button("🚢 Titanic", use_container_width=True):
            url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
            try:
                df = pd.read_csv(url)
                state.set("df", df); state.set("dataset_name", "titanic.csv")
                state.set("target_column", "Survived"); state.set("task_type", "classification")
                st.rerun()
            except Exception:
                st.warning("Could not load Titanic dataset.")

        if sample_cols[2].button("🏠 Housing", use_container_width=True):
            from sklearn.datasets import fetch_california_housing
            housing = fetch_california_housing(as_frame=True)
            df = housing.frame
            state.set("df", df); state.set("dataset_name", "california_housing.csv")
            state.set("target_column", "MedHouseVal"); state.set("task_type", "regression")
            st.rerun()

        if sample_cols[3].button("📈 Air Passengers", use_container_width=True):
            # Classic monthly airline passengers time series
            import io, urllib.request
            url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
            try:
                df = pd.read_csv(url, header=0, names=["Month", "Passengers"])
                state.set("df", df); state.set("dataset_name", "air_passengers.csv")
                state.set("target_column", "Passengers")
                state.set("task_type", "time_series")
                state.set("date_column", "Month"); st.rerun()
            except Exception:
                # Fallback: generate synthetic TS
                dates = pd.date_range("2010-01-01", periods=120, freq="MS")
                vals = (100 + np.arange(120) * 0.5
                        + 20 * np.sin(2 * np.pi * np.arange(120) / 12)
                        + np.random.normal(0, 5, 120))
                df = pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Value": vals.round(1)})
                state.set("df", df); state.set("dataset_name", "synthetic_ts.csv")
                state.set("target_column", "Value")
                state.set("task_type", "time_series")
                state.set("date_column", "Date"); st.rerun()

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

        # Experiment name
        exp_name = st.text_input(
            "🏷️ Experiment name",
            value=state.get("dataset_name", "Experiment").replace(".csv", "").replace(".xlsx", ""),
        )

        # Target column
        target = st.selectbox(
            "🎯 Prediction column (target)",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(state.get("target_column"))
            if state.get("target_column") in df.columns else 0,
        )
        state.set("target_column", target)

        # Task type (now includes time_series and text_classification)
        inferred_task = infer_task_type(df[target])
        task_opts = ["classification", "regression", "time_series", "text_classification"]
        cur_task = state.get("task_type", inferred_task)
        if cur_task not in task_opts:
            cur_task = inferred_task
        task_type = st.radio(
            "🔰 Task type",
            options=task_opts,
            index=task_opts.index(cur_task),
            horizontal=True,
        )
        state.set("task_type", task_type)

        # ── Time Series specific settings ──────────────────────────────────────
        if task_type == "time_series":
            st.markdown("<div style='background:#0f1929;border-left:3px solid #8B5CF6;padding:10px 14px;border-radius:6px;margin:8px 0;'>"
                        "<span style='color:#a78bfa;font-weight:600;font-size:12px;'>📅 Time Series Configuration</span></div>",
                        unsafe_allow_html=True)

            date_cols = ["(none)"] + df.columns.tolist()
            cur_date_col = state.get("date_column", "(none)")
            date_col_idx = date_cols.index(cur_date_col) if cur_date_col in date_cols else 0
            date_col = st.selectbox("📅 Date/Time column", date_cols, index=date_col_idx)
            state.set("date_column", date_col if date_col != "(none)" else "")

            freq_opts = {"Daily (D)": "D", "Weekly (W)": "W", "Monthly (M)": "M",
                         "Quarterly (Q)": "Q", "Hourly (H)": "H"}
            freq_label = st.selectbox("📡 Frequency", list(freq_opts.keys()), index=2)
            state.set("freq", freq_opts[freq_label])

            horizon = st.slider("🔮 Forecast horizon (steps)", 1, 60,
                                state.get("horizon", 12))
            state.set("horizon", horizon)

        # ── Text Classification specific settings ──────────────────────────
        if task_type == "text_classification":
            st.markdown("<div style='background:#0f1929;border-left:3px solid #06B6D4;padding:10px 14px;border-radius:6px;margin:8px 0;'>"
                        "<span style='color:#67e8f9;font-weight:600;font-size:12px;'>💬 Text Classification Configuration</span></div>",
                        unsafe_allow_html=True)

            text_cols = ["(none)"] + df.select_dtypes(include="object").columns.tolist()
            cur_text_col = state.get("text_column", "(none)")
            text_col_idx = text_cols.index(cur_text_col) if cur_text_col in text_cols else 0
            text_col = st.selectbox("💬 Text column (raw text)", text_cols, index=text_col_idx)
            state.set("text_column", text_col if text_col != "(none)" else "")

            ngram_opts = {"Unigrams (1,1)": (1, 1), "Bigrams (1,2)": (1, 2), "Trigrams (1,3)": (1, 3)}
            ngram_label = st.selectbox("📝 N-gram range", list(ngram_opts.keys()), index=1)
            state.set("ngram_range", ngram_opts[ngram_label])

            max_feat = st.select_slider(
                "📊 TF-IDF max features",
                options=[5000, 10000, 20000, 50000, 100000],
                value=state.get("tfidf_max_features", 20000),
            )
            state.set("tfidf_max_features", max_feat)

        st.divider()

        # Optimization metric
        metric_opts = list(OPTIMIZATION_METRICS[task_type].keys())
        cur_metric = state.get("optimization_metric", metric_opts[0])
        # reset metric if task changed
        if cur_metric not in OPTIMIZATION_METRICS[task_type].values():
            cur_metric = list(OPTIMIZATION_METRICS[task_type].values())[0]
        opt_metric_label = st.selectbox("📈 Optimization metric", metric_opts,
                                        index=metric_opts.index(
                                            next((k for k, v in OPTIMIZATION_METRICS[task_type].items()
                                                  if v == cur_metric), metric_opts[0])))
        opt_metric_key = OPTIMIZATION_METRICS[task_type][opt_metric_label]
        state.set("optimization_metric", opt_metric_key)
        
        # ── Config Mode ──────────────────────────────────────────────────────────
        st.markdown("<div style='margin-bottom:-10px; color:#64748b; font-size:13px; font-weight:600;'>🛠️ CONFIGURATION MODE</div>", unsafe_allow_html=True)
        config_mode = st.radio("config_mode_selector", ["🚀 Auto-Pilot", "⚙️ Manual"], 
                               label_visibility="collapsed", horizontal=True)
        is_manual = "Manual" in config_mode
        state.set("config_mode", "manual" if is_manual else "auto")

        if is_manual:
            from src.automl.time_series import get_ts_algorithm_registry
            from src.automl.hyperopt import get_algorithm_registry
            from src.automl.nlp_engine import get_nlp_algorithm_registry

            if task_type == "text_classification":
                reg = get_nlp_algorithm_registry()
            elif task_type == "time_series":
                reg = get_ts_algorithm_registry()
            else:
                reg = get_algorithm_registry(task_type)

            all_algos = list(reg.keys())
            selected_algos = st.multiselect("🤖 Select Algorithms", all_algos, default=all_algos[:4])
            state.set("selected_algos", selected_algos)

            st.caption("Only selected algorithms will be trained and optimized.")
        else:
            state.set("selected_algos", None)

        # Disable numeric sliders that don't apply to NLP
        is_nlp = task_type == "text_classification"
        col_a, col_b = st.columns(2)
        with col_a:
            test_size = st.slider("📦 Holdout %", 5, 30,
                                  int(state.get("test_size", 0.1) * 100), step=5,
                                  disabled=(task_type in ("time_series", "text_classification")))
            state.set("test_size", test_size / 100)
            n_folds = st.slider("🔄 CV Folds", 2, 10, state.get("n_folds", 3),
                                disabled=is_nlp)
            state.set("n_folds", n_folds)

        with col_b:
            max_algos = st.slider("🤖 Max algorithms", 2, 8, state.get("max_algorithms", 4))
            state.set("max_algorithms", max_algos)
            hpo_trials = st.slider("🔬 HPO trials", 3, 30, state.get("hpo_trials", 10))
            state.set("hpo_trials", hpo_trials)

        st.divider()

        # MLflow settings
        with st.expander("📦 MLflow Tracking"):
            mlflow_uri = st.text_input("Tracking URI",
                                       value=state.get("mlflow_tracking_uri", "mlruns"))
            state.set("mlflow_tracking_uri", mlflow_uri)
            mlflow_exp = st.text_input("Experiment Name",
                                       value=state.get("mlflow_experiment_name", "AutoML_Experiment"))
            state.set("mlflow_experiment_name", mlflow_exp)

        st.divider()

        # Launch button
        can_train = df is not None and target is not None
        if (task_type == "time_series" and not state.get("date_column")):
            st.info("💡 Select a date/time column for time series training.")
            can_train = False
        if (task_type == "text_classification" and not state.get("text_column")):
            st.info("💡 Select a text column for NLP training.")
            can_train = False

        if st.button(
            "🚀 Start AutoML Training",
            use_container_width=True,
            disabled=not can_train,
            type="primary",
        ):
            _launch_experiment(exp_name, df, task_type, target)



def _launch_experiment(exp_name: str, df, task_type: str, target: str):
    """Create and start an experiment via ExperimentManager, then navigate."""
    from src.utils import experiment_manager as em
    from src.tracking.mlflow_tracker import setup_mlflow

    # Setup MLflow
    setup_mlflow(
        tracking_uri=state.get("mlflow_tracking_uri", "mlruns"),
        experiment_name=state.get("mlflow_experiment_name", "AutoML_Experiment"),
    )

    if task_type == "time_series":
        from src.automl.ts_engine import TSEngineConfig
        config = TSEngineConfig(
            task_type="time_series",
            target_column=target,
            date_column=state.get("date_column", ""),
            horizon=state.get("horizon", 12),
            freq=state.get("freq", "M"),
            n_splits=state.get("n_folds", 5),
            max_algorithms=state.get("max_algorithms", 4),
            n_estimators_per_algo=state.get("n_estimators_per_algo", 2),
            hpo_trials=state.get("hpo_trials", 8),
            optimization_metric=state.get("optimization_metric", "rmse"),
            algorithms_to_include=state.get("selected_algos"),
        )
        exp_id = em.create_experiment(
            name=exp_name or f"TS - {state.get('dataset_name', 'dataset')}",
            df=df, config=config,
            dataset_name=state.get("dataset_name", ""),
            task_type="time_series",
            target_column=target,
            optimization_metric=state.get("optimization_metric", "rmse"),
        )
        # Unique export dir
        config.export_dir = os.path.join("exports", exp_id)
        em.start_ts_experiment(exp_id)

    elif task_type == "text_classification":
        from src.automl.nlp_engine import NLPConfig
        config = NLPConfig(
            text_column=state.get("text_column", ""),
            target_column=target,
            max_features=state.get("tfidf_max_features", 20_000),
            ngram_range=state.get("ngram_range", (1, 2)),
            max_algorithms=state.get("max_algorithms", 5),
            hpo_trials=state.get("hpo_trials", 8),
            optimization_metric=state.get("optimization_metric", "f1"),
            algorithms_to_include=state.get("selected_algos"),
        )
        exp_id = em.create_experiment(
            name=exp_name or f"NLP - {state.get('dataset_name', 'text_data')}",
            df=df, config=config,
            dataset_name=state.get("dataset_name", ""),
            task_type="text_classification",
            target_column=target,
            optimization_metric=state.get("optimization_metric", "f1"),
        )
        config.export_dir = os.path.join("exports", exp_id)
        config.experiment_id = exp_id
        config.name = exp_name or f"NLP - {state.get('dataset_name', 'text_data')}"
        em.start_nlp_experiment(exp_id)

    else:
        from src.automl.engine import EngineConfig
        config = EngineConfig(
            task_type=task_type,
            target_column=target,
            test_size=state.get("test_size", 0.1),
            n_folds=state.get("n_folds", 3),
            n_estimators_per_algo=state.get("n_estimators_per_algo", 2),
            max_algorithms=state.get("max_algorithms", 4),
            hpo_trials=state.get("hpo_trials", 10),
            optimization_metric=state.get("optimization_metric", "roc_auc"),
            export_dir="exports",
            algorithms_to_include=state.get("selected_algos"),
        )
        exp_id = em.create_experiment(
            name=exp_name or state.get("dataset_name", "Experiment"),
            df=df, config=config,
            dataset_name=state.get("dataset_name", ""),
            task_type=task_type,
            target_column=target,
            optimization_metric=state.get("optimization_metric", "roc_auc"),
        )
        # Unique export dir
        config.export_dir = os.path.join("exports", exp_id)
        em.start_experiment(exp_id)

    state.set("active_experiment_id", exp_id)
    state.set("current_page", "experiments")
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
