"""
AutoML Studio – Page: Models
List all trained models, create new ones, and manage the model registry.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from config import APP_ICON
from src.automl_engine import get_saved_models
from src.mlflow_utils import get_all_runs
from src.ui_utils import load_global_css

st.set_page_config(page_title="Models – AutoML Studio", page_icon="🤖", layout="wide")

# ── Global CSS ─────────────────────────────────────────────────────────────────
load_global_css()

st.markdown("""
<h1 style="font-size:1.8rem;font-weight:800;color:#F1F5F9;margin-bottom:0.25rem;">
    🤖 My Models
</h1>
<p style="color:#94A3B8;font-size:0.9rem;margin-top:0;">
    View and manage your trained machine learning models.
</p>
""", unsafe_allow_html=True)

# ── Create New Model CTA ───────────────────────────────────────────────────────
col_cta, _ = st.columns([2, 5])
with col_cta:
    if st.button("✨ Create New Model", key="cta_new_model"):
        st.switch_page("pages/Build.py")

st.divider()

# ── Session Models ─────────────────────────────────────────────────────────────
models = st.session_state.get("models", {})

tab_session, tab_mlflow = st.tabs(["🟢 Session Models", "📦 MLflow Registry"])

# ── Session Models Tab ──────────────────────────────────────────────────────────
with tab_session:
    if not models:
        st.markdown("""
        <div style="background:#1E293B;border:1px dashed #334155;border-radius:12px;padding:3rem;text-align:center;margin-top:1rem;">
            <div style="font-size:3rem;margin-bottom:1rem;">🤖</div>
            <p style="color:#94A3B8;font-size:1rem;">No models trained yet. Go to <b>🔨 Build</b> to train your first model.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Problem type filters
        all_types = list(set(
            m.get("metadata", {}).get("problem_type", "unknown")
            for m in models.values()
        ))
        filter_type = st.selectbox("Filter by problem type", ["All"] + all_types)

        for name, m in models.items():
            meta = m.get("metadata", {})
            pt = meta.get("problem_type", "unknown")
            if filter_type != "All" and pt != filter_type:
                continue

            has_error = "error" in m
            status_badge = (
                '<span style="background:rgba(220,38,38,0.15);color:#F87171;padding:3px 10px;border-radius:20px;font-size:0.72rem;font-weight:600;">❌ Error</span>'
                if has_error else
                '<span style="background:rgba(5,150,105,0.15);color:#34D399;padding:3px 10px;border-radius:20px;font-size:0.72rem;font-weight:600;">✅ Ready</span>'
            )

            problem_labels = {"binary": "Binary Classification", "multiclass": "Multi-class", "regression": "Regression"}
            prob_label = problem_labels.get(pt, pt.title())

            backend = meta.get("backend", "autogluon")
            fit_time = round(m.get("fit_time", 0), 1)
            best_score = abs(m.get("best_score", 0))
            best_model_str = m.get("best_model", "—")

            with st.container(border=True):
                col_title, col_status = st.columns([4, 1])
                with col_title:
                    st.markdown(f"**{name}**")
                    st.caption(f"{prob_label} • {backend.upper()}")
                with col_status:
                    if has_error:
                        st.markdown('<span class="status-badge badge-error">❌ Error</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="status-badge badge-ready">✅ Ready</span>', unsafe_allow_html=True)

                if has_error:
                    st.error(m["error"])
                else:
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Best Score", f"{best_score:.4f}")
                    mc2.metric("Best Algorithm", (best_model_str[:25] + "...") if len(best_model_str) > 25 else best_model_str)
                    mc3.metric("Fit Time", f"{fit_time}s")
                    mc4.metric("Rows", f"{meta.get('n_rows', '—'):,}" if meta.get('n_rows') else "—")

                btn_c1, btn_c2, _ = st.columns([1, 1, 5])
                with btn_c1:
                    if st.button("📈 Analyze", key=f"analyze_{name}"):
                        st.session_state["current_model"] = name
                        st.switch_page("pages/Build.py")
                with btn_c2:
                    if st.button("🗑️ Delete", key=f"del_model_{name}"):
                        del st.session_state["models"][name]
                        if st.session_state.get("current_model") == name:
                            st.session_state["current_model"] = None
                        st.rerun()

# ── MLflow Tab ──────────────────────────────────────────────────────────────────
with tab_mlflow:
    st.markdown("Recent experiment runs tracked in MLflow:")

    with st.spinner("Loading MLflow runs..."):
        runs_df = get_all_runs()

    if runs_df.empty:
        st.info("No MLflow runs found. Train a model first — all runs are automatically tracked.")
    else:
        # Select relevant columns to display
        display_cols = [c for c in [
            "run_name", "model_name", "problem_type", "build_type",
            "accuracy", "roc_auc", "rmse", "r2",
            "fit_time", "status", "start_time",
        ] if c in runs_df.columns]

        st.dataframe(
            runs_df[display_cols].rename(columns={
                "run_name": "Run Name",
                "model_name": "Model",
                "problem_type": "Problem",
                "build_type": "Build Type",
                "start_time": "Started At",
            }),
            use_container_width=True,
            hide_index=True,
        )

        mlflow_url = "http://localhost:5000"
        st.markdown(f"🔗 [Open MLflow UI]({mlflow_url}) to explore runs in detail.")
