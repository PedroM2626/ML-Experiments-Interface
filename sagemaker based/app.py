"""
AutoML Studio – Main Entry Point
Streamlit multi-page app inspired by AWS SageMaker Canvas.
"""
import sys
from pathlib import Path
import streamlit as st

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent))

from config import APP_NAME, APP_ICON, APP_VERSION, COLORS
from src.ui_utils import load_global_css

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": f"**{APP_NAME}** v{APP_VERSION} — AutoML inspired by AWS SageMaker Canvas",
    },
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
load_global_css()

# ── Session State Init ─────────────────────────────────────────────────────────
defaults = {
    "datasets": {},          # name -> DataFrame
    "dataset_profiles": {},  # name -> profile dict
    "models": {},            # name -> model result dict
    "current_dataset": None,
    "current_model": None,
    "build_step": 0,
    "build_state": {},       # ephemeral build workflow state
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown(f"""
    <div style="padding: 0.5rem 0 1.5rem; text-align: center;">
        <div style="font-size: 2rem; margin-bottom: 0.25rem;">🚀</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #F1F5F9;">{APP_NAME}</div>
        <div style="font-size: 0.7rem; color: #475569;">AutoML • No-Code Machine Learning</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Navigation links hint
    st.markdown("**NAVIGATION**")
    st.markdown("Use the pages menu above to navigate between sections.")

    st.divider()

    # Quick stats
    n_datasets = len(st.session_state["datasets"])
    n_models = len(st.session_state["models"])
    ready_models = sum(
        1 for m in st.session_state["models"].values()
        if "error" not in m
    )

    st.markdown("**WORKSPACE STATS**")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Datasets", n_datasets)
    with c2:
        st.metric("Models", n_models)

    if n_models > 0:
        st.metric("Ready Models", ready_models)

    st.divider()
    st.markdown(f"<div style='font-size:0.65rem; color:#475569;'>v{APP_VERSION} • Powered by AutoGluon</div>",
                unsafe_allow_html=True)

# ── Home Page ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 2rem;">
    <h1 style="font-size: 2rem; font-weight: 800; color: #F1F5F9; margin: 0;">
        Welcome to <span style="background: linear-gradient(135deg, #A78BFA, #60A5FA);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AutoML Studio</span>
    </h1>
    <p style="color: #94A3B8; font-size: 1rem; margin-top: 0.5rem;">
        Build machine learning models without writing code — inspired by AWS SageMaker Canvas.
    </p>
</div>
""", unsafe_allow_html=True)

# Workflow steps overview
st.markdown("### How it works")
col1, col2, col3, col4 = st.columns(4)

steps = [
    ("📊", "1. Import Data", "Upload CSV, Excel, or Parquet datasets. Preview columns and statistics automatically."),
    ("🔨", "2. Build Model", "Select a target column, configure the problem type, and let AutoML find the best model."),
    ("📈", "3. Analyze", "Review accuracy, feature importance, confusion matrix, and detailed scoring metrics."),
    ("🔮", "4. Predict", "Generate predictions for new data — single records or entire batch files."),
]

for col, (icon, title, desc) in zip([col1, col2, col3, col4], steps):
    with col:
        st.markdown(f"""
        <div class="automl-card" style="text-align: center; min-height: 180px;">
            <div style="font-size: 2.5rem; margin-bottom: 0.75rem;">{icon}</div>
            <h3 style="font-size: 0.95rem;">{title}</h3>
            <p style="font-size: 0.78rem;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# Getting started
if n_datasets == 0:
    st.markdown("""
    <div class="automl-card" style="border: 1px dashed #7C3AED; text-align: center; padding: 2rem;">
        <div style="font-size: 2rem; margin-bottom: 1rem;">👋</div>
        <h3>Get started — navigate to <b>📊 Datasets</b> to upload your first dataset</h3>
        <p>Supported formats: CSV, Excel (.xlsx), Parquet</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Recent models table
    if st.session_state["models"]:
        st.markdown("### Recent Models")
        rows = []
        for name, m in st.session_state["models"].items():
            status = "❌ Error" if "error" in m else "✅ Ready"
            meta = m.get("metadata", {})
            rows.append({
                "Model": name,
                "Status": status,
                "Problem Type": meta.get("problem_type", "—").title(),
                "Best Score": f"{abs(m.get('best_score', 0)):.4f}" if "best_score" in m else "—",
                "Fit Time (s)": f"{m.get('fit_time', 0):.1f}" if "fit_time" in m else "—",
                "Backend": meta.get("backend", "—"),
            })
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
