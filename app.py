"""
AutoML Studio — main Streamlit entrypoint.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from src.utils import state
from src.ui.pages import upload, training, results

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def _inject_css():
    st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.main { background: #0a0a14; }
.stApp { background: #0a0a14; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #070711 !important;
    border-right: 1px solid #1e293b !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6d28d9, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(109, 40, 217, 0.4) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    padding: 12px !important;
    font-size: 15px !important;
}

/* ── Inputs ── */
.stSelectbox > div, .stSlider > div {
    background: #0f1929 !important;
}

/* ── Divider ── */
hr { border-color: #1e293b !important; margin: 16px 0 !important; }

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    background: #0f1929 !important;
    border-radius: 8px !important;
    border: 1px solid #1e293b !important;
}

/* ── Code blocks ── */
.stCode pre {
    background: #070711 !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    font-size: 11px !important;
    color: #a78bfa !important;
    max-height: 240px !important;
    overflow-y: auto !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #4f46e5 !important;
    border-radius: 12px !important;
    background: #0f1929 !important;
    padding: 20px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #070711; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6366F1; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] { background: #0a0a14 !important; }
.stTabs [data-baseweb="tab"] { color: #64748b !important; }
.stTabs [aria-selected="true"] { color: #8B5CF6 !important; border-color: #8B5CF6 !important; }

/* ── Progress bar ── */
.stProgress > div > div { background: linear-gradient(90deg,#8B5CF6,#06B6D4) !important; }

/* ── Toast / alerts ── */
.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:20px 0 24px 0;">
            <div style="font-size:42px;">🤖</div>
            <div style="font-size:18px;font-weight:900;background:linear-gradient(135deg,#8B5CF6,#06B6D4);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">AutoML Studio</div>
            <div style="font-size:10px;color:#475569;margin-top:4px;">Automated Machine Learning</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        current = state.get("current_page", "upload")

        nav_items = [
            ("upload", "📁 Dataset & Config"),
            ("training", "⚡ Training Dashboard"),
            ("results", "📈 Results"),
        ]

        for page_key, label in nav_items:
            is_active = current == page_key
            btn_style = "primary" if is_active else "secondary"
            if st.button(label, use_container_width=True, key=f"nav_{page_key}"):
                state.set("current_page", page_key)
                st.rerun()

        st.divider()

        # Quick stats sidebar
        df = state.get("df")
        if df is not None:
            st.markdown("**📊 Dataset**")
            st.markdown(
                f"<div style='background:#0f1929;border-radius:8px;padding:10px;border:1px solid #1e293b;font-size:12px;color:#94a3b8;'>"
                f"📋 {state.get('dataset_name','')}<br>"
                f"📏 {df.shape[0]:,} rows × {df.shape[1]} cols<br>"
                f"🎯 Target: <span style='color:#a78bfa;'>{state.get('target_column','—')}</span><br>"
                f"🔰 Task: <span style='color:#06B6D4;'>{state.get('task_type','—').title()}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        results = state.get("results", [])
        if results:
            st.markdown("**🏆 Best Pipeline**")
            best = sorted(results, key=lambda r: r.get("primary_metric_cv", 0), reverse=True)[0]
            opt_metric = state.get("optimization_metric", "roc_auc")
            st.markdown(
                f"<div style='background:#1a0533;border-radius:8px;padding:10px;border:1px solid #6d28d9;font-size:12px;'>"
                f"<span style='color:#a78bfa;font-weight:700;'>{best.get('pipeline_id','')}</span> · {best.get('algorithm','')}<br>"
                f"CV {opt_metric.upper()}: <span style='color:#10B981;font-weight:700;'>{best.get('primary_metric_cv',0):.4f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            "<div style='font-size:10px;color:#334155;text-align:center;'>Powered by scikit-learn · XGBoost · LightGBM · Optuna · MLflow</div>",
            unsafe_allow_html=True,
        )


# ── Main routing ───────────────────────────────────────────────────────────────
def main():
    state.init_state()
    _inject_css()
    render_sidebar()

    page = state.get("current_page", "upload")

    if page == "upload":
        upload.render()
    elif page == "training":
        training.render()
    elif page == "results":
        results.render()
    else:
        upload.render()


if __name__ == "__main__" or True:
    main()

