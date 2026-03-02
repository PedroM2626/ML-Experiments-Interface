"""
AutoML Studio — main Streamlit entrypoint.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from src.utils import state
from src.utils import experiment_manager as em
from src.utils.experiment_manager import ExpStatus
from src.ui.pages import upload, training, results
from src.ui.pages import experiments

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_css():
    st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.main { background: #0a0a14; }
.stApp { background: #0a0a14; }

[data-testid="stSidebar"] {
    background: #070711 !important;
    border-right: 1px solid #1e293b !important;
}

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

.stSelectbox > div, .stSlider > div { background: #0f1929 !important; }
hr { border-color: #1e293b !important; margin: 16px 0 !important; }

[data-testid="stDataFrame"] {
    background: #0f1929 !important;
    border-radius: 8px !important;
    border: 1px solid #1e293b !important;
}

.stCode pre {
    background: #070711 !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    font-size: 11px !important;
    color: #a78bfa !important;
    max-height: 240px !important;
    overflow-y: auto !important;
}

[data-testid="stFileUploader"] {
    border: 2px dashed #4f46e5 !important;
    border-radius: 12px !important;
    background: #0f1929 !important;
    padding: 20px !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #070711; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6366F1; }

.stTabs [data-baseweb="tab-list"] { background: #0a0a14 !important; }
.stTabs [data-baseweb="tab"] { color: #64748b !important; }
.stTabs [aria-selected="true"] { color: #8B5CF6 !important; border-color: #8B5CF6 !important; }

.stProgress > div > div { background: linear-gradient(90deg,#8B5CF6,#06B6D4) !important; }
.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
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

        # Navigation with running experiment badge
        n_running = em.count_running()
        exp_badge = f" 🔵{n_running}" if n_running > 0 else ""

        nav_items = [
            ("upload",      "📁 Dataset & Config"),
            ("experiments", f"🧪 Experiments{exp_badge}"),
            ("results",     "📈 Results"),
        ]

        for page_key, label in nav_items:
            if st.button(label, use_container_width=True, key=f"nav_{page_key}",
                         type="primary" if current == page_key else "secondary"):
                state.set("current_page", page_key)
                st.rerun()

        st.divider()

        # Quick stats for active experiment
        active_id = state.get("active_experiment_id")
        if active_id:
            exp = em.get_experiment(active_id)
            if exp:
                icon, color, bg = {
                    ExpStatus.RUNNING: ("⚡", "#3b82f6", "#1e3a5f"),
                    ExpStatus.DONE: ("✅", "#10b981", "#064e3b"),
                    ExpStatus.FAILED: ("❌", "#ef4444", "#450a0a"),
                }.get(exp["status"], ("⏳", "#94a3b8", "#1e293b"))

                st.markdown("**🔍 Active Experiment**")
                best = exp.get("best_result")
                opt_metric = exp.get("optimization_metric", "roc_auc")
                st.markdown(
                    f"<div style='background:{bg};border-radius:8px;padding:10px;border:1px solid {color}22;font-size:12px;'>"
                    f"<div style='color:{color};font-weight:700;'>{icon} {exp['name'][:22]}</div>"
                    f"<div style='color:#94a3b8;margin-top:4px;'>📋 {exp['dataset_name'][:20]}<br>"
                    f"🎯 {exp['target_column']}<br>🔰 {exp['task_type'].replace('_',' ').title()}</div>"
                    + (f"<div style='color:#10b981;font-weight:700;margin-top:6px;'>"
                       f"🏆 {best.get('pipeline_id','')} — {abs(best.get('primary_metric_cv',0)):.4f} {opt_metric.upper()}</div>"
                       if best else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )

        # All experiments mini-list
        all_exps = em.list_experiments()
        if all_exps:
            st.markdown("**📊 Experiments**")
            for exp in all_exps[:4]:
                icon, color, _ = {
                    ExpStatus.RUNNING: ("⚡", "#3b82f6", ""),
                    ExpStatus.DONE: ("✅", "#10b981", ""),
                    ExpStatus.FAILED: ("❌", "#ef4444", ""),
                    ExpStatus.STOPPED: ("⏹", "#94a3b8", ""),
                }.get(exp["status"], ("⏳", "#f59e0b", ""))
                n_p = exp.get("n_pipelines_done", 0)
                st.markdown(
                    f"<div style='font-size:11px;color:{color};padding:3px 0;border-bottom:1px solid #1e293b;'>"
                    f"{icon} <strong>{exp['name'][:18]}</strong> — {n_p}p</div>",
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
    elif page == "experiments":
        experiments.render()
    elif page == "training":
        training.render()
    elif page == "results":
        results.render()
    else:
        upload.render()


if __name__ == "__main__" or True:
    main()
