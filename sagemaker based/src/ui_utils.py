"""
UI Utilities – Centralized styling and shared UI components for AutoML Studio.
"""
import streamlit as st

def load_global_css():
    """Injects central CSS styling into the page while preserving icons."""
    st.markdown("""
<style>
/* Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* --- Root Variables --- */
:root {
    --primary: #7C3AED;
    --primary-hover: #4F46E5;
    --bg-dark: #0F172A;
    --card-bg: #1E293B;
    --border: #334155;
    --text-muted: #94A3B8;
}

/* --- Base Typography --- */
/* Fix: Removed wildcard !important to prevent breaking Material Icons (keyboard_double_arrow...) */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: var(--bg-dark);
    color: #F1F5F9;
}

/* --- Sidebar --- */
[data-testid="stSidebar"] {
    background: #1E293B !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown p {
    color: var(--text-muted);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* --- Metrics & Cards --- */
[data-testid="stMetricValue"] {
    color: #F1F5F9 !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
}

/* --- Buttons --- */
.stButton > button {
    background: linear-gradient(135deg, var(--primary), var(--primary-hover)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5) !important;
}

/* Secondary/Action buttons */
button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid var(--primary) !important;
    color: #A78BFA !important;
}

/* --- Layout Elements --- */
[data-testid="stExpander"], [data-testid="stForm"] {
    background: var(--card-bg);
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 4px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--primary), var(--primary-hover)) !important;
    color: white !important;
    border-radius: 8px;
}

/* --- Navigation & Scrollbar --- */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--primary); }

/* --- Custom Components --- */
.status-badge {
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
}
.badge-ready { background: rgba(5, 150, 105, 0.15); color: #34D399; }
.badge-error { background: rgba(220, 38, 38, 0.15); color: #F87171; }
</style>
""", unsafe_allow_html=True)
