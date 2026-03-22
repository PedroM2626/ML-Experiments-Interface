"""
AutoML Studio – Page: Predictions
History of all predictions made across models.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import pandas as pd
import streamlit as st
from src.ui_utils import load_global_css

st.set_page_config(page_title="Predictions – AutoML Studio", page_icon="🔮", layout="wide")

# ── Global CSS ─────────────────────────────────────────────────────────────────
load_global_css()

st.markdown("""
<h1 style="font-size:1.8rem;font-weight:800;color:#F1F5F9;margin-bottom:0.25rem;">
    🔮 Predictions
</h1>
<p style="color:#94A3B8;font-size:0.9rem;margin-top:0;">
    History of all predictions generated across your models.
</p>
""", unsafe_allow_html=True)

history = st.session_state.get("prediction_history", [])

if not history:
    st.markdown("""
    <div style="background:#1E293B;border:1px dashed #334155;border-radius:12px;padding:3rem;text-align:center;margin-top:2rem;">
        <div style="font-size:3rem;margin-bottom:1rem;">🔮</div>
        <p style="color:#94A3B8;font-size:1rem;">No predictions yet.<br>Go to <b>🔨 Build → Predict</b> to generate predictions with a trained model.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Summary metrics
    total_predictions = sum(h.get("rows", 1) for h in history)
    batch_count = sum(1 for h in history if h.get("type") == "batch")
    single_count = sum(1 for h in history if h.get("type") == "single")

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Predictions", f"{total_predictions:,}")
    m2.metric("Batch Jobs", batch_count)
    m3.metric("Single Predictions", single_count)

    st.divider()

    # Show each prediction batch
    for i, h in enumerate(reversed(history)):
        model = h.get("model", "Unknown")
        pred_type = h.get("type", "batch")
        rows = h.get("rows", 1)
        df: pd.DataFrame = h.get("df")

        type_badge = (
            '<span style="background:rgba(37,99,235,0.15);color:#60A5FA;padding:3px 10px;border-radius:20px;font-size:0.72rem;font-weight:600;">📦 Batch</span>'
            if pred_type == "batch" else
            '<span style="background:rgba(124,58,237,0.15);color:#A78BFA;padding:3px 10px;border-radius:20px;font-size:0.72rem;font-weight:600;">🔍 Single</span>'
        )

        with st.expander(f"**{model}** — {rows:,} rows  {type_badge}", expanded=(i == 0)):
            if df is not None:
                st.dataframe(df.head(100), use_container_width=True, height=250)

                # Download button
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download CSV",
                    data=csv_bytes,
                    file_name=f"{model}_predictions_{i}.csv",
                    mime="text/csv",
                    key=f"dl_pred_{i}",
                )

    # Clear history
    st.divider()
    if st.button("🗑️ Clear Prediction History"):
        st.session_state["prediction_history"] = []
        st.rerun()
