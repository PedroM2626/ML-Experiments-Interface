"""
AutoML Studio – Page: Datasets
Upload, preview, and manage datasets with column profiling.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from config import APP_ICON, SUPPORTED_EXTENSIONS, MAX_UPLOAD_SIZE_MB
from src.data_utils import (
    load_dataset, profile_dataset, dataset_summary,
    apply_transforms, extract_datetime_features,
    drop_columns, filter_rows, detect_column_type,
)
from src.visualization import column_distribution_chart
from src.ui_utils import load_global_css

st.set_page_config(page_title="Datasets – AutoML Studio", page_icon="📊", layout="wide")

# ── Global CSS ─────────────────────────────────────────────────────────────────
load_global_css()

# ── Page Header ────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="font-size:1.8rem;font-weight:800;color:#F1F5F9;margin-bottom:0.25rem;">
    📊 Datasets
</h1>
<p style="color:#94A3B8;font-size:0.9rem;margin-top:0;">
    Import, preview, and prepare your data before building a model.
</p>
""", unsafe_allow_html=True)

# ── Upload Section ─────────────────────────────────────────────────────────────
with st.expander("⬆️ Import Data", expanded=len(st.session_state.get("datasets", {})) == 0):
    uploaded = st.file_uploader(
        "Drag & drop a file or click to browse",
        type=SUPPORTED_EXTENSIONS,
        help=f"Supported: CSV, Excel, Parquet. Max {MAX_UPLOAD_SIZE_MB}MB.",
    )
    if uploaded is not None:
        if uploaded.name not in st.session_state.get("datasets", {}):
            with st.spinner(f"Loading **{uploaded.name}**..."):
                try:
                    df = load_dataset(uploaded, filename=uploaded.name)
                    profile = profile_dataset(df)
                    if "datasets" not in st.session_state:
                        st.session_state["datasets"] = {}
                    if "dataset_profiles" not in st.session_state:
                        st.session_state["dataset_profiles"] = {}
                    st.session_state["datasets"][uploaded.name] = df
                    st.session_state["dataset_profiles"][uploaded.name] = profile
                    st.session_state["current_dataset"] = uploaded.name
                    st.success(f"✅ **{uploaded.name}** loaded — {len(df):,} rows × {len(df.columns)} columns")
                except Exception as e:
                    st.error(f"Failed to load dataset: {e}")
        else:
            st.info(f"**{uploaded.name}** is already loaded.")

# ── Dataset Selector ───────────────────────────────────────────────────────────
datasets = st.session_state.get("datasets", {})
if not datasets:
    st.markdown("""
    <div style="background:#1E293B;border:1px dashed #334155;border-radius:12px;padding:3rem;text-align:center;margin-top:2rem;">
        <div style="font-size:3rem;margin-bottom:1rem;">📂</div>
        <p style="color:#94A3B8;font-size:1rem;">No datasets yet. Upload a file above to get started.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

selected_ds = st.selectbox(
    "Active Dataset",
    list(datasets.keys()),
    index=list(datasets.keys()).index(st.session_state.get("current_dataset") or list(datasets.keys())[0]),
    key="ds_selector",
)
st.session_state["current_dataset"] = selected_ds

df: pd.DataFrame = datasets[selected_ds]
profile: dict = st.session_state["dataset_profiles"].get(selected_ds, {})
summary = dataset_summary(df)

# Delete dataset button
col_del, _ = st.columns([1, 6])
with col_del:
    if st.button("🗑️ Remove Dataset", key="del_ds"):
        del st.session_state["datasets"][selected_ds]
        if selected_ds in st.session_state.get("dataset_profiles", {}):
            del st.session_state["dataset_profiles"][selected_ds]
        st.session_state["current_dataset"] = None
        st.rerun()

st.divider()

# ── Summary Stats ──────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("📋 Rows", f"{summary['n_rows']:,}")
m2.metric("📐 Columns", summary["n_cols"])
m3.metric("📦 Cells", f"{summary['n_cells']:,}")
m4.metric("🔴 Missing", f"{summary['missing_total']:,}")
m5.metric("💾 Memory", f"{summary['memory_mb']} MB")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_data, tab_profile, tab_transform = st.tabs(["📋 Data Preview", "📊 Column Profiles", "🔧 Transformations"])

# ── Tab: Data Preview ──────────────────────────────────────────────────────────
with tab_data:
    n_preview = st.slider("Rows to preview", 10, min(500, len(df)), 50, step=10)
    st.dataframe(
        df.head(n_preview),
        use_container_width=True,
        height=400,
    )

# ── Tab: Column Profiles ───────────────────────────────────────────────────────
with tab_profile:
    st.markdown("### Column Overview")

    # Header row
    hc = st.columns([3, 1.5, 1.5, 1.5, 1.5])
    for h, label in zip(hc, ["Column", "Type", "Missing", "Unique", "Distribution"]):
        h.markdown(f"<div style='color:#64748B;font-size:0.7rem;text-transform:uppercase;font-weight:600;'>{label}</div>", unsafe_allow_html=True)

    TYPE_ICONS = {
        "Numeric": "🔢", "Binary": "⚖️", "Categorical": "🏷️",
        "Datetime": "📅", "Text": "📝", "ID": "🆔",
    }
    TYPE_COLORS = {
        "Numeric": "#60A5FA", "Binary": "#A78BFA", "Categorical": "#34D399",
        "Datetime": "#FBBF24", "Text": "#FB923C", "ID": "#94A3B8",
    }

    for col_name, col_info in profile.items():
        col_type = col_info.get("type", "Categorical")
        icon = TYPE_ICONS.get(col_type, "•")
        color = TYPE_COLORS.get(col_type, "#94A3B8")

        rc = st.columns([3, 1.5, 1.5, 1.5, 1.5])
        with rc[0]:
            is_target = col_name == st.session_state.get("build_state", {}).get("target_col")
            target_badge = " <span style='background:rgba(124,58,237,0.2);color:#A78BFA;padding:1px 6px;border-radius:4px;font-size:0.7rem;'>Target</span>" if is_target else ""
            st.markdown(
                f"<div style='padding:0.4rem 0;color:#F1F5F9;font-weight:500;font-size:0.85rem;'>{col_name}{target_badge}</div>",
                unsafe_allow_html=True,
            )
        with rc[1]:
            st.markdown(
                f"<div style='padding:0.4rem 0;'><span style='background:rgba(0,0,0,0.3);border:1px solid {color};color:{color};padding:2px 8px;border-radius:12px;font-size:0.72rem;font-weight:600;'>{icon} {col_type}</span></div>",
                unsafe_allow_html=True,
            )
        with rc[2]:
            pct = col_info.get("missing_pct", 0)
            color_miss = "#F87171" if pct > 10 else "#FBBF24" if pct > 0 else "#34D399"
            st.markdown(
                f"<div style='padding:0.4rem 0;color:{color_miss};font-size:0.85rem;'>{pct:.1f}% ({col_info.get('missing_count', 0):,})</div>",
                unsafe_allow_html=True,
            )
        with rc[3]:
            st.markdown(
                f"<div style='padding:0.4rem 0;color:#94A3B8;font-size:0.85rem;'>{col_info.get('unique_count', 0):,}</div>",
                unsafe_allow_html=True,
            )
        with rc[4]:
            mini_fig = column_distribution_chart(df[col_name], col_name, col_type)
            mini_fig.update_layout(height=80, margin=dict(l=0, r=0, t=0, b=0),
                                   xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False))
            mini_fig.update_layout(title="")
            st.plotly_chart(mini_fig, use_container_width=True, key=f"mini_{col_name}")

        st.markdown("<hr style='border-color:#1E293B;margin:0;'>", unsafe_allow_html=True)

# ── Tab: Transformations ───────────────────────────────────────────────────────
with tab_transform:
    st.markdown("### Transform Recipe")
    st.markdown("<p style='color:#94A3B8;font-size:0.85rem;'>Apply transforms to the dataset before training. Changes are tracked in the recipe.</p>", unsafe_allow_html=True)

    recipe = st.session_state.get("transform_recipe", {}).get(selected_ds, [])

    # Show current recipe
    if recipe:
        st.markdown("**Active Transforms:**")
        for i, step in enumerate(recipe):
            col_r1, col_r2 = st.columns([5, 1])
            with col_r1:
                if step["type"] == "drop_column":
                    st.markdown(f"`{i+1}.` Drop column **{step['col']}**")
                elif step["type"] == "extract_datetime":
                    st.markdown(f"`{i+1}.` Extract **{', '.join(step['features'])}** from **{step['col']}**")
                elif step["type"] == "filter_rows":
                    st.markdown(f"`{i+1}.` Filter rows: **{step['col']}** {step['operator']} **{step['value']}**")
            with col_r2:
                if st.button("❌", key=f"rm_recipe_{i}"):
                    recipe.pop(i)
                    if "transform_recipe" not in st.session_state:
                        st.session_state["transform_recipe"] = {}
                    st.session_state["transform_recipe"][selected_ds] = recipe
                    st.rerun()
        st.divider()

    # Add new transform
    transform_type = st.selectbox("Add Transform", ["— Select —", "Drop Column", "Extract Datetime Features", "Filter Rows"])

    if transform_type == "Drop Column":
        drop_col = st.selectbox("Column to drop", df.columns.tolist(), key="drop_col_sel")
        if st.button("Add to Recipe ➕", key="btn_drop"):
            recipe.append({"type": "drop_column", "col": drop_col})
            if "transform_recipe" not in st.session_state:
                st.session_state["transform_recipe"] = {}
            st.session_state["transform_recipe"][selected_ds] = recipe
            st.success(f"Added: Drop column **{drop_col}**")
            st.rerun()

    elif transform_type == "Extract Datetime Features":
        datetime_cols = [c for c, v in profile.items() if v.get("type") == "Datetime"]
        if not datetime_cols:
            st.info("No datetime columns detected in this dataset.")
        else:
            dt_col = st.selectbox("Datetime column", datetime_cols, key="dt_col_sel")
            features = st.multiselect(
                "Features to extract",
                ["Year", "Month", "Day", "Hour", "Weekday", "Quarter"],
                default=["Month", "Day"],
            )
            if features and st.button("Add to Recipe ➕", key="btn_dt"):
                recipe.append({"type": "extract_datetime", "col": dt_col, "features": features})
                if "transform_recipe" not in st.session_state:
                    st.session_state["transform_recipe"] = {}
                st.session_state["transform_recipe"][selected_ds] = recipe
                st.success(f"Added: Extract {', '.join(features)} from **{dt_col}**")
                st.rerun()

    elif transform_type == "Filter Rows":
        filter_col = st.selectbox("Column", df.columns.tolist(), key="filter_col_sel")
        filter_op = st.selectbox("Operator", [">", "<", "==", "!=", ">=", "<=", "contains"])
        filter_val = st.text_input("Value", key="filter_val_input")
        if filter_val and st.button("Add to Recipe ➕", key="btn_filter"):
            recipe.append({"type": "filter_rows", "col": filter_col, "operator": filter_op, "value": filter_val})
            if "transform_recipe" not in st.session_state:
                st.session_state["transform_recipe"] = {}
            st.session_state["transform_recipe"][selected_ds] = recipe
            st.success(f"Added: Filter {filter_col} {filter_op} {filter_val}")
            st.rerun()

    # Apply transforms preview
    if recipe:
        st.divider()
        if st.button("👁️ Preview Transformed Data", key="btn_preview_transform"):
            try:
                transformed = apply_transforms(df.copy(), recipe)
                st.markdown(f"**Preview** ({len(transformed):,} rows × {len(transformed.columns)} cols):")
                st.dataframe(transformed.head(50), use_container_width=True, height=300)
            except Exception as e:
                st.error(f"Transform error: {e}")

        if st.button("✅ Apply Transforms to Dataset", key="btn_apply_transform"):
            try:
                transformed = apply_transforms(df.copy(), recipe)
                st.session_state["datasets"][selected_ds] = transformed
                st.session_state["dataset_profiles"][selected_ds] = profile_dataset(transformed)
                st.session_state["transform_recipe"][selected_ds] = []
                st.success("Transforms applied successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to apply transforms: {e}")
