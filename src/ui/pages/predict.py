"""
Predict Page — model inference and local explainability.
Allows users to input data manually or upload a CSV to get predictions from the best model.
"""

import streamlit as st
import pandas as pd
import numpy as np
from src.utils import state, experiment_manager as em
from src.ui.components.shap_charts import render_shap_waterfall
from src.automl.explainer import compute_shap_values

def render():
    state.init_state()
    
    st.markdown(f"""
    <div style="margin-bottom:24px;">
        <h2 style="margin:0;font-size:24px;font-weight:800;color:#e2e8f0;">
            🔮 Predict
        </h2>
        <p style="margin:0;font-size:12px;color:#64748b;">
            Deploy and test your best-performing models directly.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    experiments = em.list_experiments()
    done_exps = [e for e in experiments if e["status"] == "done" and e.get("best_result")]
    
    if not done_exps:
        st.info("No completed experiments found. Train a model first!")
        return
    
    # ── Experiment Selection ──────────────────────────────────────────────────
    exp_options = {e["id"]: f"{e['name']} ({e['task_type']})" for e in done_exps}
    selected_id = st.selectbox("Select Experiment", options=list(exp_options.keys()), format_func=lambda x: exp_options[x], key="predict_exp_select")
    
    exp = em.get_experiment(selected_id)
    best = exp["best_result"]
    pipeline = best.get("pipeline")
    task_type = exp["task_type"]
    
    if not pipeline:
        st.error("Model pipeline not found. It might not have been persisted correctly.")
        return
    
    st.markdown(f"""
    <div style="background:#1e3a5f22;border:1px solid #3b82f6;border-radius:12px;padding:16px;margin-bottom:20px;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <div style="font-size:14px;font-weight:700;color:#e2e8f0;">🏆 Best Model: {best.get('algorithm', 'Unknown')}</div>
                <div style="font-size:11px;color:#94a3b8;margin-top:2px;">{best.get('pipeline_id', '')} &nbsp;·&nbsp; {best.get('transformer', '')}</div>
            </div>
            <div style="text-align:right;">
                <div style="color:#10b981;font-size:16px;font-weight:800;">{abs(best.get('primary_metric_cv', 0)):.4f}</div>
                <div style="color:#64748b;font-size:10px;">{exp.get('optimization_metric', '').upper()} (CV)</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_manual, tab_upload = st.tabs(["📝 Manual Input", "📂 Batch (CSV)"])
    
    # Get feature names from a sample (the original DF)
    df_orig = exp.get("df")
    if df_orig is None:
        st.warning("Original dataset not found in memory. Cannot infer input schema.")
        return
        
    target = exp["target_column"]
    features_df = df_orig.drop(columns=[target]) if target in df_orig.columns else df_orig
    feature_names = features_df.columns.tolist()
    
    # ── Manual Input ─────────────────────────────────────────────────────────
    with tab_manual:
        st.markdown("###### Enter values to get a prediction")
        input_data = {}
        
        # Grid layout for inputs
        cols = st.columns(3)
        for i, col_name in enumerate(feature_names[:15]): # Limit to first 15 for safety
            with cols[i % 3]:
                dtype = features_df[col_name].dtype
                if np.issubdtype(dtype, np.number):
                    min_v = float(features_df[col_name].min())
                    max_v = float(features_df[col_name].max())
                    mean_v = float(features_df[col_name].mean())
                    input_data[col_name] = st.number_input(col_name, value=mean_v, key=f"in_{col_name}")
                else:
                    unique_vals = features_df[col_name].unique().tolist()
                    input_data[col_name] = st.selectbox(col_name, options=unique_vals, key=f"in_{col_name}")
        
        if len(feature_names) > 15:
            st.caption(f"... and {len(feature_names)-15} more features (using mean/mode)")
            for col_name in feature_names[15:]:
                if np.issubdtype(features_df[col_name].dtype, np.number):
                    input_data[col_name] = float(features_df[col_name].mean())
                else:
                    input_data[col_name] = features_df[col_name].mode()[0]

        if st.button("🔮 Run Prediction", type="primary", use_container_width=True):
            X_test = pd.DataFrame([input_data])
            try:
                pred = pipeline.predict(X_test)[0]
                
                # Result card
                st.markdown(f"""
                <div style="background:#064e3b22;border:1px solid #10b981;border-radius:12px;padding:24px;text-align:center;margin:20px 0;">
                    <div style="font-size:12px;color:#94a3b8;margin-bottom:8px;">Prediction Result</div>
                    <div style="font-size:32px;font-weight:900;color:#10b981;">{pred}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # SHAP local explanation for this prediction
                st.markdown("---")
                st.markdown("##### ✨ Why this result? (Local SHAP)")
                # Compute SHAP for this single row
                try:
                    # We need the model from the pipeline (usually the last step)
                    model = pipeline.steps[-1][1]
                    # And preprocessed data
                    X_pre = pipeline.named_steps['preprocessor'].transform(X_test)
                    pre_feature_names = result_feature_names = [] # Need to get them properly
                    from ..automl.pipeline_builder import get_feature_names_after_preprocessor
                    pre_feature_names = get_feature_names_after_preprocessor(pipeline.named_steps['preprocessor'], feature_names)
                    
                    shap_vals, base_val = compute_shap_values(model, X_pre)
                    render_shap_waterfall(shap_vals[0], pre_feature_names, float(base_val))
                except Exception as e:
                    st.caption(f"SHAP explanation failed: {str(e)}")
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

    # ── Upload Input ─────────────────────────────────────────────────────────
    with tab_upload:
        st.markdown("###### Upload CSV for batch prediction")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="predict_csv_upload")
        if uploaded_file:
            df_test = pd.read_csv(uploaded_file)
            st.dataframe(df_test.head(), use_container_width=True)
            
            if st.button("🚀 Run Batch Prediction", type="primary"):
                try:
                    preds = pipeline.predict(df_test)
                    df_test["PREDICTION"] = preds
                    st.success(f"Batch completed: {len(preds)} rows.")
                    st.dataframe(df_test, use_container_width=True)
                    st.download_button("📥 Download Results", df_test.to_csv(index=False), "predictions.csv", "text/csv")
                except Exception as e:
                    st.error(f"Batch prediction error: {str(e)}")

