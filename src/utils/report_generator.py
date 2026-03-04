"""
PDF Report Generator for MLine.
Uses fpdf2 to create professional executive summaries.
"""

import os
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
from typing import Dict, List, Any, Optional

class AutoMLReport(FPDF):
    def header(self):
        # Header with gradient-like bar
        self.set_fill_color(139, 92, 246) # Purple (#8B5CF6)
        self.rect(0, 0, 210, 30, 'F')
        
        self.set_font('helvetica', 'B', 20)
        self.set_text_color(255, 255, 255)
        self.cell(0, 15, 'MLine - Executive Report', ln=True, align='C')
        self.set_font('helvetica', '', 10)
        self.cell(0, 5, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True, align='C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(150)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

    def section_title(self, title):
        self.set_font('helvetica', 'B', 14)
        self.set_text_color(50, 50, 100)
        self.cell(0, 10, title, ln=True, align='L')
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def summary_item(self, label, value):
        self.set_font('helvetica', 'B', 11)
        self.set_text_color(30, 41, 59)
        self.write(8, f"{label}: ")
        self.set_font('helvetica', '', 11)
        self.write(8, f"{value}\n")

def generate_pdf_report(
    experiment_data: Dict[str, Any],
    best_pipeline: Dict[str, Any],
    all_results: List[Dict[str, Any]],
    output_path: str
):
    """
    Creates a professional PDF report from experiment results.
    """
    pdf = AutoMLReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # 1. Executive Summary
    pdf.section_title("1. Executive Summary")
    pdf.summary_item("Experiment Name", experiment_data.get("name", "Untitled"))
    pdf.summary_item("Dataset", experiment_data.get("dataset_name", "Unknown"))
    pdf.summary_item("Task Type", experiment_data.get("task_type", "N/A").title())
    pdf.summary_item("Target Variable", experiment_data.get("target_column", "N/A"))
    pdf.summary_item("Total Pipelines Trained", str(len(all_results)))
    pdf.summary_item("Best Algorithm", best_pipeline.get("algorithm", "N/A"))
    pdf.ln(5)

    # 2. Performance Metrics
    pdf.section_title("2. Performance Overview")
    pdf.set_font('helvetica', 'B', 10)
    pdf.set_fill_color(240, 240, 255)
    
    # Create a small table for top 5 pipelines
    pdf.cell(50, 8, "Pipeline ID", 1, 0, 'C', True)
    pdf.cell(50, 8, "Algorithm", 1, 0, 'C', True)
    pdf.cell(45, 8, "CV Score", 1, 0, 'C', True)
    pdf.cell(45, 8, "Holdout Score", 1, 1, 'C', True)
    
    pdf.set_font('helvetica', '', 9)
    top_5 = sorted(all_results, key=lambda x: x.get("primary_metric_cv", 0), reverse=True)[:5]
    for r in top_5:
        pdf.cell(50, 7, str(r.get("pipeline_id")), 1)
        pdf.cell(50, 7, str(r.get("algorithm")), 1)
        pdf.cell(45, 7, f"{r.get('primary_metric_cv', 0):.4f}", 1, 0, 'R')
        pdf.cell(45, 7, f"{r.get('primary_metric_holdout', 0):.4f}", 1, 1, 'R')
    pdf.ln(10)

    # 3. Best Pipeline Deep-Dive
    pdf.section_title("3. Best Pipeline Details")
    pdf.summary_item("Pipeline ID", best_pipeline.get("pipeline_id"))
    pdf.summary_item("Preprocessing", best_pipeline.get("transformer", "PassThrough"))
    pdf.summary_item("PCA Applied", "Yes" if best_pipeline.get("use_pca") else "No")
    
    # Feature Importance Chart (if available)
    fi_data = best_pipeline.get("feature_importance")
    if fi_data and len(fi_data) > 0:
        pdf.ln(5)
        pdf.set_font('helvetica', 'B', 11)
        pdf.cell(0, 10, "Feature Importance (Top 10)", ln=True)
        
        # Plot using matplotlib and save to temp
        df_fi = pd.DataFrame(fi_data).head(10)
        plt.figure(figsize=(6, 4))
        plt.barh(df_fi['feature'][::-1], df_fi['importance'][::-1], color='#8B5CF6')
        plt.title("Gini/Permutation Importance")
        plt.tight_layout()
        
        temp_path = os.path.join(tempfile.gettempdir(), f"fi_{best_pipeline['pipeline_id']}.png")
        plt.savefig(temp_path)
        plt.close()
        
        pdf.image(temp_path, x=40, w=130)
        pdf.ln(5)

    # 4. Explained Impact (SHAP) - placeholder for future integrated SHAP images
    # Since SHAP requires a fit pipeline which might not be in this utility call,
    # we'll just note it for now or use pre-computed importance.
    
    pdf.output(output_path)
    return output_path
