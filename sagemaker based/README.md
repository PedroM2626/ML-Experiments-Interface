# 🚀 AutoML Studio

AutoML Studio is a powerful, no-code machine learning platform inspired by **AWS SageMaker Canvas**. It allows users to upload datasets, perform automated feature engineering, and train high-performance models using industry-standard AutoML frameworks.

## ✨ Features

- **Intuitive UI**: A streamlined, dark-mode interface for data scientists and analysts.
- **AutoML Engine**: Leverages **AutoGluon** with a fallback to **FLAML**, ensuring the best model for every task.
- **Data Profiling**: Automatic column type detection, distribution analysis, and missing value tracking.
- **MLOps Integrated**: Full experiment tracking, metric logging, and model versioning via **MLflow**.
- **Model Explainability**: Visualize feature importance, ROC curves, confusion matrices, and residuals.
- **Deployment Ready**: Generate single or batch predictions directly from the interface.

## 🛠️ Architecture

- **Core**: Python & Streamlit
- **ML Engine**: AutoGluon (Tabular), FLAML
- **Tracking**: MLflow
- **Visualization**: Plotly
- **Containerization**: Docker & Docker Compose

## 🚀 Getting Started

### Prerequisites

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)

### Running with Docker (Recommended)

To start both the AutoML Studio and the MLflow tracking server:

```bash
docker-compose up --build
```

- Access the Studio: [http://localhost:8501](http://localhost:8501)
- Access MLflow: [http://localhost:5000](http://localhost:5000)

### Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   streamlit run app.py
   ```

## 📋 Data Preparation Recipe

The studio supports a "Recipe" system for data transformation:
1. **Selection**: Exclude irrelevant columns.
2. **Feature Engineering**: Extract components (Year, Month, Day) from datetime columns.
3. **Filtering**: Row-level filtering based on logical operators.

## 📊 MLOps Standards

This project follows MLOps best practices:
- **Reproducibility**: Every training run logs exact hyperparameters and dataset metadata.
- **Artifacts**: Models are saved in a structured registry (`models/` and `mlruns/`).
- **Validation**: Automated metrics (AUC, R2, Accuracy) are calculated for every build.

## 🐞 Fixes & Optimizations

- **Font Glitch**: Fixed an issue where `keyboard_double_arrow_right` text appeared instead of icons due to a global CSS font override.
- **Layout Robustness**: Refactored the Models page to use native Streamlit containers, preventing DOM hierarchy breakages.
- **Centralized CSS**: Extracted duplicate styling logic into `src/ui_utils.py` for better maintainability.
- **Navigation**: Corrected internal page-to-page navigation paths.

---
Built with ❤️ by the AutoML Studio Team.
