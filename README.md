# AutoML Studio 🤖

AutoML Studio is a professional **Automated Machine Learning** platform built with Streamlit. It is deeply inspired by enterprise solutions such as **IBM WatsonX AutoAI**, Azure ML, and Vertex AI, replicate their core experience of automated pipeline discovery, interactive progress mapping, and professional model evaluation.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.41%2B-red)
![MLflow](https://img.shields.io/badge/mlflow-2.20%2B-blue)
![Version](https://img.shields.io/badge/version-1.6.0-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🌟 IBM WatsonX AutoAI Inspiration
This project aims to bridge the gap between open-source AutoML libraries and enterprise cloud platforms. Key inspirations include:
- **Dynamic Pipeline System**: An animated **Progress Map** that visualizes the data flow through feature engineering, HPO, and model selection nodes, just like in WatsonX.
- **Automated Stacking**: Automatically combines top-N pipelines into a final stacked ensemble for maximum performance.
- **Enterprise Reporting**: Professional executive PDF reports and interactive SHAP explainability charts.

---

## ✨ Key Features

### 🤖 Intelligent AutoML Engine
- **Task Detection**: Automatic detection of Classification, Regression, Time Series, and **Text Classification** tasks.
- **Auto-Pilot vs. Manual Mode**: Choose between fully autonomous training or manually selecting algorithms and transformers.
- **Top-Tier Algorithms**: XGBoost, LightGBM, GradientBoosting, RandomForest, ExtraTrees, Ridge, and more.
- **Advanced Stacking**: Auto-combination of best performers into an ensemble model.

### 🧠 Deep Learning
- **MLP (Multi-Layer Perceptron)**: `MLPClassifier` and `MLPRegressor` from sklearn with Optuna HPO (hidden layers, activation, alpha, learning rate). Available for all tabular tasks.
- **Keras / TensorFlow Networks**: Dense neural networks with configurable layers, dropout, and batch normalization. Sklearn-compatible wrapper falls back to MLP if TensorFlow is not installed.
- **Hyperparameter Search Space**: Layer counts (1–3), units (64–256), dropout (0.1–0.5), optimizer (Adam, RMSProp) are all auto-tuned.

### 💬 NLP Text Classification
- **TF-IDF Vectorization**: Configurable n-gram range (1-1 to 1-3), max features (5K–100K), and sublinear frequency scaling.
- **Classic Text Classifiers**: Logistic Regression, LinearSVC (calibrated), ComplementNB, SGD Classifier, MLP, and XGBoost.
- **Optuna HPO for Text**: Lightweight hyperparameter search with 2-fold CV for speed.
- **New Task Type**: `text_classification` integrates natively into the Upload → Train → Results → Predict flow.

### 🔍 Model Explainability (SHAP)
- **Global Importance**: Beeswarm and Bar summary charts to understand overall feature impact.
- **Local Waterfall**: Specific explanations for individual predictions — “Why was this customer classified as X?”

### ⏱️ Time Series Forecasting
- **ARIMA & Tree-based Models**: Comprehensive forecasting with specialized algorithms.
- **Time-Aware CV**: Expanding window and walk-forward cross-validation to prevent data leakage.
- **Interactive Forecasts**: Visualization of future values with 95% confidence intervals.

### 📦 Persistence & Tracking
- **SQLite Persistence**: Experiments and results are saved to a local database (`experiments.db`), surviving app restarts.
- **MLflow Integration**: Full tracking of metrics, parameters, and artifacts for every pipeline run.

### 📄 Professional Outputs
- **Predict Interface**: A dedicated page for manual data entry or CSV batch inference with local SHAP explanations.
- **Executive PDF Reports**: Generates professional summaries with metrics, top pipelines, and feature importance charts.

---

## 🚀 Quick Start

### Option 1 — Local Python Environment

```bash
# Clone the repository
git clone <repo-url>
cd midnight-hawking

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

Access the studio at **http://localhost:8501**

### Option 2 — Docker (Recommended)

```bash
docker compose up --build
```
- **AutoML Studio**: http://localhost:8501
- **MLflow UI**: http://localhost:5000

---

## 🛠️ Technology Stack

| Category | Tools |
|-----------|------------|
| **UI** | Streamlit, Plotly, CSS Glassmorphism |
| **Machine Learning** | scikit-learn, XGBoost, LightGBM, SHAP |
| **Deep Learning** | TensorFlow/Keras (optional), sklearn MLP |
| **NLP** | TF-IDF (sklearn), Naive Bayes, LinearSVC |
| **Optimization** | Optuna (TPE Sampler) |
| **Tracking** | MLflow |
| **Storage** | SQLite, joblib |
| **Reporting** | fpdf2, matplotlib |
| **Infrastructure** | Docker, Python 3.10+ |

---

## 🗂️ Project Structure

```
midnight-hawking/
├── app.py                # Streamlit entry point
├── src/
│   ├── automl/           # Core Engines (Classification, Regression, Time Series, NLP)
│   │   ├── engine.py     # Main AutoML engine
│   │   ├── nlp_engine.py # TF-IDF + classifier NLP engine [NEW in v1.4]
│   │   └── deep_learning.py  # MLP & Keras wrappers [NEW in v1.4]
│   ├── ui/               # Pages & Professional Components
│   ├── db/               # SQLite Persistence Layer
│   ├── tracking/         # MLflow Integration
│   └── utils/            # Report Generation & Managers
├── exports/              # Saved model artifacts (.pkl)
├── experiments.db        # Persistent experiment storage
└── requirements.txt
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
