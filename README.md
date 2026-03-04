# MLine 🤖

MLine is a professional **Automated Machine Learning** platform built with Streamlit. It is deeply inspired by enterprise solutions such as **IBM WatsonX AutoAI**, Azure ML, and Vertex AI, replicating their core experience of automated pipeline discovery, interactive progress mapping, and professional model evaluation.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.41%2B-red)
![MLflow](https://img.shields.io/badge/mlflow-2.20%2B-blue)
![Version](https://img.shields.io/badge/version-2.1.0-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🌟 Inspired by Enterprise AutoAI
This project bridges the gap between open-source AutoML libraries and enterprise cloud platforms. Key inspirations include:
- **Dynamic Pipeline System**: An animated **Progress Map** that visualizes the data flow through feature engineering, HPO, and model selection nodes.
- **Automated Stacking**: Automatically combines top-performing pipelines into a final stacked ensemble for maximum performance.
- **Enterprise Reporting**: Professional executive PDF reports and interactive SHAP explainability charts.

---

## ✨ Key Features

### 🤖 Intelligent AutoML Engine
- **Task Detection**: Automatic detection of Classification, Regression, Time Series, and **Text Classification** tasks.
- **Auto-Pilot vs. Manual Mode**: Choose between fully autonomous training or manually selecting algorithms and transformers.
- **Top-Tier Algorithms**: XGBoost, LightGBM, GradientBoosting, RandomForest, ExtraTrees, Neural Networks, and more.
- **Advanced Stacking**: Auto-combination of best performers into an ensemble model.

### 🧠 Deep Learning
- **MLP (Multi-Layer Perceptron)**: `MLPClassifier` and `MLPRegressor` from sklearn with Optuna HPO. Available for all tabular tasks.
- **Keras / TensorFlow Networks**: Dense neural networks with configurable layers, dropout, and batch normalization.
- **Hyperparameter Tuning**: Auto-tunes layer counts, units, dropout rates, and optimizers using Optuna.

### 💬 NLP Text Classification
- **TF-IDF Vectorization**: Configurable n-gram range, max features, and weighting strategies.
- **NLP Algorithms**: Logistic Regression, LinearSVC, ComplementNB, SGD Classifier, MLP, and XGBoost optimized for text.
- **Unified Flow**: `text_classification` task type integrates natively into the MLine ecosystem.

### � Developer Experience (NEW)
- **Consumption Code Snippets**: Every pipeline detail view now includes a copyable **Python Consumption Sample**.
- **Model Portability**: Direct instruction on how to load the `.pkl` files and predict in external environments.
- **Standardized API**: Predict/Predict_proba samples generated dynamically for each pipeline.

### � Model Explainability (SHAP)
- **Global Importance**: Beeswarm and Bar summary charts to understand overall feature impact.
- **Local Waterfall**: Specific explanations for individual predictions — “Why was this customer classified as X?”

---

## 🚀 Quick Start

### Option 1 — Local Python Environment

```bash
# Clone the repository
git clone <repo-url>
cd midnight-hawking

# Install dependencies (ensure Python 3.10+)
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

Access the studio at **http://localhost:8501**

### Option 2 — Docker (Recommended)

```bash
docker compose up --build
```
- **MLine**: http://localhost:8501
- **MLflow UI**: http://localhost:5000

---

## 🛠️ Technology Stack

| Category | Tools |
|-----------|------------|
| **UI** | Streamlit, Plotly, CSS Glassmorphism |
| **Machine Learning** | scikit-learn, XGBoost, LightGBM, SHAP |
| **Deep Learning** | TensorFlow/Keras (optional), sklearn MLP |
| **NLP** | TF-IDF (sklearn), NLTK, Naive Bayes, LinearSVC |
| **Optimization** | Optuna (TPE Sampler) |
| **Tracking** | MLflow |
| **Storage** | SQLite, joblib |
| **Infrastructure** | Docker, Python 3.10+ |

---

## 🗂️ Project Structure

```
midnight-hawking/
├── app.py                # Streamlit entry point
├── src/
│   ├── automl/           # Core Engines (Classification, Regression, Time Series, NLP)
│   ├── ui/               # Pages & Professional Components
│   │   ├── components/   # UI Fragments (Leaderboard, Charts, Detail View)
│   ├── db/               # Persistence Layer (SQLite)
│   ├── tracking/         # MLflow Integration
│   └── utils/            # Report Generation & State Management
├── exports/              # Saved model artifacts (.pkl)
├── experiments.db        # Persistent experiment storage
└── requirements.txt
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
