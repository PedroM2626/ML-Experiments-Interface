# AutoML Studio 🤖

Uma plataforma de **Automated Machine Learning** com interface Streamlit, inspirada nas principais plataformas cloud:
**Azure ML AutoML · AWS SageMaker Canvas · Google Vertex AI AutoML · IBM WatsonX AutoAI**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.41%2B-red)
![MLflow](https://img.shields.io/badge/mlflow-2.20%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## ✨ Features

### 🤖 AutoML Engine
- **Classificação, Regressão e Time Series** — detecção automática ou seleção manual
- Algoritmos: **XGBoost, LightGBM, GradientBoosting, RandomForest, ExtraTrees, SVM, Ridge, ElasticNet**
- Feature engineering automático (encoding, imputation, scaling, PCA, SelectKBest)
- **Hyperparameter Optimization** com Optuna (TPE sampler)
- Cross-validation com múltiplos folds configuráveis

### ⏱️ Time Series Forecasting
- Lag features e rolling statistics (sem data leakage)
- Decomposição temporal: mês, dia, semana, sazonalidade (encoding cíclico sin/cos)
- **Walk-forward cross-validation** (TimeSeriesSplit)
- Métricas: RMSE, MAE, MAPE, Directional Accuracy
- Gráfico interativo **Forecast vs Actuals** com intervalo de confiança 95%

### 🔄 Multi-Experimentos Simultâneos
- Execute **múltiplos experimentos em paralelo** (cada um em thread própria)
- Status em tempo real: ⚡ Running · ✅ Done · ❌ Failed · ⏹ Stopped
- Stop/Delete individual por experimento
- Sidebar com contador de experimentos ativos

### 📋 Dashboard de Experimentos
- **Cards Azure ML-style** com status badge, dataset, target, best model
- Progress Map animado com branches por pipeline
- Relationship Map (visualização de transformers)
- Pipeline Leaderboard com comparação de métricas CV vs Holdout
- Feature Importance interativo

### 📦 Tracking com MLflow
- Cada pipeline salvo como run no MLflow automaticamente
- Métricas, hiperparâmetros, artefatos e metadados registrados
- Experimento separado por dataset/task type

---

## 🚀 Quick Start

### Opção 1 — Python local

```bash
# Clone o repositório
git clone <repo-url>
cd midnight-hawking

# Instale as dependências
pip install -r requirements.txt

# Inicie o app
streamlit run app.py
```

Acesse em **http://localhost:8501**

### Opção 2 — Docker

```bash
docker compose up --build
```

Acesse em **http://localhost:8501**  
MLflow UI em **http://localhost:5000**

---

## 🗂️ Estrutura do Projeto

```
midnight-hawking/
├── app.py                        # Entrypoint Streamlit
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .streamlit/
│   └── config.toml               # Tema dark + fonte Inter
├── src/
│   ├── automl/
│   │   ├── engine.py             # Motor AutoML principal (classif/regression)
│   │   ├── ts_engine.py          # Motor Time Series
│   │   ├── time_series.py        # Feature engineering TS + métricas
│   │   ├── evaluator.py          # CV, métricas por task type
│   │   ├── hyperopt.py           # Registro de algoritmos + search spaces Optuna
│   │   ├── pipeline_builder.py   # Construção de sklearn Pipelines
│   │   └── feature_eng.py        # Préprocessamento e transformações
│   ├── tracking/
│   │   └── mlflow_tracker.py     # Setup MLflow + log de runs
│   ├── ui/
│   │   ├── pages/
│   │   │   ├── upload.py         # Dataset upload + configuração
│   │   │   ├── experiments.py    # Dashboard de experimentos
│   │   │   ├── training.py       # Visão de treinamento (legado)
│   │   │   └── results.py        # Resultados finais
│   │   └── components/
│   │       ├── progress_map.py   # Grafo animado de progresso
│   │       ├── relationship_map.py
│   │       ├── leaderboard.py    # Pipeline Leaderboard
│   │       ├── forecast_chart.py # Gráfico Forecast vs Actuals
│   │       └── pipeline_detail.py
│   └── utils/
│       ├── experiment_manager.py # Gerenciador multi-experimento
│       ├── state.py              # Session state Streamlit
│       └── data_profiler.py      # Profiling de dataset
├── tests/
│   ├── test_engine.py            # 18 testes do motor AutoML
│   └── test_time_series.py       # 18 testes de Time Series
└── exports/                      # Modelos exportados (.pkl)
```

---

## ⚙️ Configuração

Copie `.env.example` para `.env` e ajuste:

```bash
cp .env.example .env
```

```env
MLFLOW_TRACKING_URI=mlruns
MLFLOW_EXPERIMENT_NAME=AutoML_Experiment
```

---

## 📊 Datasets de Exemplo

O app inclui 4 datasets prontos no botão "Or try a sample dataset":

| Dataset | Tipo | Uso |
|---------|------|-----|
| 🌸 Iris | Classificação | 4 features, 3 classes |
| 🚢 Titanic | Classificação | sobrevivência, dados mistos |
| 🏠 California Housing | Regressão | preço de imóveis |
| 📈 Air Passengers | Time Series | passageiros mensais 1949–1960 |

---

## 🧪 Testes

```bash
# Todos os testes
python -m pytest tests/ -v

# Só engine tests
python -m pytest tests/test_engine.py -v

# Só time series tests
python -m pytest tests/test_time_series.py -v
```

**36/36 testes passando** (18 engine + 18 time series)

---

## 🔍 MLflow UI

Com o app rodando, acesse o MLflow em:

```bash
mlflow ui --port 5000
```

Ou via Docker: http://localhost:5000

---

## 🐳 Docker

```yaml
# docker-compose.yml inclui:
# - automl-studio: app Streamlit na porta 8501
# - mlflow: tracking server na porta 5000
```

```bash
# Build e start
docker compose up --build

# Só o app
docker compose up automl-studio

# Stop
docker compose down
```

---

## 🛠️ Stack Tecnológico

| Categoria | Bibliotecas |
|-----------|------------|
| UI | Streamlit 1.41+, Plotly 5.24+ |
| ML | scikit-learn 1.6+, XGBoost 2.1+, LightGBM 4.6+ |
| HPO | Optuna 4.2+ |
| Tracking | MLflow 2.20+ |
| Data | pandas 2.2+, numpy 1.26+, scipy 1.11+ |
| Infra | Docker, joblib |

---

## 📄 License

MIT License — veja [LICENSE](LICENSE) para detalhes.
