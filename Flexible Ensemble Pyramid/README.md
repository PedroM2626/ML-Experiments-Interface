# Flexible Ensemble Pyramid

Pirâmide hierárquica de ensembles com otimização via Reinforcement Learning e Algoritmo Genético (NAS), focada em classificação de texto (sentiment analysis).

## Conceito

Uma pirâmide de *N* camadas onde cada camada treina um conjunto de classificadores. As saídas probabilísticas (predict_proba) de cada camada alimentam a camada seguinte como features — criando uma hierarquia que refina progressivamente as predições.

```
Camada 1                    Camada 2                      Camada 3
┌──────┐                   ┌──────┐                      ┌──────┐
│  LR  │                   │vote_prev│                   │bag_prev│
├──────┤                   ├──────┤                      ├──────┤
│ SVC  │ ── probs ──────►  │bag_prev│ ── probs ────────► │stack_prev│
├──────┤                   ├──────┤                      ├──────┤
│  RF  │                   │boost_prev│                  │vote_prev│
├──────┤                   └──────┘                      └──────┘
│  NB  │
└──────┘
    │                          │                              │
    └─────── features acumuladas ────────►  ───►  best model
```

**Camada 1**: modelos base (LR, SVC, RF, NB, XGBoost, etc.) treinados diretamente no TF-IDF.

**Camadas 2+**: meta-ensembles que operam sobre as probabilidades da camada anterior:
- `stack_prev` — regressão logística sobre as probs anteriores
- `vote_prev` — soft voting (média das probs)
- `bag_prev` — bagging com reamostragem dos modelos anteriores
- `boost_prev` — boosting estilo SAMME com pesos por modelo

## Otimização

### RL Meta-Learner (`RLMetaLearner`)

Algoritmo epsilon-greedy com memória persistente (`pyramid_rl_knowledge.json`). A cada execução:

- **Explora** (probabilidade `epsilon`): seleciona modelos aleatoriamente
- **Explota** (probabilidade `1 - epsilon`): seleciona os modelos com melhor performance histórica para aquela camada, com ruído gaussiano (Thompson Sampling-lite)
- Atualiza médias incrementais de F1, acurácia, precisão e recall por modelo/camada
- Persiste aprendizado entre execuções — melhora com o tempo

### Neural Architecture Search (`NASController`)

É um **algoritmo genético** que evolui arquiteturas completas de ensemble:

| Parâmetro | Valor default |
|---|---|
| População | 10 arquiteturas |
| Gerações | 5 |
| Taxa de mutação | 0.1 |
| Taxa de crossover | 0.7 |

Cada arquitetura codifica: número de camadas, tipos de modelo por camada e estratégia de conexão.

**Fluxo**: população inicial aleatória → avaliação por fitness (F1 - penalidade de complexidade + bônus de diversidade) → elitismo (top ⅓) → crossover de camadas → mutação (add/remove layers, troca modelos, troca estratégias) → próxima geração.

**Híbrido RL + NAS**: no `suggest_models`, o RL consulta o NAS 70% das vezes e usa conhecimento acumulado nos outros 30%.

## Estratégias de Conexão

Controlam como as features propagam entre camadas (inspiradas em arquiteturas de deep learning):

| Estratégia | Comportamento |
|---|---|
| `dense` | Acumula **todas** as camadas anteriores (features crescentes) |
| `residual` | Usa **apenas** a saída da camada imediatamente anterior |
| `simple` | Usa apenas a camada anterior (igual residual, sem acumulação) |

## Tipos de Camada

- **heterogeneous** (default): cada camada seleciona modelos distintos do catálogo
- **homogeneous**: todas as variações de um único modelo base (ex.: `lr__v1`, `lr__v2`, ... com hiperparâmetros escalonados)

## Feature Engineering

TF-IDF com parâmetros configuráveis: max_features (até 100k), ngrams ((1,1) a (2,2)), stop_words inglês, min_df=2, mais limpeza de tweets (URLs, menções, hashtags).

## Modelos Suportados

### Base (Camada 1)
| Modelo | Tag | Instalação |
|---|---|---|
| Logistic Regression | `lr` | sklearn |
| Linear SVC (calibrado) | `svc` | sklearn |
| Multinomial Naive Bayes | `nb` | sklearn |
| Ridge Classifier (calibrado) | `ridge` | sklearn |
| Random Forest | `rf` | sklearn |
| Extra Trees | `et` | sklearn |
| AdaBoost | `ada` | sklearn |
| XGBoost | `xgb` | `pip install xgboost` |
| LightGBM | `lgbm` | `pip install lightgbm` |
| CatBoost | `catboost` | `pip install catboost` |

### Meta-Ensemble (Camadas 2+)
- `stack_prev` — stacking com Logistic Regression
- `vote_prev` — soft voting
- `bag_prev` — bagging com reamostragem
- `boost_prev` — boosting com pesos

### Hiperparâmetros com Jitter
Quando `jitter=True`, cada instância de modelo recebe variação aleatória nos hiperparâmetros (C, n_estimators, learning_rate, etc.), criando diversidade mesmo dentro do mesmo tipo de modelo.

## Experimento

- **Tracking**: MLflow (local ou DagsHub)
- **Métricas por modelo**: F1, accuracy, precision, recall, AUC, duration
- **Early stopping**: interrompe se não houver melhora por `patience` camadas
- **Reprodutibilidade**: `set_seed()` controla seeds do Python, numpy e sklearn
- **Paralelismo**: parâmetro `n_jobs` para modelos e CV

## Instalação

```bash
# Clone o repositório
git clone <repo-url>
cd "Flexible Ensemble Pyramid"

# Dependências principais
pip install numpy pandas scikit-learn matplotlib seaborn streamlit plotly python-dotenv

# Opcionais (para modelos extras)
pip install xgboost lightgbm catboost

# Tracking (opcional)
pip install mlflow dagshub
```

## Uso

### Interface Gráfica (Streamlit)

```bash
streamlit run "Flexible Ensemble Pyramid/flexible_ensemble_pyramid_ui_enhanced.py"
```

Modos de velocidade:
- **Relâmpago**: 3 camadas, sem NAS, subsample 5k (para testes rápidos)
- **Equilibrado**: 6 camadas, NAS ativo, subsample 15k
- **Completo**: 12 camadas, NAS ativo, dataset completo

### Linha de Comando

```bash
python "Flexible Ensemble Pyramid/flexible_ensemble_pyramid.py" \
    --layers 8 \
    --seed 42 \
    --epsilon 0.15 \
    --strategy residual \
    --use_nas True \
    --nas_population 15 \
    --nas_generations 8 \
    --nas_mutation 0.15 \
    --nas_crossover 0.7 \
    --metric f1 \
    --jitter True \
    --layer_type heterogeneous \
    --subsample_train 20000
```

### Parâmetros

| Parâmetro | Default | Descrição |
|---|---|---|
| `--layers` | 12 | Número de camadas |
| `--seed` | 2007 | Semente aleatória |
| `--min_models` | 2 | Mínimo de modelos por camada |
| `--max_models` | 6 | Máximo de modelos por camada |
| `--epsilon` | 0.2 | Taxa de exploração RL |
| `--patience` | 3 | Early stopping |
| `--metric` | f1 | Métrica alvo (f1, accuracy) |
| `--strategy` | dense | dense, residual, simple |
| `--layer_type` | heterogeneous | heterogeneous, homogeneous |
| `--jitter` | True | Variação de hiperparâmetros |
| `--use_nas` | False | Ativa NAS |
| `--nas_population` | 10 | Tamanho da população |
| `--nas_generations` | 5 | Número de gerações |
| `--nas_mutation` | 0.1 | Taxa de mutação |
| `--nas_crossover` | 0.7 | Taxa de crossover |
| `--subsample_train` | 0 | Amostra de treino (0 = todos) |
| `--n_jobs` | 2 | Paralelismo |

## Dataset

O dataset padrão esperado:
```
data/raw/twitter_training.csv
data/raw/twitter_validation.csv
```

CSV com colunas: `tweet_id`, `entity`, `sentiment`, `text`. Sentimentos válidos: `Positive`, `Negative`, `Neutral`, `Irrelevant`.

No Streamlit é possível fazer upload de CSV próprio (colunas obrigatórias: `text` e `sentiment`). Se só o treino for enviado, a validação é gerada por split automático.

## Estrutura do Projeto

```
Flexible Ensemble Pyramid/
├── flexible_ensemble_pyramid.py          # Lógica principal (1440 linhas)
├── flexible_ensemble_pyramid_ui_enhanced.py  # Interface Streamlit
├── pyramid_rl_knowledge.json             # Memória persistente do RL
├── pyramid_results.csv                   # Resultados da última execução
├── pyramid_run_details.json              # Metadados completos
├── nas_knowledge.json                    # Conhecimento do NAS
├── best_pyramid_model.pkl                # Melhor modelo salvo
├── mlruns/                               # Tracking local MLflow
├── data/
│   └── raw/
│       ├── twitter_training.csv
│       └── twitter_validation.csv
└── README.md
```

## Tracking

- **Local**: MLflow salva em `mlruns/` (default)
- **DagsHub**: configure `DAGSHUB_TOKEN` no `.env` para tracking remoto em `https://dagshub.com/PedroM2626/experiments.mlflow`

Métricas logadas por modelo: F1, accuracy, precision, recall, AUC, duration. Artefatos: CSV de resultados, JSON de metadados, matriz de confusão, modelo pickle.
