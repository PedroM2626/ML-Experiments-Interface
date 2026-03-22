# ML-Experiment-Projects

Repositório que agrupa 3 projetos de Machine Learning com interface em Streamlit:

- `MLine`: plataforma AutoML inspirada em IBM watsonx, com gerenciamento de experimentos, MLflow, relatórios e suporte a tabular, NLP e séries temporais.
- `sagemaker based`: studio no-code inspirado em AWS SageMaker Canvas para datasets, build, modelos e predições.
- `Flexible Ensemble Pyramid`: experimento de ensemble hierárquico com foco em NLP de sentimento e visualização das camadas do ensemble.

Além dos projetos individuais, este repositório agora possui um hub unificado em Streamlit na raiz, para abrir os 3 workspaces a partir de uma única interface.

## O que foi centralizado

- `app.py` na raiz virou o ponto de entrada principal do repositório.
- O hub sobe cada projeto em background quando necessário.
- Cada workspace é exibido dentro da interface principal, evitando ficar abrindo um app separado por vez.
- O projeto `Flexible Ensemble Pyramid` também foi ajustado para funcionar melhor neste repositório e aceitar CSVs próprios.

## Estrutura do repositório

```text
ML-Experiment-Projects/
|-- app.py
|-- README.md
|-- MLine/
|   |-- app.py
|   |-- requirements.txt
|   |-- src/
|   |-- tests/
|-- Flexible Ensemble Pyramid/
|   |-- flexible_ensemble_pyramid.py
|   |-- flexible_ensemble_pyramid_ui_enhanced.py
|   |-- README.md
|-- sagemaker based/
|   |-- app.py
|   |-- requirements.txt
|   |-- pages/
|   |-- src/
|   |-- tests/
```

## Visão geral dos projetos

### 1. MLine

Projeto de AutoML com experiência inspirada diretamente no IBM watsonx AutoAI.

Principais pontos:

- classificação, regressão, séries temporais e classificação de texto;
- fluxo completo de experimentos, pipelines, leaderboard e inferência;
- integração com MLflow;
- geração de relatórios e componentes de explicabilidade.

Pasta: `MLine/`

Entrada direta: `MLine/app.py`

### 2. MLaker

Projeto inspirado em SageMaker Canvas com foco em fluxo no-code.

Principais pontos:

- upload e profiling de datasets;
- build guiado de modelos tabulares;
- páginas separadas para datasets, modelos e histórico de predições;
- integração com AutoGluon, FLAML e MLflow.

Pasta: `sagemaker based/`

Entrada direta: `sagemaker based/app.py`

### 3. Flexible Ensemble Pyramid

Projeto experimental voltado a ensemble hierárquico para NLP.

Principais pontos:

- múltiplas camadas de modelos com estratégias diferentes;
- visualização da pirâmide, conexões e heatmaps;
- integração com MLflow;
- configuração de RL meta-learner e NAS opcional.

Pasta principal:

- `Flexible Ensemble Pyramid/`

Arquivos principais:

- `Flexible Ensemble Pyramid/flexible_ensemble_pyramid.py`
- `Flexible Ensemble Pyramid/flexible_ensemble_pyramid_ui_enhanced.py`

## Novo hub unificado

O hub principal fica no arquivo:

```bash
streamlit run app.py
```

Fluxo do hub:

- abre uma página central com visão geral dos 3 projetos;
- ao selecionar um workspace, o hub inicia o app correspondente em background;
- o app é exibido dentro da própria interface principal;
- se preferir, também é possível abrir a URL direta do workspace.

Portas padrão usadas pelo hub:

- `8501`: hub principal
- `8502`: `MLine`
- `8503`: `sagemaker based`
- `8504`: `Flexible Ensemble Pyramid`

## Como executar

### Opção 1. Rodar apenas o hub

Use esta opção se você quer uma entrada única para tudo:

```bash
streamlit run app.py
```

### Opção 2. Rodar projetos individualmente

Se quiser abrir algum projeto fora do hub:

```bash
streamlit run "MLine/app.py"
streamlit run "sagemaker based/app.py"
streamlit run "Flexible Ensemble Pyramid/flexible_ensemble_pyramid_ui_enhanced.py"
```

## Instalação recomendada

Como os projetos têm dependências diferentes, a forma mais simples é usar um ambiente virtual único e instalar os requisitos dos subprojetos.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r .\MLine\requirements.txt
pip install -r ".\sagemaker based\requirements.txt"
```

Para o projeto da pirâmide, garanta pelo menos estas bibliotecas no mesmo ambiente:

```powershell
pip install streamlit pandas numpy scikit-learn matplotlib plotly seaborn joblib python-dotenv
```

Dependências opcionais do projeto da pirâmide:

- `mlflow`
- `dagshub`

Se elas não estiverem instaladas, parte do tracking pode cair para modo simplificado.

## Uso do Flexible Ensemble Pyramid com CSV próprio

O app da pirâmide agora aceita dataset próprio pela interface.

Formato esperado:

- coluna obrigatória `text`
- coluna obrigatória `sentiment`
- colunas opcionais `tweet_id` e `entity`

Comportamento:

- se você enviar apenas um CSV de treino, o app divide automaticamente treino e validação;
- se enviar treino e validação separados, ambos são usados no fluxo;
- o dataset padrão do repositório continua sendo suportado quando os arquivos existirem localmente.

## Artefatos gerados

Dependendo do projeto usado, o repositório pode gerar ou atualizar:

- `MLine/exports/`
- `MLine/mlruns/`
- `sagemaker based/models/`
- `sagemaker based/mlruns/`
- `experiments/artifacts/` no projeto da pirâmide
- logs do hub em `.hub/logs/`

## Testes e validação

Testes já existentes no repositório:

- `MLine/tests/`
- `sagemaker based/tests/`

Comandos úteis:

```bash
pytest MLine/tests
pytest "sagemaker based/tests"
```

O hub da raiz é um orquestrador de interface. Ele não altera a lógica dos projetos filhos, apenas centraliza a execução e o acesso.

## Quando usar cada projeto

Use `MLine` quando quiser:

- explorar AutoML com foco mais amplo;
- acompanhar pipelines e experimentos com mais detalhe;
- trabalhar com problemas tabulares, texto e séries temporais.

Use `AutoML Studio` quando quiser:

- uma experiência mais parecida com um produto no-code;
- fluxo guiado de dataset para build e predição;
- navegação multipágina ao estilo Studio/Canvas.

Use `Flexible Ensemble Pyramid` quando quiser:

- experimentar ensembles hierárquicos;
- estudar como os modelos se comportam por camada;
- visualizar relações e evolução de métricas em NLP.

## Observações importantes

- O hub depende de `streamlit` instalado no ambiente atual.
- Se uma porta já estiver ocupada, o hub pode não conseguir iniciar o workspace correspondente naquela porta.
- Os projetos continuam independentes. O hub apenas os centraliza visualmente.
- Alguns apps possuem dependências pesadas, especialmente os fluxos com AutoGluon, TensorFlow, SHAP e MLflow.

## Próximos passos sugeridos

- padronizar um `requirements.txt` raiz para o repositório inteiro;
- adicionar health checks mais avançados para os workspaces filhos;
- consolidar logs e configuração em uma camada comum;
- criar uma navegação compartilhada para reduzir duplicação entre os apps.
