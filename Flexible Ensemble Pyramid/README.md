# Flexible Ensemble Pyramid

Projeto experimental de ensemble hierárquico focado em NLP de sentimento, com interface Streamlit e visualização detalhada das camadas do ensemble.

## Métodos de ensemble disponíveis

- Bagging
- Voting
- Stacking
- Boosting (sobre saídas da camada anterior)

## Catálogo de modelos

Modelos base suportados:

- Logistic Regression (`lr`)
- Linear SVC calibrado (`svc`)
- Multinomial Naive Bayes (`nb`)
- Ridge Classifier calibrado (`ridge`)
- Random Forest (`rf`)
- Extra Trees (`et`)
- AdaBoost (`ada`)
- XGBoost (`xgb`, quando `xgboost` está instalado)
- LightGBM (`lgbm`, quando `lightgbm` está instalado)
- CatBoost (`catboost`, quando `catboost` está instalado)

Modelos meta por camada:

- `stack_prev` (stacking da camada anterior)
- `bag_prev` (bagging da camada anterior)
- `vote_prev` (voting da camada anterior)
- `boost_prev` (boosting da camada anterior)

## Arquivos principais

- `flexible_ensemble_pyramid.py`: lógica principal do ensemble, tracking e treino.
- `flexible_ensemble_pyramid_ui_enhanced.py`: interface Streamlit para configuração, treino e análise visual.

## Execução direta

```bash
streamlit run "Flexible Ensemble Pyramid/flexible_ensemble_pyramid_ui_enhanced.py"
```

## Uso com CSV próprio

O app aceita CSVs com:

- coluna obrigatória `text`
- coluna obrigatória `sentiment`
- colunas opcionais `tweet_id` e `entity`

Se apenas o arquivo de treino for enviado, a validação é criada automaticamente por split.

## Dataset padrão do projeto

Quando você não usa upload, o app carrega o dataset padrão apenas deste projeto, no caminho:

- `Flexible Ensemble Pyramid/data/raw/twitter_training.csv`
- `Flexible Ensemble Pyramid/data/raw/twitter_validation.csv`
