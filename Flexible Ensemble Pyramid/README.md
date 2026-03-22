# Flexible Ensemble Pyramid

Projeto experimental de ensemble hierárquico focado em NLP de sentimento, com interface Streamlit e visualização detalhada das camadas do ensemble.

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
