# Final Data Audit

Arquivo | Existe | Linhas | Min Data | Max Data | Missing Cols | Status
---|---|---|---|---|---|---
ibovespa_clean.csv | True | 1737 | 2018-01-02 00:00:00 | 2024-12-30 00:00:00 | [] | OK
16_oof_predictions.csv | True | 2682 | 2019-08-05 00:00:00 | 2024-12-30 00:00:00 | [] | OK
18_backtest_daily_curves.csv | True | 8046 | 2019-08-05 00:00:00 | 2024-12-30 00:00:00 | [] | OK
18_backtest_results.csv | True | 6 | None | None | [] | OK
event_study_latency.csv | True | 270 | 2019-08-08 00:00:00 | 2024-12-27 00:00:00 | [] | OK
results_16_models_tfidf.json | True | 2 | None | None | [] | OK

## Nulos por coluna (apenas chaves)
### ibovespa_clean.csv
- date: 0.00% nulos
- close: 0.00% nulos
- return: 0.06% nulos

### 16_oof_predictions.csv
- model: 0.00% nulos
- day: 0.00% nulos
- close: 0.00% nulos
- proba: 0.00% nulos

### 18_backtest_daily_curves.csv
- model: 0.00% nulos
- day: 0.00% nulos
- proba: 0.00% nulos
- close: 0.00% nulos
- sentiment: 0.00% nulos
- equity: 0.00% nulos
- strategy: 0.00% nulos

### 18_backtest_results.csv
- model: 0.00% nulos
- strategy: 0.00% nulos

### event_study_latency.csv
- event_day: 0.00% nulos
- car_max_abs: 0.00% nulos

### results_16_models_tfidf.json
Nenhuma coluna-chave ou sem nulos relevantes.
