# Data Integrity Report

Período oficial: 2018-01-02 a 2024-12-31 (hard cap)

| Arquivo | Existe | Linhas | Min | Max | >2024-12-31 | <2018-01-02 | Duplicatas | Coluna data | Status |
|---------|--------|--------|-----|-----|-------------|-------------|------------|-------------|--------|
| ibovespa_clean.csv | sim | 1737 | 2018-01-02 00:00:00 | 2024-12-30 00:00:00 | 0 | 0 | 0 | date | ok |
| 16_oof_predictions.csv | sim | 2682 | 2019-08-05 00:00:00 | 2024-12-30 00:00:00 | 0 | 0 | 0 | day | ok |
| 18_backtest_daily_curves.csv | sim | 8046 | 2019-08-05 00:00:00 | 2024-12-30 00:00:00 | 0 | 0 | 0 | day | ok |
| 18_backtest_results.csv | sim | 6 | None | None | 0 | 0 | 0 | - | ok |
| event_study_latency.csv | sim | 0 | None | None | 0 | 0 | 0 | event_day | ok |

## Cobertura e interseção IBOV x Sentimento
- Dias IBOV: 1737
- Dias Sentimento: 1341
- Interseção: 1341