# Status Técnico - TCC USP (Sentimento x Ibovespa)

- **Dados locais (não versionados):** `C:/TCC_USP/data_processed/`
- **Período oficial (hard cap):** 2018-01-02 a 2024-12-31

## Comandos principais
- Pipeline mínimo (clamp datas + duplicatas básicas):  
  `python scripts/pipeline_minimal.py`
- Validação de integridade (gera `reports/data_integrity_report.md`):  
  `python scripts/data_integrity_report.py`
- Dashboard (host/porta via env vars `DASH_HOST`, `DASH_PORT`):  
  `python app_dashboard.py`
- Testes rápidos:  
  `python -m pytest -q`

## Checklist de conformidade
- `ibovespa_clean.csv`, `16_oof_predictions.csv`, `18_backtest_daily_curves.csv` dentro de 2018-01-02 → 2024-12-31.
- Backtest agregado (`18_backtest_results.csv`) sem coluna de data; valores usados apenas como métricas agregadas.
- Interseção IBOV x Sentimento registrada no relatório de integridade.
- Latência: `event_study_latency.csv` atualmente vazio → placeholder no dashboard até existirem eventos no período.

## Observações
- Dados permanecem apenas em `C:/TCC_USP` (não versionar artefatos).
- `.gitignore` bloqueia data_raw/, data_processed/, data_interim/, mlruns/ e caches.
