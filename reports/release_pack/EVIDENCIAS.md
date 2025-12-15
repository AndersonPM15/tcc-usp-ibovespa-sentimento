# EVIDENCIAS

Gerado em: 2025-12-15T17:24:41

## Datasets
- ibovespa_clean.csv: linhas=1737, min=2018-01-02, max=2024-12-30
- 16_oof_predictions.csv: linhas=2682, min=2019-08-05, max=2024-12-30
- 18_backtest_daily_curves.csv: linhas=8046, min=2019-08-05, max=2024-12-30
- event_study_latency.csv: 0 linhas
- Intersecao IBOV x Sentimento: 2682 dias

## Export de figuras
exit=0
[DEBUG] MODEL_OPTIONS carregados: ['logreg_l2', 'rf_200']
[DEBUG] RESULTS_DF shape: (4, 7)
[DEBUG] IBOV_DF shape: (1737, 10)
[DEBUG] SENTIMENT_DF shape: (1341, 4)
[DEBUG] Callback acionado: start=2018-01-02, end=2024-12-31, models=['logreg_l2', 'rf_200'], metric=auc
[DEBUG] Filtered IBOV rows=1737, SENT rows=1341, start=2018-01-02, end=2024-12-31
Figuras exportadas:
 - 01_ibovespa: html=reports\release_pack\FIGURES\01_ibovespa.html
 - 02_sentimento_medio: html=reports\release_pack\FIGURES\02_sentimento_medio.html
 - 03_comparativo_modelos: html=reports\release_pack\FIGURES\03_comparativo_modelos.html
 - 04_dispersao_sentimento_retorno: html=reports\release_pack\FIGURES\04_dispersao_sentimento_retorno.html
 - 05_correlacao_movel: html=reports\release_pack\FIGURES\05_correlacao_movel.html
 - 06_distribuicao_sentimento: html=reports\release_pack\FIGURES\06_distribuicao_sentimento.html
 - 07_latencia: html=reports\release_pack\FIGURES\07_latencia.html
 - 08_backtest: html=reports\release_pack\FIGURES\08_backtest.html


## Data integrity report
exit=0
[INFO] Relatório salvo em C:\TCC_USP\tcc-usp-ibovespa-sentimento\reports\data_integrity_report.md
[OK] Validação concluída sem dados fora do período oficial.


## Pytest
exit=0
..                                                                       [100%]
2 passed in 3.13s


## Probe HTTP (porta 8050)
exit=0
[DEBUG] MODEL_OPTIONS carregados: ['logreg_l2', 'rf_200']
[DEBUG] RESULTS_DF shape: (4, 7)
[DEBUG] IBOV_DF shape: (1737, 10)
[DEBUG] SENTIMENT_DF shape: (1341, 4)
PORT 127.0.0.1:8050 OPEN=True
HTTP 127.0.0.1:8050 STATUS=200

