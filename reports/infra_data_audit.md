## 1. Resumo executivo
- Repositório centralizado em C:/TCC_USP/tcc-usp-ibovespa-sentimento com entrypoints em app_dashboard.py, run_pipeline_complete.py e pipeline_orchestration.py; notebook pipeline 00-20 define etapas de dados/modelos.
- Dados locais disponíveis são mínimos: data_raw/ibovespa.csv (0.17 MB, 2025-11-26 16:12:35) e data_processed/ibovespa_clean.csv (0.17 MB, 2025-11-26 16:12:35) ambos com datas 2018-01-02 a 2025-11-18 → fora do escopo oficial (até 2024-12-31). data/results_registry.json (0.17 MB, 2025-11-26 17:16:53) guarda 312 logs de execuções.
- Pastas de artefatos esperados (data_interim, demais data_processed) estão vazias/ausentes; outputs críticos para dashboard (16_oof_predictions.csv, results_16_models_tfidf.json, event_study_latency.csv, 18_backtest_results.csv, tfidf_daily_matrix.npz) não existem no workspace.
- Poluição: venv completo versionado localmente; notebooks/_runs repleto de cópias (~0.75 MB cada); mlruns/ com múltiplas execuções; arquivos temporários (.tmp_diff_11.txt, tmp_dash_err.log, tmp_dash_out.log, tmp_test*.txt) e caches (__pycache__, .pytest_cache). Nenhum arquivo >1 MB fora de venv, mas o volume de runs/caches é alto.
- Risco principal: série histórica atual inclui 2025 e pode contaminar análises; falta de artefatos finais impede dashboard e backtests. Recomenda-se reset controlado dos dados e reexecução seletiva do pipeline dentro do período oficial.

## 2. Arquitetura atual (ASCII)
```
C:/TCC_USP/
├── tcc-usp-ibovespa-sentimento/ (repo)
│   ├── app_dashboard.py                (Dash app; consome ibov + OOF + latency + backtest)
│   ├── run_pipeline_complete.py        (orquestra notebooks 00-20)
│   ├── pipeline_orchestration.py       (helpers de execução)
│   ├── configs/
│   │   └── config_tcc.yaml             (BASE_PATH=C:/TCC_USP; período 2018-01-02 a 2024-12-31; arquivos_chave)
│   ├── src/
│   │   ├── io/paths.py                 (resolve BASE_PATH, DATA_RAW/PROCESSED/INTERIM, get_project_paths)
│   │   ├── config/loader.py            (load_config, get_arquivo)
│   │   ├── utils/logger.py             (logging), utils/merges.py
│   │   ├── validation/ (vazio)
│   │   └── pipeline/ (vazio)
│   ├── data_raw/                       (ibovespa.csv)
│   ├── data_processed/                 (ibovespa_clean.csv)
│   ├── data/                           (results_registry.json)
│   ├── notebooks/00-20*.ipynb          (pipeline analítico completo)
│   ├── notebooks/_runs/                (cópias executadas dos notebooks)
│   ├── mlruns/                         (artefatos MLflow; muitos runs)
│   ├── reports/                        (verificação prévia, outputs)
│   ├── venv/                           (ambiente Python completo)
│   └── testes: test_dashboard.py, test_data_period.py, verify_project.py
└── data_interim/ (esperado via config, não existe)
```

## 3. Catálogo de dados
| Caminho absoluto | Tamanho (MB) | Modificação | Linhas | Coluna(s) data | Min data | Max data | Status período | Classificação |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C:/TCC_USP/tcc-usp-ibovespa-sentimento/data_raw/ibovespa.csv | 0.17 | 2025-11-26 16:12:35 | 1960 | date | 2018-01-02 | 2025-11-18 | FORA DO ESCOPO | Bruto (download yfinance) |
| C:/TCC_USP/tcc-usp-ibovespa-sentimento/data_processed/ibovespa_clean.csv | 0.17 | 2025-11-26 16:12:35 | 1960 | date | 2018-01-02 | 2025-11-18 | FORA DO ESCOPO | Processado esperado para dashboard |
| C:/TCC_USP/tcc-usp-ibovespa-sentimento/data/results_registry.json | 0.17 | 2025-11-26 17:16:53 | 312 itens | timestamp | 2025-11-10 (mín. observada) | 2025-11-26 (máx. observada) | FORA DO ESCOPO (logs 2025) | Log de execuções (pipeline) |
| C:/TCC_USP/data_interim | - | - | - | - | - | - | NÃO EXISTE | Pasta intermediária ausente |
| C:/TCC_USP/data_processed | - | - | - | - | - | - | NÃO EXISTE | Pasta processada fora do repo ausente |
| C:/TCC_USP/data_raw | - | - | - | - | - | - | NÃO EXISTE | Pasta bruta fora do repo ausente |

### Artefatos esperados e ausentes (todos deveriam residir em C:/TCC_USP/data_processed)
- 16_oof_predictions.csv — não encontrado.
- results_16_models_tfidf.json — não encontrado.
- event_study_latency.csv (ou .parquet) — não encontrado.
- 18_backtest_results.csv — não encontrado.
- 18_backtest_daily_curves.csv — não encontrado.
- tfidf_daily_matrix.npz e tfidf_daily_vocab.json — não encontrados.

## 4. Lineage principal
| Artefato | Gerado por | Consumido por |
| --- | --- | --- |
| data_raw/ibovespa.csv | Notebook 00_data_download.ipynb (yfinance) | 01_preprocessing.ipynb; testes de período |
| data_processed/ibovespa_clean.csv | 00/01 (download + limpeza) | app_dashboard.py (IBOV_PATH); verify_project.py; testes |
| data_processed/16_oof_predictions.csv | 16_models_tfidf_baselines.ipynb | app_dashboard.py (SENTIMENT_PATH); verify_project.py |
| data_processed/results_16_models_tfidf.json | 16_models_tfidf_baselines.ipynb | app_dashboard.py (METRICS_PATH) |
| data_processed/event_study_latency.csv | 11_event_study_latency.ipynb | app_dashboard.py (LATENCY_PATH); verify_project.py |
| data_processed/18_backtest_results.csv | 18_backtest_simulation.ipynb | app_dashboard.py (BACKTEST_PATH); verify_project.py |
| data_processed/18_backtest_daily_curves.csv | 18_backtest_simulation.ipynb | app_dashboard.py (load_backtest_curves) |
| data_processed/tfidf_daily_matrix.npz (+ vocab/index) | 15_features_tfidf_daily.ipynb / 16_models_tfidf_baselines.ipynb | Modelagem TF-IDF (etapa 16); potencial insumo de métricas |
| data/results_registry.json | run_pipeline_complete.py / pipeline_orchestration.py (logging) | Auditoria manual (não usado pelo dashboard) |

## 5. Poluição e riscos
- venv/ completo dentro do repositório (diversos DLLs de TensorFlow/PyTorch >100 MB cada) — não deve ser versionado; manter fora ou em .gitignore.
- notebooks/_runs/ contém dezenas de cópias executadas (até ~0.75 MB cada, ex.: notebooks/_runs/20251126_165935_17_sentiment_validation.ipynb 0.73 MB) — manter como artefato local ou mover/limpar; não necessário para reprodução.
- mlruns/ com múltiplas execuções MLflow — avaliar tamanho total e ignorar no git.
- Arquivos temporários: .tmp_diff_11.txt (0.27 MB), tmp_dash_err.log, tmp_dash_out.log, tmp_test_file.txt, tmp_test2.txt — remover ou ignorar.
- Caches automáticos: __pycache__/, .pytest_cache/ — limpar/ignorar.
- Dados brutos/processados contêm datas até 2025-11-18 (fora do escopo oficial) — risco de contaminação analítica.

## 6. Plano recomendado de reset seguro
1) Higienizar repositório
   - Remover do controle de versão ou mover para fora: venv/, notebooks/_runs/, mlruns/, *.tmp, *.log, __pycache__/, .pytest_cache/. Adicionar estas entradas ao .gitignore antes de novo commit.
2) Reinstalar ambiente e dados-base
   - Recriar venv fora do repositório (ex.: C:/TCC_USP/venvs/tcc) e reinstalar requirements.txt.
   - Restaurar datasets brutos oficiais no período 2018-01-02 a 2024-12-31: rerun notebook 00_data_download.ipynb salvando em C:/TCC_USP/data_raw; validar datas antes de prosseguir.
3) Regerar artefatos finais necessários ao dashboard
   - Executar sequencialmente notebooks 01, 15, 16, 11, 18 apenas após confirmar bases brutas; verificar saída em data_processed/ (arquivos listados na seção 3). Rejeitar qualquer linha >2024-12-31 durante geração.

Checklist rápido
- [ ] .gitignore atualizado para venv, notebooks/_runs, mlruns, __pycache__, *.tmp, *.log
- [ ] data_raw/ibovespa.csv rebaixado para máx. 2024-12-31 e conferido
- [ ] data_processed/ibovespa_clean.csv refeito e dentro do escopo
- [ ] Artefatos: 16_oof_predictions.csv, results_16_models_tfidf.json, event_study_latency.csv, 18_backtest_results.csv, tfidf_daily_matrix.npz presentes e datados até 2024-12-31
- [ ] app_dashboard.py abre sem warnings (inputs carregados e validados)