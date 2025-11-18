# Relatório de Verificação Integral do Projeto TCC USP

**Data:** 2025-11-18 16:09:02

**Período esperado:** 2018-01-01 → 2025-01-31

## 1. Cobertura de Dados (data_processed)

| Arquivo | Existe | Min Date | Max Date | Rows | Status |
|---------|--------|----------|----------|------|--------|
| ibovespa_clean.csv | ✓ | 2024-01-02 | 2024-01-29 | 20 | OK |
| news_clean.parquet | ✓ | 2025-09-19 | 2025-10-17 | 100 | OK |
| news_multisource.parquet | ✗ | None | None | 0 | OK |
| noticias_real_clean.parquet | ✗ | None | None | 0 | OK |
| labels_y_daily.csv | ✓ | 2025-09-19 | 2025-10-17 | 16 | OK |
| 16_oof_predictions.csv | ✓ | 2025-10-02 | 2025-10-17 | 24 | OK |
| event_study_latency.csv | ✓ | None | None | 0 | OK |
| tfidf_daily_index.csv | ✓ | 2025-09-19 | 2025-10-17 | 16 | OK |
| 18_backtest_results.csv | ✓ | 1970-01-01 | 1970-01-01 | 6 | OK |

## 2. Fontes de Dados e Coleta

| Notebook | Existe | Fonte | Output | Descrição |
|----------|--------|-------|--------|-----------|
| 00_data_download.ipynb | ✓ | yfinance | ibovespa_clean.csv | Download histórico Ibovespa via yfinance |
| 05_data_collection_real.ipynb | ✓ | NewsAPI, Reuters, InfoMoney, Valor | noticias_real_clean.parquet | Coleta de notícias de fontes reais |
| 12_data_collection_multisource.ipynb | ✓ | Multiple news sources | news_multisource.parquet | Agregação de múltiplas fontes de notícias |
| 13_etl_dedup.ipynb | ✓ | CVM, processed news | news_clean.parquet | ETL e deduplicação de notícias |

## 3. Auditoria de Notebooks

| Notebook | Existe | paths.py | config | Issues |
|----------|--------|----------|--------|--------|
| 00_data_download | ✓ | ✓ | ✓ | 0 |
| 01_preprocessing | ✓ | ✓ | ✓ | 0 |
| 02_baseline_logit | ✓ | ✓ | ✓ | 1 |
| 03_tfidf_models | ✓ | ✓ | ✓ | 1 |
| 04_embeddings_models | ✓ | ✓ | ✓ | 1 |
| 05_data_collection_real | ✓ | ✓ | ✓ | 0 |
| 06_preprocessing_real | ✓ | ✓ | ✓ | 1 |
| 07_tfidf_real | ✓ | ✓ | ✓ | 1 |
| 08_embeddings_real | ✓ | ✓ | ✓ | 1 |
| 09_lstm_real | ✓ | ✓ | ✓ | 1 |
| 10_dashboard_results | ✓ | ✓ | ✓ | 1 |
| 11_event_study_latency | ✓ | ✓ | ✓ | 1 |
| 12_data_collection_multisource | ✓ | ✓ | ✓ | 1 |
| 13_etl_dedup | ✓ | ✓ | ✓ | 1 |
| 14_preprocess_ptbr | ✓ | ✓ | ✓ | 1 |
| 15_features_tfidf_daily | ✓ | ✓ | ✓ | 0 |
| 16_models_tfidf_baselines | ✓ | ✓ | ✓ | 0 |
| 17_sentiment_validation | ✓ | ✓ | ✗ | 0 |
| 18_backtest_simulation | ✓ | ✓ | ✓ | 0 |
| 19_future_extension | ✓ | ✗ | ✗ | 0 |
| 20_final_dashboard_analysis | ✓ | ✓ | ✓ | 0 |

## 4. Dashboard (app_dashboard.py)

Status do dashboard será verificado separadamente via execução manual.

## 5. Recomendações

### Arquivos Ausentes
- news_multisource.parquet
- noticias_real_clean.parquet

### Notebooks com Issues

**02_baseline_logit:**
- Hardcoded Colab path without paths.py

**03_tfidf_models:**
- Hardcoded Colab path without paths.py

**04_embeddings_models:**
- Hardcoded Colab path without paths.py

**06_preprocessing_real:**
- Hardcoded Colab path without paths.py

**07_tfidf_real:**
- Hardcoded Colab path without paths.py

**08_embeddings_real:**
- Hardcoded Colab path without paths.py

**09_lstm_real:**
- Hardcoded Colab path without paths.py

**10_dashboard_results:**
- Hardcoded Colab path without paths.py

**11_event_study_latency:**
- Hardcoded Colab path without paths.py

**12_data_collection_multisource:**
- Hardcoded Colab path without paths.py

**13_etl_dedup:**
- Hardcoded Colab path without paths.py

**14_preprocess_ptbr:**
- Hardcoded Colab path without paths.py