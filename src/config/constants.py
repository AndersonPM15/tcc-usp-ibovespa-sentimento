"""
Constantes globais do projeto TCC USP - Sentimento x Ibovespa

Conforme PLANO DE PESQUISA - valores fixos e centralizados.
"""

from datetime import date

# ==============================================================================
# PERÍODO DE ANÁLISE (PLANO DE PESQUISA - FIXO)
# ==============================================================================

START_DATE = date(2018, 1, 2)    # Primeiro pregão de 2018
END_DATE = date(2025, 12, 31)    # Último pregão de 2025

# Formatos auxiliares para diferentes APIs/bibliotecas
START_DATE_STR = "2018-01-02"
END_DATE_STR = "2025-12-31"
START_DATE_GDELT = "20180102000000"  # GDELT format
END_DATE_GDELT = "20251231235959"

# ==============================================================================
# PARÂMETROS DE MODELAGEM
# ==============================================================================

# TimeSeriesSplit com Embargo (PLANO DE PESQUISA)
N_SPLITS_TIMESERIES = 5          # Walk-forward validation com 5 folds
EMBARGO_DAYS = 1                 # Gap de 1 dia entre treino e teste
N_BOOTSTRAP_SAMPLES = 1000       # Bootstrap para intervalo de confiança 95%
RANDOM_SEED = 42                 # Seed para reprodutibilidade

# ==============================================================================
# HORÁRIOS DE PREGÃO B3
# ==============================================================================

PREGAO_START_HOUR = 10           # 10:00 BRT
PREGAO_END_HOUR = 17             # 17:00 BRT
TIMEZONE_BR = "America/Sao_Paulo"

# ==============================================================================
# FONTES DE NOTÍCIAS
# ==============================================================================

# Fontes implementadas e em uso na pipeline oficial
NEWS_SOURCES = [
    "GDELT",
    "GNews",
    "RSS",
    "NewsAPI",
]

# Fonte planejada mas NÃO implementada (trabalho futuro)
# "CVM_FR"  # CVM Fatos Relevantes - requer integração com API CVM

# ==============================================================================
# TF-IDF E FEATURES
# ==============================================================================

# Parâmetros TF-IDF (PLANO DE PESQUISA)
TFIDF_MIN_DF = 2                 # Mínimo 2 documentos
TFIDF_MAX_DF = 0.95              # Máximo 95% dos documentos
TFIDF_NGRAM_RANGE = (1, 2)       # Unigrams + bigrams
TFIDF_MAX_FEATURES = 5000        # Top 5000 features

# Rolling windows para features temporais
ROLLING_WINDOWS = [3, 7, 14, 21, 30]  # dias

# ==============================================================================
# ESTUDO DE EVENTOS
# ==============================================================================

CAR_HORIZON_DAYS = 5             # Horizonte para CAR (Cumulative Abnormal Return)
EVENT_WINDOW = (-1, 1)           # Janela de evento: [-1, +1] dias

# ==============================================================================
# CAMINHOS DE ARQUIVOS CHAVE
# ==============================================================================

# Relativamente ao diretório data/
DATA_IBOV_FILENAME = "ibov_clean.csv"
DATA_NEWS_RAW = "news_raw_multisource.parquet"
DATA_NEWS_CLEAN = "news_clean_multisource.parquet"
DATA_FEATURES_TFIDF = "tfidf_daily_matrix.parquet"
DATA_OOF_PREDICTIONS = "oof_predictions.csv"
