"""
Constantes globais do projeto TCC USP - Sentimento x Ibovespa
Centraliza datas, parâmetros e configurações fixas do plano de pesquisa.
"""

from datetime import date

# ============================================================================
# PERÍODO DE ANÁLISE (FIXO - PLANO DE PESQUISA)
# ============================================================================
# Data inicial: primeiro pregão de 2018
START_DATE = date(2018, 1, 2)

# Data final: último pregão de 2025 (ou último disponível)
END_DATE = date(2025, 12, 31)

# Formatações úteis
START_DATE_STR = START_DATE.strftime("%Y-%m-%d")
END_DATE_STR = END_DATE.strftime("%Y-%m-%d")
START_DATE_GDELT = START_DATE.strftime("%Y%m%d%H%M%S")  # Formato GDELT API
END_DATE_GDELT = END_DATE.strftime("%Y%m%d235959")

# ============================================================================
# PARÂMETROS DE MODELAGEM (PLANO DE PESQUISA)
# ============================================================================
# Walk-forward validation
N_SPLITS_TIMESERIES = 5  # Fixo conforme plano

# Embargo entre treino e teste (em dias)
EMBARGO_DAYS = 1  # Evita look-ahead bias

# Bootstrap para intervalos de confiança
N_BOOTSTRAP_SAMPLES = 1000
RANDOM_SEED = 42

# ============================================================================
# HORÁRIOS DE PREGÃO B3
# ============================================================================
# Horário oficial do pregão regular (horário de Brasília - America/Sao_Paulo)
PREGAO_START_HOUR = 10  # 10:00
PREGAO_END_HOUR = 17    # 17:00 (encerramento 17:30, mas usamos 17 como limite)

# Timezone padrão
TIMEZONE_BR = "America/Sao_Paulo"

# ============================================================================
# FONTES DE NOTÍCIAS
# ============================================================================
NEWS_SOURCES = {
    "gdelt": "GDELT Project",
    "gnews": "Google News",
    "rss": "RSS Feeds",
    "newsapi": "NewsAPI",
    "cvm": "CVM - Fatos Relevantes"
}

# Termos de busca para coleta
SEARCH_TERMS = [
    "Ibovespa",
    "Bovespa",
    "B3",
    "economia brasileira",
    "mercado financeiro brasil"
]

# ============================================================================
# TF-IDF E FEATURE ENGINEERING
# ============================================================================
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MAX_FEATURES = 5000

# ============================================================================
# ESTUDO DE EVENTOS
# ============================================================================
CAR_HORIZON_DAYS = 5  # Horizonte para Cumulative Abnormal Returns

# ============================================================================
# CAMINHOS DE ARQUIVOS CHAVE
# ============================================================================
# Definidos dinamicamente via src.io.paths, mas podemos referenciar aqui
KEY_FILES = {
    "ibov_raw": "data_raw/ibovespa.csv",
    "ibov_clean": "data_processed/ibovespa_clean.csv",
    "news_multisource": "data_processed/news_multisource.parquet",
    "news_clean": "data_processed/news_clean.parquet",
    "tfidf_matrix": "data_processed/tfidf_daily_matrix.npz",
    "labels_daily": "data_processed/labels_y_daily.csv",
}
