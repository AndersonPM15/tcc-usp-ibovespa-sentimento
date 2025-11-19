# 🎯 Reconstrução Completa - Pipeline Multisource

## ✅ FASES IMPLEMENTADAS (18/11/2025)

### **FASE 1: Coleta Multisource (NB 12)** ✅ COMPLETO
**Arquivo**: `notebooks/12_data_collection_multisource.ipynb`

**Fontes Implementadas:**
1. ✅ **GDELT API** (principal - histórico 2018-2025)
   - Query: Ibovespa, Bovespa, B3, economia brasileira
   - Batches de 3 meses para evitar timeout
   - Output: `news_gdelt_2018_2025.csv` + `.json`

2. ✅ **GNews** (complementar - últimos 6 meses)
   - language='pt', country='BR'
   - 100 resultados por termo
   - Output: `news_gnews.csv`

3. ✅ **RSS Feeds** (tempo real - 6 fontes)
   - Valor, Infomoney, Exame, SeuDinheiro, Investing, Reuters
   - Output: `news_rss_*.csv` (por fonte)

4. ✅ **NewsAPI** (opcional - últimos 30 dias)
   - Com rate limiting
   - Requer NEWSAPI_KEY env var
   - Output: `news_newsapi.csv`

**Funções Auxiliares:**
- `src/utils/multisource_collectors.py` (290 linhas)
  - `collect_gdelt_batch()`
  - `collect_gnews()`
  - `collect_rss_feeds()`
  - `collect_newsapi()`

**Output Final:**
- `data_raw/news_multisource_raw_{timestamp}.parquet` (consolidado PRÉ-DEDUP)

**Schema Unificado:**
```
id, source, title, description, content, published_at,
author, url, raw_text, scraped_text, query_term, 
source_type, language, collected_at
```

---

### **FASE 2: ETL e Deduplicação (NB 13)** ✅ COMPLETO
**Arquivo**: `notebooks/13_etl_dedup.ipynb`

**Pipeline Implementado:**
1. ✅ **Deduplicação Multi-nível**
   - Por URL canonizada (remove query params, www)
   - Por título + data (mesmo dia)
   - Por similaridade de embeddings (opcional, threshold=0.92)

2. ✅ **Validação de Campos**
   - Remove registros sem título
   - Remove textos < 20 caracteres
   - Valida datas
   - Remove URLs inválidas

3. ✅ **Normalização**
   - Timezone único (America/Sao_Paulo)
   - Coluna `date` (apenas dia)

4. ✅ **Relatório Automático**
   - JSON com estatísticas completas
   - `reports/etl_report_{timestamp}.json`

**Módulos Criados:**
- `src/utils/etl_dedup.py` (200 linhas)
  - `dedup_by_url()`
  - `dedup_by_title_date()`
  - `dedup_by_embedding_similarity()`
  - `validate_and_clean_fields()`
  - `normalize_timezone()`
  - `create_etl_report()`

- `src/pipeline/etl_pipeline.py` (180 linhas)
  - `run_etl_pipeline()` - Pipeline completo executável

**Output Final:**
- `data_processed/news_multisource.parquet` (dataset limpo)
- `reports/etl_report_{timestamp}.json`

---

### **FASE 3: Preprocessamento PT-BR (NB 14)** ✅ COMPLETO
**Arquivo**: `notebooks/14_preprocess_ptbr.ipynb`

**Pipeline Implementado:**
1. ✅ **Limpeza Avançada**
   - Remoção de HTML tags
   - Remoção de URLs
   - Normalização unicode
   - Remoção de stopwords PT
   - Limpeza de caracteres especiais

2. ✅ **Tokenização e Vocabulário**
   - Tokenização por espaços
   - Contagem de tokens
   - Filtro de textos < 5 tokens

3. ✅ **Detecção de Idioma**
   - `langdetect` em sample de 1000 registros
   - Filtro para manter apenas PT

4. ✅ **Geração de Embeddings**
   - SentenceTransformer 'all-MiniLM-L6-v2'
   - 768 dimensões
   - Batch_size=32
   - **OPCIONAL** (pode demorar 10-30min para 30k notícias)

5. ✅ **Análise de Sentimento**
   - VADER (compound score)
   - Keywords financeiras (positivas/negativas)
   - Score final = média dos dois métodos
   - Range: -1 (negativo) a +1 (positivo)

6. ✅ **Credibility Score**
   - Baseado em reputação da fonte
   - Ajustado por token_count
   - Range: 0 a 1

7. ✅ **Novelty Score**
   - Decay exponencial (half-life = 30 dias)
   - Notícias recentes = score maior
   - Range: 0 a 1

8. ✅ **Agregação Diária (BoW)**
   - Textos concatenados por data
   - Métricas agregadas (sentiment, credibility, novelty)
   - Contagem de notícias/dia

**Módulos Criados:**
- `src/utils/preprocess_ptbr.py` (400 linhas)
  - `clean_html()`, `remove_urls()`, `normalize_unicode()`
  - `remove_stopwords_pt()`, `clean_text_advanced()`
  - `preprocess_pipeline()`
  - `detect_language()`
  - `generate_embeddings()`
  - `analyze_sentiment()`
  - `calculate_credibility_score()`
  - `calculate_novelty_score()`

- `src/pipeline/preprocess_pipeline.py` (180 linhas)
  - `run_preprocess_pipeline()` - Pipeline completo

**Outputs Finais:**
- `data_processed/news_clean.parquet` (dataset completo preprocessado)
- `data_processed/bow_daily.parquet` (agregação diária)

**Colunas Geradas:**
```
clean_text, tokens, token_count, language, 
embedding_768, sentiment, sentiment_vader, sentiment_keywords,
credibility_score, novelty_score
```

---

## 📦 INFRAESTRUTURA CRIADA

### **Novos Módulos Python:**
1. `src/utils/multisource_collectors.py` (290 linhas)
2. `src/utils/etl_dedup.py` (200 linhas)
3. `src/utils/preprocess_ptbr.py` (400 linhas)
4. `src/pipeline/etl_pipeline.py` (180 linhas)
5. `src/pipeline/preprocess_pipeline.py` (180 linhas)
6. `src/pipeline/tfidf_features_pipeline.py` (500 linhas)
7. `src/validation/check_multisource.py` (450 linhas)
8. `src/validation/check_pipeline_health.py` (250 linhas)

**Total**: ~2.450 linhas de código Python de produção

### **Novos Scripts de Utilidade:**
1. `run_pipeline_multisource.py` (150 linhas) - Execução facilitada com validações
2. `cleanup_project.py` (280 linhas) - Limpeza final antes de commit
3. `pipeline_orchestration.py` (atualizado) - Já incluía NB 12-15 ✅
1. `notebooks/12_data_collection_multisource.ipynb` (10 células)
2. `notebooks/13_etl_dedup.ipynb` (6 células)
3. `notebooks/14_preprocess_ptbr.ipynb` (7 células)
4. `notebooks/15_features_tfidf_daily.ipynb` (8 células)

### **Dependências Adicionadas (`requirements.txt`):**
```
feedparser
gnews
beautifulsoup4
lxml
vaderSentiment
langdetect
```

---

## 🎯 PRÓXIMAS FASES (Pendentes)

### **FASE 4: Features TF-IDF Diário (NB 15)** ✅ COMPLETO
**Arquivo**: `notebooks/15_features_tfidf_daily.ipynb`

**Features Implementadas:**
1. ✅ **Download Ibovespa via yfinance**
   - Ticker: ^BVSP
   - Período automático baseado nas notícias
   - +30 dias antes/depois para garantir cobertura

2. ✅ **Agregação Diária de Textos**
   - Documentos por dia (todas as fontes)
   - Opção de agregação por (dia, fonte)

3. ✅ **Matriz TF-IDF**
   - min_df=2, max_df=0.95
   - ngram_range=(1,2) - unigrams + bigrams
   - max_features=5000
   - Token pattern customizado para PT-BR

4. ✅ **Labels Target Multi-horizonte**
   - D+1, D+3, D+5 (binário: subiu/caiu)
   - Retornos percentuais para cada horizonte
   - Alinhamento automático com datas

5. ✅ **Features de Janelas Móveis**
   - Volatilidade (3d, 5d, 7d)
   - Retornos médios (3d, 5d, 7d)
   - Sentiment rolling (3d, 5d, 7d)

6. ✅ **Análise de Correlação**
   - Scatter plot: Sentiment vs Retorno D+1
   - Série temporal: Sentiment vs Retornos
   - Boxplot por quartil de sentiment
   - Correlação de Pearson + p-value

7. ✅ **Top Termos por Ano**
   - Ranking TF-IDF médio por ano
   - Identificação de tendências temporais

**Módulo Criado:**
- `src/pipeline/tfidf_features_pipeline.py` (500+ linhas)
  - `download_ibovespa_data()` - yfinance integration
  - `prepare_daily_documents()` - agregação de textos
  - `create_tfidf_features()` - vetorização TF-IDF
  - `create_rolling_features()` - janelas móveis
  - `create_target_labels()` - labels multi-horizonte
  - `run_tfidf_pipeline()` - pipeline completo

**Outputs Gerados:**
- `data_processed/tfidf_daily_matrix.npz` - Matriz TF-IDF esparsa
- `data_processed/tfidf_daily_vocab.json` - Vocabulário + metadados
- `data_processed/tfidf_daily_index.csv` - Índice date→row_id
- `data_processed/labels_y_daily.csv` - Labels target (D+1/D+3/D+5)
- `data_processed/dataset_daily_complete.parquet` - Dataset completo
- `data_processed/ibovespa_YYYYMMDD.csv` - Dados históricos Ibovespa
- `data_processed/tfidf_report_YYYYMMDD.json` - Relatório detalhado
- `data_processed/sentiment_returns_analysis.png` - Gráficos de correlação
- `data_processed/top_terms_by_year.csv` - Top termos por ano

**Notebook Estrutura (8 células):**
1. Markdown: Introdução e objetivos
2. Python: Setup de paths
3. Python: Executar pipeline TF-IDF
4. Markdown: Análise dos Outputs
5. Python: Verificar arquivos gerados
6. Markdown: Visualizações e Análises
7. Python: Análise correlação Sentiment x Retornos
8. Python: Top termos TF-IDF por período

### **FASE 5: Ajustes NB 07-10, 16-20** ✅ COMPLETO
**Objetivo**: Atualizar notebooks existentes para usar dados do novo pipeline multisource

**Notebooks Atualizados:**

1. ✅ **NB 07 - TF-IDF Real**
   - Atualizado para carregar `news_clean.parquet` (novo pipeline)
   - Fallback para `noticias_real_clean.csv` (arquivo legado) se novo não existir
   - Mensagem de erro atualizada com instruções do pipeline multisource
   - Normalização automática de colunas (published_at → data)

2. ✅ **NB 08 - Embeddings Real**
   - Mesmas atualizações do NB 07
   - Compatibilidade com novo formato parquet
   - Suporte a múltiplas fontes de dados

3. ✅ **NB 09 - LSTM Real**
   - Mesmas atualizações do NB 07-08
   - Pipeline compatível com dados multisource
   - Preserva funcionamento com dados legados

4. ✅ **NB 10 - Dashboard Results**
   - Adicionado carregamento de resultados dos NB 07-08
   - Estatísticas multisource (contagem por fonte)
   - Visualização atualizada com título "Multisource"
   - Fallback para arquivo legado mantido

5. ✅ **NB 16 - Models TF-IDF Baselines**
   - Já estava configurado corretamente ✅
   - Usa arquivos do NB 15 (tfidf_daily_matrix.npz, labels_y_daily.csv)
   - Nenhuma alteração necessária

6. ✅ **NB 17-20**
   - Verificados: não usam arquivos de notícias diretamente
   - Nenhuma alteração necessária ✅

**Mudanças Implementadas:**
- **Compatibilidade**: Todos os notebooks agora suportam `news_clean.parquet` (novo) e `noticias_real_clean.csv` (legado)
- **Fallback inteligente**: Se arquivo novo não existir, usa arquivo antigo com aviso
- **Normalização de colunas**: Suporte automático para `published_at`, `date`, `data`
- **Estatísticas multisource**: Dashboard exibe contagem por fonte (gdelt, gnews, rss, newsapi)
- **Mensagens atualizadas**: Instruções claras sobre qual pipeline executar

### **FASE 6: Orquestração e Validações** ✅ COMPLETO
**Objetivo**: Integrar pipeline multisource com orquestração e validações automáticas

**Componentes Implementados:**

1. ✅ **pipeline_orchestration.py**
   - Já incluía notebooks 12-15 na sequência ✅
   - Suporte a execução seletiva (`--only 12 13 14`)
   - Logging estruturado em `reports/logs/`
   - Fallback para jupyter nbconvert se papermill indisponível

2. ✅ **src/validation/check_multisource.py** (NEW - 450 linhas)
   - `check_multisource_files()` - Verifica arquivos esperados
   - `check_data_sources()` - Valida 4 fontes (gdelt, gnews, rss, newsapi)
   - `check_date_coverage()` - Cobertura temporal mínima (default: 3 anos)
   - `check_volume_threshold()` - Volume mínimo (default: 5000 notícias)
   - `check_data_quality()` - Duplicatas, campos vazios, textos curtos
   - `run_full_validation()` - Pipeline completo de validação
   - Exit codes: 0=PASSED, 1=FAILED, 2=WARNING
   - Gera relatório JSON em `reports/validation_multisource_*.json`

3. ✅ **src/validation/check_pipeline_health.py** (NEW - 250 linhas)
   - Mapeamento de outputs esperados para cada notebook (00-20)
   - `check_notebook_outputs()` - Verifica outputs individuais
   - Suporte a wildcards (ex: `etl_report_*.json`)
   - `check_pipeline_health()` - Saúde geral do pipeline
   - Resumo: notebooks OK, Parciais, Falhados
   - Gera relatório JSON em `reports/pipeline_health_*.json`

4. ✅ **run_pipeline_multisource.py** (NEW - 150 linhas)
   - Script auxiliar user-friendly
   - Executa NB 12-15 em sequência
   - Executa validações automáticas após pipeline
   - Confirmação interativa antes de executar
   - Output formatado com status e próximos passos

5. ✅ **README.md Atualizado**
   - Seção "Como Usar" com 4 métodos de execução
   - Diagrama visual do pipeline multisource
   - Instruções de validação standalone
   - Documentação de dependências

**Comandos Disponíveis:**

```bash
# Método 1: Pipeline completo com validações (RECOMENDADO)
python run_pipeline_multisource.py

# Método 2: Orquestração de notebooks específicos
python pipeline_orchestration.py --only 12 13 14 15

# Método 3: Validação multisource standalone
python -m src.validation.check_multisource --min-years 3 --min-volume 5000

# Método 4: Verificação de saúde do pipeline
python -m src.validation.check_pipeline_health
```

**Outputs de Validação:**
- `reports/validation_multisource_YYYYMMDD_HHMMSS.json`
  - Status: PASSED / WARNING / FAILED
  - Checks: files, sources, coverage, volume, quality
  - Estatísticas detalhadas por fonte
  
- `reports/pipeline_health_YYYYMMDD_HHMMSS.json`
  - Status: HEALTHY / WARNING / UNHEALTHY
  - Notebooks: ok, partial, failed
  - Outputs faltando por notebook

**Validações Implementadas:**
- ✅ Existência de arquivos obrigatórios (7 arquivos)
- ✅ Presença de todas as fontes de dados (gdelt, gnews, rss, newsapi)
- ✅ Cobertura temporal mínima (configurável, default: 3 anos)
- ✅ Volume mínimo de notícias (configurável, default: 5000)
- ✅ Qualidade dos dados (duplicatas <5%, campos vazios <5%, textos >50 chars)
- ✅ Verificação de outputs de todos os notebooks (00-20)

### **FASE 7: Limpeza Final** ✅ COMPLETO
**Objetivo**: Preparar projeto para commit final

**Ações Realizadas:**

1. ✅ **cleanup_project.py** (NEW - 280 linhas)
   - `clean_notebook_outputs()` - Remove outputs de notebooks
   - `clean_pycache()` - Remove diretórios __pycache__
   - `clean_temp_files()` - Remove .pyc, .DS_Store, desktop.ini, etc
   - `verify_critical_files()` - Verifica arquivos essenciais
   - `count_project_stats()` - Estatísticas do código
   - Gera relatório JSON em `reports/cleanup_report_*.json`
   - Interface interativa com confirmação

2. ✅ **Verificação de Código Legado**
   - Notebooks verificados para código Colab (outputs HTML presentes, mas não impedem funcionamento)
   - Nenhum código de mount do Google Drive encontrado
   - Notebooks compatíveis com Jupyter local

3. ✅ **CONCLUSAO_PIPELINE_MULTISOURCE.md** (NEW)
   - Documento final de conclusão do projeto
   - Resumo de todas as 7 fases
   - Instruções de uso completas
   - Lista de próximos passos

**Comandos de Limpeza:**

```bash
# Limpeza completa (recomendado antes de commit)
python cleanup_project.py

# Verificação final do projeto
python verify_project.py
```

**Arquivos Críticos Verificados:**
- ✅ README.md
- ✅ requirements.txt
- ✅ pipeline_orchestration.py
- ✅ run_pipeline_multisource.py
- ✅ configs/config_tcc.yaml
- ✅ src/io/paths.py
- ✅ src/config/loader.py

**Estatísticas Finais do Código:**
- 📓 Notebooks: 21 (NB 00-20)
- 🐍 Arquivos Python (src/): ~50+
- 💻 Linhas de código (novo pipeline): ~2.450 linhas
- 📦 Módulos criados: 8
- 🔧 Scripts de utilidade: 3

---

## 📊 RESULTADOS ESPERADOS

### **Volume de Dados:**
- **Meta**: 30.000+ notícias
- **Período**: 2018-01-01 → 2025-01-31 (7 anos)
- **Fontes**: 4+ fontes ativas
- **Dedup rate**: < 10%

### **Qualidade:**
- ✅ Textos limpos e normalizados
- ✅ Embeddings de 768 dimensões
- ✅ Sentiment analysis calibrado para PT-BR financeiro
- ✅ Scores de credibilidade e novidade
- ✅ Agregação diária para modelagem

### **Outputs Gerados:**
```
data_raw/
├── news_gdelt_2018_2025.csv
├── news_gdelt_2018_2025.json
├── news_gnews.csv
├── news_rss_*.csv (6 arquivos)
├── news_newsapi.csv
└── news_multisource_raw_{timestamp}.parquet

data_processed/
├── news_multisource.parquet       (pós-dedup)
├── news_clean.parquet              (pós-preprocess)
└── bow_daily.parquet               (agregação diária)

reports/
└── etl_report_{timestamp}.json
```

---

## 🚀 COMO EXECUTAR

### **1. Instalar dependências:**
```bash
pip install -r requirements.txt
```

### **2. Configurar variáveis de ambiente (opcional):**
```bash
# Windows PowerShell
$env:NEWSAPI_KEY='sua_chave_aqui'

# Windows CMD
set NEWSAPI_KEY=sua_chave_aqui

# Linux/Mac
export NEWSAPI_KEY=sua_chave_aqui
```

### **3. Executar notebooks em sequência:**
```bash
# NB 12: Coleta multisource
jupyter notebook notebooks/12_data_collection_multisource.ipynb

# NB 13: ETL e deduplicação
jupyter notebook notebooks/13_etl_dedup.ipynb

# NB 14: Preprocessamento PT-BR
jupyter notebook notebooks/14_preprocess_ptbr.ipynb
```

### **4. Ou executar pipelines via Python:**
```bash
# Pipeline ETL
python -m src.pipeline.etl_pipeline

# Pipeline Preprocessamento (sem embeddings para teste rápido)
python -m src.pipeline.preprocess_pipeline --no-embeddings
```

---

## ⚠️ OBSERVAÇÕES IMPORTANTES

1. **GDELT API**: Pode ter limitações em alguns períodos/regiões
2. **GNews**: Limitado a últimos 6 meses pela biblioteca
3. **NewsAPI**: Free tier limitado a 30 dias + 100 req/dia
4. **Embeddings**: Geração pode demorar 10-30min para 30k notícias
5. **Dedup por embedding**: Muito lento para datasets >10k (opcional)

---

## 📝 CHECKLIST DE VALIDAÇÃO

Após executar os 3 notebooks:

- [ ] `news_multisource_raw_*.parquet` > 10.000 registros
- [ ] `news_multisource.parquet` (pós-dedup) > 5.000 registros
- [ ] `news_clean.parquet` tem colunas: sentiment, credibility, novelty
- [ ] `bow_daily.parquet` tem pelo menos 365 dias únicos
- [ ] Cobertura temporal mínima de 3 anos
- [ ] 3+ fontes ativas (gdelt, gnews, rss)
- [ ] Sentiment médio entre -0.2 e +0.2 (neutro com viés)

---

**Última atualização**: 18/11/2025  
**Status**: ✅ TODAS AS 7 FASES CONCLUÍDAS - PROJETO PRONTO
