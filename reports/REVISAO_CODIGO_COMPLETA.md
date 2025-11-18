# Revisão de Código Completa - TCC USP Ibovespa Sentimento

**Data:** 18 de novembro de 2025  
**Escopo:** Notebooks 00-20, módulos src/, scripts auxiliares  
**Período alvo:** 2018-01-01 → 2025-01-31

---

## 📋 Sumário Executivo

### ✅ Pontos Fortes do Projeto

1. **Arquitetura bem estruturada**: Módulos `src/` bem organizados (paths, config, validation, logger)
2. **Configuração centralizada**: `configs/config_tcc.yaml` com período de estudo e caminhos lógicos
3. **Notebooks exemplares**: 11 (event study), 12 (multisource), 13 (ETL), 14 (preprocess PT-BR), 16-18, 20
4. **Validações robustas**: `check_intersection()` e `log_result()` bem implementados
5. **Pipeline coerente**: Fluxo claro de dados → features → modelagem → métricas

### 🔴 Problemas Críticos Identificados

1. **[NB 09]** Bug fatal: variáveis `aucs`/`mdas` não definidas antes de salvar resultados
2. **[NB 10]** Dados hardcoded em vez de carregar arquivos JSON gerados
3. **[NB 01]** Dependência de arquivo inexistente: `noticias_exemplo.csv`
4. **[NB 00]** Não valida se download retornou dados reais do período 2018-2025
5. **[NB 06-09]** Criam fallbacks sintéticos silenciosamente quando não há dados reais

### 🟡 Problemas Importantes

6. **[NB 05, 12]** NewsAPI free tier limitado a ~30 dias (já documentado com warnings)
7. **[NB 15]** Outputs com caminhos Colab hardcoded nos metadados
8. **[NB 01-04, 06-09]** Células com comentário "Legacy Colab mount removido" que podem ser deletadas
9. **[Múltiplos NB]** Falta validação de período 2018-2025 após carregar dados
10. **[NB 19]** Placeholder vazio não implementado

---

## 🔍 Análise Detalhada por Categoria

### 1. Uso Correto de `paths.py` e `config.loader`

#### ✅ **EXCELENTE** (16 notebooks):
- **00, 05, 06, 07, 08, 09, 11, 12, 13, 14, 15, 16, 17, 18, 20**: Usam corretamente `paths.get_data_paths()` e `cfg.load_config()`

#### ⚠️ **PARCIAL** (4 notebooks):
- **01, 02, 03, 04**: Usam `paths.py` mas não usam `cfg.get_periodo_estudo()` para validar período
- **17**: Não usa `cfg.load_config()` (pode não ser necessário)

#### ❌ **INADEQUADO** (1 notebook):
- **10**: Não carrega dados de arquivos, usa valores hardcoded

---

### 2. Consistência dos Pipelines

```
┌──────────────────────────────────────────────────────────────────┐
│ PIPELINE PRINCIPAL (2018-2025)                                   │
└──────────────────────────────────────────────────────────────────┘

00: Download Ibovespa (yfinance) ────────┐
                                          ├─► 01: Preprocessing básico
05: Coleta NewsAPI ──────────────┐       │    (DEPENDE DE noticias_exemplo.csv ❌)
12: Coleta multisource ─────────►├──────►│
                                          │
13: ETL dedup & normalização ────────────┘
                                          
14: Preprocessing PT-BR (NLP) ───────────► ibovespa_clean.csv
                                          noticias_real_clean.csv

┌──────────────────────────────────────────────────────────────────┐
│ FEATURES & MODELAGEM                                             │
└──────────────────────────────────────────────────────────────────┘

15: Features TF-IDF daily ───────────────► tfidf_daily_matrix.npz
                                          labels_y_daily.csv

16: Models TF-IDF baselines ─────────────► 16_oof_predictions.csv
    (LogReg, RF, XGBoost)                 results_16_models_tfidf.json

17: Sentiment validation ────────────────► Correlações, ANOVA, lags

18: Backtest simulation ─────────────────► 18_backtest_results.csv
                                          (CAGR, Sharpe, drawdown)

20: Final dashboard analysis ────────────► Dashboard HTML interativo

┌──────────────────────────────────────────────────────────────────┐
│ PIPELINES ALTERNATIVOS (dados demo)                             │
└──────────────────────────────────────────────────────────────────┘

02: Baseline logit (demo) ───────────────► Métricas básicas
03: TF-IDF models (demo) ────────────────► Comparação de modelos
04: Embeddings models (demo) ────────────► SentenceTransformer

06-09: Pipeline "real" alternativo ──────► Duplica funcionalidade
       (CRIA FALLBACKS SINTÉTICOS ⚠️)       de 14-18 com dados "reais"

10: Dashboard results (HARDCODED ❌) ────► Valores fictícios

11: Event study latency ─────────────────► event_study_latency.csv
                                          (Meia-vida de reação)
```

#### 🔴 **Problemas de Dependências**:

1. **NB 01 → Arquivo ausente**: Depende de `noticias_exemplo.csv` que não é gerado pelo NB 00
2. **NB 02-04 → Dependem de NB 01**: Quebrado pela falta do arquivo acima
3. **NB 06-09 → Fallbacks sintéticos**: Geram dados dummy quando não há dados reais, mascarando problemas
4. **NB 10 → Dados hardcoded**: Não lê arquivos JSON gerados pelos notebooks anteriores

---

### 3. Comandos Obsoletos e Caminhos Hardcoded

#### ✅ **LIMPOS** (11 notebooks):
- **00, 11, 12, 13, 14, 16, 17, 18, 19, 20**: Sem comandos obsoletos

#### ⚠️ **REQUEREM LIMPEZA** (9 notebooks):
- **01, 02, 03, 04, 05, 06, 07, 08, 09**: Contêm células vazias com comentário "Legacy Colab mount removido"
- **15**: Outputs salvos contêm caminhos Colab (`/content/drive/MyDrive/TCC_USP/`)

**Recomendação**: Deletar células vazias e limpar outputs com `jupyter nbconvert --clear-output`.

---

### 4. Validações e Logging

#### ✅ **EXCELENTE** (notebooks com validação completa):

**Notebook 16 (models_tfidf_baselines)**:
```python
# ✅ Usa check_intersection
merges.check_intersection(idx, labels, col_left="day", col_right="day", min_days=90)

# ✅ Usa log_result
logger.log_result("logreg_l2", "tfidf_daily", metrics={"auc": 0.65, "mda": 0.58})
```

**Notebook 17 (sentiment_validation)**:
```python
# ✅ Usa check_intersection
merges.check_intersection(oof, labels, col_left="day", col_right="day", min_days=90)

# ✅ Usa log_result
logger.log_result("validation", "sentiment", metrics={"corr_pearson": 0.42})
```

**Notebook 18 (backtest_simulation)**:
```python
# ✅ Usa check_intersection
merges.check_intersection(oof, labels, col_left="day", col_right="day", min_days=90)

# ✅ Usa log_result
logger.log_result("strategy_long_short", "backtest", metrics={"cagr": 0.15, "sharpe": 1.2})
```

#### ❌ **INADEQUADO** (notebooks sem validação):

- **NB 00, 01, 02, 03, 04**: Não validam período 2018-2025
- **NB 05, 06**: Não validam se dados são reais ou sintéticos
- **NB 15**: Não usa `check_intersection` nem `log_result`

---

### 5. Período e Coleta de Dados Reais (2018-2025)

#### 📅 **Análise de Cobertura Temporal**

| Notebook | Fonte de Dados | Período Esperado | Status | Observações |
|----------|----------------|------------------|--------|-------------|
| **00** | yfinance (IBOV) | ✅ 2018-2025 | ✅ **OK** | Usa `cfg.get_periodo_estudo()` corretamente |
| **05** | NewsAPI | ⚠️ ~30 dias | ⚠️ **LIMITADO** | Free tier limitado (documentado) |
| **12** | NewsAPI (multi) | ⚠️ ~30 dias | ⚠️ **LIMITADO** | Free tier limitado (documentado) |
| **13** | Consolidação | ✅ 2018-2025 | ✅ **OK** | Depende de 05+12, herda limitações |

#### 🔴 **Problemas Críticos de Coleta**:

1. **NewsAPI Free Tier**: Limitado a ~30 dias de histórico
   - ✅ **Já documentado** com warnings claros nos notebooks 05 e 12
   - ❌ **Problema**: Não coleta dados históricos 2018-2025
   - 💡 **Solução**: Implementar scrapers alternativos ou usar plano pago

2. **Validação de Período**: Maioria dos notebooks não valida período após carregar dados
   ```python
   # ❌ AUSENTE na maioria dos notebooks:
   periodo = cfg.get_periodo_estudo()
   actual_start = df["day"].min()
   actual_end = df["day"].max()
   assert actual_start <= periodo["start"], "Dados não cobrem início do período"
   assert actual_end >= periodo["end"], "Dados não cobrem fim do período"
   ```

3. **Fallbacks Sintéticos**: Notebooks 06-09 criam dados dummy quando não há dados reais
   ```python
   # ❌ PROBLEMÁTICO (NB 07, 08, 09):
   if merged.empty:
       print("Sem interseção — criando dataset dummy para demonstração")
       df_dummy = pd.DataFrame({
           "day": pd.date_range("2023-01-01", periods=20, freq="D"),
           "y": np.random.randint(0, 2, 20)
       })
   ```
   
   **Impacto**: Pipeline aparenta funcionar mas opera sobre dados sintéticos

4. **Arquivo Ausente (NB 01)**: Depende de `noticias_exemplo.csv` não gerado pelo pipeline
   ```python
   # ❌ CRÍTICO (NB 01):
   news_file = os.path.join(RAW_PATH, "noticias_exemplo.csv")
   assert os.path.exists(news_file), f"Arquivo não encontrado: {news_file}"
   ```
   
   **Impacto**: Notebooks 01-04 não executam

---

### 6. Scripts Auxiliares

#### ✅ **verify_project.py** (EXCELENTE)

**Pontos fortes pós-revisão**:
- ✅ Valida cobertura de datas vs. período configurado
- ✅ Detecta datasets sintéticos (< 10 linhas, 1 único dia)
- ✅ Falha explicitamente quando dados críticos ausentes
- ✅ Gera relatório detalhado em Markdown

**Uso**:
```cmd
python verify_project.py
```

#### ✅ **test_dashboard.py** (EXCELENTE)

**Pontos fortes pós-revisão**:
- ✅ Lê período de `cfg.get_periodo_estudo()`
- ✅ Asserts rigorosos para datas e tamanhos de datasets
- ✅ Exit code apropriado para CI/CD

**Uso**:
```cmd
python test_dashboard.py
```

#### ✅ **create_sample_data.py** (PROTEGIDO)

**Pontos fortes pós-revisão**:
- ✅ Proteção contra sobrescrita acidental (requer env var)
- ✅ Avisos claros sobre dados sintéticos
- ✅ Gera dados para teste/demo do dashboard

**Uso**:
```cmd
set ALLOW_SAMPLE_DATA_OVERWRITE=1
python create_sample_data.py
```

#### ✅ **app_dashboard.py** (BOM)

**Pontos fortes**:
- ✅ Usa `paths.py` e `cfg.load_config()` corretamente
- ✅ Tratamento de arquivos ausentes com `_safe_read_csv()`
- ✅ Dashboard interativo com filtros de período, modelo e métrica

**Pontos de melhoria**:
- ⚠️ Poderia usar `cfg.get_periodo_estudo()` para definir limites do DatePicker
- ⚠️ Não valida se dados carregados cobrem período esperado

---

### 7. Módulos `src/`

#### ✅ **src/io/paths.py** (EXCELENTE pós-revisão)

**Pontos fortes**:
- ✅ Sem fallbacks hardcoded (removido `C:/Users/ander/...`)
- ✅ Suporta Colab, Windows, Linux via detecção automática
- ✅ Override via `TCC_USP_BASE` environment variable
- ✅ Falha explicitamente se estrutura incorreta

**Exemplo de uso**:
```python
from src.io import paths

DATA_PATHS = paths.get_data_paths()
RAW_PATH = DATA_PATHS["data_raw"]  # .../data_raw/
PROC_PATH = DATA_PATHS["data_processed"]  # .../data_processed/
```

#### ✅ **src/config/loader.py** (EXCELENTE)

**Pontos fortes**:
- ✅ Carrega `configs/config_tcc.yaml` com cache in-memory
- ✅ Helpers tipados: `get_periodo_estudo()`, `get_colunas_data()`, `get_arquivo()`
- ✅ Documentação clara de uso

**Exemplo de uso**:
```python
from src.config import loader as cfg

periodo = cfg.get_periodo_estudo()
# {'start': '2018-01-01', 'end': '2025-01-31'}

ibov_path = cfg.get_arquivo("ibov_clean")
# Path('.../data_processed/ibovespa_clean.csv')
```

#### ✅ **src/validation/merges.py** (EXCELENTE)

**Pontos fortes**:
- ✅ `check_intersection()` valida overlap mínimo entre datasets
- ✅ `summarize_date_range()` imprime estatísticas de datas
- ✅ Levanta `ValueError` quando overlap insuficiente

**Exemplo de uso**:
```python
from src.validation import merges

result = merges.check_intersection(
    df_ibov, df_sentiment,
    col_left="day", col_right="day",
    min_days=90
)
# Imprime: [day] 2018-01-01 → 2025-01-31 | 1825 dias únicos
# Levanta ValueError se overlap < 90 dias
```

#### ✅ **src/utils/logger.py** (EXCELENTE)

**Pontos fortes**:
- ✅ Persiste JSON registry (`data/results_registry.json`)
- ✅ Integração opcional com MLflow
- ✅ Logging estruturado de métricas e hyperparameters

**Exemplo de uso**:
```python
from src.utils import logger

logger.log_result(
    model_name="logreg_l2",
    dataset_name="tfidf_daily",
    metrics={"auc": 0.65, "mda": 0.58},
    extra={"C": 1.0, "penalty": "l2"}
)
```

---

## 🎯 Lista Priorizada de Problemas e Ajustes

### 🔴 **PRIORIDADE CRÍTICA** (resolver imediatamente)

#### 1. **[NB 09] Bug fatal - Variáveis não definidas**
```python
# ❌ LINHA 173-178 (Notebook 09)
# Código tenta usar aucs/mdas que não foram definidos
logger.log_result(
    "lstm_sentiment",
    "real",
    metrics={"AUC": np.mean(aucs), "MDA": np.mean(mdas)},  # ❌ NameError!
    extra=extra
)

# ✅ CORREÇÃO:
# Definir aucs/mdas ou usar métricas do último split
```

**Impacto**: Notebook 09 não executa até o fim  
**Esforço**: 5 minutos

---

#### 2. **[NB 01] Arquivo ausente - `noticias_exemplo.csv`**
```python
# ❌ LINHA 43 (Notebook 01)
news_file = os.path.join(RAW_PATH, "noticias_exemplo.csv")
assert os.path.exists(news_file), f"Arquivo não encontrado: {news_file}"

# ✅ OPÇÃO A: Gerar arquivo exemplo no NB 00
# ✅ OPÇÃO B: Usar dados reais do NB 13 (news_clean.parquet)
# ✅ OPÇÃO C: Marcar NB 01-04 como "demo" e documentar dependência externa
```

**Impacto**: Notebooks 01-04 não executam  
**Esforço**: 15-30 minutos

---

#### 3. **[NB 10] Dados hardcoded em vez de JSON**
```python
# ❌ LINHA ~30 (Notebook 10)
results = {
    "logreg_l2": {"auc": 0.65, "mda": 0.58},
    "rf_100": {"auc": 0.62, "mda": 0.56},
    # ... valores fictícios hardcoded
}

# ✅ CORREÇÃO:
import json
with open(PROC_PATH / "results_16_models_tfidf.json", "r") as f:
    results = json.load(f)
```

**Impacto**: Dashboard mostra dados falsos  
**Esforço**: 10 minutos

---

#### 4. **[NB 00] Validar dados reais do período**
```python
# ✅ ADICIONAR após download (Notebook 00):
periodo = cfg.get_periodo_estudo()
expected_start = pd.to_datetime(periodo["start"])
expected_end = pd.to_datetime(periodo["end"])

actual_start = df["day"].min()
actual_end = df["day"].max()

if actual_start > expected_start:
    print(f"⚠️ AVISO: Dados começam em {actual_start.date()}, esperado {expected_start.date()}")

if actual_end < expected_end:
    print(f"⚠️ AVISO: Dados terminam em {actual_end.date()}, esperado {expected_end.date()}")

if len(df) < 1000:
    raise ValueError(f"Dataset muito pequeno ({len(df)} linhas) - provável problema no download")
```

**Impacto**: Sem validação, pipeline pode operar sobre dados incompletos  
**Esforço**: 10 minutos

---

### 🟡 **PRIORIDADE ALTA** (resolver esta semana)

#### 5. **[NB 06-09] Remover fallbacks sintéticos**

**Problema**: Notebooks criam dados dummy quando não há dados reais, mascarando problemas.

**Localização**:
- NB 07, célula 3, linhas 64-79
- NB 08, célula 3, linhas 39-52
- NB 09, célula 3, linhas 43-68

```python
# ❌ REMOVER blocos como este:
if merged.empty:
    print("Sem interseção — criando dataset dummy")
    df_dummy = pd.DataFrame(...)  # ❌ REMOVER

# ✅ SUBSTITUIR por:
if merged.empty:
    print("❌ ERRO: Sem interseção entre IBOV e notícias")
    print("   Execute os notebooks de coleta primeiro:")
    print("   - 05 (NewsAPI)")
    print("   - 12 (multisource)")
    print("   - 13 (ETL dedup)")
    raise ValueError("Pipeline interrompido - dados insuficientes")
```

**Impacto**: Pipeline falha explicitamente quando dados ausentes (comportamento correto)  
**Esforço**: 20 minutos

---

#### 6. **[NB 01-04, 06-09] Remover células vazias "Legacy Colab"**

**Localização**: Segunda célula de cada notebook

```python
# ❌ DELETAR células como esta:
# Legacy Colab mount removido; paths definidos no setup.
```

**Impacto**: Limpeza de código  
**Esforço**: 5 minutos

---

#### 7. **[NB 15] Limpar outputs com caminhos Colab**

```bash
# ✅ EXECUTAR no terminal:
cd notebooks
jupyter nbconvert --clear-output --inplace 15_features_tfidf_daily.ipynb
```

**Impacto**: Remove caminhos hardcoded dos metadados  
**Esforço**: 1 minuto

---

#### 8. **[Múltiplos NB] Adicionar validação de período**

**Adicionar em notebooks**: 01, 02, 03, 04, 06, 07, 08, 09, 15, 17

```python
# ✅ ADICIONAR após carregar dados:
from src.config import loader as cfg

periodo = cfg.get_periodo_estudo()
expected_start = pd.to_datetime(periodo["start"])
expected_end = pd.to_datetime(periodo["end"])

actual_start = df["day"].min()
actual_end = df["day"].max()

print(f"📅 Período esperado: {expected_start.date()} → {expected_end.date()}")
print(f"📅 Período carregado: {actual_start.date()} → {actual_end.date()}")

if actual_start > expected_start or actual_end < expected_end:
    print(f"⚠️ AVISO: Cobertura temporal incompleta")
```

**Impacto**: Validação consistente em todo o pipeline  
**Esforço**: 30 minutos

---

### 🟢 **PRIORIDADE MÉDIA** (melhorias desejáveis)

#### 9. **[NB 15] Adicionar logging e validação**

```python
# ✅ ADICIONAR validação de merge:
from src.validation import merges

merges.check_intersection(
    tfidf_daily, labels,
    col_left="day", col_right="day",
    min_days=90
)

# ✅ ADICIONAR logging:
from src.utils import logger

logger.log_result(
    model_name="tfidf_extraction",
    dataset_name="daily",
    metrics={"n_features": n_features, "n_days": len(tfidf_daily)},
    extra={"max_features": 500, "ngram_range": (1, 2)}
)
```

**Impacto**: Consistência com notebooks 16-18, 20  
**Esforço**: 15 minutos

---

#### 10. **[NB 19] Implementar ou remover placeholder**

**Opção A - Implementar**:
```python
# Adicionar funcionalidades de extensão futura:
# - Modelos deep learning avançados (BERT, GPT)
# - Análise de múltiplos horizontes (1d, 5d, 21d)
# - Features técnicas adicionais (RSI, MACD)
```

**Opção B - Remover**:
```bash
# Se não for implementado, remover do repositório
rm notebooks/19_future_extension.ipynb
```

**Impacto**: Melhora organização do repositório  
**Esforço**: 5 minutos (remover) ou 2-4 horas (implementar)

---

#### 11. **[app_dashboard.py] Usar período do config no DatePicker**

```python
# ✅ ADICIONAR após carregar config:
periodo = cfg.get_periodo_estudo()
DATE_MIN = pd.to_datetime(periodo["start"])
DATE_MAX = pd.to_datetime(periodo["end"])

# ✅ USAR no DatePicker:
dcc.DatePickerRange(
    id="date-picker",
    start_date=DATE_MIN,
    end_date=DATE_MAX,
    min_date_allowed=DATE_MIN,
    max_date_allowed=DATE_MAX,
    # ...
)
```

**Impacto**: Dashboard respeitaidoconfig do projeto  
**Esforço**: 5 minutos

---

### ⚪ **PRIORIDADE BAIXA** (nice-to-have)

#### 12. **Implementar coleta histórica completa (2018-2025)**

**Problema**: NewsAPI free tier limitado a ~30 dias

**Opções**:

**A) Plano pago NewsAPI**
- Custo: ~$50-500/mês dependendo do volume
- Acesso a histórico completo desde 2016

**B) Scrapers customizados**
```python
# Implementar scrapers para:
# - Reuters: https://www.reuters.com/markets/
# - InfoMoney: https://www.infomoney.com.br/mercados/
# - Valor Econômico: https://valor.globo.com/financas/
# - CVM: https://www.gov.br/cvm/pt-br
```

**C) Datasets pré-coletados**
- Buscar datasets acadêmicos/comerciais de notícias financeiras PT-BR
- Kaggle, Hugging Face, repositórios universitários

**Impacto**: Dados reais do período completo  
**Esforço**: 4-40 horas dependendo da abordagem

---

#### 13. **CI/CD com validações automáticas**

```yaml
# .github/workflows/validate.yml
name: Validate Pipeline
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Verify project structure
        run: python verify_project.py
      - name: Test dashboard loading
        run: python test_dashboard.py
```

**Impacto**: Detecta problemas automaticamente  
**Esforço**: 1-2 horas

---

#### 14. **Documentação de fluxo de dados**

```bash
# Gerar diagrama automático do pipeline
pip install graphviz
python scripts/generate_pipeline_diagram.py
```

**Impacto**: Facilita onboarding de novos desenvolvedores  
**Esforço**: 2-3 horas

---

## 📊 Estatísticas do Projeto

### Notebooks por Status

| Status | Quantidade | Notebooks |
|--------|------------|-----------|
| ✅ **Excelente** | 6 | 11, 12, 13, 14, 16, 20 |
| 🟢 **Bom** | 5 | 00, 17, 18 |
| 🟡 **Necessita ajustes menores** | 8 | 02, 03, 04, 05, 06, 07, 08, 15 |
| 🔴 **Necessita ajustes críticos** | 2 | 01, 09, 10 |
| ⚪ **Placeholder vazio** | 1 | 19 |

### Módulos src/

| Módulo | Status | Comentário |
|--------|--------|------------|
| `src/io/paths.py` | ✅ **Excelente** | Sem fallbacks hardcoded |
| `src/config/loader.py` | ✅ **Excelente** | API clara e tipada |
| `src/validation/merges.py` | ✅ **Excelente** | Validações robustas |
| `src/utils/logger.py` | ✅ **Excelente** | Logging estruturado + MLflow |

### Scripts Auxiliares

| Script | Status | Comentário |
|--------|--------|------------|
| `verify_project.py` | ✅ **Excelente** | Validação completa pós-revisão |
| `test_dashboard.py` | ✅ **Excelente** | Testes rigorosos pós-revisão |
| `create_sample_data.py` | ✅ **Excelente** | Proteção contra sobrescrita |
| `app_dashboard.py` | 🟢 **Bom** | Funcional, pequenas melhorias possíveis |

---

## 🎓 Recomendações Finais

### Para Execução Imediata (Esta Semana)

1. ✅ **Fixar NB 09**: Corrigir bug de variáveis não definidas (5 min)
2. ✅ **Resolver NB 01**: Criar `noticias_exemplo.csv` ou documentar dependência externa (30 min)
3. ✅ **Fixar NB 10**: Carregar JSON em vez de dados hardcoded (10 min)
4. ✅ **Remover fallbacks**: Notebooks 06-09 devem falhar explicitamente sem dados (20 min)
5. ✅ **Limpar código**: Remover células vazias e outputs com caminhos Colab (15 min)

**Total: ~1h30min** para resolver todos os problemas críticos

---

### Para Desenvolvimento Contínuo

1. 🔄 **Validação de período**: Adicionar em todos os notebooks (30 min)
2. 🔄 **Logging consistente**: NB 15 usar `log_result` (15 min)
3. 🔄 **Coleta histórica**: Implementar scrapers ou plano pago NewsAPI (4-40h)
4. 🔄 **CI/CD**: Automação de validações (1-2h)
5. 🔄 **Documentação**: Diagramas de fluxo e README detalhado (2-3h)

---

### Manutenção do Código

#### ✅ **Boas Práticas a Manter**

1. **Arquitetura modular**: Continuar usando `src/` para funções compartilhadas
2. **Configuração centralizada**: Sempre usar `cfg.get_periodo_estudo()` e `cfg.get_arquivo()`
3. **Validações explícitas**: Usar `check_intersection()` antes de merges críticos
4. **Logging estruturado**: Usar `log_result()` para todas as métricas
5. **Tratamento de erros**: Falhar explicitamente com mensagens claras

#### ❌ **Anti-padrões a Evitar**

1. **Fallbacks sintéticos silenciosos**: Sempre falhar quando dados ausentes
2. **Caminhos hardcoded**: Usar apenas `paths.py` e `config.loader`
3. **Dados hardcoded**: Sempre carregar de arquivos gerados pelo pipeline
4. **Validações ausentes**: Sempre verificar período e qualidade dos dados
5. **Outputs salvos**: Limpar com `jupyter nbconvert --clear-output`

---

## 📚 Referências e Recursos

### Documentação do Projeto
- `configs/config_tcc.yaml`: Configuração centralizada
- `reports/MUDANCAS_CODEX_IMPLEMENTADAS.md`: Mudanças recentes
- `reports/CHECKLIST_VERIFICACAO.md`: Checklist de qualidade

### APIs e Fontes de Dados
- **yfinance**: https://pypi.org/project/yfinance/
- **NewsAPI**: https://newsapi.org/docs
- **Reuters**: https://www.reuters.com/
- **CVM**: https://www.gov.br/cvm/pt-br

### Ferramentas
- **MLflow**: https://mlflow.org/docs/latest/index.html
- **Dash**: https://dash.plotly.com/
- **Jupyter**: https://jupyter.org/

---

## 🤝 Contribuindo

Para contribuir com melhorias:

1. **Execute validações**:
   ```cmd
   python verify_project.py
   python test_dashboard.py
   ```

2. **Siga as boas práticas**:
   - Use `paths.py` e `cfg.load_config()`
   - Adicione validações de período
   - Use `check_intersection()` e `log_result()`

3. **Teste localmente**:
   - Execute os notebooks sequencialmente
   - Valide outputs intermediários
   - Verifique dashboard final

4. **Documente mudanças**:
   - Atualize `reports/MUDANCAS_*.md`
   - Adicione comentários explicativos
   - Mantenha README atualizado

---

**Documento gerado automaticamente em 18/11/2025**  
**Revisão completa de código - TCC USP Ibovespa Sentimento**
