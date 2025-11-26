# 📊 TCC USP – Impacto do Sentimento de Notícias na Previsão do Ibovespa

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange.svg?logo=jupyter)
![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow.svg)
![License](https://img.shields.io/badge/Licença-Acadêmica-green.svg)

---

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Objetivos](#-objetivos)
- [Arquitetura do Pipeline](#-arquitetura-do-pipeline)
- [Requisitos do Sistema](#-requisitos-do-sistema)
- [Instalação](#-instalação)
- [Como Executar](#-como-executar)
- [Estrutura de Diretórios](#-estrutura-de-diretórios)
- [Saídas Esperadas](#-saídas-esperadas)
- [Notebooks](#-notebooks)
- [Metodologia](#-metodologia)
- [Métricas de Avaliação](#-métricas-de-avaliação)
- [Cronograma](#-cronograma)
- [Referências](#-referências)
- [Licença](#-licença)
- [Créditos e Contato](#-créditos-e-contato)

---

## 🎯 Visão Geral

Este trabalho de conclusão de curso investiga a **relação entre o sentimento de notícias financeiras em português brasileiro e a direção do índice Ibovespa**.

O projeto adota uma abordagem de **baselines transparentes e reprodutíveis**, priorizando técnicas clássicas de NLP (TF-IDF) combinadas com modelos de machine learning interpretáveis (Regressão Logística, Random Forest, XGBoost).

### Diferenciais Metodológicos

| Aspecto | Descrição |
|---------|-----------|
| **Pipeline ponta a ponta** | Da coleta de dados brutos até o backtest de estratégias |
| **Foco em português brasileiro** | Pré-processamento especializado para PT-BR |
| **Validação temporal rigorosa** | Walk-forward para evitar data leakage |
| **Reprodutibilidade** | Código modular, configurável e documentado |
| **Latência T½** | Mensuração do tempo de absorção da informação pelo mercado |

### Resumo Técnico

O projeto processa **notícias financeiras em português brasileiro** coletadas de múltiplas fontes (GDELT, portais de notícias) no período de **2018 a 2024**, totalizando aproximadamente **2.556 dias de cobertura**.

O pipeline transforma texto bruto em uma **matriz TF-IDF esparsa** agregada por dia, que alimenta modelos de classificação binária (alta/baixa do Ibovespa). A avaliação utiliza **validação walk-forward** com janelas expansíveis, simulando um cenário realista de previsão.

### Palavras-chave

`análise de sentimento` · `NLP em português` · `mercado financeiro` · `Ibovespa` · `TF-IDF` · `machine learning` · `walk-forward validation` · `finanças quantitativas` · `event study`

---

## 🎯 Objetivos

### Objetivo Geral

Avaliar se o **sentimento extraído de notícias financeiras em português** possui poder preditivo sobre a **direção de curto prazo do Ibovespa**, utilizando metodologia transparente e reprodutível.

### Objetivos Específicos

1. **Construir um pipeline de dados** integrando notícias financeiras multisource e cotações do Ibovespa
2. **Desenvolver features de sentimento** baseadas em TF-IDF com agregação temporal diária
3. **Treinar e avaliar modelos baseline** (Logistic Regression, Random Forest, XGBoost) com validação walk-forward
4. **Quantificar o ganho informacional** do sentimento textual sobre modelos puramente técnicos
5. **Investigar a latência de absorção** da informação pelo mercado brasileiro
6. **Documentar limitações e extensões** para trabalhos futuros com LLMs e deep learning

---

## 🏗️ Arquitetura do Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PIPELINE TCC USP                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │   COLETA     │───▶│     ETL      │───▶│  FEATURES    │───▶│  MODELOS   │ │
│  │  (Notebook   │    │  (Notebook   │    │  (Notebook   │    │ (Notebook  │ │
│  │   00, 12)    │    │     13)      │    │   14, 15)    │    │    16)     │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘ │
│        │                    │                   │                   │        │
│        ▼                    ▼                   ▼                   ▼        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │  data_raw/   │    │data_interim/ │    │data_processed│    │  reports/  │ │
│  │  - ibovespa  │    │  - news_     │    │  - tfidf_    │    │ - metrics  │ │
│  │  - news_     │    │    clean     │    │    matrix    │    │ - curves   │ │
│  │    multisrc  │    │              │    │  - labels    │    │ - backtest │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘ │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        BACKTEST & DASHBOARD                           │   │
│  │                    (Notebooks 18, 20 + app_dashboard.py)              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stack Tecnológico

| Categoria | Tecnologias |
|-----------|-------------|
| **Coleta** | GDELT API, yfinance, requests |
| **Processamento** | pandas, spaCy (pt_core_news_sm), scikit-learn |
| **Modelagem** | scikit-learn, XGBoost, MLflow (tracking) |
| **Visualização** | Plotly, Dash |
| **Orquestração** | Jupyter notebooks, Prefect (opcional) |

---

## 💻 Requisitos do Sistema

### Pré-requisitos Obrigatórios

- **Python** 3.10 ou superior
- **Git** instalado e configurado
- **Pip** atualizado (recomendado: >= 23.0)
- **8 GB RAM** mínimo (16 GB recomendado para processamento completo)
- **10 GB de espaço em disco** para dados e modelos

### Sistemas Operacionais Suportados

| Sistema | Status | Observações |
|---------|--------|-------------|
| Windows 10/11 | ✅ Testado | Ambiente principal de desenvolvimento |
| Linux (Ubuntu 20.04+) | ✅ Compatível | Testado em WSL2 |
| macOS (12+) | ⚠️ Compatível | Não testado extensivamente |
| Google Colab | ✅ Compatível | Estrutura adaptada para Drive |

---

## 🛠️ Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/AndersonPM15/tcc-usp-ibovespa-sentimento.git
cd tcc-usp-ibovespa-sentimento
```

### 2. Crie o ambiente virtual

```bash
python -m venv venv
```

### 3. Ative o ambiente virtual

**Windows (CMD):**
```cmd
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### 4. Instale as dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure variáveis de ambiente (opcional)

```bash
cp .env.example .env
# Edite o arquivo .env com suas credenciais, se necessário
```

**Variáveis disponíveis:**
```ini
DB_URL=              # URL do banco de dados (opcional)
BQ_PROJECT_ID=       # ID do projeto BigQuery (opcional)
LLM_API_KEY=         # Chave de API para LLMs (extensões futuras)
```

---

## 🚀 Como Executar

### Scripts Disponíveis

| Script | Descrição | Uso típico |
|--------|-----------|------------|
| `run_pipeline_complete.py` | Executa notebooks 13-15 em sequência | Pipeline principal |
| `pipeline_orchestration.py` | Orquestra todos os notebooks oficiais | Execução completa |
| `app_dashboard.py` | Inicia dashboard interativo | Visualização de resultados |
| `check_raw_data.py` | Verifica integridade dos dados brutos | Diagnóstico |
| `check_news.py` | Valida coleta de notícias | Diagnóstico |

### Execução Rápida

```bash
# 1. Ative o ambiente virtual
venv\Scripts\activate

# 2. Execute o pipeline completo (ETL → Preprocessamento → TF-IDF)
python run_pipeline_complete.py
```

### Verificação de Dados

```bash
# Verificar dados brutos
python check_raw_data.py

# Verificar coleta de notícias
python check_news.py
```

### Dashboard Interativo

```bash
python app_dashboard.py
```

Acesse em: **http://localhost:8050**

### Orquestração Completa

```bash
# Executar todos os notebooks do pipeline oficial
python pipeline_orchestration.py
```

---

## 📂 Estrutura de Diretórios

O projeto utiliza uma estrutura organizada em `C:\TCC_USP\` (Windows) ou no Google Drive (Colab):

```
C:\TCC_USP\
│
├── data_raw\                        # Dados brutos (imutáveis)
├── data_processed\                  # Dados prontos para modelagem
├── data_interim\                    # Dados intermediários (cache)
├── reports\                         # Relatórios e logs
│
└── tcc-usp-ibovespa-sentimento\     # Repositório Git
    ├── notebooks\                   # Notebooks Jupyter
    ├── src\                         # Código Python modular
    │   ├── io\                      # Entrada/saída (paths.py)
    │   ├── config\                  # Constantes e configurações
    │   ├── pipeline\                # Lógica do pipeline
    │   ├── utils\                   # Funções auxiliares
    │   └── validation\              # Validação de dados
    ├── configs\                     # Arquivos YAML
    │   └── config_tcc.yaml
    ├── run_pipeline_complete.py
    ├── pipeline_orchestration.py
    ├── app_dashboard.py
    ├── requirements.txt
    └── README.md
```

### Descrição das Pastas

| Pasta | Descrição |
|-------|-----------|
| `data_raw/` | Dados brutos baixados de APIs (GDELT, yfinance). **Nunca são modificados.** |
| `data_processed/` | Dados limpos e transformados, prontos para modelagem. |
| `data_interim/` | Arquivos intermediários do pipeline (cache, checkpoints). |
| `reports/` | Relatórios gerados, logs de execução e outputs finais. |

> **Nota:** No Google Colab, a estrutura equivalente fica em `/content/drive/MyDrive/TCC_USP/`.

---

## 📤 Saídas Esperadas

### `data_raw/` — Dados Brutos

| Arquivo | Descrição |
|---------|-----------|
| `ibovespa.csv` | Série histórica do Ibovespa (yfinance) |
| `news_multisource.parquet` | Notícias consolidadas de múltiplas fontes |
| `gdelt_historical.parquet` | Dados históricos do GDELT (2018-2024) |

### `data_interim/` — Dados Intermediários

| Arquivo | Descrição |
|---------|-----------|
| `news_clean_multisource.parquet` | Notícias após ETL e deduplicação |

### `data_processed/` — Dados Processados

| Categoria | Arquivo | Descrição |
|-----------|---------|-----------|
| **Texto** | `news_clean.parquet` | Notícias limpas e normalizadas |
| | `bow_daily.parquet` | Bag-of-Words agregado por dia |
| **TF-IDF** | `tfidf_daily_matrix.npz` | Matriz TF-IDF esparsa |
| | `tfidf_daily_index.csv` | Índice de datas |
| | `tfidf_daily_vocab.json` | Vocabulário do vetorizador |
| **Labels** | `labels_y_daily.csv` | Direção do Ibovespa (subida/queda) |
| | `ibov_clean.csv` | Série do Ibovespa alinhada |
| **Modelos** | `results_16_models_tfidf.json` | Métricas dos modelos |
| | `16_roc_curves.html` | Curvas ROC interativas |
| **Backtest** | `18_backtest_results.csv` | Resultados walk-forward |
| | `18_backtest_equity.html` | Curva de equity |
| **Dashboard** | `20_final_dashboard.html` | Dashboard final |

### `reports/` — Relatórios

| Arquivo | Descrição |
|---------|-----------|
| `coverage_report_*.csv` | Relatório de cobertura temporal |
| `pipeline_summary.txt` | Resumo da execução |

---

## 📓 Notebooks

### Pipeline Oficial

Notebooks executados pelo `pipeline_orchestration.py`:

| Notebook | Descrição | Etapa |
|----------|-----------|-------|
| `00_data_download.ipynb` | Download e validação inicial | Coleta |
| `13_etl_dedup.ipynb` | ETL e deduplicação multisource | ETL |
| `14_preprocess_ptbr.ipynb` | Lematização e processamento PT-BR | Features |
| `15_features_tfidf_daily.ipynb` | Construção de features TF-IDF | Features |
| `16_models_tfidf_baselines.ipynb` | Modelagem (LogReg, RF, XGB) | Modelagem |
| `18_backtest_simulation.ipynb` | Simulação de estratégias | Avaliação |
| `20_final_dashboard_analysis.ipynb` | Preparação do dashboard | Visualização |

### Notebooks Experimentais

Utilizados em fases exploratórias, **não fazem parte do pipeline final**:

<details>
<summary>Clique para expandir</summary>

- `02_baseline_logit.ipynb` — Experimento preliminar de baseline
- `03_tfidf_models.ipynb` — Testes iniciais de TF-IDF
- `04_embeddings_models.ipynb` — Experimentos com embeddings
- `05_data_collection_real.ipynb` — Coleta antiga de dados
- `06_preprocessing_real.ipynb` — Pré-processamento antigo
- `07_tfidf_real.ipynb` — TF-IDF preliminar
- `08_embeddings_real.ipynb` — Embeddings em dados reais
- `09_lstm_real.ipynb` — Experimentos com LSTM
- `10_dashboard_results.ipynb` — Análise preliminar
- `11_event_study_latency.ipynb` — Estudo de eventos
- `12_data_collection_multisource.ipynb` — Coleta multisource
- `17_sentiment_validation.ipynb` — Validação de sentimento
- `19_future_extension.ipynb` — Trabalhos futuros

</details>

---

## 📊 Metodologia

### Etapas do Pipeline

1. **Coleta de Dados**
   - Notícias financeiras via GDELT API
   - Cotações do Ibovespa via yfinance

2. **Pré-processamento**
   - Limpeza e normalização de textos
   - Tokenização com spaCy (pt_core_news_sm)
   - Remoção de stopwords e lematização

3. **Engenharia de Features**
   - Vetorização TF-IDF
   - Agregação temporal diária

4. **Modelagem**
   - Regressão Logística
   - Random Forest
   - XGBoost

5. **Validação**
   - Walk-forward com janelas expansíveis
   - Métricas: AUC-ROC, Accuracy, MDA

6. **Backtest**
   - Simulação de estratégias de trading
   - Cálculo de Sharpe Ratio

---

## 🎯 Métricas de Avaliação

| Métrica | Descrição | Uso |
|---------|-----------|-----|
| **AUC-ROC** | Área sob a curva ROC | Discriminação do modelo |
| **Accuracy** | Taxa de acerto geral | Avaliação básica |
| **MDA** | Mean Directional Accuracy | Acerto de direção |
| **Sharpe Ratio** | Retorno ajustado ao risco | Backtest |
| **T½ Latency** | Tempo de absorção da informação | Event study |

---

## 📅 Cronograma

| Fase | Atividade | Prazo |
|:----:|-----------|:-----:|
| 1 | Revisão Bibliográfica e Plano de Pesquisa | Mar/2025 |
| 2 | Coleta e Tratamento de Dados | Abr/2025 |
| 3 | Implementação do Pipeline | Mai/2025 |
| 4 | Experimentos e Validação | Jun/2025 |
| 5 | Redação Final do TCC | Jul/2025 |
| 6 | Defesa | Ago/2025 |

---

## 📚 Referências

- Tetlock, P. C. (2007). *Giving Content to Investor Sentiment: The Role of Media in the Stock Market.*
- Nassirtoussi, A. K., et al. (2015). *Text Mining for Market Prediction: A Systematic Review.*
- Estudos recentes sobre **LLMs aplicados a finanças e previsão de mercado**.

---

## 📜 Licença

Este repositório é de uso **acadêmico e de pesquisa**.

O projeto integra o **Trabalho de Conclusão de Curso (TCC)** do MBA em Business Intelligence & Analytics da **ECA/USP**.

> ⚠️ **Aviso:** O código e os dados são disponibilizados apenas para fins educacionais. Não constitui recomendação de investimento.

---

## 🤝 Contribuição

Contribuições são bem-vindas! Para contribuir:

1. Faça um fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

---

## 👤 Créditos e Contato

| | |
|---|---|
| **Aluno** | Anderson P. M. |
| **Curso** | MBA em Business Intelligence & Analytics |
| **Instituição** | ECA/USP – Escola de Comunicações e Artes |
| **Modalidade** | Projeto Aplicado em Business Intelligence e Analytics |
| **Orientador** | _(a ser definido pela USP)_ |
| **Repositório** | [github.com/AndersonPM15/tcc-usp-ibovespa-sentimento](https://github.com/AndersonPM15/tcc-usp-ibovespa-sentimento) |

---

<div align="center">

**⭐ Se este projeto foi útil, considere dar uma estrela no repositório!**

</div>
