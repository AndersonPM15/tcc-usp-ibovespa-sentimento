# Análise do Sentimento de Notícias em Português e seu Efeito no Ibovespa

**Trabalho de Conclusão de Curso — MBA em Business Intelligence & Analytics — ECA/USP**

**Autor:** Anderson Pantoja Machado  
**Orientador:** Prof. Vinicius Rocha Biscaro  
**Instituição:** Universidade de São Paulo (USP)

---

## Descrição

Este projeto investiga empiricamente se o sentimento extraído de notícias financeiras em português antecipa a direção do retorno diário do Ibovespa (T+1). O pipeline vai da coleta e limpeza de notícias multisource até a validação de modelos classificadores (TF-IDF + Regressão Logística / Random Forest) com walk-forward e estudo de eventos (CAR/latência).

**Pergunta de pesquisa:** o sentimento de notícias publicadas em T₀ está associado à direção do retorno do Ibovespa em T₀+1?

**Hipóteses:**
- H1: sentimento negativo em T₀ associa-se a retornos negativos em T₀+1.
- H2: sentimento melhora o desempenho versus modelos puramente técnicos (ganho em AUC/MDA).
- H3 (exploratória): a latência de incorporação de sentimento varia por fonte e horário de publicação (CAR).

**Período de análise:** 2018-01-02 a 2024-12-31 (hard cap); período efetivo ajustado pela interseção das séries (sentimento e backtest iniciam em 2019-08).

---

## Estrutura do Repositório

```
.
├── app_dashboard.py            # Dashboard interativo (Dash/Plotly, 8 figuras)
├── main.py                     # Ponto de entrada stub do pipeline
├── pipeline_orchestration.py   # Orquestra execução sequencial dos notebooks 00→20
│
├── assets/
│   └── styles.css              # Tema visual do dashboard
│
├── configs/
│   └── config_tcc.yaml         # Parâmetros globais: período, colunas, arquivos-chave
│
├── notebooks/                  # Pipeline analítico (21 notebooks numerados 00→20)
│   ├── 00_data_download.ipynb
│   ├── 01_preprocessing.ipynb
│   ├── ...
│   └── 20_final_dashboard_analysis.ipynb
│
├── scripts/                    # Utilitários: exportação, diagnóstico, validação
│   ├── create_sample_data.py   # Gera dados sintéticos para testes
│   ├── data_integrity_report.py
│   ├── export_dashboard_figures.py
│   ├── export_tcc_figures.py   # Exportação headless das figuras da banca
│   ├── generate_event_study_latency.py
│   ├── generate_release_pack.py
│   ├── pipeline_minimal.py     # Pipeline mínimo (clamp de datas)
│   ├── port_http_probe.py
│   ├── preflight_check.py
│   ├── run_pipeline_complete.py # Executa ETL → features (notebooks 13–15)
│   └── verify_project.py       # Verificação integral de artefatos e notebooks
│
├── src/                        # Módulos Python reutilizáveis
│   ├── config/
│   │   ├── constants.py        # Constantes globais (período, TF-IDF, eventos)
│   │   └── loader.py           # Leitura de config_tcc.yaml
│   ├── io/
│   │   └── paths.py            # Resolução de caminhos (local/Colab)
│   ├── utils/
│   │   ├── gdelt_collector.py  # Coleta via GDELT 2.0
│   │   ├── logger.py           # Logging estruturado para MLflow
│   │   └── newsapi_collector.py # Coleta via NewsAPI
│   └── validation/
│       └── merges.py           # Validação de interseção de séries temporais
│
├── tests/                      # Testes automatizados (pytest)
│   ├── conftest.py
│   ├── test_dashboard.py
│   └── test_data_period.py
│
├── reports/                    # Relatórios de auditoria e figuras finais
│   ├── figures/                # PNGs e CSVs exportados para a banca
│   ├── final_data_audit.md
│   ├── final_sanity_checks.md
│   ├── final_graph_validation.md
│   ├── final_runtime_checks.md
│   └── ...
│
├── data/                       # Metadados leves (rastreados no git)
│   └── results_registry.json
│
├── .env.example                # Variáveis de ambiente necessárias (template)
├── requirements.txt            # Dependências Python
└── LICENSE
```

> Os diretórios `data_raw/`, `data_processed/` e `data_interim/` **não são versionados** (`.gitignore`). Devem ser mantidos localmente ou recriados via pipeline conforme descrito abaixo.

---

## Stack

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.x |
| Dashboard | Dash + Plotly |
| ETL / Séries temporais | Pandas, NumPy |
| Modelos baseline | Scikit-learn (Regressão Logística, Random Forest) |
| Embeddings / LSTM | Transformers, Sentence-Transformers, TensorFlow/PyTorch |
| Dados de mercado | yfinance |
| Coleta de notícias | GDELT 2.0 (público), NewsAPI (API paga) |
| NLP PT-BR | NLTK, spaCy |
| Estatísticas | Statsmodels |
| Visualização | Matplotlib, Seaborn, Plotly |
| Orquestração | Papermill (com fallback para nbconvert) |
| Rastreamento de experimentos | MLflow |

---

## Configuração do Ambiente

### 1. Clonar o repositório

```bash
git clone https://github.com/AndersonPM15/tcc-usp-ibovespa-sentimento.git
cd tcc-usp-ibovespa-sentimento
```

### 2. Criar e ativar ambiente virtual

```bat
python -m venv venv
.\venv\Scripts\activate
```

### 3. Instalar dependências

```bat
pip install -r requirements.txt
```

### 4. Configurar variáveis de ambiente

Copie `.env.example` para `.env` e preencha os valores necessários:

```bat
copy .env.example .env
```

O caminho base dos dados pode ser sobrescrito com a variável:

```
TCC_USP_BASE=C:\seu\caminho\para\TCC_USP
```

Se não definida, o módulo `src/io/paths.py` usa `C:/TCC_USP` como padrão.

---

## Obtenção dos Dados

### Dados de mercado (Ibovespa)

Baixados via `yfinance` pelo notebook `00_data_download.ipynb`. Não requerem credenciais.

### Dados de notícias

As notícias financeiras em português foram coletadas de múltiplas fontes durante o período de análise. Por restrições de licença das fontes, os dados brutos não são distribuídos neste repositório.

**Para reproduzir a coleta:**

1. **GDELT 2.0** (público, sem chave): execute `12_data_collection_multisource.ipynb`. O coletor está em `src/utils/gdelt_collector.py`.
2. **NewsAPI** (plano pago para histórico > 30 dias): configure `NEWSAPI_KEY` no `.env` e execute `05_data_collection_real.ipynb`.
3. **RSS e fontes adicionais**: configuradas no notebook `12`.

O notebook `13_etl_dedup.ipynb` consolida e deduplica todas as fontes em `data_processed/news_clean_multisource.parquet`.

---

## Reprodução do Pipeline

Execute os notebooks na ordem numérica usando o orquestrador:

```bat
python pipeline_orchestration.py
```

Ou um subconjunto específico:

```bat
python pipeline_orchestration.py --only 13 14 15 16 17 18
```

### Ordem recomendada

| Faixa | Notebooks | Descrição |
|---|---|---|
| 00 | `00_data_download` | Download do Ibovespa via yfinance |
| 01–04 | `01`–`04` | Preprocessamento e modelos com dados sintéticos (prova de conceito) |
| 05–09 | `05`–`09` | Coleta e modelagem com dados reais (fonte única) |
| 12–15 | `12`–`15` | Coleta multisource, ETL, deduplicação, features TF-IDF diárias |
| 16–18 | `16`–`17`–`18` | Modelos baseline, validação do sentimento, backtest |
| 19–20 | `19`–`20` | Extensões futuras e dashboard final |

### Execução do subpipeline ETL → features

```bat
python scripts/run_pipeline_complete.py
```

---

## Dashboard

Para iniciar o dashboard interativo:

```bat
.\venv\Scripts\python.exe app_dashboard.py --host 127.0.0.1 --port 8050 --open
```

O dashboard exibe 8 figuras: série do Ibovespa com eventos, sentimento diário, comparativo de modelos, dispersão sentimento–retorno, correlação móvel (60d/90d), distribuição de sentimento, latência (CAR/estudo de eventos) e backtest vs benchmark.

### Exportação de figuras para a banca

```bat
python scripts/export_tcc_figures.py --strategy long_only_60
```

Gera 11 PNGs determinísticos em `reports/figures/`. Com `--run_robustness`, gera mais 3 figuras de robustez (total = 14).

---

## Reprodutibilidade e Validação Temporal

- **Walk-forward validation:** `TimeSeriesSplit` com 5 folds e embargo de 1 dia (sem vazamento temporal).
- **Hard cap temporal:** nenhum dado posterior a 2024-12-31 entra no pipeline.
- **Seed fixo:** `RANDOM_SEED = 42` para todos os modelos estocásticos.
- **Relatórios de auditoria:** disponíveis em `reports/` — `final_data_audit.md`, `final_sanity_checks.md`, `final_graph_validation.md`, `final_runtime_checks.md`.

Para verificação integral dos artefatos:

```bat
python scripts/verify_project.py
```

---

## Dados, Ética e Conformidade

- Dados de notícias não são redistribuídos; respeitar os termos e licenças de cada fonte.
- Metadados e trechos curtos são utilizados; o conteúdo completo das notícias não é armazenado no repositório.
- O pipeline evita vazamento temporal via clamping e walk-forward estrito.

---

## Como Citar (ABNT — sugestão)

MACHADO, Anderson Pantoja. **Análise do sentimento de notícias em português e seu efeito no Ibovespa: evidência empírica com baselines transparentes e estudo de eventos**. Trabalho de Conclusão de Curso (MBA em Business Intelligence & Analytics) — Escola de Comunicações e Artes, Universidade de São Paulo, São Paulo, 2026.

---

## Licença

Distribuído sob a licença **MIT**. Veja o arquivo `LICENSE`.
