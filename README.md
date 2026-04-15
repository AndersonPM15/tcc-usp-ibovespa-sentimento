# Dashboard — Sentimento de Notícias (PT-BR) × Ibovespa (USP)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Dash](https://img.shields.io/badge/Dash-Plotly-informational)
![Status](https://img.shields.io/badge/Status-Validado%20%7C%20v1.2--legend--labels-success)

## Identificação acadêmica (USP)
- **Autor:** Anderson Pantoja Machado
- **Orientador:** Prof. Vinicius Rocha Biscaro
- **Instituição:** Universidade de São Paulo (USP)
- **Curso/Programa:** MBA em Business Intelligence & Analytics — ECA/USP
- **Título do TCC:** *Análise do Sentimento de Notícias em Português e seu Efeito no Ibovespa: Evidência Empírica com Baselines Transparentes e Estudo de Eventos*

> Projeto: investigar se o sentimento em notícias financeiras em português antecipa a direção do retorno diário do Ibovespa, usando baselines transparentes (TF-IDF + Regressão Logística/Random Forest), métricas AUC/MDA e estudo de eventos (CAR/latência).

---

## Sumário
- [Visão geral](#visão-geral)
- [Pergunta de pesquisa e hipóteses](#pergunta-de-pesquisa-e-hipóteses)
- [Tecnologias](#tecnologias)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Como rodar (Windows)](#como-rodar-windows)
- [Como usar o dashboard](#como-usar-o-dashboard)
- [Como exportar as figuras (banca)](#como-exportar-as-figuras-banca)
- [Validação e rastreabilidade](#validação-e-rastreabilidade)
- [Dados, ética e conformidade](#dados-ética-e-conformidade)
- [Como citar (ABNT — sugestão)](#como-citar-abnt--sugestão)
- [Licença](#licença)

---

## Visão geral
- Dashboard (Dash/Plotly) com 8 figuras: Ibovespa/eventos, sentimento diário, comparativo de modelos, dispersão, correlação móvel, distribuição de sentimento, latência (CAR), backtest.
- Exportação headless gera **14 figuras finais** para banca (11 base + 3 de robustez).
- Hard cap temporal: **2018-01-02 a 2024-12-31**; período efetivo ajustado pela interseção das séries (sentimento/backtest iniciam em 2019-08).
- Conteúdo: app + CSS, scripts utilitários, exportação headless de figuras e relatórios de validação em `reports/`.

---

## Pergunta de pesquisa e hipóteses
- **Pergunta:** o sentimento de notícias publicadas em T₀ está associado à direção do retorno do Ibovespa em T₀+1?
- **Hipóteses (síntese):**
  - H1: sentimento negativo em T₀ associa-se a retornos negativos em T₀+1.
  - H2: sentimento melhora desempenho vs modelos apenas técnicos (ganho em AUC/MDA).
  - H3 (exploratória): latência de incorporação varia por fonte/horário (CAR).

---

## Tecnologias
**Linguagens**
- Python (principal)
- CSS (tema/layout do dashboard)

**Principais bibliotecas**
- Dash + Plotly (dashboard e gráficos)
- Pandas/NumPy (ETL, joins, séries temporais)
- Scikit-learn (baselines: Regressão Logística, Random Forest)

**Boas práticas**
- Usar ambiente virtual (`venv`) e pinagem em `requirements.txt` (se presente).
- **Não versionar dados** (`data_processed/` ignorado).
- Reprodutibilidade com scripts e logs; exportação headless determinística.

---

## Estrutura do repositório
```text
.
├── app_dashboard.py                 # app Dash/Plotly (8 figuras + modo exportação)
├── assets/
│   └── styles.css                   # tema visual (USP-like) + export-mode
├── data_processed/                  # DADOS (NÃO versionados)
├── reports/                         # auditorias, validações e figuras finais
│   ├── figures/                     # PNGs finais gerados via script (11 base + 3 robustez)
│   ├── final_data_audit.md
│   ├── final_sanity_checks.md
│   ├── final_graph_validation.md
│   ├── final_runtime_checks.md
│   ├── how_to_export_figures.md
│   ├── dashboard_blueprint.md
│   └── dashboard_graph_index.json
├── scripts/                         # utilitários (export headless, diagnósticos)
└── README.md
```
> `data_processed/` não é versionado; manter os arquivos locais em `C:\TCC_USP\data_processed\`.

---

## Como rodar (Windows)
### 1) Ambiente virtual (venv)
```bat
python -m venv venv
.\venv\Scripts\activate
```

### 2) Instalar dependências
- Se existir `requirements.txt`:
  ```bat
  pip install -r requirements.txt
  ```
- Se não existir, instale manualmente (dash, plotly, pandas, numpy, scikit-learn etc.) e gere:
  ```bat
  pip freeze > requirements.txt
  ```

### 3) Rodar o dashboard
```bat
cd C:\TCC_USP\tcc-usp-ibovespa-sentimento
.\venv\Scripts\python.exe app_dashboard.py --host 127.0.0.1 --port 8050 --find-port --open
```

### 4) Probe (validação rápida)
> Só funciona com o servidor rodando e na porta correta (ajuste se `--find-port` mudar a porta).
```bat
.\venv\Scripts\python.exe app_dashboard.py --probe --host 127.0.0.1 --port 8050
```

---

## Como usar o dashboard
- **Período de Análise:** filtra todos os 8 gráficos (clamp dentro do hard cap).
- **Modelo:** dropdown (identificadores: `logreg_l2` = *Média simples do sentimento*, `rf_200` = *Média ponderada por volume*); afeta comparativo, backtest e KPIs. As legendas dos gráficos exibem os nomes descritivos.
- **Métrica:** `AUC`, `MDA`, `Sharpe` (comparativo e KPIs).
  - AUC: discriminação (ROC) do classificador.
  - MDA: acerto direcional médio (↑/↓).
  - Sharpe: desempenho risco-retorno da estratégia (backtest).
- **Modo Exportação:** 1 coluna, altura ampliada (~900px), cabeçalho/controles ocultos para recorte/PNG manual.

---

## Como exportar as figuras (banca)
### Exportação headless (recomendado)
Gera **11 PNGs base** determinísticos em `reports/figures/`:
```bat
.\venv\Scripts\python.exe scripts\export_tcc_figures.py --strategy long_only_60
```
Arquivos gerados:
- `Figura_1_ibov_eventos.png`
- `Figura_2_sentimento_medio_diario.png`
- `Figura_3_comparativo_modelos.png`
- `Figura_4_dispersao_sentimento_retorno.png`
- `Figura_5_correlacao_movel_60d_90d.png`
- `Figura_6_distribuicao_sentimento.png`
- `Figura_7A_latencia_boxplot.png`
- `Figura_7B_event_time_CAAR.png`
- `Figura_8_backtest_vs_benchmark.png`
- `Tabela_1_metricas.png`
- `Tabela_intersecao_periodo.png`

### Robustez (complemento para banca)
Gera **+3 figuras** adicionais (total = **14 PNGs**):
```bat
.\venv\Scripts\python.exe scripts\export_tcc_figures.py --strategy long_only_60 --run_robustness
```
- `Tabela_2_robustez_backtest.csv/png`
- `Tabela_3_metricas_extendidas.csv/png`
- `Figura_9_robustez_correlacao.png`

### Exportação manual (dashboard)
1) Rodar o app com `--open` e ativar **Modo Exportação**.
2) Cenário TCC: período padrão (interseção), modelo = melhor por Sharpe, métrica = `sharpe`.
3) Em cada card, clique no ícone de câmera (Plotly) e salve PNG (largura ≥ 1600px).
> Guia detalhado: `reports/how_to_export_figures.md`.

---

## Validação e rastreabilidade
Relatórios em `reports/`:
- `final_data_audit.md` — datas, colunas, nulos.
- `final_sanity_checks.md` — retornos, distribuição de sentimento, interseção, backtest, latência.
- `final_graph_validation.md` — 8/8 gráficos OK no estado padrão.
- `final_runtime_checks.md` — `py_compile`, `pytest`, probe HTTP 200.
- `dashboard_blueprint.md` e `dashboard_graph_index.json` — mapa do layout/figuras.

Versão estável: tag **v1.0-dashboard** (commit core: `cbee9db`).
Versão atual: **v1.2** — rótulos de legendas e eixos das Figuras 2, 3, 4 e 8 atualizados para nomes descritivos (`Média simples do sentimento` / `Média ponderada por volume`).

---

## Dados, ética e conformidade
- Dados locais em `C:\TCC_USP\data_processed\` (não versionados).
- Respeitar termos/licenças das fontes; não redistribuir conteúdo protegido.
- Priorizar metadados/trechos curtos quando aplicável; citar fontes no TCC.
- Evitar vazamento temporal: usar clamping e validação walk-forward.

---

## Como citar (ABNT — sugestão)
MACHADO, Anderson Pantoja. **Análise do sentimento de notícias em português e seu efeito no Ibovespa: evidência empírica com baselines transparentes e estudo de eventos**. Trabalho de Conclusão de Curso (MBA em Business Intelligence & Analytics) — Escola de Comunicações e Artes, Universidade de São Paulo, São Paulo, 2026. (Ajuste ano/local conforme a entrega oficial.)

---

## Licença
Distribuído sob a licença **MIT**. Veja o arquivo `LICENSE`.
