# Dashboard — Sentimento de Notícias (PT-BR) × Ibovespa (USP)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Dash](https://img.shields.io/badge/Dash-Plotly-informational)
![Status](https://img.shields.io/badge/Status-Validado%20%7C%20v1.0--dashboard-success)

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
- [Como exportar as 8 figuras](#como-exportar-as-8-figuras)
- [Validação e rastreabilidade](#validação-e-rastreabilidade)
- [Dados, ética e conformidade](#dados-ética-e-conformidade)
- [Como citar (ABNT — sugestão)](#como-citar-abnt--sugestão)
- [Licença](#licença)

---

## Visão geral
- Dashboard (Dash/Plotly) com 8 figuras: Ibovespa/eventos, sentimento diário, comparativo de modelos, dispersão, correlação móvel, distribuição de sentimento, latência (CAR), backtest.
- Hard cap temporal: **2018-01-02 a 2024-12-31**; período efetivo ajustado pela interseção das séries (sentimento/backtest iniciam em 2019-08).
- Conteúdo: código do app, CSS, scripts utilitários, relatórios de validação em `reports/`, guia de exportação para o TCC.

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
- Reprodutibilidade com scripts e relatórios; usar `--find-port` para evitar conflito de porta; probe para checar HTTP 200.

---

## Estrutura do repositório
```text
.
├── app_dashboard.py                 # app Dash/Plotly (8 figuras + modo exportação)
├── assets/
│   └── styles.css                   # tema visual (USP-like) + export-mode
├── data_processed/                  # DADOS (NÃO versionados)
├── reports/                         # auditorias, validações e blueprint do dashboard
│   ├── final_data_audit.md
│   ├── final_sanity_checks.md
│   ├── final_graph_validation.md
│   ├── final_runtime_checks.md
│   ├── how_to_export_figures.md
│   ├── dashboard_blueprint.md
│   └── dashboard_graph_index.json
├── scripts/                         # utilitários (diagnóstico/latência/preflight)
└── README.md
```
> `data_processed/` não é versionado; mantenha os arquivos locais em `C:\TCC_USP\data_processed\`.

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
- **Modelo:** dropdown single (ex.: `logreg_l2`, `rf_200`); afeta comparativo, backtest e KPIs.
- **Métrica:** `AUC`, `MDA`, `Sharpe` (comparativo, badge, KPIs).
  - AUC: discriminação (ROC) do classificador.
  - MDA: acerto direcional médio (↑/↓).
  - Sharpe: desempenho risco-retorno da estratégia (backtest).
- **Modo Exportação:** 1 coluna, altura ampliada (~900px), cabeçalho/controles ocultos para recorte/PNG.

---

## Como exportar as 8 figuras
1) Rodar o dashboard com `--open` e ativar **Modo Exportação**.  
2) Cenário TCC: período padrão (interseção), modelo = melhor por Sharpe (ex.: `logreg_l2`), métrica = `sharpe`.  
3) Em cada card, clique no ícone de câmera (Plotly) e salve em PNG (largura ≥ 1600px) com nomes:
   - `fig01_ibov_eventos.png`
   - `fig02_sentimento_medio.png`
   - `fig03_comparativo_modelos.png`
   - `fig04_scatter_sent_retorno.png`
   - `fig05_corr_movel.png`
   - `fig06_dist_sentimento.png`
   - `fig07_event_study_latencia.png`
   - `fig08_backtest_equity.png`
4) Checklist: legenda não cortada; rodapé “Fonte | Período | N” visível; nenhum placeholder.  
> Guia detalhado: `reports/how_to_export_figures.md`.

---

## Validação e rastreabilidade
Relatórios em `reports/`:
- `final_data_audit.md` — datas, colunas, nulos (max ≤ 2024-12-30/27).
- `final_sanity_checks.md` — retornos, distribuição de sentimento, interseção, backtest, latência.
- `final_graph_validation.md` — 8/8 gráficos OK no estado padrão.
- `final_runtime_checks.md` — `py_compile`, `pytest`, probe HTTP 200.
- `dashboard_blueprint.md` e `dashboard_graph_index.json` — mapa do layout/figuras.

Versão estável: tag **v1.0-dashboard** (commit core: `cbee9db`).

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
