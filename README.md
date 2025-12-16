# Sentimento de Notícias x Ibovespa — Dashboard TCC USP

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white)]()
[![Dash/Plotly](https://img.shields.io/badge/Dash%20%2F%20Plotly-UI%20Interativo-20B2AA.svg?logo=plotly&logoColor=white)]()
[![Status](https://img.shields.io/badge/Status-Dashboard%20v1.0--dashboard-brightgreen.svg)]()

## Sumário
- [Identificação Acadêmica](#identificação-acadêmica)
- [Visão Geral](#visão-geral)
- [Tecnologias e Boas Práticas](#tecnologias-e-boas-práticas)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Como Rodar (Windows)](#como-rodar-windows)
- [Como Usar o Dashboard](#como-usar-o-dashboard)
- [Exportar as 8 Figuras](#exportar-as-8-figuras)
- [Validação e Qualidade](#validação-e-qualidade)
- [Dados e Ética](#dados-e-ética)
- [Licença e Contato](#licença-e-contato)

## Identificação Acadêmica
- **Autor:** [[PREENCHER PELO ALUNO]]
- **Orientador(a):** [[PREENCHER PELO ALUNO]]
- **Instituição:** Universidade de São Paulo (USP)
- **Curso/MBA/Programa:** [[PREENCHER PELO ALUNO]]
- **Ano/Semestre:** [[PREENCHER PELO ALUNO]]
- **Título do TCC:** [[PREENCHER PELO ALUNO]]
- **Tema:** Sentimento de notícias e impacto/predição no Ibovespa (2018–2024; período efetivo = interseção das séries)

## Visão Geral
- Problema: verificar se o sentimento diário de notícias em PT-BR tem relação ou poder preditivo sobre o Ibovespa.
- Hipótese: séries de sentimento agregadas podem antecipar movimento/retorno do índice.
- Contribuição: dashboard interativo (8 figuras) + trilha de validação de dados e backtest reprodutível.
- Escopo temporal (hard cap): **2018-01-02 a 2024-12-31**. Sentimento/backtest iniciam em 2019-08 pela interseção das séries.
- Conteúdo do repositório: app_dashboard (Dash/Plotly), assets (CSS), scripts utilitários, relatórios de validação em `reports/`, documentação de exportação.

## Tecnologias e Boas Práticas
- **Linguagens:** Python (principal), HTML/CSS (assets/).
- **Bibliotecas principais:** Dash, Plotly, Pandas, NumPy; (se aplicável) Scikit-learn para métricas/modelos; urllib/socket para probe.
- **Boas práticas:** uso de `venv`; pinagem em requirements.txt (se existir); **não versionar dados** (`data_processed/` ignorado); reprodutibilidade com scripts e relatórios; usar `--find-port` para evitar conflito de porta; probe para checar HTTP 200.

## Estrutura do Repositório
```
.
├── app_dashboard.py            # Dashboard (Dash/Plotly)
├── assets/                    # CSS e assets estáticos
├── data_processed/            # NÃO versionado (dados locais)
├── reports/                   # Relatórios de auditoria/validação/export
├── scripts/                   # Utilitários (latência, preflight, etc.)
├── README.md                  # Este arquivo
└── requirements.txt           # (se existir) dependências
```
- `app_dashboard.py`: lógica de carregamento, callbacks e layout dos 8 gráficos.
- `assets/`: estilos (inclui ajustes para Modo Exportação).
- `reports/`: auditorias finais (dados, sanity, gráficos, runtime), blueprint/index das figuras e guia de exportação.
- `scripts/`: ferramentas de diagnóstico/geração (não roda pipeline pesado).
- `data_processed/`: dados locais esperados (ibovespa_clean.csv, 16_oof_predictions.csv, backtest, latência) — ficam fora do Git.

## Como Rodar (Windows)
```powershell
# 1) Criar e ativar venv
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2) Instalar dependências
pip install --upgrade pip
pip install -r requirements.txt   # se o arquivo existir

# 3) Rodar o dashboard
cd C:\TCC_USP\tcc-usp-ibovespa-sentimento
.\venv\Scripts\python.exe app_dashboard.py --host 127.0.0.1 --port 8050 --find-port --open
```
- `--find-port` escolhe a próxima porta livre se 8050 estiver ocupada.
- Probe (somente com o servidor rodando e porta correta; ajuste se o find-port mudar):
  ```powershell
  .\venv\Scripts\python.exe app_dashboard.py --probe --host 127.0.0.1 --port 8050
  ```

## Como Usar o Dashboard
- **Período:** DatePickerRange (clamp 2018-01-02 a 2024-12-31); padrão = interseção das séries (evita gráficos vazios).
- **Modelo:** dropdown single (ex.: `logreg_l2`, `rf_200`); afeta comparativo, backtest e KPIs.
- **Métrica:** `AUC`, `MDA`, `Sharpe`; usada no comparativo, badge e KPIs.
- **Modelos (resumo):** `logreg_l2` (baseline linear), `rf_200` (árvore/ensemble).
- **Modo Exportação:** 1 coluna, altura ampliada (~900px), oculta cabeçalho/controles/KPIs — ideal para recorte/PNG.

## Exportar as 8 Figuras
Passo a passo (resumo do `reports/how_to_export_figures.md`):
1) Suba o app com `--open`, ative **Modo Exportação**.
2) Cenário TCC: período padrão (2019-08-08 → 2024-12-27), modelo = melhor por Sharpe (ex.: `logreg_l2`), métrica = `sharpe`.
3) Use o ícone de câmera (Plotly) em cada card e salve com largura ≥ 1600px:
   - `fig01_ibov_eventos.png`
   - `fig02_sentimento_medio.png`
   - `fig03_comparativo_modelos.png`
   - `fig04_scatter_sent_retorno.png`
   - `fig05_corr_movel.png`
   - `fig06_dist_sentimento.png`
   - `fig07_event_study_latencia.png`
   - `fig08_backtest_equity.png`
Checklist: legenda não cortada, rodapé “Fonte | Período | N” visível, nenhum placeholder.

## Validação e Qualidade
- `reports/final_data_audit.md` — datas/colunas/nulos (max ≤ 2024-12-30/27).
- `reports/final_sanity_checks.md` — retornos, distribuição de sentimento, interseção, backtest, latência.
- `reports/final_graph_validation.md` — 8/8 gráficos OK no estado padrão.
- `reports/final_runtime_checks.md` — py_compile, pytest, probe HTTP 200.
- `reports/dashboard_blueprint.md` e `reports/dashboard_graph_index.json` — mapa do layout/figuras.
- Versão marcada: tag **v1.0-dashboard**; commit core: `cbee9db`.

## Dados e Ética
- Dados locais em `C:\TCC_USP\data_processed\` (não versionados).
- Use apenas fontes permitidas/licenças adequadas; cite as fontes no TCC.
- Evite expor credenciais; mantenha `.env` fora do Git; respeite privacidade e termos de uso.

## Licença e Contato
- Licença: [[PREENCHER]]
- Contato: [[PREENCHER PELO ALUNO]] (email)
