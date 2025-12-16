# Dashboard TCC USP – Sentimento de Notícias x Ibovespa

## Resumo do projeto
- Relaciona sentimento de notícias em PT-BR com o Ibovespa (NLP + finanças).
- Período oficial (hard cap): **2018-01-02 a 2024-12-31** (nunca extrapolar 2025).
- Dashboard validado com **8 figuras** (Ibovespa, sentimento, comparativo, dispersão, correlação móvel, distribuição, latência, backtest).
- Foco em transparência e reprodutibilidade; dados locais fora do Git.

## Requisitos e instalação
- Python 3.11 (recomendado), Git, pip atualizado.
- Criar e ativar venv (PowerShell):
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- Instalar dependências (se existir requirements.txt):
  ```powershell
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
  (Se não houver requirements.txt, instale manualmente: pandas, plotly, dash, etc.)

## Como rodar o dashboard
```powershell
cd C:\TCC_USP\tcc-usp-ibovespa-sentimento
.\venv\Scripts\python.exe app_dashboard.py --host 127.0.0.1 --port 8050 --find-port --open
```
- `--find-port` escolhe a próxima porta livre se 8050 estiver ocupada.
- Probe (apenas com servidor rodando; ajuste porta se mudar):
  ```powershell
  .\venv\Scripts\python.exe app_dashboard.py --probe --host 127.0.0.1 --port 8050
  ```

## Controles do cabeçalho
- **Período**: DatePickerRange (clamp 2018-01-02 a 2024-12-31); padrão = interseção das séries para evitar vazios.
- **Modelo**: dropdown single (ex.: `logreg_l2`, `rf_200`) usado no comparativo e backtest.
- **Métrica**: `AUC`, `MDA`, `Sharpe` (alimenta comparativo, badge e KPIs).

## Modo Exportação (recorte/PNG)
- Toggle “Modo Exportação” coloca 1 coluna, aumenta a altura dos gráficos (900px) e oculta cabeçalho/KPIs para recorte.
- Use este modo para baixar as figuras em alta qualidade.

## Como exportar as 8 figuras
1) Suba o app com `--open`, ative **Modo Exportação**.
2) Fixe o cenário TCC: período padrão (2019-08-08 → 2024-12-27), modelo = melhor por Sharpe (ex.: `logreg_l2`), métrica = `sharpe`.
3) Em cada card, clique no ícone de câmera (Plotly) e salve com estes nomes (width ≥ 1600px sugerido):
   - `fig01_ibov_eventos.png`
   - `fig02_sentimento_medio.png`
   - `fig03_comparativo_modelos.png`
   - `fig04_scatter_sent_retorno.png`
   - `fig05_corr_movel.png`
   - `fig06_dist_sentimento.png`
   - `fig07_event_study_latencia.png`
   - `fig08_backtest_equity.png`
(Detalhe completo em `reports/how_to_export_figures.md`.)

## Validação e relatórios
- `reports/final_data_audit.md` – datas, colunas, nulos (8/8 OK, max ≤ 2024-12-30/27).
- `reports/final_sanity_checks.md` – retornos, distribuição de sentimento, interseção, backtest, latência.
- `reports/final_graph_validation.md` – 8/8 gráficos OK no estado padrão.
- `reports/final_runtime_checks.md` – py_compile, pytest, probe HTTP 200.
- `reports/dashboard_blueprint.md` e `reports/dashboard_graph_index.json` – mapa do layout/figuras.

## Dados e versionamento
- Dados locais em `C:\TCC_USP\data_processed\` (ibovespa_clean.csv, 16_oof_predictions.csv, resultados/backtest/latência). **Não são versionados.**
- O repositório contém apenas código, scripts e relatórios.

## Versão
- Tag: `v1.0-dashboard`
- Commit final: `cbee9db`
