# Como exportar as 8 figuras do dashboard

## Roteiro fechado (cenário TCC principal)
1) Suba o app e abra no navegador (período padrão já é a interseção dos dados):
   ```
   cd C:\TCC_USP\tcc-usp-ibovespa-sentimento
   .\venv\Scripts\python.exe app_dashboard.py --host 127.0.0.1 --port 8050 --find-port --open
   ```
2) Ative **Modo Exportação** no toggle do topo (1 coluna, altura 900px, controles ocultos).
3) Fixe os filtros:
   - Período: deixe o padrão (2019-08-08 → 2024-12-27) para evitar vazios.
   - Modelo: escolha o **melhor por Sharpe** (ex.: `logreg_l2`, se listado como melhor).
   - Métrica: selecione `sharpe` (para alinhar badge e comparativo).

## Exportar as 8 figuras (camera da modebar)
Para cada card “Figura X – …”, passe o mouse, clique no ícone de câmera (Export to PNG) e salve com os nomes abaixo. Recomende width ≥ 1600px (ou o máximo oferecido) para alta qualidade.

- Figura 1: `fig01_ibov_eventos.png`
- Figura 2: `fig02_sentimento_medio.png`
- Figura 3: `fig03_comparativo_modelos.png`
- Figura 4: `fig04_scatter_sent_retorno.png`
- Figura 5: `fig05_corr_movel.png`
- Figura 6: `fig06_dist_sentimento.png`
- Figura 7: `fig07_event_study_latencia.png`
- Figura 8: `fig08_backtest_equity.png`

## Checklist antes de baixar
- Todos os 8 cards com dados (sem placeholders); latência preenchida.
- Rodapés “Fonte | Período | N” coerentes com `C:\TCC_USP\data_processed`.
- Probe opcional para garantir serviço ativo:
  ```
  .\venv\Scripts\python.exe app_dashboard.py --probe --host 127.0.0.1 --port 8050
  ```
