# Dashboard Blueprint – “Sentimento de Notícias x Ibovespa”

## Visão geral do layout
- Contêiner raiz `page-container` com cards em grid 2 colunas (mobile: 1).
- Card superior (controls-card): título, toggle **Modo Exportação**, barra global de controles (período, modelo único, métrica), KPIs, indicador de filtros e “Última interação”, bloco de interpretação.
- Sequência de figuras (todas com cabeçalho “Figura X – …” e rodapé “Fonte | Período | N”):
  1. Ibovespa com Eventos
  2. Sentimento Médio Diário
  3. Comparativo de Modelos + Tabela
  4. Dispersão Sentimento x Retorno Diário
  5. Correlação Móvel (60d/90d)
  6. Distribuição do Sentimento
  7. Latência por Fonte/Daypart
  8. Curva de Backtest
- Rodapés dinâmicos por figura (meta-text): fonte(s), período filtrado e N pós-filtro.

## Filtros e efeitos
- `date-range` (DatePickerRange) – min/max: 2018-01-02 / 2024-12-31; padrão = interseção dos datasets carregados (hoje 2019-08-08 → 2024-12-27). Afeta todos os gráficos (filtra IBOV, sentimento, backtest, latência).
- `model-filter` (Dropdown, single-select) – opções carregadas de `RESULTS_DF` (ex.: logreg_l2, rf_200); padrão = primeiro modelo. Afeta: comparativo (filtro de barras/tabela), backtest (filtra curvas) e KPIs/melhor modelo.
- `metric-filter` (Dropdown) – opções: auc, mda, sharpe; padrão: auc. Afeta: comparativo (barras/tabela), badge, KPI “melhor modelo”.
- `export-toggle` (Checklist) – quando ligado, adiciona classe `export-mode` que oculta o card de controles e aumenta a altura dos gráficos para recorte/print.

## Tabela dos 8 gráficos

| Ordem/Título | Graph id | Callback | Filtros usados | Datasets/colunas esperadas | Regra de data / vazio | Layout/config |
| --- | --- | --- | --- | --- | --- | --- |
| 1. Ibovespa com Eventos | `ibov-graph` | `update_dashboard` | date-range | `ibovespa_clean.csv`: day, close, return (esperado). `event_study_latency.csv`: event_day, fonte/event_name (opcional) | filtro por day dentro do range; se vazio → placeholder “Sem dados no intervalo selecionado” | PLOTLY_CONFIG; altura 460 (620 em export) |
| 2. Sentimento Médio Diário | `sentiment-graph` | `update_dashboard` | date-range | `16_oof_predictions.csv` agregado: day, sentiment | filtro por day; se vazio → placeholder | PLOTLY_CONFIG; altura 460/620 |
| 3. Comparativo de Modelos | `model-comparison-graph` (+ `model-table`) | `update_dashboard` | date-range (indireto via métricas), model-filter (single), metric-filter | `results_16_models_tfidf.json`: model, auc, mda; `18_backtest_results.csv`: model, strategy, cagr, sharpe | não há filtro de data direto; se métrica/seleção zerar → placeholder | PLOTLY_CONFIG; altura 460/620; tabela com sort/filter nativos |
| 4. Dispersão Sentimento x Retorno Diário | `scatter-graph` | `update_dashboard` | date-range | Merge de sentimento filtrado (`day`, `sentiment`) com IBOV filtrado (`day`, `return`) | se merge vazio → placeholder “Sem dados…” | PLOTLY_CONFIG; altura 460/620 |
| 5. Correlação Móvel (60d/90d) | `rolling-corr-graph` | `update_dashboard` | date-range | Mesmo merge do gráfico 4; usa rolling corr 60/90 | se vazio → placeholder | PLOTLY_CONFIG; altura 460/620 |
| 6. Distribuição do Sentimento | `sentiment-dist-graph` | `update_dashboard` | date-range | Sentimento filtrado (`sentiment`) | se vazio → placeholder | PLOTLY_CONFIG; altura 460/620 |
| 7. Latência por Fonte/Daypart | `latency-graph` | `update_dashboard` | date-range | `event_study_latency.csv`: event_day, fonte, car_max_abs (usada no eixo y) | se vazio → placeholder acadêmico | PLOTLY_CONFIG; altura 460/620 |
| 8. Curva de Backtest | `backtest-graph` | `update_dashboard` | date-range, model-filter (single) | `18_backtest_daily_curves.csv`: day, model, strategy, equity (ou strategy_ret → equity calculado no load) | se vazio → placeholder | PLOTLY_CONFIG; altura 460/620 |

## Riscos de quebra (atuais)
- Se algum dataset estiver ausente/vazio fora de `event_study_latency.csv`, o placeholder aparece; o KPI “melhor modelo” fica “—” quando não há métrica disponível.
- Filtro de período fora da interseção pode zerar gráficos (placeholders) mesmo com dados locais; range padrão já usa a interseção para mitigar.
- `return` do IBOV é esperado no CSV; se ausente, scatter/rolling dependem dessa coluna e podem retornar placeholder.

## Validação rápida
- Probe: `.\venv\Scripts\python.exe app_dashboard.py --probe --host 127.0.0.1 --port 8050` → STATUS=200 confirmado na execução mais recente.
