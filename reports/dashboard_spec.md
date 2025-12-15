# Dashboard Spec — TCC USP “Sentimento de Notícias x Ibovespa”

## Escopo e período
- Período oficial: 2018-01-02 a 2024-12-31 (hard cap em todos os loaders).
- Fonte preferencial dos dados: `C:\TCC_USP\data_processed\`.
- Nenhum dado é versionado no repositório.

## Controles globais
- DatePickerRange (`date-range`): filtra todos os gráficos.
- Dropdown de modelos (`model-filter`): afeta sentimento, comparativo, backtest.
- Dropdown de métrica (`metric-filter`): afeta comparativo (barras, badge, tabela).

## Gráficos (8)

1) Ibovespa com Eventos (`ibov-graph`)
- Dataset: `C:\TCC_USP\data_processed\ibovespa_clean.csv`
- Colunas: `day`, `close` (ou `adj_close`), opcional evento `event_day`/`event_name`.
- Transformação: clamp período; ordena por `day`; scatter de eventos se existir `event_day`.
- Valor pro TCC: contexto de mercado e marcação temporal para cruzar com sentimento.

2) Sentimento Médio Diário (`sentiment-graph`)
- Dataset: `C:\TCC_USP\data_processed\16_oof_predictions.csv`
- Colunas: `day`, `proba` → `sentiment = proba*2-1`.
- Transformação: média diária, contagem `n_obs`; clamp período.
- Valor: série agregada de sentimento para inspeção temporal e alinhamento com retornos.

3) Comparativo de Modelos + Tabela (`model-comparison-graph`, `model-table`)
- Dataset: `C:\TCC_USP\data_processed\results_16_models_tfidf.json` e opcional `18_backtest_results.csv`.
- Colunas: `model`, métricas `auc`, `mda`, `sharpe`, `strategy`, `cagr`.
- Transformação: ordena por métrica selecionada; tabela preenche ausentes com “—”.
- Valor: evidencia performance relativa e melhor modelo por métrica (AUC/MDA/Sharpe).

4) Dispersão Sentimento x Retorno Diário (`scatter-graph`)
- Dataset: merge de sentimento diário com Ibovespa.
- Colunas: `sentiment`, `return` (retorno diário do IBOV), `day`.
- Transformação: inner join por `day`; Pearson corr.; scatter.
- Valor: relação contemporânea entre sinal de sentimento e retorno diário (direção/força).

5) Correlação Móvel 60d/90d (`rolling-corr-graph`)
- Dataset: mesmo merge do item 4.
- Colunas: `sentiment`, `return`, `day`.
- Transformação: rolling corr. 60 e 90 dias; clamp período.
- Valor: estabilidade temporal do sinal de sentimento em janelas mais longas.

6) Distribuição do Sentimento (`sentiment-dist-graph`)
- Dataset: série de sentimento diário.
- Colunas: `sentiment`.
- Transformação: histograma + boxplot; clamp período.
- Valor: evidencia viés de sentimento, outliers e dispersão.

7) Latência por Fonte/Daypart (`latency-graph`)
- Dataset: `C:\TCC_USP\data_processed\event_study_latency.csv`
- Colunas típicas: `event_day`, `fonte`, métricas de latência/CAR.
- Transformação: clamp período; barras por `fonte`.
- Valor: medir timing de informação; **se vazio**: card mostra “Latência informacional não disponível nesta versão por ausência de eventos estruturados; ver relatório.”

8) Curva de Backtest (`backtest-graph`)
- Dataset: `C:\TCC_USP\data_processed\18_backtest_daily_curves.csv`
- Colunas: `day`, `model`, `strategy`, `equity` (ou derivada de `strategy_ret`).
- Transformação: clamp período; cumprod de retornos se necessário.
- Valor: comportamento da estratégia vs tempo; se ausente → placeholder.

## Limitações e notas
- Interseção Ibovespa x Sentimento: 1341 dias (atual); eventos/latência podem estar indisponíveis.
- Latência: se não houver `event_study_latency.csv` preenchido, o card é ocultado e exibida mensagem textual.
- Backtest: depende de curvas diárias em `18_backtest_daily_curves.csv`; se ausente, gráfico mostra placeholder.
