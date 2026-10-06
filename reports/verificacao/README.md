# Verificações pós-submissão

> **Rótulo: pós-submissão; não consta do artigo.** Os números desta pasta foram calculados
> depois da submissão ao XXIX SemeAd (2026) e não substituem os do artigo, que continuam
> reproduzidos por `tests/test_article_reproduction.py`. Cada verificação refaz uma análise
> do artigo com o método descrito no texto ou com uma inferência mais robusta, usando as
> mesmas previsões fora da amostra.

Gerado por `python scripts/run_post_submission_checks.py` (cerca de 2 minutos), com a variável
`TCC_USP_BASE` apontando para os dados. Bootstraps com semente 42: os resultados são idênticos a
cada execução. As opções novas estão no mesmo código do artigo (por exemplo,
`threshold_mode="rolling"` no backtest e `window_unit="trading_days"` no estudo de eventos), e
o padrão de cada uma reproduz o artigo. Testes: `tests/test_post_submission.py`.

## Resumo

| Verificação | Resultado | Em relação ao artigo |
|---|---|---|
| **(a)** Benchmark no mesmo período das estratégias | No período das estratégias, o Ibovespa rendeu **3,25% a.a. (Sharpe 0,255)**, e não 6,49% (0,382). As estratégias da Tabela 3 ficam em −3,16% e −4,22% | **CONFIRMA**: as estratégias perdem para o Ibovespa. A diferença é menor que a mostrada na Tabela 3 |
| **(b)** H1 com IC por bootstrap em blocos e Newey-West | Sentimento × retorno de D+1: r = **0,020** (LR) e **0,003** (RF), IC em blocos inclui zero e Newey-West p = 0,44 e 0,91. O r = −0,127 é com o retorno **do mesmo dia** e é significativo (IC [−0,177; −0,084]; p = 2×10⁻⁶) | **CONFIRMA** que não há associação preditiva. **CONTRADIZ** que o r ≈ −0,13 seja não significativo e se refira a D+1 |
| **(c)** Backtest sem *look-ahead* (limiares dos 60 pregões anteriores) | As 12 configurações da Tabela 4 passam a ter CAGR negativo (de −8,3% a −0,5%) e Sharpe de −0,49 a +0,05. Nenhuma supera o Ibovespa | **CONFIRMA** que não há valor econômico. Os dois casos positivos da Tabela 4 (RF, lag 2: Sharpe 0,307 e 0,383) desaparecem |
| **(d)** Estudo de eventos com retorno anormal e janela em pregões | Com os 135 + 135 eventos em todo τ, o CAAR só é significativo no próprio dia do evento. Em τ = 4, o grupo positivo vai de ≈ −2,9% (IC exclui zero no artigo) para **−0,41%**, com IC [−1,29%; +0,56%] | **CONTRADIZ** o efeito acumulado nos dias seguintes mostrado na Fig. 7B. **CONFIRMA** que não há reação defasada explorável |
| **(e)** Modelo técnico e texto + técnico | AUC de 0,489 a 0,502, com todos os IC contendo 0,5. Diferenças em relação ao só-texto entre −0,007 e +0,008, sem significância. Todas as MDAs ficam abaixo de "sempre alta" (0,517) | **CONFIRMA** que o sentimento não melhora a previsão em relação a um modelo técnico, e vice-versa |

**Leitura geral:** as conclusões centrais do artigo se mantêm, porque não há poder preditivo nem
valor econômico. Dois resultados citados não se sustentam:

- o r ≈ −0,13 é contemporâneo e significativo, não preditivo;
- o CAAR negativo em τ = 4 da Fig. 7B.

---

## (a) Benchmark no mesmo período — `a_benchmark_mesmo_periodo.csv`

**Problema no artigo:** o buy-and-hold da Tabela 3 cobre 03/01/2018–30/12/2024, mas as
estratégias só operam de 05/08/2019 a 30/12/2024.

**Método:** usamos a mesma função de métricas do artigo (`_compute_metrics`: CAGR com 252
pregões/ano e Sharpe sem taxa livre de risco). As estratégias têm 1.341 linhas, que são os 1.346
pregões do período menos 5 dias sem notícias. Por isso há três versões do benchmark:

| Cenário | Período | n | CAGR | Sharpe | Máx. drawdown |
|---|---|---|---|---|---|
| Ibovespa, como na Tabela 3 do artigo | 03/01/2018–30/12/2024 | 1.736 | 6,49% | 0,382 | −46,8% |
| **Ibovespa, mesmas linhas e retorno das estratégias (D→D+1)** | 05/08/2019–30/12/2024 | 1.341 | **3,25%** | **0,255** | −46,8% |
| Ibovespa, todos os pregões do período | 05/08/2019–30/12/2024 | 1.346 | 3,50% | 0,264 | −46,8% |
| Ibovespa, convenção da Figura 8 (D−1→D nas datas das estratégias) | 05/08/2019–30/12/2024 | 1.341 | 4,34% | 0,296 | −46,8% |
| Regressão Logística, Tabela 3 (lag 0, quantil 0,90) | 05/08/2019–30/12/2024 | 1.341 | −3,16% | −0,080 | −45,5% |
| Random Forest, Tabela 3 (lag 0, quantil 0,90) | 05/08/2019–30/12/2024 | 1.341 | −4,22% | −0,132 | −50,0% |

A comparação direta é com a linha em negrito, porque usa os mesmos dias e o mesmo retorno das
estratégias. A Figura 8 pula os retornos de outros 5 pregões, que juntos somaram −4,1%, e por
isso mostra um Ibovespa melhor (4,34%).

**Conclusão:** CONFIRMA que as estratégias perdem para o buy-and-hold. A distância em CAGR cai
de 9,7–10,7 pontos percentuais (Tabela 3) para 6,4–7,5 pontos.

---

## (b) H1: correlação, bootstrap em blocos, Newey-West e estacionariedade

**Problema no artigo:** o texto atribui o r ≈ −0,13 ao retorno de D+1 e diz que o IC por
bootstrap em blocos inclui zero e que o Newey-West dá p > 0,05. No código, porém, o −0,13 é a
correlação com o retorno do mesmo dia, e não havia bootstrap em blocos nem Newey-West.

**Método:**

- sentimento = 2p − 1 fora da amostra, refeito da matriz TF-IDF e conferido com o artigo
  (diferença < 10⁻⁹);
- retorno do mesmo dia (D−1→D) e do dia seguinte (D→D+1);
- dois períodos: o do artigo (05/08/2019–30/12/2024, n = 1.341, principal) e o completo
  (até 17/11/2025, n = 1.552);
- **bootstrap em blocos móveis:** pares (sentimento, retorno) reamostrados em blocos
  contíguos de n^(1/3) dias (11 ou 12), com sensibilidade a blocos de 5, 10 e 20; 2.000
  reamostragens, semente 42, IC percentil de 95%;
- **Newey-West:** regressão r = α + β·s com erros HAC (núcleo de Bartlett), com
  floor(4·(n/100)^(2/9)) = 7 defasagens e sensibilidade a 5 e 10;
- **ADF e KPSS:** no sentimento e no retorno.

**Resultados no período do artigo (n = 1.341; bloco de 11; Newey-West com 7 defasagens):**

| Modelo | Retorno | Pearson | IC 95% em blocos | Spearman | IC 95% em blocos | β (HAC) | t | p |
|---|---|---|---|---|---|---|---|---|
| Regressão Logística | D→D+1 | 0,020 | [−0,031; 0,078] | 0,025 | [−0,027; 0,080] | 0,0047 | 0,78 | 0,44 |
| Random Forest | D→D+1 | 0,003 | [−0,053; 0,057] | 0,006 | [−0,043; 0,057] | 0,0015 | 0,11 | 0,91 |
| Regressão Logística | mesmo dia | −0,127 | [−0,177; −0,084] | −0,149 | [−0,205; −0,097] | −0,0294 | −4,71 | 2×10⁻⁶ |
| Random Forest | mesmo dia | −0,140 | [−0,200; −0,073] | −0,125 | [−0,170; −0,075] | −0,0693 | −3,23 | 0,001 |

As conclusões não mudam com:

- blocos de 5, 10 ou 20 dias;
- 5 ou 10 defasagens;
- o período completo até 2025.

**Estacionariedade (`b_h1_estacionariedade.csv`):**

- **Retorno do Ibovespa:** estacionário pelos dois testes (ADF p < 10⁻²⁰; KPSS p ≥ 0,10).
- **Sentimento:** o ADF rejeita raiz unitária (p ≤ 0,03), mas o KPSS rejeita estacionariedade.
  - **Causa:** o nível do sentimento muda entre os 4 blocos do walk-forward, porque cada bloco
    usa um modelo re-treinado (média do RF: 0,087 → 0,063 → 0,056 → 0,035).
  - **Sem a média de cada bloco:** o KPSS deixa de rejeitar (p ≥ 0,10).
  - **Regressão com o sentimento sem a média do bloco:** equivale a um efeito fixo por bloco e
    dá as mesmas conclusões (D+1: p = 0,45 e 0,88; mesmo dia: p < 0,001).

**Conclusão:**

- **CONFIRMA:** o sentimento não antecipa o retorno de D+1. A correlação é praticamente zero,
  e os ICs robustos e o Newey-West não rejeitam a ausência de associação.
- **CONTRADIZ:** o r ≈ −0,13 citado é do mesmo dia e é estatisticamente significativo mesmo com
  inferência robusta à autocorrelação.
- **Interpretação (inferida):** o escore fica mais alto nos dias em que o mercado cai, um padrão
  de reversão aprendido pelo classificador. Isso não se traduz em previsão do dia seguinte.

---

## (c) Backtest sem *look-ahead* — `c_backtest_sem_lookahead.csv`

**Problema no artigo:** o texto descreve limiares numa janela móvel de 60 pregões, mas o
código usa os quantis de p do período inteiro, o que inclui informação futura.

**Método:** a mesma grade da Tabela 4 (2 modelos × lags 0, 1 e 2 × quantis 0,90 e 0,95),
com a mesma regra e o mesmo custo (0,0005 por unidade de turnover).

- **Entrada:** compra se p ≥ quantil *q* dos p dos **60 pregões anteriores** (sem o dia
  corrente).
- **Saída:** vende se p ≤ mediana dos 60 anteriores.
- **Início:** nos primeiros 60 pregões não há limiar, e a estratégia fica fora do mercado.

| Limiar | Sharpe médio | Sharpe (mín.–máx.) | CAGR (mín.–máx.) | CAGR > 0 | Sharpe > Ibovespa (0,255) |
|---|---|---|---|---|---|
| Artigo: quantis do período inteiro | 0,012 | −0,201 a 0,383 | −5,1% a +5,1% | 3 de 12 | 2 de 12 |
| **Sem look-ahead: 60 pregões anteriores** | **−0,268** | **−0,493 a 0,046** | **−8,3% a −0,5%** | **0 de 12** | **0 de 12** |

Com look-ahead, os dois casos que superavam o Ibovespa eram RF, lag 2, com quantis 0,90 e
0,95 (Sharpe 0,307 e 0,383). Sem look-ahead, eles ficam em 0,046 e −0,076.

**Conclusão:** CONFIRMA a ausência de valor econômico, agora sem informação futura. Os casos
positivos da Tabela 4 não devem ser destacados, porque dependiam do look-ahead.

---

## (d) Estudo de eventos com retorno anormal — `d_eventos_retorno_anormal.csv`

**Problema no artigo:** o texto descreve retorno anormal (média de t−60 a t−1) e janela de
t = 0 a 4. O código usa retornos brutos e janela em dias corridos. Com isso, só entram em cada
τ os eventos seguidos de τ pregões sem fim de semana, e o número de eventos cai de 135 para
24 e 15 em τ = 4.

**Método:**

- **Eventos:** os mesmos 270 do artigo (135 de sentimento extremo positivo e 135 negativo).
- **Retorno anormal:** retorno menos a média dos 60 pregões anteriores ao evento.
- **Janela em pregões:**
  - [0, τ], que inclui o retorno do dia do evento;
  - [1, τ], só os dias seguintes.
- **IC:** bootstrap i.i.d. dos eventos, 2.000 reamostragens, semente 42.
- **Dados:** Ibovespa até 18/11/2025, para que os eventos de dezembro/2024 tenham τ completo.
- **Controle:** a 1ª parte do CSV replica exatamente a Fig. 7B do repositório (diferença 0,0).

| Cenário | Grupo | τ = 0 | τ = 1 | τ = 4 | n em τ = 4 |
|---|---|---|---|---|---|
| Artigo (bruto, dias corridos) | negativo | +0,34% [0,15; 0,53] | +0,30% [0,00; 0,61] | −0,41% [−1,14; 0,39] | 24 |
| Artigo (bruto, dias corridos) | positivo | −0,61% [−1,09; −0,15] | −0,15% [−0,72; 0,51] | −2,87% [−6,57; 0,21] | 15 |
| Anormal, pregões, [0, τ] | negativo | +0,32% [0,12; 0,51] | +0,35% [0,09; 0,60] | +0,24% [−0,17; 0,67] | 135 |
| Anormal, pregões, [0, τ] | positivo | −0,59% [−1,09; −0,13] | −0,14% [−0,68; 0,45] | −0,41% [−1,29; 0,56] | 135 |
| Anormal, pregões, [1, τ] | negativo | — | +0,03% [−0,13; 0,19] | −0,08% [−0,44; 0,29] | 135 |
| Anormal, pregões, [1, τ] | positivo | — | +0,46% [0,02; 0,93] | +0,18% [−0,64; 1,00] | 135 |

Os ICs estão em pontos percentuais. A tabela completa (τ = 0 a 4) está no CSV.

**Leitura:**

- **No dia do evento (τ = 0):** o efeito é significativo. Isso é coerente com a correlação
  contemporânea da verificação (b): o sentimento extremo positivo coincide com queda do
  Ibovespa no mesmo dia.
- **Nos dias seguintes ([1, τ]):** nenhum CAAR é significativo, exceto o grupo positivo em
  τ = 1 (+0,46%). Esse resultado fica no limite e é 1 em 8 testes, o que é compatível com o
  acaso a 5%.

**Conclusão:**

- **CONTRADIZ** a queda acumulada de ≈ −2,9% em τ = 4 no grupo positivo da Fig. 7B. Com o
  método descrito no artigo e todos os eventos, ela vira −0,41% e não é significativa.
- **CONFIRMA** que não há reação defasada explorável ao sentimento extremo.

---

## (e) Modelo técnico e texto + técnico — `e_modelos_texto_tecnico.csv`, `e_diferencas_auc.csv`

**Problema no artigo:** o texto descreve variáveis técnicas (retornos defasados e volatilidade
histórica), mas o código só usava o TF-IDF.

**Método:**

- **Variáveis técnicas no fechamento de D:** retornos de D−4 a D (5 defasagens) e
  volatilidade dos últimos 5 e 20 retornos.
- **Modelos e validação:** os mesmos 2 modelos e o mesmo walk-forward da Tabela 2 (4 blocos,
  1.552 dias fora da amostra).
- **Padronização:** só nas colunas técnicas, ajustada dentro de cada bloco de treino.
- **Diferenças de AUC:** bootstrap em blocos pareado (mesmos dias para os dois modelos; bloco
  de 12; 2.000 reamostragens; semente 42).
- **Controle:** o modelo só-texto reproduz a Tabela 2 exatamente (AUC 0,5015 e 0,4913).

| Variáveis | Modelo | AUC | IC 95% | MDA | AUC até 30/12/2024 |
|---|---|---|---|---|---|
| Só texto (Tabela 2) | Regressão Logística | 0,502 | [0,474; 0,532] | 0,503 | 0,501 |
| Só texto (Tabela 2) | Random Forest | 0,491 | [0,463; 0,521] | 0,512 | 0,498 |
| Técnico | Regressão Logística | 0,494 | [0,465; 0,523] | 0,488 | 0,493 |
| Técnico | Random Forest | 0,489 | [0,461; 0,518] | 0,503 | 0,489 |
| Texto + técnico | Regressão Logística | 0,502 | [0,473; 0,532] | 0,491 | 0,502 |
| Texto + técnico | Random Forest | 0,499 | [0,470; 0,529] | 0,510 | 0,507 |

MDA de quem prevê "sempre alta" nos mesmos dias: **0,517**.

| Comparação | Modelo | Diferença de AUC | IC 95% em blocos |
|---|---|---|---|
| Técnico − só texto | Regressão Logística | −0,007 | [−0,050; 0,035] |
| Técnico − só texto | Random Forest | −0,003 | [−0,040; 0,031] |
| Texto + técnico − só texto | Regressão Logística | +0,001 | [−0,022; 0,024] |
| Texto + técnico − só texto | Random Forest | +0,008 | [−0,016; 0,032] |

**Conclusão:** CONFIRMA a H2 como não sustentada.

- Nenhum dos conjuntos de variáveis se distingue do acaso.
- Somar o texto ao técnico, ou o técnico ao texto, não muda a AUC de forma significativa.
- Nenhuma MDA supera a regra trivial de "sempre alta".

---

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `a_benchmark_mesmo_periodo.csv` | Buy-and-hold em quatro versões e as estratégias da Tabela 3 |
| `b_h1_correlacoes.csv` | Pearson e Spearman (p-valores assintóticos) |
| `b_h1_bootstrap_blocos.csv` | IC 95% por bootstrap em blocos móveis (blocos de n^(1/3), 5, 10 e 20) |
| `b_h1_newey_west.csv` | Regressão com erros HAC (7, 5 e 10 defasagens; sentimento bruto e sem a média do bloco) |
| `b_h1_estacionariedade.csv` | ADF e KPSS |
| `c_backtest_sem_lookahead.csv` | Grade da Tabela 4 com limiares do período inteiro e dos 60 pregões anteriores |
| `d_eventos_retorno_anormal.csv` | CAAR: replicação da Fig. 7B e dois cenários com retorno anormal em pregões |
| `e_modelos_texto_tecnico.csv` | AUC/MDA dos modelos só texto, técnico e texto + técnico |
| `e_diferencas_auc.csv` | Diferença de AUC em relação ao só-texto, com IC por bootstrap em blocos |

Todas as tabelas trazem a coluna `rotulo` = "pós-submissão; não consta do artigo".
