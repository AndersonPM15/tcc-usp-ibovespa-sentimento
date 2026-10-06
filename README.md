# Análise do Sentimento de Notícias e seu Efeito no Ibovespa

**Artigo:** *Análise do Sentimento de Notícias e seu Efeito no Ibovespa: Um Estudo Empírico com
Baselines Transparentes* — XXIX SemeAd (2026), artigo 160.
**Trabalho de Conclusão de Curso:** MBA em Business Intelligence & Analytics — ECA/USP.
**Autor:** Anderson Pantoja Machado · **Orientador:** Prof. Vinicius Rocha Biscaro

**Versão do código usada no artigo:** commit `0b3f78b` (tag `v1.0-semead2026`). Commits
posteriores reorganizam o código sem alterar nenhum número do artigo, o que é verificado por
testes automáticos (ver [Reprodução](#reprodução)).

---

## O que o pipeline faz

1. **Notícias:** manchetes em português coletadas do GDELT 2.0 (notebook 12), de 02/01/2018 a
   19/11/2025: 91.941 manchetes de 508 domínios, após deduplicação pela URL normalizada, ou
   por data + título quando não há URL (notebook 13).
2. **Pré-processamento (notebook 14):** remoção de URLs, e-mails e números, minúsculas e
   stopwords do NLTK. O spaCy `pt_core_news_lg` (lematização) é usado se estiver instalado;
   na rodada que gerou o artigo ele não estava, então o texto não foi lematizado.
3. **Texto → vetor (notebook 15):** um documento por dia (todas as manchetes do dia
   concatenadas) e TF-IDF com `min_df=2`, `max_df=0.95`, n-gramas 1–2 e sem limite de termos
   (45.473 termos).
4. **Escore (notebook 16):** dois classificadores preveem a alta do Ibovespa entre D e D+1
   usando apenas o TF-IDF do dia D:
   - Regressão Logística (`saga`, L2, C=1, `class_weight="balanced"`);
   - Random Forest (200 árvores, profundidade máxima 5).
   As probabilidades *p* são geradas fora da amostra, em validação *walk-forward* com janela
   expansiva (`TimeSeriesSplit` com 4 blocos de teste de 388 pregões, treino inicial de 390
   pregões, sem embargo). O sentimento diário é **2p − 1**.
5. **Análises (`scripts/export_tcc_figures.py`):** correlação com retornos, backtests e
   estudo de eventos sobre 05/08/2019–30/12/2024 (1.341 pregões).

> **Rótulos no artigo:** as séries chamadas no artigo de "média simples do sentimento" e
> "média ponderada por volume" correspondem, respectivamente, à **Regressão Logística** e ao
> **Random Forest**. O código não agrega notícias nem pondera por volume.

### Como cada resultado do artigo é calculado

| Resultado | Arquivo em `reports/figures/` | Cálculo no código |
|---|---|---|
| Tabela 1 (amostra) | `Tabela_intersecao_periodo.csv` | 1.737 pregões (2018–2024); 1.341 dias com sentimento |
| Tabela 2 (AUC/MDA) | `Tabela_1_metricas.csv` | Notebook 16, sobre 1.552 dias fora da amostra (05/08/2019–17/11/2025). MDA = acurácia de (p ≥ 0,5). IC 95% por bootstrap i.i.d. (1.000 reamostragens, semente 42) |
| Tabela 3 (backtest, cenário padrão) | `Tabela_3_metricas_extendidas.csv` | Compra se p ≥ quantil 0,90 de p no período inteiro; vende se p ≤ mediana; posição aplicada com 1 dia de defasagem (lag 0); custo de 0,0005 por turnover. Benchmark: buy-and-hold de 2018 a 2024 |
| Tabela 4 (robustez) | `Tabela_2_robustez_backtest.csv` | Mesma regra, com quantis 0,90 e 0,95 e lags 0, 1 e 2 (defasagem efetiva = lag + 1) |
| Figura 1 | `Figura_1_ibov_eventos.png` | Eventos de sentimento extremo cujo \|CAR\| está no percentil 90 ou acima |
| Figura 3 | `Figura_3_comparativo_modelos.png` | Sharpe da regra `long_only_60`: compra se p ≥ 0,60, vende se p ≤ 0,40, mantém entre os dois |
| Figura 4 | `Figura_4_dispersao_sentimento_retorno.png` | Pearson entre o sentimento da Regressão Logística e o retorno do **mesmo dia** (D−1 → D) |
| Figura 7B | `Figura_7B_event_time_CAAR.*` | Eventos = dias com a média do sentimento dos dois modelos no percentil 90 ou acima, ou no 10 ou abaixo. CAR = soma dos retornos brutos de D0 até D0+τ **dias corridos** |
| Figura 8 | `Figura_8_backtest_vs_benchmark.*` | Curvas da regra `long_only_60` e do Ibovespa nos mesmos 1.341 dias |

---

## Dados

Os dados não são versionados no repositório (licenças das fontes de notícias). Para reproduzir
os números do artigo, a pasta apontada por `TCC_USP_BASE` deve conter `data_processed/` com:

| Arquivo | Conteúdo |
|---|---|
| `16_oof_predictions.csv` | p diário fora da amostra da Regressão Logística e do Random Forest (até 30/12/2024) |
| `tfidf_daily_matrix.npz`, `tfidf_daily_index.csv` | Matriz TF-IDF diária (2.771 dias × 45.473 termos) |
| `labels_y_daily.csv` | Alvo (alta em D+1) e retorno de D para D+1 |
| `ibovespa_clean.csv` | Ibovespa diário de 2018 a 2024 |

Configure a variável copiando `.env.example` para `.env`, ou com `export TCC_USP_BASE=...`.

As verificações pós-submissão usam também o Ibovespa até 18/11/2025, lido do histórico do git
(versão de `ibovespa_clean.csv` no blob `1e9e2ec`). Por isso, precisam de um clone com histórico
completo.

---

## Reprodução

```bash
python3.11 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
export TCC_USP_BASE=/caminho/para/TCC_USP                # pasta que contém data_processed/

pytest                                                   # confere os números do artigo
python scripts/export_tcc_figures.py --strategy long_only_60 --run_robustness   # regenera reports/figures
python scripts/run_post_submission_checks.py             # verificações pós-submissão (reports/verificacao)
```

O teste `tests/test_article_reproduction.py` confere as Tabelas 1 a 4, as Figuras 3, 4 e 8 e o
estudo de eventos contra referências congeladas em `tests/fixtures/article/`. A Tabela 2 é
recalculada a partir da matriz TF-IDF, executando o código do notebook 16. Sem os dados, os
testes são pulados.

**Exceção documentada — Figura 7B:** no artigo, o bootstrap dos intervalos de confiança não
tinha semente fixa. O código agora usa a semente 42, então os ICs podem diferir dos publicados a
partir da 3ª casa decimal. O teste aceita até 0,0025 nas médias e 0,01 nos ICs. No grupo
positivo em τ=4 (15 eventos), o limite superior do IC muda de sinal conforme o sorteio.

### Versões de reprodução (2026)

Testadas em ambiente limpo com **Python 3.11.15**:

| Arquivo | Uso | Principais versões |
|---|---|---|
| `requirements.txt` | Reproduzir o artigo e as verificações | numpy 2.4.6, pandas 2.3.3, scipy 1.17.1, scikit-learn 1.9.1, statsmodels 0.15.0, matplotlib 3.11.2 |
| `requirements-dev.txt` | Testes, lint e pré-commit | pytest 9.1.1, ruff 0.16.10, pre-commit 4.6.2 |
| `requirements-optional.txt` | Recoleta, pré-processamento, dashboard e orquestração | nltk 3.10.3, spacy 3.8.16, dash 4.4.1, papermill 2.7.0 |

Os notebooks-protótipo de embeddings e LSTM (04, 08 e 09) usam TensorFlow e
sentence-transformers. Eles não fazem parte do artigo e essas bibliotecas não estão fixadas.

---

## Verificações pós-submissão

**Pós-submissão; não consta do artigo.** Em [`reports/verificacao/`](reports/verificacao/README.md)
há cinco análises feitas depois da submissão, com as mesmas previsões fora da amostra:

- **(a)** benchmark no mesmo período das estratégias;
- **(b)** H1 com IC por bootstrap em blocos, Newey-West e ADF/KPSS;
- **(c)** backtest sem *look-ahead* (limiares dos 60 pregões anteriores);
- **(d)** estudo de eventos com retorno anormal e janela em pregões;
- **(e)** modelo técnico e modelo texto + técnico.

Elas confirmam as conclusões centrais do artigo: não há poder preditivo nem valor econômico.
Também mostram dois pontos que não se sustentam:

- o r ≈ −0,13 é contemporâneo e significativo, não preditivo;
- o CAAR negativo em τ = 4 da Figura 7B.

As opções novas ficam no mesmo código do artigo, por exemplo `threshold_mode="rolling"` e
`window_unit="trading_days"`. O padrão de cada uma continua reproduzindo o artigo.

---

## Padrões do repositório

- **Idioma:** nomes de funções, variáveis, módulos e testes em inglês, seguindo a convenção do
  Python e das bibliotecas usadas. Comentários, docstrings, mensagens, documentação e
  relatórios em português.
- **Formatação e lint:** ruff, com linhas de 100 colunas (configuração em `pyproject.toml`).
- **Antes de cada commit:** `pre-commit install` ativa lint, formatação e o teste de
  reprodução do artigo (`.pre-commit-config.yaml`).
- **Segredos:** chaves de API ficam no `.env`, que não é versionado (`src/config/secrets.py`).

---

## Estrutura

```
app_dashboard.py              Dashboard interativo (Dash/Plotly)
pipeline_orchestration.py     Executa os notebooks em sequência (papermill/nbconvert)
configs/config_tcc.yaml       Período do estudo e nomes de arquivos
notebooks/
  12_…ipynb a 18_…ipynb, 20_  Pipeline do artigo: coleta, ETL, pré-processamento, TF-IDF,
                              modelos, validação e backtest
  00_…ipynb a 11_…, 19_       Protótipos de desenvolvimento (dados sintéticos ou fonte única),
                              fora do pipeline do artigo
scripts/export_tcc_figures.py Figuras e tabelas do artigo (reports/figures)
scripts/generate_event_study_latency.py  Eventos do estudo de eventos
scripts/run_post_submission_checks.py    Verificações pós-submissão (reports/verificacao)
src/                          Módulos reutilizáveis (caminhos, configuração, coletores, validação)
  models/walk_forward.py      Walk-forward dos classificadores (mesma lógica do notebook 16)
  features/technical.py       Variáveis técnicas (retornos defasados, volatilidade)
  analysis/h1_tests.py        Bootstrap em blocos, Newey-West, ADF/KPSS
tests/                        Testes (pytest), incluindo a reprodução do artigo
reports/figures/              Figuras e tabelas exportadas
reports/verificacao/          Resultados pós-submissão (não constam do artigo)
```

---

## Licença e citação

Código sob licença MIT (`LICENSE`). Os dados de notícias pertencem às respectivas fontes e não
são redistribuídos.

MACHADO, Anderson Pantoja. Análise do Sentimento de Notícias e seu Efeito no Ibovespa: Um
Estudo Empírico com Baselines Transparentes. In: XXIX SemeAd — Seminários em Administração
FEA-USP, 2026. Artigo 160.
