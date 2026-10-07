# Figuras da apresentação

Geradas por `python -m tcc presentation-figures` (código em `src/tcc/presentation/`).
Cada figura tem PNG (300 dpi), SVG e o CSV com os dados plotados (ponto decimal no
CSV; vírgula decimal nas figuras). Tamanho de slide 11,3 × 5,3 pol. (largura total:
17,7 × 5,3 pol.), sem título dentro da figura.

> Arquivos com `pos-submissao` no nome trazem resultados calculados depois da
> submissão (verificações em `reports/verificacao/`); não constam do artigo.

| Arquivo | O que mostra |
|---|---|
| `F1_ibovespa_periodos_manchetes` | Ibovespa de 2018 a 2025 com as faixas do estudo e manchetes por dia (base limpa) |
| `F2_sentimento_diario` | Sentimento diário (2p − 1) dos dois modelos, fronteiras dos blocos do walk-forward e histograma |
| `F3_walk_forward` | As 4 etapas do walk-forward, com treino e teste em datas reais |
| `F4a_dispersao_sentimento_retorno_pos-submissao` | Sentimento da Regressão Logística × retorno do mesmo dia e do dia seguinte, com reta de MQO, r e IC 95% por bootstrap em blocos (verificação b) |
| `F4b_correlacao_movel` | Correlação móvel de 60 e 90 pregões entre sentimento e retorno do mesmo dia (Figura 5 do artigo) |
| `F5a_curvas_roc` | Curvas ROC fora da amostra, com a AUC e o IC 95% da Tabela 2 |
| `F5b_auc_por_bloco_pos-submissao` | AUC de cada bloco de teste do walk-forward, com IC 95% (bootstrap i.i.d.) |
| `F6_caar_retorno_anormal_pos-submissao` | CAAR com retorno anormal, τ = 0 a 4 pregões e IC 95% (verificação d) |
| `F7a_patrimonio_limiares_artigo_pos-submissao` | Patrimônio (início = 1,0) das estratégias da Tabela 3 e do Ibovespa nas mesmas linhas (verificação a); sinais de 05/08/2019 a 30/12/2024, cada ponto datado pelo pregão em que o retorno se realiza |
| `F7b_patrimonio_sem_lookahead_pos-submissao` | O mesmo, com limiares sem look-ahead (verificação c) |
| `F8_sharpe_12_configuracoes_pos-submissao` | Sharpe das 12 configurações da Tabela 4, com os limiares do artigo e sem look-ahead, e o Sharpe do Ibovespa no mesmo período (verificações a e c) |
| `F9_auc_token_none_pos-submissao` | AUC com e sem o token espúrio "None" (bug 12), com IC 95% |
| `F1_cobertura_por_ano` | Tabela de apoio à F1 (só CSV): dias com manchete, média, mediana e total de manchetes por dia em cada ano da base limpa (2025 até 19/11) |
