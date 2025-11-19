# 📊 TCC USP – Impacto do Sentimento de Notícias na Previsão do Ibovespa  

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)  
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange.svg?logo=jupyter)  
![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow.svg)  
![License](https://img.shields.io/badge/Licença-Acadêmica-green.svg)  

**Aluno:** Anderson P. M.  
**Curso:** MBA em Business Intelligence & Analytics – ECA/USP  
**Modalidade:** Projeto Aplicado em Business Intelligence e Analytics  
**Orientador:** _(a ser definido pela USP)_  

---

## 📌 Resumo do Projeto
Este projeto investiga **o impacto do sentimento de notícias financeiras na previsão do índice Ibovespa**, por meio de uma abordagem híbrida que combina:  

- **Modelos de linguagem de larga escala (LLMs)**  
- **Análise técnica do mercado financeiro brasileiro**  

O diferencial está na criação de um **pipeline ponta a ponta**, desde a coleta e pré-processamento de dados até a avaliação de modelos preditivos.  

Destaques:  
- Validação **walk-forward**  
- Medição do **índice de latência T½** (velocidade de absorção da informação pelo mercado)  

O estudo visa gerar contribuições relevantes para a **literatura acadêmica** e aplicações práticas em **Inteligência de Mercado e Finanças Quantitativas**.

---

## 🎯 Objetivos
1. Construir um pipeline de dados integrando **notícias financeiras** e **cotações do Ibovespa**.  
2. Aplicar **análise de sentimento via LLMs** para extração de sinal textual.  
3. Combinar variáveis de **sentimento + técnicas** em modelos preditivos.  
4. Avaliar o impacto da informação textual no desempenho dos modelos.  
5. Investigar a **latência de absorção da informação** no mercado brasileiro.  

---

## 📂 Estrutura do Repositório
```bash
├── data/               # Conjunto de dados
│   ├── raw/            # Dados brutos (originais)
│   ├── interim/        # Dados intermediários (pré-processados)
│   └── processed/      # Dados finais prontos para modelagem
│
├── notebooks/          # Notebooks Jupyter/Colab
│   ├── eda/            # Análises exploratórias
│   └── models/         # Testes de modelos
│
├── src/                # Código fonte modular
│   ├── io/             # Entrada e saída de dados
│   ├── nlp/            # Funções de NLP e análise de sentimento
│   ├── features/       # Engenharia de variáveis
│   ├── modeling/       # Treinamento e avaliação
│   ├── backtest/       # Estratégias de validação (ex: walk-forward)
│   └── utils/          # Funções auxiliares
│
├── reports/            # Relatórios e resultados
│   └── figures/        # Gráficos e imagens
│
├── configs/            # Arquivos de configuração
├── main.py             # Script principal do pipeline
├── requirements.txt    # Dependências do projeto
├── .env.example        # Exemplo de variáveis de ambiente
└── README.md           # Este documento
```

---

## 📓 Organização dos Notebooks

### **Notebooks Oficiais (Pipeline TCC)**
Estes notebooks fazem parte do fluxo oficial de análise do TCC e são executados pelo `pipeline_orchestration.py`:

- **00_data_download.ipynb** - Download e validação inicial dos dados
- **01_preprocessing.ipynb** - Pré-processamento básico
- **13_etl_dedup.ipynb** - ETL e deduplicação de notícias multisource
- **14_preprocess_ptbr.ipynb** - Lematização e processamento de texto PT-BR
- **15_features_tfidf_daily.ipynb** - Construção de features TF-IDF agregadas por dia
- **16_models_tfidf_baselines.ipynb** - Modelagem principal (Logistic Regression, Random Forest)
- **18_backtest_simulation.ipynb** - Simulação de estratégias de trading
- **20_final_dashboard_analysis.ipynb** - Preparação dos dados para o dashboard

### **Notebooks Experimentais**
Estes notebooks foram utilizados em fases exploratórias e experimentos preliminares, mas **não fazem parte do pipeline final do TCC**:

- **02_baseline_logit.ipynb** - Experimento preliminar de baseline
- **03_tfidf_models.ipynb** - Testes iniciais de TF-IDF (substituído pelo 16)
- **04_embeddings_models.ipynb** - Experimentos com embeddings (abordagem alternativa)
- **05_data_collection_real.ipynb** - Coleta antiga de dados reais (substituído pelo 12)
- **06_preprocessing_real.ipynb** - Pré-processamento antigo (substituído pelo 14)
- **07_tfidf_real.ipynb** - TF-IDF preliminar (substituído pelo 15+16)
- **08_embeddings_real.ipynb** - Embeddings em dados reais (não usado no TCC final)
- **09_lstm_real.ipynb** - Experimentos com LSTM (não incluído no TCC final)
- **10_dashboard_results.ipynb** - Análise preliminar (substituído pelo 20)
- **11_event_study_latency.ipynb** - Estudo de eventos e latência (análise complementar)
- **12_data_collection_multisource.ipynb** - Coleta multisource (preparatória para o 13)
- **17_sentiment_validation.ipynb** - Validação exploratória de sentimento
- **19_future_extension.ipynb** - Planejamento de trabalhos futuros

---

## ⚙️ Requisitos
Instale as dependências com:  

```bash
pip install -r requirements.txt
```

---

## 🔑 Variáveis de Ambiente
Crie um arquivo `.env` na raiz do projeto, baseado no modelo `.env.example`:  

```ini
DB_URL=
BQ_PROJECT_ID=
LLM_API_KEY=
```

---

## 📊 Metodologia
- **Coleta de Dados:** notícias financeiras e séries históricas do Ibovespa.  
- **Pré-processamento:** limpeza, normalização e tokenização de textos.  
- **Análise de Sentimento:** embeddings e LLMs para classificação.  
- **Modelagem:** regressão logística, árvores de decisão, ensembles e deep learning.  
- **Validação:** backtest walk-forward + métricas AUC, Accuracy e MDA.  
- **Latência T½:** mensuração da absorção da informação nos preços.  

---

## 📅 Cronograma de Entregas
| Fase | Atividade | Prazo |
|------|-----------|-------|
| 1 | Revisão Bibliográfica e Plano de Pesquisa | Março/2025 |
| 2 | Coleta e Tratamento de Dados | Abril/2025 |
| 3 | Implementação do Pipeline (Sentimento + Técnica) | Maio/2025 |
| 4 | Experimentos e Validação | Junho/2025 |
| 5 | Redação Final do TCC | Julho/2025 |
| 6 | Defesa | Agosto/2025 |

---

## 📚 Referências Iniciais
- Tetlock, P. C. (2007). *Giving Content to Investor Sentiment: The Role of Media in the Stock Market.*  
- Nassirtoussi, A. K., et al. (2015). *Text Mining for Market Prediction: A Systematic Review.*  
- Estudos recentes sobre **LLMs aplicados a finanças e previsão de mercado**.  

---

## 📜 Licença
Uso **acadêmico e de pesquisa**.  
Este repositório integra o **Trabalho de Conclusão de Curso (TCC)** no MBA em Business Intelligence & Analytics – **ECA/USP**.
