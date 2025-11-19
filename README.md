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

## ⚙️ Requisitos
Instale as dependências com:  

```bash
pip install -r requirements.txt
```

### Dependências Principais
- **pandas, numpy**: Manipulação de dados
- **scikit-learn**: Machine Learning
- **sentence-transformers**: Embeddings de texto
- **yfinance**: Dados financeiros do Ibovespa
- **feedparser, gnews, beautifulsoup4**: Coleta multisource de notícias
- **vaderSentiment, langdetect**: Análise de sentimento PT-BR
- **mlflow**: Rastreamento de experimentos
- **plotly, matplotlib**: Visualizações

---

## 🚀 Como Usar

### 1️⃣ Pipeline Completo (Recomendado)
Execute o pipeline multisource completo (notebooks 12-15) com validações automáticas:

```bash
python run_pipeline_multisource.py
```

Este script irá:
- ✅ Coletar notícias de 4 fontes (GDELT, GNews, RSS, NewsAPI)
- ✅ Aplicar ETL e deduplicação
- ✅ Preprocessar textos em PT-BR com sentiment analysis
- ✅ Gerar features TF-IDF diárias e labels target
- ✅ Validar volume, cobertura temporal e qualidade dos dados

### 2️⃣ Notebooks Individuais
Execute notebooks específicos:

```bash
# Pipeline multisource
jupyter notebook notebooks/12_data_collection_multisource.ipynb
jupyter notebook notebooks/13_etl_dedup.ipynb
jupyter notebook notebooks/14_preprocess_ptbr.ipynb
jupyter notebook notebooks/15_features_tfidf_daily.ipynb

# Modelagem
jupyter notebook notebooks/16_models_tfidf_baselines.ipynb
```

### 3️⃣ Orquestração via Python
Execute notebooks programaticamente:

```bash
# Todos os notebooks
python pipeline_orchestration.py

# Apenas notebooks específicos
python pipeline_orchestration.py --only 12 13 14 15

# Continuar mesmo com erros
python pipeline_orchestration.py --continue-on-fail
```

### 4️⃣ Validações Standalone
Valide os dados sem executar o pipeline:

```bash
# Validação multisource (fontes, volume, cobertura)
python -m src.validation.check_multisource --min-years 3 --min-volume 5000

# Saúde geral do pipeline
python -m src.validation.check_pipeline_health
```

---

## 📊 Estrutura do Pipeline Multisource

```
┌─────────────────────────────────────────────────────┐
│  NB 12: Coleta Multisource                          │
│  - GDELT (2018-2025, histórico completo)            │
│  - GNews (últimos 6 meses)                          │
│  - RSS (6 fontes brasileiras em tempo real)         │
│  - NewsAPI (últimos 30 dias, opcional)              │
│  Output: news_multisource_raw_*.parquet             │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│  NB 13: ETL e Deduplicação                          │
│  - Dedup por URL, título+data, embeddings           │
│  - Validação de campos obrigatórios                 │
│  - Normalização de timezone                         │
│  Output: news_multisource.parquet                   │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│  NB 14: Preprocessamento PT-BR                      │
│  - Limpeza avançada (HTML, URLs, stopwords)         │
│  - Embeddings 768-dim (SentenceTransformer)         │
│  - Sentiment (VADER + keywords financeiros)         │
│  - Scores: credibilidade e novelty                  │
│  Output: news_clean.parquet, bow_daily.parquet      │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│  NB 15: Features TF-IDF Diário                      │
│  - Download Ibovespa via yfinance                   │
│  - Matriz TF-IDF com ngrams (1,2)                   │
│  - Labels multi-horizonte (D+1, D+3, D+5)           │
│  - Rolling features (volatilidade, sentiment)       │
│  Output: tfidf_daily_matrix.npz, labels_y_daily.csv │
└─────────────────────────────────────────────────────┘
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

## 🔐 Ética, LGPD e Uso de Dados

### **Conformidade com LGPD e Ética em Pesquisa**

Este projeto segue rigorosamente as diretrizes da **Lei Geral de Proteção de Dados (LGPD - Lei nº 13.709/2018)** e princípios éticos de pesquisa acadêmica:

#### **1. Fontes de Dados**
- ✅ **Coleta exclusiva de fontes públicas**: Notícias de portais jornalísticos, APIs abertas (GDELT, RSS), e comunicados oficiais (CVM - Fatos Relevantes).
- ✅ **Sem dados pessoais sensíveis**: Não coletamos, armazenamos ou processamos dados pessoais identificáveis (CPF, e-mail, endereços, etc.).
- ✅ **Metadados apenas**: Armazenamos apenas título, data, fonte, URL e trechos textuais das notícias (não o conteúdo integral).

#### **2. Uso de APIs e Termos de Serviço**
- ⚠️ O usuário é responsável por:
  - Obter suas próprias chaves de API (NewsAPI, etc.) respeitando os termos de uso de cada serviço.
  - Não redistribuir conteúdo protegido por direitos autorais.
  - Respeitar rate limits e políticas de acesso das fontes.

#### **3. Finalidade Exclusivamente Acadêmica**
- 📚 Este projeto é desenvolvido como **Trabalho de Conclusão de Curso (TCC)** no MBA em Business Intelligence & Analytics – ECA/USP.
- 🚫 **Uso comercial não autorizado**: Os dados, modelos e resultados não devem ser usados para fins lucrativos sem autorização explícita.
- 📖 Resultados podem ser publicados em meios acadêmicos (artigos, conferências) respeitando citações e direitos autorais.

#### **4. Transparência e Reprodutibilidade**
- 📊 Todo o pipeline é documentado e versionado (logs com timestamps, relatórios JSON).
- 🔍 Não há "caixa-preta": Código aberto permite auditoria de métodos e dados.
- ⚙️ Logs NÃO incluem dados pessoais, apenas estatísticas agregadas (contagens, fontes, períodos).

#### **5. Limitações e Responsabilidade**
- ⚠️ Os modelos preditivos desenvolvidos são para **fins educacionais e de pesquisa**, não constituindo recomendação de investimento.
- ⚠️ O autor não se responsabiliza por uso indevido dos dados, modelos ou resultados por terceiros.
- ⚠️ Dados de mercado (Ibovespa) são públicos via yfinance, mas sujeitos a termos da B3 e fornecedores.

#### **6. Contato e Dúvidas**
Para questões sobre privacidade, ética ou uso dos dados:
- 📧 Contato: [Inserir e-mail institucional USP]
- 🏛️ Instituição: MBA BI & Analytics – ECA/USP

---

**Declaração de Conformidade**: Este projeto foi desenvolvido em conformidade com as normas da LGPD, não processa dados pessoais sensíveis, e utiliza apenas informações públicas para fins de pesquisa acadêmica. O código está disponível para revisão e auditoria.

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
