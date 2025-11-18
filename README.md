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

---

## 🚀 Getting Started

### 1. Download dos Dados do Ibovespa

Para começar, execute o notebook de download de dados:

```bash
python -m papermill notebooks/00_data_download.ipynb \
    notebooks/_runs/$(date +%Y%m%d_%H%M%S)_00_data_download.ipynb \
    -k python3
```

Este notebook:
- Baixa dados históricos do Ibovespa (^BVSP) e BOVA11.SA via yfinance
- Período: 2018-01-01 a 2025-01-31 (configurável em `configs/config_tcc.yaml`)
- Inclui fallback automático para dados de amostra caso o download falhe
- Gera arquivos CSV em `data_raw/`

📖 **Consulte o guia completo:** [docs/00_data_download_guide.md](docs/00_data_download_guide.md)

### 2. Próximos Notebooks

Após o download, execute os notebooks seguintes na ordem:
- `01_preprocessing.ipynb`: Limpeza e preparação dos dados
- `02_baseline_logit.ipynb`: Modelos baseline
- ... (demais notebooks conforme pipeline)

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
