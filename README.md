# TCC USP – Impacto do Sentimento de Notícias na Previsão do Ibovespa  

**Aluno:** Anderson P. M.  
**Curso:** MBA em Business Intelligence & Analytics – ECA/USP  
**Modalidade:** Projeto Aplicado em Business Intelligence e Analytics  
**Orientador:** _(a ser definido pela USP)_  

---

## 📌 Resumo do Projeto
Este projeto tem como objetivo investigar **o impacto do sentimento de notícias financeiras na previsão do índice Ibovespa**, utilizando uma abordagem híbrida que combina **modelos de linguagem de larga escala (LLMs)** com **análise técnica do mercado brasileiro**.  

O diferencial está na criação de um **pipeline ponta a ponta**, desde a coleta e tratamento de dados até a avaliação de modelos preditivos com métricas robustas, incluindo:  
- Validação **walk-forward**  
- Medição do **índice de latência T½** (velocidade de absorção da informação pelo mercado)  

O estudo busca contribuir tanto para a literatura acadêmica quanto para aplicações práticas em **Inteligência de Mercado e Finanças Quantitativas**.

---

## 🎯 Objetivos
1. Implementar um pipeline de dados integrando **notícias financeiras** e **cotações do Ibovespa**.  
2. Aplicar **análise de sentimento com LLMs** para extrair sinal de texto.  
3. Combinar sentimento + variáveis técnicas para treinar modelos preditivos.  
4. Avaliar o impacto da informação textual no desempenho do modelo.  
5. Investigar a latência da absorção de informações pelo mercado brasileiro.  

---

## 📂 Estrutura do Repositório
```bash
├── data/               # Dados utilizados no projeto
│   ├── raw/            # Dados brutos (extraídos da fonte original)
│   ├── interim/        # Dados intermediários (pré-processados)
│   └── processed/      # Dados finais prontos para modelagem
│
├── notebooks/          # Notebooks Jupyter/Colab
│   ├── eda/            # Análises exploratórias
│   └── models/         # Experimentos com modelos
│
├── src/                # Código fonte modular
│   ├── io/             # Scripts de entrada e saída de dados
│   ├── nlp/            # Funções de NLP e análise de sentimento
│   ├── features/       # Engenharia de variáveis
│   ├── modeling/       # Treinamento e avaliação de modelos
│   ├── backtest/       # Estratégias de validação (ex: walk-forward)
│   └── utils/          # Funções auxiliares
│
├── reports/            # Relatórios e resultados
│   └── figures/        # Gráficos e imagens geradas
│
├── configs/            # Arquivos de configuração
├── main.py             # Script principal do pipeline
├── requirements.txt    # Dependências do projeto
├── .env.example        # Exemplo de variáveis de ambiente
└── README.md           # Este documento
