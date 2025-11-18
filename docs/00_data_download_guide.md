# Download de Dados do Ibovespa

Este documento explica como executar o notebook `00_data_download.ipynb` para baixar dados históricos do Ibovespa.

## Visão Geral

O notebook `00_data_download.ipynb` está configurado para baixar dados históricos do Ibovespa (^BVSP) e do ETF BOVA11.SA usando a biblioteca `yfinance`. O período de download é definido no arquivo `configs/config_tcc.yaml`.

## Período Configurado

- **Início**: 2018-01-01
- **Fim**: 2025-01-31
- **Total de dias úteis**: ~1850 registros

## Pré-requisitos

1. Python 3.8+
2. Dependências instaladas (via `requirements.txt`):
   - pandas
   - yfinance
   - jupyter
   - papermill

```bash
pip install -r requirements.txt
```

## Execução

### Opção 1: Via Papermill (Linha de Comando)

```bash
# Criar diretório de output se não existir
mkdir -p notebooks/_runs

# Executar o notebook
python -m papermill notebooks/00_data_download.ipynb \
    notebooks/_runs/$(date +%Y%m%d_%H%M%S)_00_data_download.ipynb \
    -k python3
```

### Opção 2: Via Jupyter Notebook

1. Abrir o Jupyter:
   ```bash
   jupyter notebook
   ```

2. Navegar até `notebooks/00_data_download.ipynb`

3. Executar todas as células (Cell → Run All)

### Opção 3: Via VS Code

1. Abrir `notebooks/00_data_download.ipynb` no VS Code
2. Selecionar o kernel Python apropriado
3. Executar todas as células

## Comportamento com Fallback

O notebook inclui tratamento de erros robusto:

1. **Tentativa de Download via yfinance**: Primeiro tenta baixar dados reais da Yahoo Finance
2. **Fallback Automático**: Se o download falhar (sem internet, erro de API, etc.), automaticamente:
   - Chama o script `scripts/generate_sample_data.py`
   - Gera dados de amostra realistas usando movimento Browniano geométrico
   - Os dados gerados têm a mesma estrutura dos dados reais

## Arquivos Gerados

Os dados são salvos em `data_raw/`:

- **ibovespa.csv**: Dados do índice Ibovespa (^BVSP)
- **bova11.csv**: Dados do ETF BOVA11.SA

### Estrutura dos Arquivos CSV

Cada arquivo contém as seguintes colunas:

- `day`: Data (formato YYYY-MM-DD)
- `open`: Preço de abertura
- `high`: Preço máximo do dia
- `low`: Preço mínimo do dia
- `close`: Preço de fechamento
- `adj_close`: Preço de fechamento ajustado
- `volume`: Volume negociado
- `source_ticker`: Ticker de origem (^BVSP ou BOVA11.SA)

## Validação dos Dados

Após a execução, você pode validar os dados gerados:

```python
import pandas as pd
from pathlib import Path

base_path = Path("data_raw")
for file in ["ibovespa.csv", "bova11.csv"]:
    path = base_path / file
    if path.exists():
        df = pd.read_csv(path, parse_dates=["day"])
        print(f"{file}:")
        print(f"  Records: {len(df)}")
        print(f"  Date range: {df['day'].min()} to {df['day'].max()}")
        print(f"  File size: {path.stat().st_size / 1024:.1f} KB")
        print()
```

## Próximos Passos

Após executar o notebook 00, você pode prosseguir com os demais notebooks:

1. `01_preprocessing.ipynb`: Limpeza e preparação dos dados
2. `02_baseline_logit.ipynb`: Modelos baseline
3. ... (demais notebooks conforme necessário)

Os dados processados serão salvos em `data_processed/` à medida que você executa os notebooks subsequentes.

## Solução de Problemas

### Erro: "No kernel name found in notebook"

**Solução**: Especifique o kernel ao executar com papermill:
```bash
python -m papermill notebooks/00_data_download.ipynb output.ipynb -k python3
```

### Erro: "Could not resolve host: guce.yahoo.com"

**Solução**: Sem internet ou firewall bloqueando. O notebook automaticamente usará o fallback de dados de amostra.

### Erro: "Sem dados retornados para ^BVSP"

**Solução**: Verifique se o período configurado em `configs/config_tcc.yaml` está correto. O notebook automaticamente usará o fallback de dados de amostra.

## Notas Técnicas

- Os dados de amostra são gerados de forma determinística (seed=42)
- A volatilidade dos dados de amostra (~1.8% diário para Ibovespa) é realista
- O movimento é modelado com drift positivo (~0.03% diário)
- Os dados de amostra são adequados para testes e desenvolvimento
- Para análises finais, recomenda-se usar dados reais sempre que possível
