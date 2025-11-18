# Pipeline de Dados - Fluxo do Notebook 00

```
┌─────────────────────────────────────────────────────────────────┐
│                   00_data_download.ipynb                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Ler config YAML │
                    │  (2018-2025)     │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Tentar yfinance  │
                    │   download       │
                    └──────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                  Sucesso            Falha
                    │                   │
                    │                   ▼
                    │         ┌──────────────────┐
                    │         │ Fallback:        │
                    │         │ generate_sample  │
                    │         │    _data.py      │
                    │         └──────────────────┘
                    │                   │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Salvar em       │
                    │  data_raw/       │
                    │                  │
                    │  • ibovespa.csv  │
                    │  • bova11.csv    │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Validar dados   │
                    │  (1850 records)  │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Pronto para      │
                    │ notebook 01      │
                    └──────────────────┘
```

## Estrutura dos Dados

### Entrada
- **Configuração**: `configs/config_tcc.yaml`
  - `periodo_estudo.start`: "2018-01-01"
  - `periodo_estudo.end`: "2025-01-31"

### Saída
- **Localização**: `data_raw/`
- **Arquivos**:
  - `ibovespa.csv` (^BVSP) - 1850 registros
  - `bova11.csv` (BOVA11.SA) - 1850 registros

### Estrutura CSV
| Coluna       | Tipo      | Descrição                    |
|--------------|-----------|------------------------------|
| day          | datetime  | Data do pregão               |
| open         | float     | Preço de abertura            |
| high         | float     | Preço máximo                 |
| low          | float     | Preço mínimo                 |
| close        | float     | Preço de fechamento          |
| adj_close    | float     | Preço ajustado               |
| volume       | int       | Volume negociado             |
| source_ticker| string    | Ticker de origem             |

## Próximos Notebooks

```
data_raw/
  └─> 01_preprocessing.ipynb
        └─> data_processed/
              └─> 02_baseline_logit.ipynb
                    └─> 03_tfidf_models.ipynb
                          └─> ... (04-18)
```
