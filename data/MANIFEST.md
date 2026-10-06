# Manifesto dos dados

Arquivos de entrada do pipeline, na pasta apontada por `TCC_USP_BASE`. Os dados não são
versionados (licenças das fontes de notícias). Gerado por `python -m tcc manifest --write`;
`python -m tcc manifest` confere a sua pasta. O SHA-256 abaixo é o do conteúdo (tabela em
CSV canônico ou valores da matriz), que não depende do sistema em que o arquivo foi
gravado; o do arquivo está em `MANIFEST.json`.

| Arquivo | Origem | Usado por | Período | Linhas | SHA-256 |
|---|---|---|---|---|---|
| `data_processed/ibovespa_clean.csv` | Yahoo Finance, ^BVSP sem ajuste (`python -m tcc download-ibovespa`) | `reproduce` | 2018-01-02 a 2025-11-18 | 1.960 | `d5a6ba96f52b2ad61ed12734a267a7810ff6e3e0552df5f72f73a7f261111552` |
| `data_processed/tfidf_daily_matrix.npz` | TF-IDF diário das notícias limpas (`python -m tcc build-tfidf`) | `reproduce` | — | 2.771 × 45.473 | `610a1fdfedbb5143412ddcb94fb1cc53d574ccc21ba5530848c11b14de2dfb69` |
| `data_processed/tfidf_daily_index.csv` | dia de cada linha da matriz TF-IDF (`python -m tcc build-tfidf`) | `reproduce` | 2018-01-02 a 2025-11-19 | 2.771 | `0a9e48e83b9ec5c97e617b1893beb946efa53619f433691b597afb0c0403cf10` |
| `data_interim/news_clean_multisource.parquet` | manchetes do GDELT deduplicadas (`python -m tcc clean-news`) | `build-tfidf` | 2018-01-02 a 2025-11-19 | 91.941 | `b94a786816f27348af1786a954010938efece7d4b0fb65d24759cdbdb86214ee` |
| `data_raw/news_multisource.parquet` | coleta do GDELT DOC 2.0 (`python -m tcc collect-news`) | `clean-news` | — | — | pendente: arquivo não disponível nesta máquina; rode python -m tcc manifest --write onde ele estiver |
