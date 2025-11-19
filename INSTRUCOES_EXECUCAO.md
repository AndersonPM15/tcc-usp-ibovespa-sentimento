# 🚀 INSTRUÇÕES DE EXECUÇÃO - PIPELINE MULTISOURCE

## 📋 Pré-requisitos

1. **Python 3.10+** instalado
2. **Dependências instaladas:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Variáveis de ambiente (opcional):**
   ```bash
   # Windows CMD
   set NEWSAPI_KEY=sua_chave_aqui
   
   # Windows PowerShell
   $env:NEWSAPI_KEY='sua_chave_aqui'
   
   # Linux/Mac
   export NEWSAPI_KEY=sua_chave_aqui
   ```
   > Nota: NewsAPI é opcional. Pipeline funciona sem ele usando outras 3 fontes.

---

## ⚡ EXECUÇÃO RÁPIDA (Recomendado)

### Método 1: Script Automatizado
Execute o pipeline completo com validações automáticas:

```bash
python run_pipeline_multisource.py
```

Este script irá:
- ✅ Executar notebooks 12→13→14→15 em sequência
- ✅ Validar dados multisource (fontes, volume, cobertura)
- ✅ Verificar saúde geral do pipeline
- ✅ Gerar relatórios de validação

**Tempo estimado**: ~20-40 minutos (depende da velocidade da internet e CPU)

---

## 🎯 EXECUÇÃO DETALHADA

### Método 2: Notebooks Individuais

Execute cada notebook manualmente para maior controle:

#### **Passo 1: Coleta Multisource**
```bash
jupyter notebook notebooks/12_data_collection_multisource.ipynb
```
- Coleta notícias de GDELT, GNews, RSS, NewsAPI
- Output: `news_multisource_raw_*.parquet`
- Tempo: ~10-15 minutos

#### **Passo 2: ETL e Deduplicação**
```bash
jupyter notebook notebooks/13_etl_dedup.ipynb
```
- Deduplica e valida dados
- Output: `news_multisource.parquet`
- Tempo: ~3-5 minutos

#### **Passo 3: Preprocessamento PT-BR**
```bash
jupyter notebook notebooks/14_preprocess_ptbr.ipynb
```
- Limpa textos, gera embeddings, sentiment
- Output: `news_clean.parquet`, `bow_daily.parquet`
- Tempo: ~10-20 minutos (embeddings são lentos)

#### **Passo 4: Features TF-IDF**
```bash
jupyter notebook notebooks/15_features_tfidf_daily.ipynb
```
- Gera matriz TF-IDF e labels
- Output: `tfidf_daily_matrix.npz`, `labels_y_daily.csv`
- Tempo: ~2-5 minutos

---

## 🤖 EXECUÇÃO PROGRAMÁTICA

### Método 3: Orquestração via Python

Execute notebooks via script sem abrir Jupyter:

```bash
# Todos os notebooks multisource
python pipeline_orchestration.py --only 12 13 14 15

# Com log detalhado
python pipeline_orchestration.py --only 12 13 14 15 > pipeline.log 2>&1

# Continuar mesmo com erros
python pipeline_orchestration.py --only 12 13 14 15 --continue-on-fail
```

---

## ✅ VALIDAÇÕES

### Método 4: Validação Standalone

Execute validações sem rodar o pipeline:

#### **Validação Multisource**
```bash
python -m src.validation.check_multisource --min-years 3 --min-volume 5000
```

Verifica:
- ✅ Arquivos esperados existem
- ✅ 4 fontes de dados presentes (gdelt, gnews, rss, newsapi)
- ✅ Cobertura temporal >= 3 anos
- ✅ Volume >= 5000 notícias
- ✅ Qualidade dos dados (duplicatas, campos vazios)

Exit codes:
- `0` = PASSED (tudo OK)
- `1` = FAILED (erro crítico)
- `2` = WARNING (avisos)

#### **Saúde do Pipeline**
```bash
python -m src.validation.check_pipeline_health
```

Verifica:
- ✅ Outputs de todos os notebooks (00-20)
- ✅ Arquivos de resultados (.json, .parquet, .npz)
- ✅ Status: HEALTHY / WARNING / UNHEALTHY

---

## 🧹 LIMPEZA (Antes de Commit)

### Método 5: Limpeza do Projeto

```bash
python cleanup_project.py
```

Remove:
- 🗑️ Outputs de notebooks
- 🗑️ Diretórios __pycache__
- 🗑️ Arquivos temporários (.pyc, .DS_Store, desktop.ini)

Verifica:
- ✅ Arquivos críticos existem
- 📊 Estatísticas do código

---

## 📊 VERIFICAÇÃO FINAL

Execute o script de verificação completa:

```bash
python verify_project.py
```

Este script:
- ✅ Verifica datas de todos os arquivos
- ✅ Audita notebooks
- ✅ Testa funcionalidade do dashboard
- ✅ Gera relatório completo

---

## 🎯 FLUXO COMPLETO RECOMENDADO

```bash
# 1. Executar pipeline multisource
python run_pipeline_multisource.py

# 2. Executar notebooks de modelagem (opcional)
python pipeline_orchestration.py --only 16 17 18

# 3. Verificar resultados no dashboard
jupyter notebook notebooks/10_dashboard_results.ipynb

# 4. Antes de commit: limpar e verificar
python cleanup_project.py
python verify_project.py
```

---

## ❓ TROUBLESHOOTING

### Problema: "Arquivo não encontrado"
**Solução**: Execute os notebooks anteriores primeiro. Sequência: 12→13→14→15

### Problema: "ModuleNotFoundError: src"
**Solução**: Execute do diretório raiz do projeto, não de `notebooks/`

### Problema: Embeddings muito lentos
**Solução**: No NB 14, configure `generate_emb=False` para pular embeddings

### Problema: Poucos dados retornados
**Solução**: 
- Verifique conexão com internet
- GDELT pode ter limitações temporárias
- Execute novamente após algumas horas

### Problema: NewsAPI retorna erro
**Solução**: NewsAPI é opcional. Pipeline funciona sem ele. Para usar:
1. Obtenha chave gratuita em https://newsapi.org
2. Configure variável de ambiente `NEWSAPI_KEY`

---

## 📈 RESULTADOS ESPERADOS

Após execução completa, você terá:

### **Dados Processados** (`data_processed/`)
- ✅ `news_multisource_raw_*.parquet` - ~30.000+ notícias
- ✅ `news_multisource.parquet` - Dedupado (~27.000+)
- ✅ `news_clean.parquet` - Preprocessado com features
- ✅ `tfidf_daily_matrix.npz` - Matriz TF-IDF
- ✅ `labels_y_daily.csv` - Labels D+1/D+3/D+5
- ✅ `dataset_daily_complete.parquet` - Dataset completo

### **Relatórios** (`reports/`)
- ✅ `etl_report_*.json`
- ✅ `tfidf_report_*.json`
- ✅ `validation_multisource_*.json`
- ✅ `pipeline_health_*.json`

### **Logs** (`reports/logs/`)
- ✅ `pipeline_run_*.log`

---

## 🎓 PRÓXIMOS PASSOS

1. **Modelagem**: Execute notebooks 16-20
2. **Dashboard**: Visualize resultados no NB 10 ou 20
3. **Análises**: Event study (NB 11), backtest (NB 18)
4. **Documentação**: Revisar relatórios em `reports/`

---

## 📞 SUPORTE

- **Documentação**: Veja `README.md` e `reports/CONCLUSAO_PIPELINE_MULTISOURCE.md`
- **Logs**: Verifique `reports/logs/` para erros detalhados
- **Validação**: Use scripts de validação para diagnosticar problemas

---

**Última atualização**: 18/11/2025  
**Status**: Pipeline testado e funcional
