"""
AUDITORIA DA PIPELINE DE DADOS - TCC USP
Objetivo: Identificar onde a base está sendo estrangulada de ~2900 dias para 20 dias
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.io import paths
from src.config.constants import START_DATE, END_DATE

DATA_PATHS = paths.get_data_paths()
RAW = DATA_PATHS['data_raw']
PROCESSED = DATA_PATHS['data_processed']

print("="*100)
print("AUDITORIA DA PIPELINE DE DADOS - TCC USP")
print("="*100)
print(f"\nPeríodo planejado: {START_DATE} a {END_DATE}")
print(f"Dias teóricos: {(END_DATE - START_DATE).days + 1}")
print(f"\nRAW: {RAW}")
print(f"PROCESSED: {PROCESSED}")

audit_results = []

print("\n" + "="*100)
print("1. IBOVESPA.CSV (BRUTO)")
print("="*100)
try:
    ibov_raw = pd.read_csv(RAW / 'ibovespa.csv', parse_dates=['date'])
    print(f"✅ Shape: {ibov_raw.shape}")
    print(f"   Colunas: {list(ibov_raw.columns)}")
    print(f"   Min date: {ibov_raw['date'].min()}")
    print(f"   Max date: {ibov_raw['date'].max()}")
    print(f"   Dias distintos: {ibov_raw['date'].nunique()}")
    audit_results.append({
        'Arquivo': 'ibovespa.csv (RAW)',
        'Min Date': str(ibov_raw['date'].min()),
        'Max Date': str(ibov_raw['date'].max()),
        '# Linhas': len(ibov_raw),
        '# Dias': ibov_raw['date'].nunique()
    })
except Exception as e:
    print(f"❌ ERRO: {e}")

print("\n" + "="*100)
print("2. NEWS_MULTISOURCE.PARQUET (BRUTO)")
print("="*100)
try:
    news_raw = pd.read_parquet(RAW / 'news_multisource.parquet')
    if 'date' not in news_raw.columns and 'published_at' in news_raw.columns:
        news_raw['date'] = pd.to_datetime(news_raw['published_at'], errors='coerce')
    else:
        news_raw['date'] = pd.to_datetime(news_raw['date'], errors='coerce')
    
    print(f"✅ Shape: {news_raw.shape}")
    print(f"   Colunas: {list(news_raw.columns)}")
    print(f"   Min date: {news_raw['date'].min()}")
    print(f"   Max date: {news_raw['date'].max()}")
    print(f"   Dias distintos: {news_raw['date'].dt.date.nunique()}")
    if 'source' in news_raw.columns:
        print(f"   Fontes: {news_raw['source'].value_counts().to_dict()}")
    
    audit_results.append({
        'Arquivo': 'news_multisource.parquet (RAW)',
        'Min Date': str(news_raw['date'].min()),
        'Max Date': str(news_raw['date'].max()),
        '# Linhas': len(news_raw),
        '# Dias': news_raw['date'].dt.date.nunique()
    })
except Exception as e:
    print(f"❌ ERRO: {e}")

print("\n" + "="*100)
print("3. IBOV_CLEAN.CSV (PROCESSADO)")
print("="*100)
try:
    ibov_clean = pd.read_csv(PROCESSED / 'ibov_clean.csv')
    # Identificar coluna de data
    date_col = None
    for col in ['day', 'date', 'Date', 'DATA']:
        if col in ibov_clean.columns:
            date_col = col
            break
    
    if date_col:
        ibov_clean[date_col] = pd.to_datetime(ibov_clean[date_col])
        print(f"✅ Shape: {ibov_clean.shape}")
        print(f"   Colunas: {list(ibov_clean.columns)}")
        print(f"   Min date: {ibov_clean[date_col].min()}")
        print(f"   Max date: {ibov_clean[date_col].max()}")
        print(f"   Dias distintos: {ibov_clean[date_col].nunique()}")
        
        audit_results.append({
            'Arquivo': 'ibov_clean.csv (PROCESSED)',
            'Min Date': str(ibov_clean[date_col].min()),
            'Max Date': str(ibov_clean[date_col].max()),
            '# Linhas': len(ibov_clean),
            '# Dias': ibov_clean[date_col].nunique()
        })
        
        # DIAGNÓSTICO CRÍTICO
        if len(ibov_clean) < 50:
            print(f"\n   🚨 PROBLEMA CRÍTICO: Apenas {len(ibov_clean)} linhas!")
            print(f"   📋 Primeiras linhas:")
            print(ibov_clean.head(10))
    else:
        print(f"❌ Coluna de data não encontrada em: {list(ibov_clean.columns)}")
except Exception as e:
    print(f"❌ ERRO: {e}")

print("\n" + "="*100)
print("4. NEWS_CLEAN.PARQUET (PROCESSADO)")
print("="*100)
try:
    news_clean = pd.read_parquet(PROCESSED / 'news_clean.parquet')
    date_col = None
    for col in ['day', 'date', 'Date']:
        if col in news_clean.columns:
            date_col = col
            break
    
    if date_col:
        news_clean[date_col] = pd.to_datetime(news_clean[date_col])
        print(f"✅ Shape: {news_clean.shape}")
        print(f"   Colunas: {list(news_clean.columns)[:10]}...")  # Primeiras 10 colunas
        print(f"   Min date: {news_clean[date_col].min()}")
        print(f"   Max date: {news_clean[date_col].max()}")
        print(f"   Dias distintos: {news_clean[date_col].dt.date.nunique()}")
        
        audit_results.append({
            'Arquivo': 'news_clean.parquet (PROCESSED)',
            'Min Date': str(news_clean[date_col].min()),
            'Max Date': str(news_clean[date_col].max()),
            '# Linhas': len(news_clean),
            '# Dias': news_clean[date_col].dt.date.nunique()
        })
except Exception as e:
    print(f"❌ ERRO: {e}")

print("\n" + "="*100)
print("5. TFIDF_DAILY_INDEX.CSV (PROCESSADO - Features diárias)")
print("="*100)
try:
    tfidf_index = pd.read_csv(PROCESSED / 'tfidf_daily_index.csv', parse_dates=['day'])
    print(f"✅ Shape: {tfidf_index.shape}")
    print(f"   Colunas: {list(tfidf_index.columns)}")
    print(f"   Min date: {tfidf_index['day'].min()}")
    print(f"   Max date: {tfidf_index['day'].max()}")
    print(f"   Dias distintos: {tfidf_index['day'].nunique()}")
    
    audit_results.append({
        'Arquivo': 'tfidf_daily_index.csv (PROCESSED)',
        'Min Date': str(tfidf_index['day'].min()),
        'Max Date': str(tfidf_index['day'].max()),
        '# Linhas': len(tfidf_index),
        '# Dias': tfidf_index['day'].nunique()
    })
    
    if tfidf_index['day'].nunique() < 50:
        print(f"\n   🚨 PROBLEMA: Apenas {tfidf_index['day'].nunique()} dias com features!")
except Exception as e:
    print(f"❌ ERRO: {e}")

print("\n" + "="*100)
print("6. LABELS_Y_DAILY.CSV (PROCESSADO - Targets)")
print("="*100)
try:
    y_labels = pd.read_csv(PROCESSED / 'labels_y_daily.csv', parse_dates=['day'])
    print(f"✅ Shape: {y_labels.shape}")
    print(f"   Colunas: {list(y_labels.columns)}")
    print(f"   Min date: {y_labels['day'].min()}")
    print(f"   Max date: {y_labels['day'].max()}")
    print(f"   Dias distintos: {y_labels['day'].nunique()}")
    
    if 'target_1d' in y_labels.columns:
        print(f"   Distribuição target: {y_labels['target_1d'].value_counts().to_dict()}")
    
    audit_results.append({
        'Arquivo': 'labels_y_daily.csv (PROCESSED)',
        'Min Date': str(y_labels['day'].min()),
        'Max Date': str(y_labels['day'].max()),
        '# Linhas': len(y_labels),
        '# Dias': y_labels['day'].nunique()
    })
except Exception as e:
    print(f"❌ ERRO: {e}")

print("\n" + "="*100)
print("7. 16_OOF_PREDICTIONS.CSV (PROCESSADO - Predições)")
print("="*100)
try:
    oof = pd.read_csv(PROCESSED / '16_oof_predictions.csv', parse_dates=['day'])
    print(f"✅ Shape: {oof.shape}")
    print(f"   Colunas: {list(oof.columns)}")
    print(f"   Min date: {oof['day'].min()}")
    print(f"   Max date: {oof['day'].max()}")
    print(f"   Dias distintos: {oof['day'].nunique()}")
    
    if 'model' in oof.columns:
        print(f"   Modelos: {oof['model'].unique()}")
    
    audit_results.append({
        'Arquivo': '16_oof_predictions.csv (PROCESSED)',
        'Min Date': str(oof['day'].min()),
        'Max Date': str(oof['day'].max()),
        '# Linhas': len(oof),
        '# Dias': oof['day'].nunique()
    })
    
    if oof['day'].nunique() < 50:
        print(f"\n   🚨 PROBLEMA CRÍTICO: Apenas {oof['day'].nunique()} dias de predições!")
except Exception as e:
    print(f"❌ ERRO: {e}")

# TABELA RESUMO
print("\n" + "="*100)
print("TABELA RESUMO DA AUDITORIA")
print("="*100)
audit_df = pd.DataFrame(audit_results)
print(audit_df.to_string(index=False))

# Salvar
audit_df.to_csv(PROCESSED / 'AUDIT_pipeline_summary.csv', index=False)
print(f"\n✅ Tabela salva em: {PROCESSED / 'AUDIT_pipeline_summary.csv'}")

# DIAGNÓSTICO
print("\n" + "="*100)
print("DIAGNÓSTICO DO ESTRANGULAMENTO")
print("="*100)

try:
    ibov_raw_days = ibov_raw['date'].nunique()
    ibov_clean_days = ibov_clean[date_col].nunique() if date_col else 0
    tfidf_days = tfidf_index['day'].nunique()
    oof_days = oof['day'].nunique()
    
    print(f"\n📊 FLUXO DA PIPELINE:")
    print(f"   1. Ibovespa BRUTO: {ibov_raw_days} dias")
    print(f"   2. Ibovespa LIMPO: {ibov_clean_days} dias (PERDA: {ibov_raw_days - ibov_clean_days} dias)")
    print(f"   3. TF-IDF Features: {tfidf_days} dias")
    print(f"   4. Predições OOF: {oof_days} dias (PERDA: {tfidf_days - oof_days} dias)")
    
    print(f"\n🚨 PROBLEMAS IDENTIFICADOS:")
    if ibov_clean_days < 50:
        print(f"   ❌ CRÍTICO: ibov_clean.csv tem apenas {ibov_clean_days} dias (esperado: ~{ibov_raw_days})")
        print(f"      → O arquivo foi sobrescrito com dados de teste/exemplo!")
    
    if oof_days < 50:
        print(f"   ❌ CRÍTICO: Predições OOF têm apenas {oof_days} dias (esperado: ~{tfidf_days})")
        print(f"      → Os modelos foram treinados apenas em uma amostra mínima!")
    
    if tfidf_days < 100:
        print(f"   ⚠️ AVISO: TF-IDF tem apenas {tfidf_days} dias de features")
        print(f"      → Verificar notebooks 14 (preprocess) e 15 (features)")
        
except Exception as e:
    print(f"❌ Erro no diagnóstico: {e}")

print("\n" + "="*100)
print("FIM DA AUDITORIA")
print("="*100)
