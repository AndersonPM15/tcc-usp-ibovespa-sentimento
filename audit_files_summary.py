"""
Auditoria dos arquivos principais da pipeline após coleta
"""
import pandas as pd
from pathlib import Path

print("="*80)
print("AUDITORIA DE ARQUIVOS - PIPELINE TCC")
print("="*80)

base_path = Path(r"C:\Users\ander\OneDrive\TCC_USP")

# 1. news_multisource.parquet (RAW)
print("\n📰 1. NEWS_MULTISOURCE.PARQUET (dados brutos de notícias)")
print("-"*80)
try:
    news_file = base_path / "data_raw" / "news_multisource.parquet"
    df_news = pd.read_parquet(news_file)
    
    print(f"✅ Arquivo encontrado: {news_file}")
    print(f"   Total de notícias: {len(df_news):,}")
    print(f"   Data mínima: {df_news['date'].min()}")
    print(f"   Data máxima: {df_news['date'].max()}")
    print(f"   Dias distintos: {df_news['date'].nunique():,}")
    print(f"   Fontes únicas: {df_news['source'].nunique()}")
    print(f"   Colunas: {df_news.columns.tolist()}")
except FileNotFoundError:
    print(f"❌ Arquivo não encontrado: {news_file}")
except Exception as e:
    print(f"❌ Erro ao ler: {e}")

# 2. tfidf_daily_matrix.parquet (features de sentimento)
print("\n📊 2. TFIDF_DAILY_MATRIX.PARQUET (features de sentimento diário)")
print("-"*80)

# Tentar múltiplos nomes possíveis
possible_names = [
    "tfidf_daily_matrix.parquet",
    "tfidf_daily_index.csv",
    "15_tfidf_daily.parquet",
    "tfidf_features_daily.parquet",
]

tfidf_found = False
for fname in possible_names:
    for folder in ["data_processed", "data_interim"]:
        try:
            tfidf_file = base_path / folder / fname
            if tfidf_file.exists():
                df_tfidf = pd.read_parquet(tfidf_file) if fname.endswith(".parquet") else pd.read_csv(tfidf_file)
                
                print(f"✅ Arquivo encontrado: {tfidf_file}")
                print(f"   Shape: {df_tfidf.shape}")
                
                # Identificar coluna de data
                date_cols = [c for c in df_tfidf.columns if 'date' in c.lower() or 'day' in c.lower()]
                if date_cols:
                    date_col = date_cols[0]
                    df_tfidf[date_col] = pd.to_datetime(df_tfidf[date_col], errors='coerce')
                    print(f"   Data mínima: {df_tfidf[date_col].min()}")
                    print(f"   Data máxima: {df_tfidf[date_col].max()}")
                    print(f"   Dias distintos: {df_tfidf[date_col].nunique():,}")
                else:
                    print(f"   ⚠️ Coluna de data não identificada")
                    print(f"   Linhas (dias): {len(df_tfidf):,}")
                
                print(f"   Colunas: {df_tfidf.columns.tolist()[:10]}...")
                tfidf_found = True
                break
        except Exception as e:
            continue
    if tfidf_found:
        break

if not tfidf_found:
    print(f"❌ Nenhum arquivo de features TF-IDF encontrado")
    print(f"   Arquivos procurados: {possible_names}")

# 3. ibov_clean.csv (dados de mercado)
print("\n📈 3. IBOV_CLEAN.CSV (dados de pregão Ibovespa)")
print("-"*80)
try:
    ibov_file = base_path / "data_processed" / "ibov_clean.csv"
    df_ibov = pd.read_csv(ibov_file)
    
    # Identificar coluna de data
    date_cols = [c for c in df_ibov.columns if 'date' in c.lower() or 'day' in c.lower()]
    if date_cols:
        date_col = date_cols[0]
    else:
        date_col = df_ibov.columns[0]  # Primeira coluna
    
    df_ibov[date_col] = pd.to_datetime(df_ibov[date_col], errors='coerce')
    
    print(f"✅ Arquivo encontrado: {ibov_file}")
    print(f"   Total de pregões: {len(df_ibov):,}")
    print(f"   Data mínima: {df_ibov[date_col].min()}")
    print(f"   Data máxima: {df_ibov[date_col].max()}")
    print(f"   Dias distintos: {df_ibov[date_col].nunique():,}")
    print(f"   Colunas: {df_ibov.columns.tolist()}")
except FileNotFoundError:
    print(f"❌ Arquivo não encontrado: {ibov_file}")
except Exception as e:
    print(f"❌ Erro ao ler: {e}")

# 4. Resumo de alinhamento
print("\n" + "="*80)
print("📊 RESUMO DE ALINHAMENTO")
print("="*80)

try:
    if 'df_news' in locals():
        news_days = df_news['date'].nunique()
    else:
        news_days = 0
    
    if tfidf_found and date_cols:
        tfidf_days = df_tfidf[date_col].nunique()
    else:
        tfidf_days = 0
    
    if 'df_ibov' in locals():
        ibov_days = df_ibov[date_col].nunique()
    else:
        ibov_days = 0
    
    print(f"Notícias (dias):        {news_days:>6,}")
    print(f"Sentimento TF-IDF:      {tfidf_days:>6,}")
    print(f"Pregão Ibovespa:        {ibov_days:>6,}")
    print(f"\n⚠️ Interseção esperada: ~{min(news_days, ibov_days) if news_days > 0 and ibov_days > 0 else 0} dias")
    print(f"   (dias com notícias E pregão)")
except Exception as e:
    print(f"❌ Erro no resumo: {e}")

print("\n" + "="*80)
