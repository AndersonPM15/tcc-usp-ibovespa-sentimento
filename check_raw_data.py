import pandas as pd
from pathlib import Path

RAW = Path("C:/Users/ander/OneDrive/TCC_USP/data_raw")

def check_file(filename, skip_rows=0):
    print("="*80)
    print(filename.upper())
    try:
        df = pd.read_csv(RAW / filename, skiprows=skip_rows)
        print(f"✅ Linhas: {len(df)}")
        print(f"   Colunas: {list(df.columns)[:8]}")
        
        # Tentar encontrar coluna de data
        date_cols = ['day', 'Date', 'date', 'data', 'DATA', 'publishedAt', 'published_at']
        date_col = next((c for c in date_cols if c in df.columns), None)
        
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df_clean = df.dropna(subset=[date_col])
            print(f"   Min date: {df_clean[date_col].min()}")
            print(f"   Max date: {df_clean[date_col].max()}")
            print(f"   Dias distintos: {df_clean[date_col].nunique()}")
        else:
            print(f"   ⚠️ Coluna de data não encontrada")
    except Exception as e:
        print(f"❌ Erro: {e}")

# Verificar todos os arquivos
check_file("bova11.csv")
check_file("ibovespa.csv", skip_rows=1)
check_file("noticias_real.csv")
check_file("noticias_exemplo.csv")

print("\n" + "="*80)
print("ARQUIVOS PARQUET")
print("="*80)
try:
    df_news_ms = pd.read_parquet(RAW / "news_multisource.parquet")
    print(f"news_multisource.parquet: {df_news_ms.shape}")
    if 'date' in df_news_ms.columns:
        df_news_ms['date'] = pd.to_datetime(df_news_ms['date'], errors='coerce')
        print(f"   Min: {df_news_ms['date'].min()}")
        print(f"   Max: {df_news_ms['date'].max()}")
        print(f"   Dias: {df_news_ms['date'].dt.date.nunique()}")
except Exception as e:
    print(f"❌ news_multisource.parquet: {e}")
