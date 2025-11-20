"""
Download histórico completo do Ibovespa (^BVSP) para TCC
Período: 2018-01-02 → 2025-11-19 (alinhado com base de notícias)
"""
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

# Configuração
TICKER = "^BVSP"  # Ibovespa
START_DATE = "2018-01-02"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Paths
BASE_PATH = Path(__file__).parent
DATA_RAW = BASE_PATH / "data_raw"
DATA_PROC = BASE_PATH / "data_processed"

DATA_RAW.mkdir(exist_ok=True)
DATA_PROC.mkdir(exist_ok=True)

print("="*80)
print("DOWNLOAD HISTÓRICO IBOVESPA (^BVSP)")
print("="*80)
print(f"Período: {START_DATE} → {END_DATE}")
print(f"Ticker: {TICKER}")
print("\n🌐 Baixando dados do Yahoo Finance...")

try:
    # Download via yfinance
    df = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        progress=True,
        auto_adjust=False,  # Manter preços originais + Adj Close
    )
    
    if df.empty:
        raise ValueError("Nenhum dado retornado pelo Yahoo Finance")
    
    # Reset index (date vira coluna)
    df = df.reset_index()
    
    # Tratar MultiIndex (yfinance pode retornar MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Renomear colunas para padrão
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
    
    # Garantir coluna 'date' (não 'Date')
    if 'index' in df.columns:
        df = df.rename(columns={'index': 'date'})
    
    # Converter date para datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Adicionar coluna de retorno diário
    df['return'] = df['close'].pct_change()
    
    # Adicionar coluna binária (1 = subiu, 0 = caiu)
    df['direction'] = (df['return'] > 0).astype(int)
    
    print("\n" + "="*80)
    print("DADOS BAIXADOS COM SUCESSO")
    print("="*80)
    print(f"Total de registros: {len(df):,}")
    print(f"Período efetivo: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Dias de pregão: {len(df):,}")
    print("="*80)
    
    # Estatísticas básicas
    print("\n📊 Estatísticas do período:")
    print(f"   Fechamento mínimo: {df['close'].min():,.2f}")
    print(f"   Fechamento máximo: {df['close'].max():,.2f}")
    print(f"   Fechamento médio: {df['close'].mean():,.2f}")
    print(f"   Retorno total: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.2f}%")
    print(f"   Volatilidade (std retornos): {df['return'].std():.4f}")
    
    # Amostra
    print("\n📄 Amostra dos dados:")
    print(df[['date', 'open', 'high', 'low', 'close', 'volume', 'return', 'direction']].head(10))
    
    # Salvar arquivos
    raw_file = DATA_RAW / "ibovespa.csv"
    clean_file = DATA_PROC / "ibovespa_clean.csv"
    
    # Raw: todos os dados
    df.to_csv(raw_file, index=False, encoding="utf-8")
    print(f"\n✅ Salvo RAW: {raw_file}")
    
    # Clean: apenas colunas essenciais
    df_clean = df[['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'return', 'direction']].copy()
    df_clean.to_csv(clean_file, index=False, encoding="utf-8")
    print(f"✅ Salvo CLEAN: {clean_file}")
    
    print("\n" + "="*80)
    print("✅ DOWNLOAD CONCLUÍDO COM SUCESSO!")
    print(f"   Arquivo: ibovespa_clean.csv")
    print(f"   Registros: {len(df_clean):,} dias de pregão")
    print(f"   Período: {df_clean['date'].min().date()} → {df_clean['date'].max().date()}")
    print("="*80)
    
except Exception as e:
    print("\n❌ ERRO NO DOWNLOAD:")
    print(f"   {type(e).__name__}: {e}")
    print("\nVerifique:")
    print("  - Conexão com internet")
    print("  - Disponibilidade do Yahoo Finance")
    print("  - Ticker ^BVSP está correto")
    raise
