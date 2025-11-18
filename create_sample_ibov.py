"""
Script temporário para baixar dados do Ibovespa (2018-2025) para o dashboard
"""
import pandas as pd
import yfinance as yf
from pathlib import Path
from src.io import paths

# Baixar dados do Ibovespa
print("📥 Baixando dados do Ibovespa (2018-2025)...")
ibov = yf.download("^BVSP", start="2018-01-01", end="2025-12-31", progress=False)

# Preparar DataFrame
ibov_df = ibov.reset_index()
# Flatten multi-level columns if present
if isinstance(ibov_df.columns, pd.MultiIndex):
    ibov_df.columns = [col[0] if isinstance(col, tuple) else col for col in ibov_df.columns]
ibov_df.columns = [str(col).lower().replace(" ", "_") for col in ibov_df.columns]
ibov_df = ibov_df.rename(columns={"date": "day"})

# Criar diretório se não existir
data_paths = paths.get_data_paths()
output_dir = data_paths["data_processed"]
output_dir.mkdir(parents=True, exist_ok=True)

# Salvar arquivo
output_file = output_dir / "ibov_clean.csv"
ibov_df.to_csv(output_file, index=False)

print(f"✅ Dados salvos em: {output_file}")
print(f"📊 Período: {ibov_df['day'].min()} a {ibov_df['day'].max()}")
print(f"📈 Total de registros: {len(ibov_df)}")
