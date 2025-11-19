import pandas as pd

df = pd.read_parquet(r"C:\Users\ander\OneDrive\TCC_USP\data_raw\news_multisource.parquet")
print(f"Shape: {df.shape}")
print(f"\nColunas: {df.columns.tolist()}")
print(f"\nPeríodo: {df['date'].min()} → {df['date'].max()}")
print(f"Dias únicos: {df['date'].nunique()}")
print(f"\nPrimeiras linhas:\n{df.head(3)}")
