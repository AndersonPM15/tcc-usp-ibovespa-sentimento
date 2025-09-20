from google.colab import drive
drive.mount('/content/drive')

import os
BASE = "/content/drive/MyDrive/TCC_USP"
RAW = os.path.join(BASE, "data_raw")
os.makedirs(RAW, exist_ok=True)
print("Salvar em:", RAW)

import yfinance as yf
import pandas as pd

ibov = yf.download("^BVSP", start="2015-01-01", end=None, progress=False)
ibov.to_csv(os.path.join(RAW, "ibovespa.csv"))
print("OK: ibovespa.csv salvo.")

noticias_df = pd.DataFrame({
    "published_at": ["2025-01-01T09:00:00", "2025-01-02T10:30:00"],
    "title": ["Bolsa sobe com otimismo no mercado", "Ibovespa cai após decisão do FED"],
    "source": ["Exemplo", "Exemplo"]
})
noticias_df.to_csv(os.path.join(RAW, "noticias_exemplo.csv"), index=False)
print("OK: noticias_exemplo.csv salvo.")

import subprocess, shlex
subprocess.run(shlex.split(f'ls -lh "{RAW}"'))
