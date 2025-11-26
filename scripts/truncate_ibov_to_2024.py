"""
Trunca o arquivo ibovespa_clean.csv para o período oficial do TCC.

Uso:
    python scripts/truncate_ibov_to_2024.py

O script:
  - Lê C:/TCC_USP/data_processed/ibovespa_clean.csv
  - Converte a coluna de data para datetime (coluna "date")
  - Mantém apenas registros com date <= 2024-12-31
  - Sobrescreve o arquivo no mesmo caminho
"""

from pathlib import Path

import pandas as pd


CSV_PATH = Path("C:/TCC_USP/data_processed/ibovespa_clean.csv")
END_DATE = pd.Timestamp("2024-12-31")


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    if "date" not in df.columns:
        raise KeyError("Coluna 'date' não encontrada em ibovespa_clean.csv")

    df["date"] = pd.to_datetime(df["date"])
    original_rows = len(df)

    df = df[df["date"] <= END_DATE]
    truncated_rows = len(df)

    df.to_csv(CSV_PATH, index=False)
    print(
        f"Truncagem concluída: {original_rows} -> {truncated_rows} linhas "
        f"(até {END_DATE.date()})"
    )


if __name__ == "__main__":
    main()
