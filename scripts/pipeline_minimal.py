#!/usr/bin/env python
"""
Pipeline mínimo para garantir artefatos essenciais dentro do período oficial
2018-01-02 a 2024-12-31. Operações:
- Carrega os principais arquivos em data_processed (base C:/TCC_USP)
- Faz clamp de datas no intervalo oficial e remove duplicidades básicas
- Persiste os arquivos já normalizados

Saídas no console indicam contagens antes/depois e se houve truncamento.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import sys

import pandas as pd

# Garantir que src esteja no PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.constants import START_DATE, END_DATE
from src.io import paths

OFFICIAL_START = pd.Timestamp(START_DATE)
OFFICIAL_END = pd.Timestamp(END_DATE)
DATE_CANDIDATES = ["day", "date", "Date", "data", "Data", "event_day"]

DATA_PATHS = paths.get_data_paths(create=True)
DATA_PROCESSED: Path = DATA_PATHS["data_processed"]


def _find_date_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _clamp_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    return df.loc[(df[date_col] >= OFFICIAL_START) & (df[date_col] <= OFFICIAL_END)]


def _process_csv(
    filename: str,
    dedup_subset: Optional[List[str]] = None,
    required: bool = True,
    date_required: bool = True,
) -> Dict[str, object]:
    path = DATA_PROCESSED / filename
    result: Dict[str, object] = {
        "file": filename,
        "exists": path.exists(),
        "rows_before": 0,
        "rows_after": 0,
        "status": "ok",
        "date_col": None,
    }

    if not path.exists():
        result["status"] = "missing" if required else "skipped"
        print(f"[WARN] {filename}: arquivo não encontrado ({path})")
        return result

    df = pd.read_csv(path)
    result["rows_before"] = len(df)

    date_col = _find_date_column(df, DATE_CANDIDATES) if date_required else None
    if date_required and date_col is None:
        result["status"] = "no_date_col"
        print(f"[ERROR] {filename}: nenhuma coluna de data localizada.")
        return result

    if date_col:
        df = _clamp_dates(df, date_col)
        df = df.sort_values(date_col)
        result["date_col"] = date_col

    if dedup_subset:
        dedup_cols = [c for c in dedup_subset if c in df.columns]
        if dedup_cols:
            df = df.drop_duplicates(subset=dedup_cols)

    result["rows_after"] = len(df)
    df.to_csv(path, index=False)

    msg = f"[OK] {filename}: {result['rows_before']} -> {result['rows_after']}"
    if date_col:
        msg += f" (coluna de data: {date_col})"
    print(msg)
    return result


def main() -> int:
    print("=== Pipeline mínimo (clamp 2018-01-02 a 2024-12-31) ===")
    summary = []

    summary.append(_process_csv("ibovespa_clean.csv", dedup_subset=["date", "day"]))
    summary.append(
        _process_csv(
            "16_oof_predictions.csv",
            dedup_subset=["model", "fold", "day", "row_id"],
        )
    )
    summary.append(
        _process_csv(
            "18_backtest_daily_curves.csv",
            dedup_subset=["model", "strategy", "day", "fold"],
        )
    )
    summary.append(
        _process_csv(
            "18_backtest_results.csv",
            required=False,
            date_required=False,
            dedup_subset=["model", "strategy"],
        )
    )
    summary.append(
        _process_csv(
            "event_study_latency.csv",
            dedup_subset=["event_day", "titulo"],
            required=False,
        )
    )

    # JSON de métricas (sem datas) apenas confirma existência.
    metrics_path = DATA_PROCESSED / "results_16_models_tfidf.json"
    if metrics_path.exists():
        print(f"[OK] results_16_models_tfidf.json: encontrado em {metrics_path}")
    else:
        print(f"[WARN] results_16_models_tfidf.json: não encontrado em {metrics_path}")

    # Retorna 0 se nenhum item obrigatório faltou ou falhou.
    failures = [s for s in summary if s["status"] not in {"ok", "skipped"}]
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
