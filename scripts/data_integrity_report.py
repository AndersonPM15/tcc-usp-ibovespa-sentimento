#!/usr/bin/env python
"""
Validação de integridade dos dados (hard cap 2018-01-02 a 2024-12-31).

Saídas:
- Console com min/max/rows por dataset e checagem de futuros/anteriores.
- reports/data_integrity_report.md com o mesmo resumo.

Exit code != 0 quando existir:
    * arquivo obrigatório ausente
    * datas fora do intervalo oficial
    * duplicidade em arquivos que deveriam ser únicos
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
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

DATA_PROCESSED: Path = paths.DATA_PROCESSED
REPORTS_DIR: Path = paths.get_project_paths()["reports"]
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _find_date_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _validate_dataset(
    file_path: Path,
    required: bool = True,
    unique_on: Optional[List[str]] = None,
    require_date: bool = True,
) -> Dict[str, object]:
    name = file_path.name
    result: Dict[str, object] = {
        "file": name,
        "exists": file_path.exists(),
        "required": required,
        "rows": 0,
        "min": None,
        "max": None,
        "future_rows": 0,
        "before_rows": 0,
        "duplicates": 0,
        "date_col": None,
        "status": "ok",
    }

    if not file_path.exists():
        result["status"] = "missing" if required else "optional-missing"
        return result

    df = pd.read_csv(file_path)
    result["rows"] = len(df)
    date_col = _find_date_column(df, DATE_CANDIDATES) if require_date else None
    result["date_col"] = date_col

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        valid = df[date_col].dropna()
        if not valid.empty:
            result["min"] = valid.min()
            result["max"] = valid.max()
            result["future_rows"] = (df[date_col] > OFFICIAL_END).sum()
            result["before_rows"] = (df[date_col] < OFFICIAL_START).sum()
        if unique_on:
            cols = [c for c in unique_on if c in df.columns]
            if cols:
                result["duplicates"] = df.duplicated(subset=cols).sum()
    elif require_date:
        result["status"] = "no-date-col"

    if (
        (result["future_rows"] > 0)
        or (result["before_rows"] > 0)
        or (result["status"] == "no-date-col" and required)
        or (result["duplicates"] > 0)
    ):
        result["status"] = "fail"
    return result


def _build_intersection(
    ibov_path: Path, sentiment_path: Path
) -> Dict[str, object]:
    if not ibov_path.exists() or not sentiment_path.exists():
        return {
            "ibov_days": 0,
            "sentiment_days": 0,
            "intersect_days": 0,
        }
    ibov = pd.read_csv(ibov_path, parse_dates=["date"])
    sentiment = pd.read_csv(sentiment_path, parse_dates=["day"])
    ibov_days = set(ibov["date"].dt.normalize())
    sent_days = set(sentiment["day"].dt.normalize())
    intersect = ibov_days.intersection(sent_days)
    return {
        "ibov_days": len(ibov_days),
        "sentiment_days": len(sent_days),
        "intersect_days": len(intersect),
    }


def _write_report(
    results: List[Dict[str, object]],
    intersection: Dict[str, object],
    report_path: Path,
) -> None:
    lines: List[str] = [
        "# Data Integrity Report",
        "",
        f"Período oficial: {OFFICIAL_START.date()} a {OFFICIAL_END.date()} (hard cap)",
        "",
        "| Arquivo | Existe | Linhas | Min | Max | >2024-12-31 | <2018-01-02 | Duplicatas | Coluna data | Status |",
        "|---------|--------|--------|-----|-----|-------------|-------------|------------|-------------|--------|",
    ]
    for r in results:
        lines.append(
            "| {file} | {exists} | {rows} | {min} | {max} | {future} | {before} | {dups} | {date_col} | {status} |".format(
                file=r["file"],
                exists="sim" if r["exists"] else "não",
                rows=r["rows"],
                min=r["min"],
                max=r["max"],
                future=r["future_rows"],
                before=r["before_rows"],
                dups=r["duplicates"],
                date_col=r["date_col"] or "-",
                status=r["status"],
            )
        )

    lines.extend(
        [
            "",
            "## Cobertura e interseção IBOV x Sentimento",
            f"- Dias IBOV: {intersection['ibov_days']}",
            f"- Dias Sentimento: {intersection['sentiment_days']}",
            f"- Interseção: {intersection['intersect_days']}",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Relatório salvo em {report_path}")


def main() -> int:
    datasets = [
        {
            "path": DATA_PROCESSED / "ibovespa_clean.csv",
            "required": True,
            "unique_on": ["date", "day"],
        },
        {
            "path": DATA_PROCESSED / "16_oof_predictions.csv",
            "required": True,
            "unique_on": ["model", "fold", "day", "row_id"],
        },
        {
            "path": DATA_PROCESSED / "18_backtest_daily_curves.csv",
            "required": True,
            "unique_on": ["model", "strategy", "day", "fold"],
        },
        {
            "path": DATA_PROCESSED / "18_backtest_results.csv",
            "required": False,
            "unique_on": ["model", "strategy"],
            "require_date": False,
        },
        {
            "path": DATA_PROCESSED / "event_study_latency.csv",
            "required": False,
            "unique_on": ["event_day", "titulo"],
        },
    ]

    results = [
        _validate_dataset(
            d["path"],
            d.get("required", True),
            d.get("unique_on"),
            d.get("require_date", True),
        )
        for d in datasets
    ]

    intersection = _build_intersection(
        DATA_PROCESSED / "ibovespa_clean.csv",
        DATA_PROCESSED / "16_oof_predictions.csv",
    )

    report_path = REPORTS_DIR / "data_integrity_report.md"
    _write_report(results, intersection, report_path)

    issues = [
        r
        for r in results
        if r["status"] in {"missing", "fail", "no-date-col"}
        or (r["required"] and not r["exists"])
    ]

    if issues:
        print("[ERROR] Problemas encontrados na validação:")
        for r in issues:
            print(f" - {r['file']}: status={r['status']}")
        return 1

    print("[OK] Validação concluída sem dados fora do período oficial.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
