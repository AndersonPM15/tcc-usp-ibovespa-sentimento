"""
Date-range validation helpers for safely merging Ibovespa and news sentiment series.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

__all__ = ["summarize_date_range", "check_intersection"]


@dataclass(frozen=True)
class DateSummary:
    """Container for date statistics (mostly for internal use / testing)."""

    min_date: Optional[pd.Timestamp]
    max_date: Optional[pd.Timestamp]
    n_days: int
    n_records: int
    n_missing: int


def _ensure_datetime(series: pd.Series, col_name: str) -> pd.Series:
    """Convert a series to daily resolution datetimes."""
    if series.empty:
        raise ValueError(f"DataFrame sem registros para coluna '{col_name}'.")
    dates = pd.to_datetime(series, errors="coerce")
    if dates.isna().all():
        raise ValueError(f"Coluna '{col_name}' não pôde ser convertida para datetime.")
    return dates.dt.floor("D")


def summarize_date_range(df: pd.DataFrame, col: str = "day") -> DateSummary:
    """
    Print a summary (min/max/n dias/registros) for the provided date column.

    Returns the `DateSummary` dataclass for optional downstream checks.
    """
    if col not in df.columns:
        raise KeyError(f"Coluna '{col}' não encontrada em {list(df.columns)}")

    dates = _ensure_datetime(df[col], col)
    summary = DateSummary(
        min_date=dates.min(),
        max_date=dates.max(),
        n_days=int(dates.nunique()),
        n_records=len(df),
        n_missing=int(df[col].isna().sum()),
    )
    min_repr = summary.min_date.date() if pd.notna(summary.min_date) else "NaT"
    max_repr = summary.max_date.date() if pd.notna(summary.max_date) else "NaT"
    print(
        f"[{col}] {min_repr} → {max_repr} | {summary.n_days} dias únicos | "
        f"{summary.n_records} registros | missing={summary.n_missing}"
    )
    return summary


def check_intersection(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    col_left: str = "day",
    col_right: str = "day",
    min_days: int = 60,
) -> Dict[str, object]:
    """
    Validate that two date series have a meaningful intersection.

    Parameters
    ----------
    df_left, df_right : pd.DataFrame
        Dataframes to compare (e.g., Ibovespa vs. sentimento diário).
    col_left, col_right : str, default "day"
        Columns containing the date information in each dataframe.
    min_days : int, default 60
        Minimum number of overlapping days considered acceptable.

    Returns
    -------
    dict
        Keys: `left`, `right`, `common_days`, `days_list`.

    Raises
    ------
    ValueError
        When there is zero overlap between the series.
    """
    left_summary = summarize_date_range(df_left, col_left)
    right_summary = summarize_date_range(df_right, col_right)

    left_dates = _ensure_datetime(df_left[col_left], col_left).dropna().unique()
    right_dates = _ensure_datetime(df_right[col_right], col_right).dropna().unique()

    common = pd.Index(left_dates).intersection(pd.Index(right_dates))
    n_common = common.size

    print(f"Dias em comum: {n_common}")
    if n_common == 0:
        raise ValueError(
            "Sem interseção entre as séries informadas. "
            "Revise o período de coleta ou o pré-processamento antes de seguir com o merge."
        )
    if n_common < min_days:
        warnings.warn(
            f"Amostra curta: apenas {n_common} dias em comum (< {min_days}). "
            "Resultados estatísticos podem ficar instáveis.",
            stacklevel=2,
        )

    return {
        "left": left_summary,
        "right": right_summary,
        "common_days": n_common,
        "days_list": common.sort_values(),
    }
