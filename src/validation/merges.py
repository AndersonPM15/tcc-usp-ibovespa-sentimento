"""
Validation helpers to make sure sentiment/IBOV merges always have overlapping dates.
These checks should run near the start of each notebook before any heavy processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd

__all__ = ["check_ibov_news_intersection", "assert_min_intersection"]


@dataclass(frozen=True)
class IntersectionStats:
    start_ibov: pd.Timestamp
    end_ibov: pd.Timestamp
    start_news: pd.Timestamp
    end_news: pd.Timestamp
    days_common: int


def _prepare_dates(df: pd.DataFrame, date_col: str) -> pd.Series:
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in dataframe with columns={list(df.columns)}")
    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.isna().all():
        raise ValueError(f"Column '{date_col}' could not be parsed as datetime.")
    return dates


def _intersection_stats(
    df_ibov: pd.DataFrame,
    df_news: pd.DataFrame,
    date_col_ibov: str,
    date_col_news: str,
) -> IntersectionStats:
    ibov_dates = _prepare_dates(df_ibov, date_col_ibov)
    news_dates = _prepare_dates(df_news, date_col_news)

    ibov_range = (ibov_dates.min(), ibov_dates.max())
    news_range = (news_dates.min(), news_dates.max())

    common = pd.Index(ibov_dates.dt.normalize().unique()).intersection(
        news_dates.dt.normalize().unique()
    )

    return IntersectionStats(
        start_ibov=ibov_range[0],
        end_ibov=ibov_range[1],
        start_news=news_range[0],
        end_news=news_range[1],
        days_common=int(common.size),
    )


def check_ibov_news_intersection(
    df_ibov: pd.DataFrame,
    df_news: pd.DataFrame,
    date_col_ibov: str = "day",
    date_col_news: str = "day",
) -> IntersectionStats:
    """
    Print range summaries for both series and raise if there is zero overlap.
    Returns the stats object for downstream reuse.
    """
    stats = _intersection_stats(df_ibov, df_news, date_col_ibov, date_col_news)

    print(
        f"IBOV range: {stats.start_ibov.date()} → {stats.end_ibov.date()}  "
        f"({len(df_ibov)} registros)"
    )
    print(
        f"News range: {stats.start_news.date()} → {stats.end_news.date()}  "
        f"({len(df_news)} registros)"
    )
    print(f"Dias em comum: {stats.days_common}")

    if stats.days_common == 0:
        raise ValueError(
            "Sem interseção entre datas de Ibovespa e notícias. "
            "Revise o período de coleta ou o pré-processamento antes de seguir com o merge."
        )
    return stats


def assert_min_intersection(
    df_ibov: pd.DataFrame,
    df_news: pd.DataFrame,
    min_days: int = 60,
    date_col_ibov: str = "day",
    date_col_news: str = "day",
) -> IntersectionStats:
    """
    Ensure there is a minimum number of overlapping days before modeling.
    Raises ValueError when the overlap is below `min_days`.
    """
    stats = check_ibov_news_intersection(
        df_ibov=df_ibov,
        df_news=df_news,
        date_col_ibov=date_col_ibov,
        date_col_news=date_col_news,
    )
    if stats.days_common < min_days:
        raise ValueError(
            f"Interseção insuficiente ({stats.days_common} dias). "
            f"São necessários pelo menos {min_days} dias para prosseguir."
        )
    return stats
