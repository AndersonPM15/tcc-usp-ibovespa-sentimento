from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.constants import START_DATE, END_DATE
from src.io.paths import DATA_PROCESSED

END_TS = pd.Timestamp(END_DATE)
START_TS = pd.Timestamp(START_DATE)
REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "data_quality_report.md"

DATE_CANDIDATES = ["day", "date", "Data", "data", "event_day"]


def normalize_day(df: pd.DataFrame, candidates: List[str] = DATE_CANDIDATES) -> Tuple[pd.DataFrame, str | None]:
    if df.empty:
        return df, None
    chosen = None
    for col in candidates:
        if col in df.columns:
            chosen = col
            break
    if chosen is None:
        return pd.DataFrame(), None
    df = df.copy()
    df["day"] = pd.to_datetime(df[chosen], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["day"])
    df = df[(df["day"] >= START_TS) & (df["day"] <= END_TS)]
    df = df.sort_values("day").drop_duplicates(subset=["day"])
    return df, chosen


def load_artifact(path: Path, parse_dates: List[str] | None = None) -> tuple[pd.DataFrame, bool]:
    if not path.exists():
        return pd.DataFrame(), False
    if path.suffix.lower() == ".json":
        try:
            return pd.read_json(path), True
        except Exception:
            return pd.DataFrame(), True
    try:
        return pd.read_csv(path, parse_dates=parse_dates), True
    except Exception:
        # fallback without parse_dates
        try:
            return pd.read_csv(path), True
        except Exception:
            return pd.DataFrame(), True


def load_latency(data_dir: Path) -> tuple[pd.DataFrame, bool]:
    parquet_path = data_dir / "latency_events.parquet"
    csv_path = data_dir / "event_study_latency.csv"
    fallback_csv = data_dir / "latency_events.csv"
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path), True
        except Exception:
            return pd.DataFrame(), True
    elif csv_path.exists():
        return load_artifact(csv_path)
    else:
        return load_artifact(fallback_csv)


def stats_for(name: str, df: pd.DataFrame, exists: bool) -> Dict:
    info: Dict[str, object] = {"name": name, "exists": exists and not df.empty or exists}
    if df.empty or "day" not in df.columns:
        return info
    info["rows"] = len(df)
    info["min_date"] = df["day"].min()
    info["max_date"] = df["day"].max()
    info["rows_over_end"] = int((df["day"] > END_TS).sum())
    info["dup_days"] = int(len(df) - df["day"].nunique())
    return info


def render_report(entries: List[Dict], intersection: int) -> str:
    lines = ["# Data Quality Report", f"Período esperado: {START_TS.date()} a {END_TS.date()}", ""]
    lines.append("| Dataset | Exists | Rows | Min day | Max day | >END_DATE | Duplicated days |")
    lines.append("|--------|--------|------|---------|---------|----------|-----------------|")
    for info in entries:
        lines.append(
            "| {name} | {exists} | {rows} | {min} | {max} | {over} | {dup} |".format(
                name=info.get("name"),
                exists=info.get("exists", False),
                rows=info.get("rows", 0),
                min=info.get("min_date", "-") if info.get("exists") else "-",
                max=info.get("max_date", "-") if info.get("exists") else "-",
                over=info.get("rows_over_end", 0),
                dup=info.get("dup_days", 0),
            )
        )
    lines.append("")
    lines.append(f"Interseção IBOV x SENTIMENTO (dias): {intersection}")
    return "\n".join(lines)


def main() -> int:
    data_dir = Path(DATA_PROCESSED)

    ibov_raw, ibov_exists = load_artifact(data_dir / "ibovespa_clean.csv")
    ibov_df, _ = normalize_day(ibov_raw)

    sentiment_raw, sent_exists = load_artifact(data_dir / "16_oof_predictions.csv", parse_dates=["day"])
    sentiment_df, _ = normalize_day(sentiment_raw)

    results_df, results_exists = load_artifact(data_dir / "results_16_models_tfidf.json")
    latency_raw, latency_exists = load_latency(data_dir)
    latency_df, _ = normalize_day(latency_raw)

    backtest_results, backtest_results_exists = load_artifact(data_dir / "18_backtest_results.csv")
    backtest_curves, backtest_curves_exists = load_artifact(data_dir / "18_backtest_daily_curves.csv", parse_dates=["day"])
    backtest_curves, _ = normalize_day(backtest_curves)

    entries = [
        stats_for("IBOV", ibov_df, ibov_exists),
        stats_for("SENTIMENT", sentiment_df, sent_exists),
        stats_for("RESULTS_16", results_df, results_exists),
        stats_for("LATENCY", latency_df, latency_exists),
        stats_for("BACKTEST_RESULTS", backtest_results, backtest_results_exists),
        stats_for("BACKTEST_CURVES", backtest_curves, backtest_curves_exists),
    ]

    intersection = 0
    if not ibov_df.empty and not sentiment_df.empty:
        intersection = len(set(ibov_df["day"].dt.date) & set(sentiment_df["day"].dt.date))

    report_text = render_report(entries, intersection)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    has_over_end = any(info.get("rows_over_end", 0) > 0 for info in entries)
    has_intersection_issue = intersection == 0

    if has_over_end or has_intersection_issue:
        print(report_text)
        return 1

    print(report_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
