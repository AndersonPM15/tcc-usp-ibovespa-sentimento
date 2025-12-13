from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import loader as cfg
from src.config.constants import END_DATE, START_DATE
from src.io import paths as path_utils
OFFICIAL_START = pd.Timestamp(START_DATE)
OFFICIAL_END = pd.Timestamp(END_DATE)


def _filter_period(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return pd.DataFrame(columns=df.columns)
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=[col])
    df = df[(df[col] >= OFFICIAL_START) & (df[col] <= OFFICIAL_END)]
    return df.sort_values(col)


def _write_csv(df: pd.DataFrame, targets: Sequence[Path]) -> None:
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target, index=False)


def _write_json(payload: Dict, targets: Sequence[Path]) -> None:
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_parquet(df: pd.DataFrame, targets: Sequence[Path]) -> None:
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(target, index=False)


def _ensure_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df[columns]


def process_ibov(base_paths: Dict[str, Path], repo_dp: Path | None) -> Dict[str, object]:
    target = cfg.get_arquivo("ibov_clean", base_paths["base"])
    repo_target = repo_dp / "ibovespa_clean.csv" if repo_dp else None
    raw_path = base_paths["data_raw"] / "ibovespa.csv"

    if target.exists():
        df = pd.read_csv(target)
        source = target
    elif raw_path.exists():
        df = pd.read_csv(raw_path)
        source = raw_path
    else:
        df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])
        source = None

    before = len(df)
    if "date" not in df.columns and "day" in df.columns:
        df = df.rename(columns={"day": "date"})

    df = _filter_period(df, "date")
    if "return" not in df.columns and "close" in df.columns:
        df["return"] = df["close"].pct_change()
    if "direction" not in df.columns and "return" in df.columns:
        df["direction"] = (df["return"] > 0).astype(int)

    ordered_cols = ["date", "open", "high", "low", "close", "adj_close", "volume", "return", "direction"]
    df = _ensure_columns(df, ordered_cols)
    df = df.drop_duplicates(subset=["date"])

    targets = [target]
    if repo_target:
        targets.append(repo_target)
    _write_csv(df, targets)
    return {"name": "ibovespa_clean", "path": str(target), "source": str(source) if source else "-", "before": before, "after": len(df)}


def process_oof(base_paths: Dict[str, Path], repo_dp: Path | None) -> Dict[str, object]:
    target = base_paths["data_processed"] / "16_oof_predictions.csv"
    repo_target = repo_dp / "16_oof_predictions.csv" if repo_dp else None
    cols = ["model", "fold", "row_id", "day", "y", "ret_next", "close", "proba"]

    df = pd.read_csv(target) if target.exists() else pd.DataFrame(columns=cols)
    before = len(df)

    if "day" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "day"})

    df = _filter_period(df, "day")
    if not df.empty:
        df = df.sort_values(["day", "model", "fold"]).reset_index(drop=True)

    df = _ensure_columns(df, cols)
    targets = [target]
    if repo_target:
        targets.append(repo_target)
    _write_csv(df, targets)
    return {"name": "16_oof_predictions", "path": str(target), "before": before, "after": len(df)}


def process_backtest_curves(base_paths: Dict[str, Path], repo_dp: Path | None) -> Dict[str, object]:
    target = base_paths["data_processed"] / "18_backtest_daily_curves.csv"
    repo_target = repo_dp / "18_backtest_daily_curves.csv" if repo_dp else None

    df = pd.read_csv(target) if target.exists() else pd.DataFrame(columns=["day"])
    before = len(df)

    if "day" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "day"})

    df = _filter_period(df, "day")
    targets = [target]
    if repo_target:
        targets.append(repo_target)
    _write_csv(df, targets)
    return {"name": "18_backtest_daily_curves", "path": str(target), "before": before, "after": len(df)}


def process_backtest_results(base_paths: Dict[str, Path], repo_dp: Path | None) -> Dict[str, object]:
    target = cfg.get_arquivo("backtest_results", base_paths["base"])
    repo_target = repo_dp / "18_backtest_results.csv" if repo_dp else None

    df = pd.read_csv(target) if target.exists() else pd.DataFrame(
        columns=["model", "strategy", "n_days", "cagr", "sharpe", "max_dd", "hit_ratio"]
    )
    before = len(df)

    targets = [target]
    if repo_target:
        targets.append(repo_target)
    _write_csv(df, targets)
    return {"name": "18_backtest_results", "path": str(target), "before": before, "after": len(df)}


def process_results_json(base_paths: Dict[str, Path], repo_dp: Path | None) -> Dict[str, object]:
    ref_matrix = cfg.get_arquivo("tfidf_daily_matrix", base_paths["base"])
    target = ref_matrix.with_name("results_16_models_tfidf.json")
    repo_target = repo_dp / "results_16_models_tfidf.json" if repo_dp else None

    if target.exists():
        payload = json.loads(target.read_text(encoding="utf-8"))
        before = len(payload.get("models", {}))
    else:
        payload = {"notebook": "16_models_tfidf_baselines", "generated_at": None, "n_samples": None, "models": {}}
        before = 0

    targets = [target]
    if repo_target:
        targets.append(repo_target)
    _write_json(payload, targets)
    return {"name": "results_16_models_tfidf", "path": str(target), "before": before, "after": len(payload.get("models", {}))}


def process_latency(base_paths: Dict[str, Path], repo_dp: Path | None) -> Dict[str, object]:
    csv_main = cfg.get_arquivo("latency_events", base_paths["base"])
    parquet_path = base_paths["data_processed"] / "latency_events.parquet"
    fallback_csv = base_paths["data_processed"] / "latency_events.csv"

    candidates = [csv_main, parquet_path, fallback_csv]
    source = next((p for p in candidates if p.exists()), csv_main)

    if source.suffix.lower() == ".parquet":
        df = pd.read_parquet(source)
    elif source.exists():
        df = pd.read_csv(source)
    else:
        df = pd.DataFrame(columns=["event_day"])

    before = len(df)
    date_col = "event_day" if "event_day" in df.columns else ("day" if "day" in df.columns else None)
    if date_col:
        df = _filter_period(df, date_col)
        if date_col != "event_day":
            df = df.rename(columns={date_col: "event_day"})
    if "event_day" not in df.columns:
        df["event_day"] = pd.NaT
    df = df.sort_values("event_day")

    csv_targets = [csv_main]
    if repo_dp:
        csv_targets.append(repo_dp / "event_study_latency.csv")
    _write_csv(df, csv_targets)
    if parquet_path.exists() or source.suffix.lower() == ".parquet":
        _write_parquet(df, [parquet_path])

    return {"name": "event_study_latency", "path": str(source), "before": before, "after": len(df)}


def run_pipeline(copy_to_repo: bool = True) -> List[Dict[str, object]]:
    data_paths = path_utils.get_data_paths(create=True)
    repo_dp = ROOT / "data_processed" if copy_to_repo else None
    if repo_dp:
        repo_dp.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, object]] = []
    summaries.append(process_ibov(data_paths, repo_dp))
    summaries.append(process_oof(data_paths, repo_dp))
    summaries.append(process_backtest_results(data_paths, repo_dp))
    summaries.append(process_backtest_curves(data_paths, repo_dp))
    summaries.append(process_results_json(data_paths, repo_dp))
    summaries.append(process_latency(data_paths, repo_dp))
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pipeline minimo para garantir artefatos chave dentro do periodo oficial (2018-01-02 a 2024-12-31)."
    )
    parser.add_argument("--no-copy-repo", action="store_true", help="Nao copiar artefatos para o repo local data_processed/.")
    args = parser.parse_args()

    summaries = run_pipeline(copy_to_repo=not args.no_copy_repo)
    print("=== Pipeline minimo concluido ===")
    for item in summaries:
        print(
            f"- {item['name']}: {item['before']} -> {item['after']} linhas (path: {item['path']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
