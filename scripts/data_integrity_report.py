from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import loader as cfg
from src.config.constants import END_DATE, START_DATE
from src.io import paths as path_utils

START_TS = pd.Timestamp(START_DATE)
END_TS = pd.Timestamp(END_DATE)
FUTURE_TS = pd.Timestamp("2025-01-01")
REPORT_PATH_DEFAULT = ROOT / "reports" / "data_integrity_report.md"

DATE_CANDIDATES = ("day", "date", "data", "Data", "event_day")


@dataclass
class DatasetSpec:
    name: str
    filename: str
    logical_key: Optional[str] = None
    date_required: bool = True
    date_candidates: Iterable[str] = DATE_CANDIDATES
    unique_keys: Optional[Iterable[str]] = None
    optional: bool = False
    alt_filenames: Iterable[str] = ()


def resolve_path(spec: DatasetSpec, base_path: Path, data_processed: Path) -> Path:
    """
    Resolve o caminho do artefato, priorizando config (logical_key) e caindo para
    data_processed/<filename> ou alternativas definidas.
    """
    candidates: List[Path] = []
    if spec.logical_key:
        try:
            candidates.append(cfg.get_arquivo(spec.logical_key, base_path))
        except KeyError:
            pass
    candidates.append(data_processed / spec.filename)
    for alt in spec.alt_filenames:
        candidates.append(data_processed / alt)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            return pd.json_normalize(payload)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def pick_date_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        lowered = str(col).lower()
        if "date" in lowered or "day" in lowered:
            return col
    return None


def evaluate_dataset(spec: DatasetSpec, base_path: Path, data_processed: Path) -> Dict[str, object]:
    path = resolve_path(spec, base_path, data_processed)
    df = load_frame(path)

    result: Dict[str, object] = {
        "name": spec.name,
        "path": str(path),
        "exists": path.exists(),
        "rows": len(df),
        "date_col": None,
        "min_date": None,
        "max_date": None,
        "before_start": 0,
        "after_end": 0,
        "future_rows": 0,
        "duplicates": 0,
        "issues": [],
        "optional": spec.optional,
    }

    if not path.exists():
        if not spec.optional:
            result["issues"].append("missing")
        return result

    if df.empty:
        result["issues"].append("empty")
        return result

    date_col = pick_date_column(df, spec.date_candidates) if spec.date_required else None
    result["date_col"] = date_col

    if spec.date_required:
        if not date_col:
            result["issues"].append("date_column_missing")
            return result

        dt = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
        result["min_date"] = dt.min()
        result["max_date"] = dt.max()
        result["before_start"] = int((dt < START_TS).sum())
        result["after_end"] = int((dt > END_TS).sum())
        result["future_rows"] = int((dt >= FUTURE_TS).sum())

        if result["before_start"] > 0:
            result["issues"].append("dates_before_start")
        if result["after_end"] > 0:
            result["issues"].append("dates_after_end")
        if result["future_rows"] > 0:
            result["issues"].append("future_dates")

        if spec.unique_keys:
            subset = [col if col in df.columns else date_col for col in spec.unique_keys]
            dup_count = int(df.duplicated(subset=subset).sum())
            result["duplicates"] = dup_count
            if dup_count > 0:
                result["issues"].append("duplicate_keys")

    return result


def build_markdown(results: List[Dict[str, object]], intersection_days: int) -> str:
    lines = [
        "# Data Integrity Report",
        f"Periodo oficial: {START_TS.date()} a {END_TS.date()} (bloqueio >= 2025-01-01)",
        "",
        "| Dataset | Existe | Linhas | Min | Max | >=2025 | Duplicatas | Path | Status |",
        "|---------|--------|--------|-----|-----|--------|------------|------|--------|",
    ]
    for res in results:
        status = "OK" if not res["issues"] else ", ".join(res["issues"])
        min_date = res["min_date"] if res["min_date"] is not None else "-"
        max_date = res["max_date"] if res["max_date"] is not None else "-"
        lines.append(
            f"| {res['name']} | {res['exists']} | {res['rows']} | {min_date} | {max_date} | "
            f"{res.get('future_rows', 0)} | {res.get('duplicates', 0)} | {res['path']} | {status} |"
        )
    lines.append("")
    lines.append(f"Interseccao IBOV x SENTIMENTO (dias): {intersection_days}")
    return "\n".join(lines)


def compute_intersection(results: List[Dict[str, object]]) -> int:
    ibov = next((r for r in results if r["name"] == "ibovespa_clean"), None)
    sent = next((r for r in results if r["name"] == "16_oof_predictions"), None)
    if not ibov or not sent:
        return 0

    ibov_path = Path(ibov["path"])
    sent_path = Path(sent["path"])
    if not ibov_path.exists() or not sent_path.exists():
        return 0

    ibov_df = load_frame(ibov_path)
    sent_df = load_frame(sent_path)
    if ibov_df.empty or sent_df.empty:
        return 0

    ibov_day_col = pick_date_column(ibov_df, DATE_CANDIDATES)
    sent_day_col = pick_date_column(sent_df, DATE_CANDIDATES)
    if not ibov_day_col or not sent_day_col:
        return 0

    ibov_days = set(pd.to_datetime(ibov_df[ibov_day_col], errors="coerce").dt.date.dropna())
    sent_days = set(pd.to_datetime(sent_df[sent_day_col], errors="coerce").dt.date.dropna())
    return len(ibov_days & sent_days)


def should_fail(res: Dict[str, object]) -> bool:
    if res["optional"]:
        if not res["exists"] or res.get("rows", 0) == 0:
            return False
    return bool(res["issues"])


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valida os principais artefatos do TCC (2018-01-02 a 2024-12-31).")
    parser.add_argument("--report-path", default=str(REPORT_PATH_DEFAULT), help="Caminho do markdown de saida.")
    parser.add_argument("--no-report", action="store_true", help="Nao escrever arquivo de report.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv or [])

    data_paths = path_utils.get_data_paths(create=False)
    base_path = data_paths["base"]
    data_processed = data_paths["data_processed"]

    datasets = [
        DatasetSpec(name="ibovespa_clean", filename="ibovespa_clean.csv", logical_key="ibov_clean", unique_keys=("day",)),
        DatasetSpec(name="16_oof_predictions", filename="16_oof_predictions.csv"),
        DatasetSpec(name="results_16_models_tfidf", filename="results_16_models_tfidf.json", date_required=False),
        DatasetSpec(
            name="event_study_latency",
            filename="event_study_latency.csv",
            logical_key="latency_events",
            alt_filenames=("latency_events.parquet", "latency_events.csv"),
            date_candidates=("event_day", "day", "date"),
            optional=True,
        ),
        DatasetSpec(name="18_backtest_results", filename="18_backtest_results.csv", logical_key="backtest_results", date_required=False),
        DatasetSpec(
            name="18_backtest_daily_curves",
            filename="18_backtest_daily_curves.csv",
            date_candidates=("day", "date"),
            optional=False,
        ),
    ]

    results: List[Dict[str, object]] = [
        evaluate_dataset(spec, base_path, data_processed) for spec in datasets
    ]
    intersection_days = compute_intersection(results)

    print("=== Validacao de artefatos (2018-01-02 a 2024-12-31) ===")
    for res in results:
        status = "OK" if not res["issues"] else ", ".join(res["issues"])
        min_date = res["min_date"] if res["min_date"] is not None else "-"
        max_date = res["max_date"] if res["max_date"] is not None else "-"
        print(
            f"- {res['name']}: exists={res['exists']} rows={res['rows']} "
            f"min={min_date} max={max_date} future>={FUTURE_TS.date()}={res.get('future_rows', 0)} "
            f"dups={res.get('duplicates', 0)} status={status}"
        )
    print(f"Interseccao IBOV x SENTIMENTO (dias): {intersection_days}")

    if not args.no_report:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(build_markdown(results, intersection_days), encoding="utf-8")
        print(f"Relatorio salvo em: {report_path}")

    has_failures = any(should_fail(res) for res in results) or intersection_days == 0
    return 1 if has_failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
