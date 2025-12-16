#!/usr/bin/env python
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(r"C:\TCC_USP\data_processed")
REPORT = ROOT / "reports" / "data_audit_report.md"
LOG = ROOT / "reports" / "preflight_log.txt"

START = pd.Timestamp("2018-01-02")
END = pd.Timestamp("2024-12-31")


def run_cmd(cmd: list[str]) -> Tuple[int, str]:
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    return res.returncode, (res.stdout or "") + (res.stderr or "")


def ensure_latency() -> pd.DataFrame:
    lat_path = DATA_DIR / "event_study_latency.csv"
    needs_regen = not lat_path.exists()
    if not needs_regen:
        try:
            lat = pd.read_csv(lat_path)
            if lat.empty or "event_day" not in lat.columns:
                needs_regen = True
            else:
                lat["event_day"] = pd.to_datetime(lat["event_day"], errors="coerce")
                lat = lat.dropna(subset=["event_day"])
                if lat.empty:
                    needs_regen = True
                else:
                    lat = lat[(lat["event_day"] >= START) & (lat["event_day"] <= END)]
                    if lat.empty:
                        needs_regen = True
        except Exception:
            needs_regen = True
    if needs_regen:
        code, out = run_cmd([sys.executable, "scripts/generate_event_study_latency.py"])
        LOG.write_text(out, encoding="utf-8")
    return pd.read_csv(lat_path)


def dataset_info(path: Path, preferred_col: str | None) -> Dict[str, str]:
    if not path.exists():
        return {"status": "missing"}
    if path.suffix == ".json":
        try:
            js = pd.read_json(path)
            return {"status": "ok", "rows": len(js)}
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": str(exc)}
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}
    if df.empty:
        return {"status": "empty"}
    col = None
    for c in [preferred_col, "day", "date", "Data", "DATE"]:
        if c and c in df.columns:
            col = c
            break
    out: Dict[str, str] = {"status": "ok", "rows": str(len(df))}
    if col:
        dt = pd.to_datetime(df[col], errors="coerce").dropna()
        if not dt.empty:
            out["min"] = str(dt.min().date())
            out["max"] = str(dt.max().date())
    return out


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)

    lat_df = ensure_latency()

    artifacts = {
        "ibovespa_clean.csv": {"col": "day"},
        "16_oof_predictions.csv": {"col": "day"},
        "results_16_models_tfidf.json": {"col": None},
        "18_backtest_daily_curves.csv": {"col": "day"},
        "18_backtest_results.csv": {"col": None},
        "event_study_latency.csv": {"col": "event_day"},
    }
    lines_log = []
    for name, cfg in artifacts.items():
        info = dataset_info(DATA_DIR / name, cfg["col"])
        lines_log.append(f"{name}: {info}")
    LOG.write_text("\n".join(lines_log), encoding="utf-8")

    # smoke callback
    sys.path.insert(0, str(ROOT))
    import app_dashboard as dash_app  # type: ignore

    class DummyCtx:
        @property
        def triggered_id(self):
            return "preflight"

    dash_app.ctx = DummyCtx()
    start = str(dash_app.DEFAULT_START.date())
    end = str(dash_app.DEFAULT_END.date())
    outputs = dash_app.update_dashboard(start, end, list(dash_app.MODEL_OPTIONS), "auc")
    (
        ibov_fig,
        sentiment_fig,
        comparison_fig,
        table_data,
        indicator_content,
        metric_badge,
        ui_status,
        scatter_fig,
        rolling_fig,
        dist_fig,
        latency_fig,
        backtest_fig,
    ) = outputs

    def has_data(fig) -> bool:
        return hasattr(fig, "data") and len(fig.data) > 0

    checks = {
        "ibov": has_data(ibov_fig),
        "sentiment": has_data(sentiment_fig),
        "comparison": has_data(comparison_fig),
        "scatter": has_data(scatter_fig),
        "rolling": has_data(rolling_fig),
        "dist": has_data(dist_fig),
        "latency": has_data(latency_fig),
        "backtest": has_data(backtest_fig),
    }

    status_lines = [f"{k}: {'OK' if v else 'FAIL'}" for k, v in checks.items()]
    LOG.write_text(LOG.read_text(encoding="utf-8") + "\nSMOKE CALLBACK:\n" + "\n".join(status_lines), encoding="utf-8")

    # Audit report
    report = [
        f"# Data Audit - {datetime.now().date()}",
        "Periodo oficial: 2018-01-02 a 2024-12-31",
        "",
        "## Artefatos",
    ]
    for name, cfg in artifacts.items():
        info = dataset_info(DATA_DIR / name, cfg["col"])
        report.append(f"- {name}: {info}")
    report.append("")
    report.append("## Graficos (8)")
    for k, v in checks.items():
        report.append(f"- {k}: {'OK' if v else 'FAIL'}")
    report.append("")
    report.append(f"DEFAULT_START={dash_app.DEFAULT_START.date()} DEFAULT_END={dash_app.DEFAULT_END.date()}")
    REPORT.write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
