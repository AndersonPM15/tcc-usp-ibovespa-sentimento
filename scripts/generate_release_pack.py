#!/usr/bin/env python
from __future__ import annotations

import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(r"C:\TCC_USP\data_processed")
RELEASE_DIR = ROOT / "reports" / "release_pack"
FIG_DIR = RELEASE_DIR / "FIGURES"


def run_cmd(cmd: List[str]) -> Tuple[int, str]:
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    return res.returncode, (res.stdout or "") + (res.stderr or "")


def _pick_date_col(df: pd.DataFrame, preferred: str | None) -> str | None:
    if preferred and preferred in df.columns:
        return preferred
    aliases = ["day", "date", "Data", "DATE"]
    for c in aliases:
        if c in df.columns:
            return c
    return None


def dataset_summary(path: Path, date_col: str | None) -> str:
    if not path.exists():
        return f"{path.name}: arquivo ausente"
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        return f"{path.name}: erro ao ler ({exc.__class__.__name__})"
    if df.empty:
        return f"{path.name}: 0 linhas"
    use_col = _pick_date_col(df, date_col)
    if use_col:
        dates = pd.to_datetime(df[use_col], errors="coerce").dropna()
        if dates.empty:
            return f"{path.name}: linhas={len(df)}, datas inválidas"
        return f"{path.name}: linhas={len(df)}, min={dates.min().date()}, max={dates.max().date()}"
    return f"{path.name}: linhas={len(df)} (sem coluna de data)"


def intersection_summary(ibov_path: Path, sent_path: Path) -> str:
    if not ibov_path.exists() or not sent_path.exists():
        return "Interseção: faltam arquivos"
    try:
        ibov = pd.read_csv(ibov_path)
        sent = pd.read_csv(sent_path)
    except Exception as exc:  # noqa: BLE001
        return f"Interseção: erro ao ler ({exc.__class__.__name__})"
    ibov_col = _pick_date_col(ibov, "day")
    sent_col = _pick_date_col(sent, "day")
    if ibov.empty or sent.empty or not ibov_col or not sent_col:
        return "Interseção: dados vazios ou sem coluna day"
    ibov["__d"] = pd.to_datetime(ibov[ibov_col], errors="coerce")
    sent["__d"] = pd.to_datetime(sent[sent_col], errors="coerce")
    merged = ibov[["__d"]].merge(sent[["__d"]], on="__d", how="inner").dropna()
    return f"Intersecao IBOV x Sentimento: {len(merged)} dias"


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_runbook() -> None:
    content = f"""# RUNBOOK — Release Pack Dashboard

## Pré-requisitos
- Python do venv: .\\\\venv\\\\Scripts\\\\python.exe
- Dados locais em C:\\\\TCC_USP\\\\data_processed (período 2018-01-02 a 2024-12-31).

## Passos recomendados
1) (Opcional) Regenerar artefatos mínimos
   .\\\\venv\\\\Scripts\\\\python.exe scripts\\\\pipeline_minimal.py
2) Validação de integridade
   .\\\\venv\\\\Scripts\\\\python.exe scripts\\\\data_integrity_report.py
3) Testes
   .\\\\venv\\\\Scripts\\\\python.exe -m pytest -q
4) Probe do dashboard (HTTP 200 esperado)
   .\\\\venv\\\\Scripts\\\\python.exe app_dashboard.py --probe --host 127.0.0.1 --port 8050
5) Subir dashboard
   .\\\\venv\\\\Scripts\\\\python.exe app_dashboard.py --host 127.0.0.1 --port 8050 --find-port --open
6) Exportar figuras (8 gráficos)
   .\\\\venv\\\\Scripts\\\\python.exe scripts\\\\export_dashboard_figures.py

## Notas
- Nenhum dado é versionado; use sempre C:\\\\TCC_USP\\\\data_processed.
- Latência: se o dataset estiver vazio, o card é tratado academicamente (mensagem textual).
"""
    write_file(RELEASE_DIR / "RUNBOOK.md", content)


def build_checklist() -> None:
    content = """# CHECKLIST_REPROD

[ ] Dados presentes em C:\\TCC_USP\\data_processed (até 2024-12-31, sem 2025+)
[ ] scripts/data_integrity_report.py executado (ou evidência de min/max datas)
[ ] pytest -q passou
[ ] app_dashboard.py --probe retornou HTTP 200
[ ] app_dashboard.py sobe com --find-port --open
[ ] Figuras exportadas em reports/release_pack/FIGURES (8 gráficos)
[ ] EVIDENCIAS.md e RUNBOOK.md atualizados
"""
    write_file(RELEASE_DIR / "CHECKLIST_REPROD.md", content)


def main() -> None:
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Exporta figuras
    exp_code, exp_out = run_cmd([sys.executable, "scripts/export_dashboard_figures.py"])

    # Data integrity (opcional)
    di_code, di_out = (1, "scripts/data_integrity_report.py não encontrado")
    di_script = ROOT / "scripts" / "data_integrity_report.py"
    if di_script.exists():
        di_code, di_out = run_cmd([sys.executable, "scripts/data_integrity_report.py"])

    # Pytest
    pytest_code, pytest_out = run_cmd([sys.executable, "-m", "pytest", "-q"])

    # Probe com servidor temporário
    def find_free_port(start: int = 8050) -> int:
        port = start
        while port < start + 20:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    return port
            port += 1
        return start

    port = find_free_port(8050)
    server = subprocess.Popen(
        [sys.executable, "app_dashboard.py", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(4)
    probe_code, probe_out = run_cmd([sys.executable, "app_dashboard.py", "--probe", "--host", "127.0.0.1", "--port", str(port)])
    server.terminate()
    try:
        server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()

    # Dataset summaries
    ibov_path = DATA_DIR / "ibovespa_clean.csv"
    sent_path = DATA_DIR / "16_oof_predictions.csv"
    backtest_path = DATA_DIR / "18_backtest_daily_curves.csv"
    latency_path = DATA_DIR / "event_study_latency.csv"
    summaries = [
        dataset_summary(ibov_path, "day"),
        dataset_summary(sent_path, "day"),
        dataset_summary(backtest_path, "day"),
        dataset_summary(latency_path, "event_day"),
        intersection_summary(ibov_path, sent_path),
    ]

    evidencias = f"""# EVIDENCIAS

Gerado em: {datetime.now().isoformat(timespec='seconds')}

## Datasets
- """ + "\n- ".join(summaries) + f"""

## Export de figuras
exit={exp_code}
{exp_out}

## Data integrity report
exit={di_code}
{di_out}

## Pytest
exit={pytest_code}
{pytest_out}

## Probe HTTP (porta {port})
exit={probe_code}
{probe_out}
"""
    write_file(RELEASE_DIR / "EVIDENCIAS.md", evidencias)

    build_runbook()
    build_checklist()


if __name__ == "__main__":
    main()
