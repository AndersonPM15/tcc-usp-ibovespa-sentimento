"""
Pipeline orchestration script for the TCC USP project.

Runs the notebooks 00→20 in sequence, logs progress, and stores executed
versions under `notebooks/_runs/`.

Usage:
    python pipeline_orchestration.py
    python pipeline_orchestration.py --continue-on-fail
    python pipeline_orchestration.py --only 16 17 18
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

try:
    import papermill as pm

    HAVE_PAPERMILL = True
except ImportError:  # pragma: no cover - papermill optional
    HAVE_PAPERMILL = False

from src.config import loader as cfg
from src.io import paths as path_utils
from src.utils.logger import log_result


NOTEBOOK_SEQUENCE: Sequence[str] = [
    "00_data_download",
    "01_preprocessing",
    "02_baseline_logit",
    "03_tfidf_models",
    "04_embeddings_models",
    "05_data_collection_real",
    "06_preprocessing_real",
    "07_tfidf_real",
    "08_embeddings_real",
    "09_lstm_real",
    "10_dashboard_results",
    "11_event_study_latency",
    "12_data_collection_multisource",
    "13_etl_dedup",
    "14_preprocess_ptbr",
    "15_features_tfidf_daily",
    "16_models_tfidf_baselines",
    "17_sentiment_validation",
    "18_backtest_simulation",
    "19_future_extension",
    "20_final_dashboard_analysis",
]


def _setup_logging() -> Path:
    project_paths = path_utils.get_project_paths()
    log_dir = project_paths["reports"] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_run_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_file


def _ensure_pythonpath(repo_root: Path) -> None:
    """
    Guarantee that the repository root is present on PYTHONPATH.

    This avoids `ModuleNotFoundError: src` when notebooks are executed via
    nbconvert (fallback mode) even quando o ambiente nǜo tem papermill.
    """
    current = os.environ.get("PYTHONPATH", "")
    paths = [p for p in current.split(os.pathsep) if p]
    repo_str = str(repo_root)
    if repo_str not in paths:
        os.environ["PYTHONPATH"] = os.pathsep.join([repo_str] + paths)


def _execute_notebook(
    notebook_name: str,
    base_path: Path,
    runs_dir: Path,
) -> None:
    project_paths = path_utils.get_project_paths()
    repo_root = project_paths["repo_root"]
    notebooks_dir = project_paths["notebooks"]

    _ensure_pythonpath(repo_root)

    src = notebooks_dir / f"{notebook_name}.ipynb"
    if not src.exists():
        raise FileNotFoundError(f"Notebook não encontrado: {src}")

    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = runs_dir / f"{timestamp}_{notebook_name}.ipynb"

    logging.info("Executando %s", notebook_name)
    start = datetime.now()
    try:
        if HAVE_PAPERMILL:
            pm.execute_notebook(
                input_path=str(src),
                output_path=str(output),
                parameters={
                    "base_path": str(base_path),
                    "run_id": timestamp,
                },
                progress_bar=False,
                report_mode=True,
            )
        else:  # pragma: no cover - fallback para ambientes sem papermill
            import subprocess

            cmd = [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                str(src),
                "--output",
                output.name,
                "--output-dir",
                str(runs_dir),
            ]
            env = os.environ.copy()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    except Exception as exc:  # noqa: BLE001
        duration = (datetime.now() - start).total_seconds()
        logging.error("Falha em %s (%.1fs): %s", notebook_name, duration, exc)
        metrics = {"status": 0, "duration_s": duration}
        log_result(
            model_name=notebook_name,
            dataset_name="pipeline",
            metrics=metrics,
            extra={"error": str(exc)},
        )
        raise

    duration = (datetime.now() - start).total_seconds()
    logging.info("Concluído %s em %.1fs", notebook_name, duration)
    metrics = {"status": 1, "duration_s": duration}
    log_result(
        model_name=notebook_name,
        dataset_name="pipeline",
        metrics=metrics,
        extra={"output_notebook": str(output)},
    )


def run_pipeline(
    notebooks: Iterable[str],
    continue_on_fail: bool = False,
) -> List[str]:
    data_paths = path_utils.get_data_paths()
    base_path = data_paths["base"]
    runs_dir = path_utils.get_project_paths()["notebooks"] / "_runs"

    completed: List[str] = []
    for nb_name in notebooks:
        try:
            _execute_notebook(nb_name, base_path=base_path, runs_dir=runs_dir)
            completed.append(nb_name)
        except Exception:
            if not continue_on_fail:
                raise
            logging.warning("Continuando após falha em %s", nb_name)
    return completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa o pipeline 00→20.")
    parser.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="Não interrompe o pipeline quando um notebook falhar.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Lista de notebooks para executar (ex.: 16 17 18).",
    )
    return parser.parse_args()


def main() -> None:
    log_file = _setup_logging()
    args = parse_args()
    if args.only:
        notebooks = [f"{nb if nb.endswith('.ipynb') else nb}".replace(".ipynb", "") for nb in args.only]
    else:
        notebooks = list(NOTEBOOK_SEQUENCE)

    logging.info("Iniciando pipeline com %d notebooks.", len(notebooks))
    logging.info("Log: %s", log_file)
    logging.info("Ambiente base_path: %s", path_utils.get_data_paths()["base"])

    try:
        completed = run_pipeline(notebooks, continue_on_fail=args.continue_on_fail)
    except Exception as exc:  # noqa: BLE001
        logging.error("Pipeline interrompido: %s", exc)
        logging.info("Notebooks concluídos antes da falha: %s", completed if 'completed' in locals() else [])
        sys.exit(1)

    logging.info("Pipeline finalizado. Notebooks concluídos: %s", completed)


if __name__ == "__main__":
    main()
