"""
Prefect-based orchestration for the TCC USP sentiment pipeline.

Usage
-----
python pipeline_orchestration.py
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable

try:
    import papermill as pm

    HAVE_PAPERMILL = True
except ImportError:
    HAVE_PAPERMILL = False

from prefect import flow, task, get_run_logger

from src.config import loader as config_loader
from src.io import paths as path_utils

# ------------------------------------------------------------------------------
# Global paths / logging setup
# ------------------------------------------------------------------------------

DATA_PATHS = path_utils.get_data_paths()
PROJECT_PATHS = path_utils.get_project_paths()
NOTEBOOKS_DIR = PROJECT_PATHS["notebooks"]
RUNS_DIR = NOTEBOOKS_DIR / "_runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = PROJECT_PATHS["reports"] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"pipeline_{TIMESTAMP}.log"


def _log_to_file(message: str) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(message + "\n")


def run_notebook(notebook_name: str, parameters: Dict[str, str] | None = None) -> Path:
    """
    Execute a notebook via papermill (if installed) or jupyter nbconvert.

    Returns
    -------
    Path
        Location of the executed notebook output.
    """
    parameters = parameters or {}
    src = NOTEBOOKS_DIR / f"{notebook_name}.ipynb"
    if not src.exists():
        raise FileNotFoundError(f"Notebook não encontrado: {src}")

    output = RUNS_DIR / f"{TIMESTAMP}_{notebook_name}.ipynb"
    msg = f"[{datetime.now():%H:%M:%S}] Executando {src.name}"
    print(msg)
    _log_to_file(msg)

    if HAVE_PAPERMILL:
        pm.execute_notebook(
            input_path=str(src),
            output_path=str(output),
            parameters=parameters,
            log_output=True,
        )
    else:
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
            str(RUNS_DIR),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            _log_to_file(result.stdout)
            _log_to_file(result.stderr)
            raise RuntimeError(
                f"Falha ao executar {src.name} via nbconvert. "
                f"stdout: {result.stdout}\n stderr: {result.stderr}"
            )
    done_msg = f"[{datetime.now():%H:%M:%S}] Concluído {src.name}"
    print(done_msg)
    _log_to_file(done_msg)
    return output


def run_group(notebooks: Iterable[str], parameters: Dict[str, str]) -> None:
    for nb in notebooks:
        run_notebook(nb, parameters=parameters)


# ------------------------------------------------------------------------------
# Prefect tasks
# ------------------------------------------------------------------------------

@task(name="Protótipo 00-04")
def tarefa_00_01_02_03_04(params: Dict[str, str]) -> None:
    notebooks = [
        "00_data_download",
        "01_preprocessing",
        "02_baseline_logit",
        "03_tfidf_models",
        "04_embeddings_models",
    ]
    run_group(notebooks, params)


@task(name="Dados reais 05-10")
def tarefa_05_06_07_08_09_10(params: Dict[str, str]) -> None:
    notebooks = [
        "05_data_collection_real",
        "06_preprocessing_real",
        "07_tfidf_real",
        "08_embeddings_real",
        "09_lstm_real",
        "10_dashboard_results",
    ]
    run_group(notebooks, params)


@task(name="Multisource 11-15")
def tarefa_11_12_13_14_15(params: Dict[str, str]) -> None:
    notebooks = [
        "11_event_study_latency",
        "12_data_collection_multisource",
        "13_etl_dedup",
        "14_preprocess_ptbr",
        "15_features_tfidf_daily",
    ]
    run_group(notebooks, params)


@task(name="Modelos finais 16-20")
def tarefa_16_17_18_19_20(params: Dict[str, str]) -> None:
    notebooks = [
        "16_models_tfidf_baselines",
        "17_sentiment_validation",
        "18_backtest_simulation",
        "19_future_extension",
        "20_final_dashboard_analysis",
    ]
    run_group(notebooks, params)


# ------------------------------------------------------------------------------
# Flow definition
# ------------------------------------------------------------------------------

def _build_parameters() -> Dict[str, str]:
    cfg = config_loader.load_config()
    periodo = cfg.get("periodo_estudo", {})
    params = {
        "base_path": str(DATA_PATHS["base"]),
        "proc_path": str(DATA_PATHS["data_processed"]),
        "raw_path": str(DATA_PATHS["data_raw"]),
        "interim_path": str(DATA_PATHS["data_interim"]),
        "periodo_start": periodo.get("start"),
        "periodo_end": periodo.get("end"),
    }
    return params


@flow(name="tcc_pipeline_orchestration")
def orchestrate_pipeline() -> None:
    logger = get_run_logger()
    logger.info("Iniciando pipeline TCC USP")
    logger.info(f"Logs em: {LOG_FILE}")
    params = _build_parameters()
    logger.info(f"Parâmetros base: {params}")

    res_proto = tarefa_00_01_02_03_04.submit(params)
    res_proto.result()

    res_real = tarefa_05_06_07_08_09_10.submit(params)
    res_real.result()

    res_multi = tarefa_11_12_13_14_15.submit(params)
    res_multi.result()

    res_final = tarefa_16_17_18_19_20.submit(params)
    res_final.result()

    logger.info("Pipeline concluído com sucesso ✅")


if __name__ == "__main__":
    orchestrate_pipeline()
