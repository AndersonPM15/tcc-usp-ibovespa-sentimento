"""
Result logging utilities for model experiments.

Features
--------
- Persists a JSON registry (`data/results_registry.json`) aggregating all calls.
- Optionally logs to MLflow if the `mlflow` package is available in the env.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import mlflow  # type: ignore

    _HAVE_MLFLOW = True
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None  # type: ignore
    _HAVE_MLFLOW = False

from src.io import paths as path_utils

PROJECT_PATHS = path_utils.get_project_paths()
DATA_DIR = PROJECT_PATHS.get("data", Path(__file__).resolve().parents[2] / "data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_FILE = DATA_DIR / "results_registry.json"


def _load_registry() -> List[Dict[str, Any]]:
    if not REGISTRY_FILE.exists():
        return []
    try:
        with REGISTRY_FILE.open("r", encoding="utf-8") as fh:
            content = json.load(fh)
            if isinstance(content, list):
                return content
    except json.JSONDecodeError:
        pass
    return []


def _persist_registry(entries: List[Dict[str, Any]]) -> None:
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_FILE.open("w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)


def _log_to_mlflow(model_name: str, dataset_name: str, metrics: Dict[str, Any], extra: Dict[str, Any]) -> None:
    if not _HAVE_MLFLOW:
        return

    run_name = f"{model_name}_{dataset_name}"
    active_run = mlflow.active_run()
    should_close = False
    if active_run is None:
        mlflow.start_run(run_name=run_name)
        should_close = True

    tags = {"model": model_name, "dataset": dataset_name}
    mlflow.set_tags(tags)

    # Log metrics (cast to float when possible)
    metrics_float = {}
    for key, value in metrics.items():
        try:
            metrics_float[key] = float(value)
        except (TypeError, ValueError):
            continue
    if metrics_float:
        mlflow.log_metrics({f"{dataset_name}.{k}": v for k, v in metrics_float.items()})

    # Log params from extra info (store as strings)
    if extra:
        params = {f"{dataset_name}.{k}": str(v) for k, v in extra.items()}
        mlflow.log_params(params)

    if should_close:
        mlflow.end_run()


def log_result(
    model_name: str,
    dataset_name: str,
    metrics: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Append a new entry to the registry and optionally log to MLflow.

    Parameters
    ----------
    model_name : str
        Identifier for the model (e.g., "logreg_l2").
    dataset_name : str
        Name of the dataset or experiment split (e.g., "tfidf_daily").
    metrics : dict
        Dictionary with metric names → values (will be stored as-is).
    extra : dict, optional
        Additional metadata such as hyperparameters, notes, paths.

    Returns
    -------
    dict
        The entry that was just stored (useful for debugging/tests).
    """
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("metrics precisa ser um dicionário com ao menos uma chave.")

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": model_name,
        "dataset": dataset_name,
        "metrics": metrics,
        "extra": extra or {},
    }

    registry = _load_registry()
    registry.append(entry)
    _persist_registry(registry)

    _log_to_mlflow(model_name, dataset_name, metrics, entry["extra"])

    return entry


def load_results() -> List[Dict[str, Any]]:
    """
    Load the aggregated results registry.

    Returns
    -------
    list of dict
        Each dict contains `timestamp`, `model`, `dataset`, `metrics`, `extra`.
    """
    return _load_registry()
