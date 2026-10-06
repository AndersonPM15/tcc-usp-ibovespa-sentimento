"""Classificadores do artigo em validação walk-forward (Tabela 2) e série diária de p.

Com a matriz TF-IDF como entrada, reproduz exatamente a Tabela 2 e as probabilidades fora
da amostra usadas em todas as análises (notebook 16 da versão do artigo).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from tcc.config import RANDOM_SEED
from tcc.stats import iid_bootstrap_ci

N_SPLITS = 4


def baseline_models() -> dict[str, Any]:
    """Os dois classificadores do artigo, com os hiperparâmetros do notebook 16.

    A penalidade L2 do notebook é o padrão da LogisticRegression; o argumento `penalty`
    foi descontinuado no scikit-learn 1.8 e por isso não é passado (resultado idêntico).
    """
    return {
        "logreg_l2": LogisticRegression(
            solver="saga",
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "rf_200": RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }


def mean_directional_accuracy(
    y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5
) -> float:
    """MDA: fração de dias em que a direção prevista (p ≥ 0,5) acerta a realizada."""
    if len(y_true) == 0:
        return np.nan
    return float(((proba >= threshold).astype(int) == y_true).mean())


def walk_forward_predictions(
    features: Any, y: np.ndarray, models: dict[str, Any], n_splits: int = N_SPLITS
) -> dict[str, np.ndarray]:
    """Probabilidades fora da amostra com janela expansiva (TimeSeriesSplit, sem embargo).

    Linhas usadas só para treino (1º bloco) ficam com NaN.
    """
    splitter = TimeSeriesSplit(n_splits=n_splits)
    predictions = {}
    for name, base_model in models.items():
        proba = np.full(len(y), np.nan)
        for train_idx, test_idx in splitter.split(features):
            if np.unique(y[train_idx]).size < 2:
                continue
            model = clone(base_model)
            model.fit(features[train_idx], y[train_idx])
            proba[test_idx] = model.predict_proba(features[test_idx])[:, 1]
        predictions[name] = proba
    return predictions


def fold_ids(n_obs: int, n_splits: int = N_SPLITS) -> np.ndarray:
    """Bloco de teste de cada linha (0, 1, …); −1 nas linhas usadas só para treino."""
    ids = np.full(n_obs, -1)
    for fold, (_, test_idx) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(np.zeros(n_obs))):
        ids[test_idx] = fold
    return ids


def summarize_predictions(y: np.ndarray, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    """AUC e MDA fora da amostra com IC 95% por bootstrap i.i.d. (formato da Tabela 2)."""
    rows = []
    for name, proba in predictions.items():
        valid = ~np.isnan(proba)
        y_valid, p_valid = y[valid], proba[valid]
        auc_low, auc_high = iid_bootstrap_ci(
            y_valid, p_valid, roc_auc_score, requires_two_classes=True
        )
        mda_low, mda_high = iid_bootstrap_ci(y_valid, p_valid, mean_directional_accuracy)
        rows.append(
            {
                "model": name,
                "auc": float(roc_auc_score(y_valid, p_valid)),
                "auc_low": auc_low,
                "auc_high": auc_high,
                "mda": mean_directional_accuracy(y_valid, p_valid),
                "mda_low": mda_low,
                "mda_high": mda_high,
                "n_obs": int(valid.sum()),
            }
        )
    return pd.DataFrame(rows)


def oof_frame(labelled: pd.DataFrame, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    """Série diária de p por modelo (formato longo), só nos dias fora da amostra.

    `labelled` traz `day`, `y` e `ret_next` dos dias com rótulo, na ordem das previsões.
    """
    folds = fold_ids(len(labelled))
    frames = [
        labelled[["day", "y", "ret_next"]].assign(model=name, fold=folds, proba=proba)
        for name, proba in predictions.items()
    ]
    oof = pd.concat(frames, ignore_index=True).dropna(subset=["proba"])
    return oof.sort_values(["model", "day"]).reset_index(drop=True)
