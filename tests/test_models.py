"""Walk-forward dos classificadores e Tabela 2 (tcc.models)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tcc import datasets, market, models
from tcc.config import Settings

# Tabela 2 do artigo (rodada de 26/11/2025; valores completos do JSON de resultados)
TABLE2 = {
    "logreg_l2": {
        "auc": 0.5015378221113882,
        "auc_low": 0.4739878216337317,
        "auc_high": 0.5319980068370906,
        "mda": 0.5025773195876289,
    },
    "rf_200": {
        "auc": 0.4912635078969243,
        "auc_low": 0.46285592310485124,
        "auc_high": 0.5211218602628767,
        "mda": 0.5115979381443299,
    },
}


def test_fold_ids_match_article_walk_forward() -> None:
    ids = models.fold_ids(1942)
    assert (ids[:390] == -1).all()
    assert [int((ids == fold).sum()) for fold in range(4)] == [388] * 4


def test_walk_forward_schedule_dates_match_fold_ids() -> None:
    days = pd.Series(pd.bdate_range("2018-01-02", periods=1942))
    schedule = models.walk_forward_schedule(days)
    assert schedule["n_train"].tolist() == [390, 778, 1166, 1554]
    assert schedule["n_test"].tolist() == [388] * 4
    assert (schedule["train_start"] == days.iloc[0]).all()
    ids = models.fold_ids(len(days))
    for step in schedule.itertuples(index=False):
        in_block = days[ids == step.step - 1]
        assert (in_block.min(), in_block.max()) == (step.test_start, step.test_end)
        assert step.train_end == days[days < step.test_start].max()


def test_mean_directional_accuracy() -> None:
    assert models.mean_directional_accuracy(np.array([1, 0, 1]), np.array([0.6, 0.4, 0.4])) == (
        pytest.approx(2 / 3)
    )
    assert np.isnan(models.mean_directional_accuracy(np.array([]), np.array([])))


def test_walk_forward_only_predicts_after_first_training_block() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(50, 3))
    y = (features[:, 0] > 0).astype(int)
    predictions = models.walk_forward_predictions(features, y, models.baseline_models())
    for proba in predictions.values():
        assert np.isnan(proba[:10]).all()
        assert not np.isnan(proba[10:]).any()


def test_oof_frame_keeps_only_out_of_sample_days() -> None:
    labelled = pd.DataFrame(
        {"day": pd.bdate_range("2024-01-01", periods=10), "y": [0, 1] * 5, "ret_next": 0.01}
    )
    proba = np.r_[np.full(2, np.nan), np.linspace(0.4, 0.6, 8)]
    oof = models.oof_frame(labelled, {"logreg_l2": proba})
    assert len(oof) == 8
    assert set(oof.columns) >= {"model", "day", "y", "ret_next", "proba", "fold"}


def test_table2_reproduced_from_tfidf(settings: Settings) -> None:
    matrix, index = datasets.read_tfidf(settings)
    labels = market.build_labels(index, datasets.read_ibovespa(settings))
    labelled = labels["y"].notna().to_numpy()
    y = labels.loc[labelled, "y"].astype(int).to_numpy()
    predictions = models.walk_forward_predictions(matrix[labelled], y, models.baseline_models())
    summary = models.summarize_predictions(y, predictions).set_index("model")
    for model, expected in TABLE2.items():
        for column, value in expected.items():
            assert summary.loc[model, column] == pytest.approx(value, abs=1e-12), (model, column)
    assert (summary["n_obs"] == 1552).all()
