"""Seleção de notebooks do orquestrador (`--only`)."""

from __future__ import annotations

import pytest

import pipeline_orchestration as orchestration


def test_only_accepts_numbers_and_file_names() -> None:
    assert orchestration.resolve_notebook_names(["16", "17_sentiment_validation.ipynb"]) == [
        "16_models_tfidf_baselines",
        "17_sentiment_validation",
    ]


def test_only_rejects_notebooks_outside_the_pipeline() -> None:
    with pytest.raises(ValueError):
        orchestration.resolve_notebook_names(["05"])


def test_pipeline_notebooks_exist() -> None:
    notebooks_dir = orchestration.path_utils.get_project_paths()["notebooks"]
    for name in orchestration.NOTEBOOK_SEQUENCE:
        assert (notebooks_dir / f"{name}.ipynb").exists(), name
