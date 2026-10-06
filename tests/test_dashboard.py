"""Teste de fumaça do dashboard: o callback devolve figuras mesmo sem os dados completos."""

import pytest

go = pytest.importorskip("plotly.graph_objects")
pytest.importorskip("dash")


def test_update_additional_graphs_returns_figures() -> None:
    from app_dashboard import update_additional_graphs

    corr_fig, latency_fig, backtest_fig = update_additional_graphs("2020-01-02", "2020-12-30", None)

    assert isinstance(corr_fig, go.Figure)
    assert isinstance(latency_fig, go.Figure)
    assert isinstance(backtest_fig, go.Figure)
