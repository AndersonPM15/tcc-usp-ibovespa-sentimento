"""
Teste de regressão: o código deve reproduzir os números publicados no artigo
(XXIX SemeAd, 2026, artigo 160).

Cobertura: Tabelas 1, 2, 3 e 4, Figuras 3, 4 e 8 e o estudo de eventos (Fig. 7B).

Os dados processados não são versionados (licença das fontes). Defina a variável
de ambiente TCC_USP_BASE apontando para a pasta que contém `data_processed/` com:
16_oof_predictions.csv, ibovespa_clean.csv, labels_y_daily.csv,
tfidf_daily_matrix.npz e tfidf_daily_index.csv. Sem esses arquivos, os testes
são pulados (skip) com aviso.

Exceção documentada: a Fig. 7B do artigo veio de um bootstrap sem semente fixa.
Em 200 repetições, o bootstrap se afastou do artigo até 0,0019 nas médias e até
0,0068 nos ICs (grupo positivo, τ=4, apenas 15 eventos). Por isso, nessa figura,
o nº de eventos é exato, as médias são comparadas até a 3ª casa decimal
(tolerância 0,0025) e os ICs até a 2ª casa decimal (tolerância 0,01).
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "article"
REQUIRED_FILES = (
    "16_oof_predictions.csv",
    "ibovespa_clean.csv",
    "labels_y_daily.csv",
    "tfidf_daily_matrix.npz",
    "tfidf_daily_index.csv",
)

# Números publicados (Tabela 2 = rodada de 26/11/2025; valores completos do JSON de resultados)
TABLE2_EXPECTED = {
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
FIGURE4_PEARSON_R = -0.12740289189868295
FIGURE3_SHARPE = {"logreg_l2": -0.15541758121670646, "rf_200": 0.2134549284542668}
FIGURE8_CAGR = {"logreg_l2": -0.026479016125162458, "rf_200": 0.02216856378698573}


def _data_dir() -> Path:
    base = os.environ.get("TCC_USP_BASE")
    if not base:
        pytest.skip("TCC_USP_BASE não definida: dados do artigo indisponíveis.")
    data_dir = Path(base) / "data_processed"
    missing = [name for name in REQUIRED_FILES if not (data_dir / name).exists()]
    if missing:
        pytest.skip(f"Arquivos ausentes em {data_dir}: {missing}")
    return data_dir


def _load_script(name: str, relative_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def data_dir() -> Path:
    return _data_dir()


@pytest.fixture(scope="module")
def export(data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    """Script de exportação com entrada nos dados do artigo e saída em pasta temporária."""
    module = _load_script("export_tcc_figures", "scripts/export_tcc_figures.py")
    module.BASE_DATA = data_dir
    module.OUTPUT_DIR = tmp_path_factory.mktemp("figuras")
    return module


@pytest.fixture(scope="module")
def ibov(export: ModuleType) -> pd.DataFrame:
    return export.load_ibov()


@pytest.fixture(scope="module")
def oof(export: ModuleType) -> pd.DataFrame:
    return export.load_oof_predictions()


@pytest.fixture(scope="module")
def daily_sentiment(export: ModuleType) -> pd.DataFrame:
    return export.load_sentiment_daily(export.load_sentiment())


def test_table1_sample_sizes(ibov: pd.DataFrame, daily_sentiment: pd.DataFrame) -> None:
    """Tabela 1: 1.737 pregões (2018–2024) e 1.341 dias com sentimento."""
    ibov_days = set(ibov["day"])
    sentiment_days = set(daily_sentiment["day"])
    assert len(ibov_days) == 1737
    assert len(sentiment_days) == 1341
    assert len(ibov_days & sentiment_days) == 1341


def test_table2_auc_mda_reproduced_from_tfidf(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tabela 2: re-executa as células de cálculo do notebook 16 sobre a matriz TF-IDF."""
    monkeypatch.setenv("TCC_USP_BASE", str(data_dir.parent))
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    notebook = json.loads(
        (REPO_ROOT / "notebooks/16_models_tfidf_baselines.ipynb").read_text("utf-8")
    )
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    namespace: dict = {"display": lambda *args, **kwargs: None}
    for index, cell in enumerate(code_cells[:6]):  # setup → modelos + walk-forward; não grava nada
        exec(
            compile("".join(cell["source"]), f"nb16_cell_{index + 1}", "exec"),
            namespace,
        )
    summary = namespace["summary_df"].set_index("model")
    for model, expected in TABLE2_EXPECTED.items():
        for column, value in expected.items():
            assert summary.loc[model, column] == pytest.approx(value, abs=1e-9), (
                model,
                column,
            )
        assert summary.loc[model, "n_obs"] == 1552


def test_table3_default_scenario(export: ModuleType, oof: pd.DataFrame, ibov: pd.DataFrame) -> None:
    """Tabela 3: cenário padrão (lag 0, quantil 0,90) idêntico ao do artigo."""
    export.generate_table3_metricas_extendidas(oof, ibov, "long_only_60", 0, 0.90)
    produced = pd.read_csv(export.OUTPUT_DIR / "Tabela_3_metricas_extendidas.csv").astype(str)
    expected = pd.read_csv(FIXTURES / "table3_default_scenario.csv").astype(str)
    pd.testing.assert_frame_equal(produced, expected)


def test_table4_robustness_grid(export: ModuleType, oof: pd.DataFrame, ibov: pd.DataFrame) -> None:
    """Tabela 4: grade lag × quantil idêntica à do artigo."""
    export.generate_table2_robustez(oof, ibov, "long_only_60", [0, 1, 2], [0.90, 0.95])
    produced = pd.read_csv(export.OUTPUT_DIR / "Tabela_2_robustez_backtest.csv").astype(str)
    expected = pd.read_csv(FIXTURES / "table4_robustness_grid.csv").astype(str)
    pd.testing.assert_frame_equal(produced, expected)


def test_figure4_correlation(daily_sentiment: pd.DataFrame, ibov: pd.DataFrame) -> None:
    """Figura 4: r de Pearson (Regressão Logística × retorno do mesmo dia) = −0,13."""
    lr = daily_sentiment[daily_sentiment["model"] == "logreg_l2"]
    merged = pd.merge(lr, ibov[["day", "ret"]], on="day").dropna()
    assert len(merged) == 1341
    assert merged["sentiment"].corr(merged["ret"]) == pytest.approx(FIGURE4_PEARSON_R, abs=1e-12)


def test_figures3_and_8_backtest(export: ModuleType, oof: pd.DataFrame, ibov: pd.DataFrame) -> None:
    """Figuras 3 e 8: curvas diárias, Sharpe e CAGR da regra long_only_60 (limiar fixo)."""
    equity, stats, _ = export.compute_backtest_mark_to_market(oof, ibov, "long_only_60")
    expected = pd.read_csv(FIXTURES / "figure8_equity_curves.csv")
    produced_dates = pd.to_datetime(equity["date"]).dt.strftime("%Y-%m-%d").to_numpy()
    assert (produced_dates == expected["date"].to_numpy()).all()
    columns = ["equity_ibov", "equity_logreg_l2", "equity_rf_200"]
    np.testing.assert_allclose(
        equity[columns].to_numpy(), expected[columns].to_numpy(), rtol=0, atol=1e-12
    )
    for model in ("logreg_l2", "rf_200"):
        assert stats[model]["sharpe"] == pytest.approx(FIGURE3_SHARPE[model], abs=1e-12)
        assert stats[model]["cagr"] == pytest.approx(FIGURE8_CAGR[model], abs=1e-12)


def test_event_study_events(data_dir: Path) -> None:
    """Estudo de eventos: os 270 eventos (p90/p10) e seus CARs são regenerados sem diferença."""
    module = _load_script("generate_event_study_latency", "scripts/generate_event_study_latency.py")
    ibov = module.load_ibov(data_dir / "ibovespa_clean.csv")
    sentiment = module.load_sentiment(data_dir / "16_oof_predictions.csv")
    events = module.generate_latency(ibov, sentiment)
    expected = pd.read_csv(FIXTURES / "event_study_events.csv")
    assert len(events) == len(expected) == 270
    assert (events["event_day"].astype(str).to_numpy() == expected["event_day"].to_numpy()).all()
    assert (events["event_name"].to_numpy() == expected["event_name"].to_numpy()).all()
    np.testing.assert_allclose(events["car_value"], expected["car_value"], rtol=0, atol=1e-12)


def test_figure7b_caar(export: ModuleType, ibov: pd.DataFrame) -> None:
    """Fig. 7B: nº de eventos por τ exato; médias até 0,0025 e ICs até 0,01 (ver docstring do módulo)."""
    events = pd.read_csv(FIXTURES / "event_study_events.csv")
    events["event_day"] = pd.to_datetime(events["event_day"])
    events["polarity"] = events["event_name"].str.contains("pos").map({True: "pos", False: "neg"})
    np.random.seed(42)  # determinismo do teste; o script também fixa a semente
    export.figure_caar_event_time(events, ibov)
    produced = pd.read_csv(export.OUTPUT_DIR / "Figura_7B_event_time_CAAR.csv")
    expected = pd.read_csv(FIXTURES / "figure7b_caar.csv")
    for polarity in ("neg", "pos"):
        assert (produced[f"n_events_{polarity}"] == expected[f"n_events_{polarity}"]).all()
        np.testing.assert_allclose(
            produced[f"caar_{polarity}_mean"], expected[f"caar_{polarity}_mean"], atol=2.5e-3
        )
        for bound in ("ci_low", "ci_high"):
            column = f"caar_{polarity}_{bound}"
            np.testing.assert_allclose(produced[column], expected[column], atol=1e-2)
