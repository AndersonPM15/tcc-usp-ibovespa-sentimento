"""Figuras da apresentação (tcc.presentation): padrão visual, desenho e números.

Os testes de desenho redesenham cada figura a partir do CSV versionado em
`reports/apresentacao/` e rodam também sem os dados. Com os dados (e a base limpa de
notícias, usada por F1 e F9), as tabelas são recalculadas e comparadas com os CSVs
versionados e com os números do artigo e das verificações pós-submissão.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import partial
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.text import Text

from tcc import reproduce
from tcc.config import Settings
from tcc.datasets import NEWS_CLEAN_FILE
from tcc.presentation import build, charts, style

REPORTS = Path(__file__).resolve().parents[1] / "reports"
COMMITTED = REPORTS / "apresentacao"
DATE_COLUMNS = (
    "dia",
    "data",
    "inicio",
    "fim",
    "treino_inicio",
    "treino_fim",
    "teste_inicio",
    "teste_fim",
)
FULL_WIDTH = ("F1_", "F2_", "F3_", "F4a_", "F8_")
TOLERANCE = 1e-9
F7A = "F7a_patrimonio_limiares_artigo_pos-submissao"
F7B = "F7b_patrimonio_sem_lookahead_pos-submissao"


def _read(stem: str, folder: Path = COMMITTED) -> pd.DataFrame:
    """CSV de uma figura com as datas convertidas (o Ibovespa fica vazio fora dos pregões)."""
    frame = pd.read_csv(folder / f"{stem}.csv", keep_default_na=False, na_values={"ibovespa": [""]})
    for column in DATE_COLUMNS:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column])
    return frame


def _drawers() -> dict[str, Callable[[pd.DataFrame], Figure]]:
    limits = charts.equity_limits(_read(F7A), _read(F7B))
    return {
        "F1_ibovespa_periodos_manchetes": charts.ibovespa_and_headlines,
        "F2_sentimento_diario": charts.daily_sentiment,
        "F3_walk_forward": charts.walk_forward,
        "F4a_dispersao_sentimento_retorno_pos-submissao": charts.sentiment_scatter,
        "F4b_correlacao_movel": charts.rolling_correlation,
        "F5a_curvas_roc": charts.roc_curves,
        "F5b_auc_por_bloco_pos-submissao": charts.auc_by_block,
        "F6_caar_retorno_anormal_pos-submissao": charts.event_study,
        F7A: partial(charts.equity_curves, limits=limits),
        F7B: partial(charts.equity_curves, limits=limits),
        "F8_sharpe_12_configuracoes_pos-submissao": charts.sharpe_heatmaps,
        "F9_auc_token_none_pos-submissao": charts.none_token_auc,
    }


COMMITTED_STEMS = sorted(path.stem for path in COMMITTED.glob("*.csv"))  # figuras e tabelas
FIGURE_STEMS = sorted(path.stem for path in COMMITTED.glob("*.png"))
F5B = "F5b_auc_por_bloco_pos-submissao"
F6 = "F6_caar_retorno_anormal_pos-submissao"
F9 = "F9_auc_token_none_pos-submissao"


# --------------------------------------------------------------------------- padrão visual


def test_numbers_are_written_in_portuguese() -> None:
    assert style.decimal(-0.1274) == "−0,127"
    assert style.decimal(91941, 0) == "91.941"
    assert style.decimal(1234.5, 1) == "1.234,5"
    assert style.percent(-0.03160) == "−3,16%"
    assert style.interval(-0.17728, -0.08413) == "[−0,177; −0,084]"
    assert style.month_year(pd.Timestamp("2019-08-05")) == "ago/2019"
    assert style.date_br(pd.Timestamp("2025-11-17")) == "17/11/2025"


def test_montserrat_is_bundled_with_a_fallback_font() -> None:
    assert style.FONT_FILE.exists()
    assert (style.FONT_FILE.parent / "OFL.txt").exists()  # licença da fonte
    assert style.font_family() == (style.FONT, style.FALLBACK_FONT)


@pytest.mark.parametrize("full_width", [False, True])
def test_save_writes_png_svg_and_csv_in_slide_size(tmp_path: Path, full_width: bool) -> None:
    data = pd.DataFrame({"x": [0.5, 1.5], "y": [1.25, 0.5]})
    for attempt in ("a", "b"):
        with style.slide_style():
            fig, ax = style.new_figure(full_width=full_width)
            ax.plot(data["x"], data["y"])
            style.save(fig, data, tmp_path / attempt, "figura")
    height, width = mpimg.imread(tmp_path / "a" / "figura.png").shape[:2]
    size = style.FULL_WIDTH_SIZE if full_width else style.FIGURE_SIZE
    assert (width, height) == (round(size[0] * 300), round(size[1] * 300))
    svg = (tmp_path / "a" / "figura.svg").read_bytes()
    assert svg == (tmp_path / "b" / "figura.svg").read_bytes()  # sem data nem ids aleatórios
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "a" / "figura.csv"), data)


@pytest.mark.parametrize("stem", FIGURE_STEMS)
def test_committed_data_redraws_in_slide_format(stem: str, tmp_path: Path) -> None:
    frame = _read(stem)
    with style.slide_style():
        fig = _drawers()[stem](frame)
        fig.canvas.draw()
        texts = [text.get_text() for text in fig.findobj(Text) if text.get_text()]
        assert fig.get_suptitle() == ""
        assert all(ax.get_title() == "" for ax in fig.axes)
        png = style.save(fig, frame, tmp_path, stem)[0]
    # vírgula decimal e sinal de menos tipográfico em todos os textos (ponto só no milhar)
    assert not [text for text in texts if re.search(r"\d\.\d{1,2}(?!\d)", text)]
    assert not [text for text in texts if re.search(r"(?<!\w)-\d", text)]
    height, width = mpimg.imread(png).shape[:2]
    size = style.FULL_WIDTH_SIZE if stem.startswith(FULL_WIDTH) else style.FIGURE_SIZE
    assert (width, height) == (round(size[0] * 300), round(size[1] * 300))


def test_post_submission_results_are_labelled_in_the_file_name() -> None:
    labelled = {stem.split("_")[0] for stem in FIGURE_STEMS if build.POST_SUBMISSION in stem}
    assert labelled == {"F4a", "F5b", "F6", "F7a", "F7b", "F8", "F9"}


def test_event_study_uses_the_polarity_colors() -> None:
    with style.slide_style():
        fig = charts.event_study(_read(F6))
    ax = fig.axes[0]
    lines = {
        to_hex(line.get_color()).upper() for line in ax.get_lines() if line.get_marker() == "o"
    }
    assert lines == {color.upper() for color in style.POLARITY_COLORS.values()}
    assert not lines & {color.upper() for color in style.MODEL_COLORS.values()}
    bands = [band for band in ax.collections if isinstance(band, PolyCollection)]
    assert len(bands) == 2 and all(band.get_alpha() == style.CI_ALPHA for band in bands)


@pytest.mark.parametrize("stem", [F5B, F9])
def test_value_labels_stay_clear_of_the_auc_reference_line(stem: str) -> None:
    with style.slide_style():
        fig = _drawers()[stem](_read(stem))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
        ax = fig.axes[0]
        line_y = ax.transData.transform((0, 0.5))[1]
        labels = [text for text in ax.texts if re.fullmatch(r"0,\d{3}", text.get_text())]
        boxes = [text.get_window_extent(renderer) for text in labels]
    assert len(labels) == len(_read(stem))  # um rótulo por ponto
    assert all(box.y0 > line_y or box.y1 < line_y for box in boxes)


def test_headline_coverage_by_year() -> None:
    coverage = _read(build.COVERAGE_STEM).set_index("ano")
    assert coverage.loc[2018, "mediana_por_dia"] == 72
    assert coverage.loc[2024, "mediana_por_dia"] == 10
    assert coverage["total_manchetes"].sum() == 91941
    assert coverage.index.tolist() == list(range(2018, 2026))


# --------------------------------------------------------------------------- com os dados


@pytest.fixture(scope="module")
def produced(
    settings: Settings,
    article_results: reproduce.ArticleResults,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    if not (settings.interim_dir / NEWS_CLEAN_FILE).exists():
        pytest.skip(f"{NEWS_CLEAN_FILE} indisponível: F1 e F9 não são geradas")
    local = Settings(base_dir=settings.base_dir, reports_dir=tmp_path_factory.mktemp("reports"))
    _, skipped = build.write_figures(local, article_results)
    assert not skipped
    return local.presentation_dir


def test_every_committed_figure_is_reproduced(produced: Path) -> None:
    names = sorted(path.name for path in produced.iterdir())
    assert names == sorted(path.name for path in COMMITTED.iterdir())
    assert (produced / "README.md").read_text() == (COMMITTED / "README.md").read_text()


@pytest.mark.parametrize("stem", COMMITTED_STEMS)
def test_presentation_data_unchanged(produced: Path, stem: str) -> None:
    committed, recalculated = _read(stem), _read(stem, produced)
    assert list(recalculated.columns) == list(committed.columns)
    assert recalculated.shape == committed.shape
    for column in committed.columns:
        if pd.api.types.is_numeric_dtype(committed[column]):
            np.testing.assert_allclose(
                recalculated[column], committed[column], rtol=0, atol=TOLERANCE, err_msg=column
            )
        else:
            assert (recalculated[column].astype(str) == committed[column].astype(str)).all(), column


def test_key_numbers_match_article_and_verifications(produced: Path) -> None:
    table2 = pd.read_csv(REPORTS / "figures" / "Tabela_2_auc_mda_valores.csv")
    roc = _read("F5a_curvas_roc", produced).drop_duplicates("modelo")
    np.testing.assert_allclose(roc["auc"], table2["auc"], atol=TOLERANCE)
    np.testing.assert_allclose(roc["auc_ic95_inferior"], table2["auc_low"], atol=TOLERANCE)

    bootstrap = pd.read_csv(REPORTS / "verificacao" / "b_h1_bootstrap_blocos.csv")
    main = bootstrap[
        bootstrap["periodo"].str.startswith("artigo")
        & (bootstrap["modelo"] == "Regressão Logística")
        & (bootstrap["estatistica"] == "pearson")
        & (bootstrap["regra_bloco"] != "sensibilidade")
    ]
    scatter = _read("F4a_dispersao_sentimento_retorno_pos-submissao", produced)
    panels = scatter.drop_duplicates("retorno_de")
    np.testing.assert_allclose(panels["r"], main["estimativa"], atol=TOLERANCE)
    np.testing.assert_allclose(panels["ic95_inferior"], main["ic95_inferior"], atol=TOLERANCE)
    assert panels["n"].tolist() == [1341, 1341]

    events = _read("F6_caar_retorno_anormal_pos-submissao", produced)
    first_day = events[events["tau_pregoes"] == 0].set_index("grupo")["caar"]
    assert first_day.round(4).to_dict() == {
        "Sentimento positivo extremo": -0.0059,
        "Sentimento negativo extremo": 0.0032,
    }

    for stem, cagr in ((F7A, -0.0316), (F7B, -0.0504)):
        curves = _read(stem, produced)
        logistic = curves[curves["serie"] == "Regressão Logística"]
        assert round(float(logistic["cagr"].iloc[0]), 4) == cagr
        assert logistic["patrimonio"].iloc[0] == 1.0
        ibovespa = curves[curves["serie"] == "Ibovespa buy-and-hold"]
        assert round(float(ibovespa["sharpe"].iloc[0]), 3) == 0.255

    sharpe = _read("F8_sharpe_12_configuracoes_pos-submissao", produced)
    ranges = sharpe.groupby("limiar")["sharpe"].agg(["min", "max"]).round(3)
    assert sorted(map(tuple, ranges.to_numpy())) == [(-0.493, 0.046), (-0.201, 0.383)]

    none_token = _read("F9_auc_token_none_pos-submissao", produced).set_index(["versao", "modelo"])
    fixed = none_token.loc['sem "None" (corrigido)', "auc"].round(4).to_dict()
    assert fixed == {"Regressão Logística": 0.5001, "Random Forest": 0.4974}

    schedule = _read("F3_walk_forward", produced)
    assert schedule["pregoes_treino"].tolist() == [390, 778, 1166, 1554]
    headlines = _read("F1_ibovespa_periodos_manchetes", produced)
    assert headlines["manchetes"].sum() == 91941
