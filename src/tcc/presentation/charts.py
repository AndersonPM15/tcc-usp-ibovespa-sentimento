"""Desenho das figuras da apresentação a partir das tabelas de `tcc.presentation.data`.

Cada função recebe a tabela que vai para o CSV da figura e devolve a figura pronta, no
padrão de `tcc.presentation.style` (sem título: o título fica no slide).
"""

from __future__ import annotations

from typing import Any

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator

from tcc.config import MODEL_LABELS
from tcc.presentation import data as slide_data
from tcc.presentation.style import (
    AXIS_PT,
    CI_ALPHA,
    IBOVESPA_COLOR,
    MODEL_COLORS,
    MUTED_COLOR,
    TEXT_COLOR,
    date_br,
    decimal,
    interval,
    month_year,
    new_figure,
    number_ticks,
    percent,
    year_ticks,
)

BAND_COLORS = {
    slide_data.TRAINING: "#E5E7EB",
    slide_data.ARTICLE_TEST: "#DCE8FA",
    slide_data.H2_ONLY_TEST: "#EDE7F6",
}
TRAIN_BAR_COLOR = "#D1D5DB"
TEST_BAR_COLOR = "#374151"
EVENT_COLORS = {slide_data.POSITIVE_EVENTS: "#1B7F3B", slide_data.NEGATIVE_EVENTS: "#B42318"}
VERSION_MARKERS = {slide_data.WITH_NONE: "o", slide_data.WITHOUT_NONE: "D"}
HEATMAP_LIMIT = 0.5  # escala de cor comum aos dois painéis da F8 (Sharpe de −0,5 a 0,5)


def _zero_line(ax: Axes, value: float = 0.0) -> None:
    ax.axhline(value, color=MUTED_COLOR, linewidth=1.2, linestyle="--", zorder=1)


def _span_center(start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Meio do intervalo, na escala de datas do matplotlib."""
    return float(mdates.date2num(start + (end - start) / 2))


def _years_from_january(ax: Axes, first: pd.Timestamp, last: pd.Timestamp) -> None:
    """Eixo de datas de 1º de janeiro do 1º ano até o último dia, com um rótulo por ano."""
    start = pd.Timestamp(year=pd.Timestamp(first).year, month=1, day=1)
    ax.set_xlim(float(mdates.date2num(start)), float(mdates.date2num(last)))
    year_ticks(ax.xaxis)


def _value_label(ax: Axes, text: str, xy: tuple[float, float], side: int, color: str) -> None:
    """Valor escrito ao lado de um ponto (à esquerda se `side` < 0), sobre fundo branco."""
    ax.annotate(
        text,
        xy,
        xytext=(side * 12, 0),
        textcoords="offset points",
        ha="right" if side < 0 else "left",
        va="center",
        fontsize=AXIS_PT,
        color=color,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
    )


def _date_spans(frame: pd.DataFrame, key: str) -> dict[Any, tuple[pd.Timestamp, pd.Timestamp]]:
    """Primeiro e último dia de cada grupo (faixa do estudo ou bloco do walk-forward)."""
    return {
        group: (pd.Timestamp(days.min()), pd.Timestamp(days.max()))
        for group, days in frame.groupby(key)["dia"]
    }


# --------------------------------------------------------------------------- F1


def ibovespa_and_headlines(frame: pd.DataFrame) -> Figure:
    """F1: Ibovespa com as faixas do estudo e, embaixo, as manchetes por dia."""
    fig, (top, bottom) = new_figure(
        full_width=True, nrows=2, sharex=True, gridspec_kw={"height_ratios": [2.4, 1]}
    )
    prices = frame.dropna(subset=["ibovespa"])
    top.plot(prices["dia"], prices["ibovespa"], color=IBOVESPA_COLOR, linewidth=1.6)
    bottom.bar(frame["dia"], frame["manchetes"], width=1.0, color=MUTED_COLOR, linewidth=0)
    spans = _date_spans(frame[frame["faixa"] != ""], "faixa")
    training_days = int(frame["pregoes_treino_inicial"].iloc[0])
    for band, color in BAND_COLORS.items():
        for ax in (top, bottom):
            ax.axvspan(*spans[band], color=color, linewidth=0, zorder=0)
    labels = {
        slide_data.TRAINING: f"Treino inicial: {decimal(training_days, 0)} pregões",
        slide_data.ARTICLE_TEST: f"H1 e H3: {_span_text(*spans[slide_data.ARTICLE_TEST])}",
    }
    for band, label in labels.items():
        top.text(
            _span_center(*spans[band]),
            0.96,
            label,
            transform=transforms.blended_transform_factory(top.transData, top.transAxes),
            ha="center",
            va="top",
            fontsize=AXIS_PT,
        )
    _h2_bracket(top, spans[slide_data.ARTICLE_TEST][0], spans[slide_data.H2_ONLY_TEST][1])
    top.set_ylim(prices["ibovespa"].min() * 0.85, prices["ibovespa"].max() * 1.42)
    top.set_ylabel("Ibovespa (pontos)")
    number_ticks(top.yaxis, 0)
    bottom.set_ylabel("Manchetes\npor dia")
    number_ticks(bottom.yaxis, 0)
    bottom.text(
        0.005,
        0.92,
        f"{decimal(frame['manchetes'].sum(), 0)} manchetes",
        transform=bottom.transAxes,
        ha="left",
        va="top",
        fontsize=AXIS_PT,
    )
    _years_from_january(bottom, frame["dia"].min(), frame["dia"].max())
    return fig


def _span_text(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{month_year(start)}–{month_year(end)}"


def _h2_bracket(ax: Axes, start: pd.Timestamp, end: pd.Timestamp) -> None:
    """Chave horizontal sobre o período de teste de H2 (ago/2019 a 17/11/2025)."""
    blended = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.annotate(
        "",
        xy=(mdates.date2num(start), 0.70),
        xytext=(mdates.date2num(end), 0.70),
        xycoords=blended,
        textcoords=blended,
        arrowprops={
            "arrowstyle": "|-|",
            "color": TEXT_COLOR,
            "linewidth": 1.4,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    ax.text(
        _span_center(start, end),
        0.72,
        f"H2: {month_year(start)} a {date_br(end)}",
        transform=blended,
        ha="center",
        va="bottom",
        fontsize=AXIS_PT,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )


# --------------------------------------------------------------------------- F2


def daily_sentiment(frame: pd.DataFrame) -> Figure:
    """F2: sentimento diário dos dois modelos, blocos do walk-forward e histograma ao lado."""
    fig, (series_ax, histogram_ax) = new_figure(
        full_width=True, ncols=2, sharey=True, gridspec_kw={"width_ratios": [4.2, 1]}
    )
    bins = np.linspace(frame["sentimento"].min(), frame["sentimento"].max(), 41)
    for model, color in MODEL_COLORS.items():
        series = frame[frame["modelo"] == model]
        series_ax.plot(series["dia"], series["sentimento"], color=color, linewidth=0.9, label=model)
        histogram_ax.hist(
            series["sentimento"], bins=bins, orientation="horizontal", color=color, alpha=0.55
        )
    for block, (start, end) in _date_spans(frame, "bloco").items():
        if block > 1:
            series_ax.axvline(start, color=MUTED_COLOR, linewidth=1.2, linestyle="--")
        series_ax.text(
            _span_center(start, end),
            1.0,
            f"Bloco {block}",
            transform=transforms.blended_transform_factory(
                series_ax.transData, series_ax.transAxes
            ),
            ha="center",
            va="bottom",
            fontsize=AXIS_PT,
        )
    _zero_line(series_ax)
    series_ax.set_xlim(frame["dia"].min(), frame["dia"].max())
    series_ax.set_ylabel("Sentimento diário (2p − 1)")
    number_ticks(series_ax.yaxis, 1)
    year_ticks(series_ax.xaxis)
    series_ax.legend(loc="lower left", ncols=2)
    histogram_ax.set_xlabel("Dias")
    number_ticks(histogram_ax.xaxis, 0)
    return fig


# --------------------------------------------------------------------------- F3


def walk_forward(schedule: pd.DataFrame) -> Figure:
    """F3: as 4 etapas do walk-forward, com treino e teste no eixo de datas."""
    fig, ax = new_figure(full_width=True)
    for step in schedule.to_dict("records"):
        row = step["etapa"]
        train_start, train_end = mdates.date2num([step["treino_inicio"], step["treino_fim"]])
        test_start, test_end = mdates.date2num([step["teste_inicio"], step["teste_fim"]])
        ax.barh(row, train_end - train_start, left=train_start, height=0.5, color=TRAIN_BAR_COLOR)
        ax.barh(row, test_end - test_start, left=test_start, height=0.5, color=TEST_BAR_COLOR)
        ax.text(
            (train_start + train_end) / 2,
            row,
            f"Treino: {decimal(step['pregoes_treino'], 0)} pregões",
            ha="center",
            va="center",
            fontsize=AXIS_PT,
        )
        ax.text(
            (test_start + test_end) / 2,
            row,
            f"Teste: {step['pregoes_teste']} pregões",
            ha="center",
            va="center",
            color="white",
            fontsize=AXIS_PT,
        )
        ax.text(
            (test_start + test_end) / 2,
            row + 0.29,
            _span_text(step["teste_inicio"], step["teste_fim"]),
            ha="center",
            va="top",
            fontsize=AXIS_PT,
        )
    ax.set_yticks(schedule["etapa"], [f"Etapa {step}" for step in schedule["etapa"]])
    ax.set_ylim(schedule["etapa"].max() + 0.6, schedule["etapa"].min() - 0.5)
    ax.xaxis_date()
    _years_from_january(ax, schedule["treino_inicio"].min(), schedule["teste_fim"].max())
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    return fig


# --------------------------------------------------------------------------- F4


def sentiment_scatter(frame: pd.DataFrame) -> Figure:
    """F4a: dispersão sentimento × retorno (mesmo dia e dia seguinte), reta, r e IC."""
    fig, axes = new_figure(full_width=True, ncols=2, sharey=True)
    color = MODEL_COLORS[MODEL_LABELS["logreg_l2"]]
    for ax, (timing, panel) in zip(axes, frame.groupby("retorno_de", sort=False), strict=True):
        ax.scatter(
            panel["sentimento"], panel["retorno"], s=16, color=color, alpha=0.35, edgecolors="none"
        )
        first = panel.iloc[0]
        grid = np.linspace(panel["sentimento"].min(), panel["sentimento"].max(), 50)
        (line,) = ax.plot(
            grid, first["intercepto"] + first["inclinacao"] * grid, color=TEXT_COLOR, linewidth=2.2
        )
        dots = Line2D([], [], marker="o", linestyle="", color=color, alpha=0.6)
        ax.legend(
            [dots, line],
            [
                f"Dias (n = {decimal(first['n'], 0)})",
                f"r = {decimal(first['r'])}; IC 95% {interval(first['ic95_inferior'], first['ic95_superior'])}",
            ],
            title=f"Retorno do {timing}",
            loc="lower left",
            alignment="left",
        )
        _zero_line(ax)
        ax.set_xlabel("Sentimento da Regressão Logística (2p − 1)")
        number_ticks(ax.xaxis, 1)
    axes[0].set_ylabel("Retorno diário do Ibovespa")
    number_ticks(axes[0].yaxis, 0, scale=100, suffix="%")
    return fig


def rolling_correlation(frame: pd.DataFrame) -> Figure:
    """F4b: correlação móvel (60 e 90 pregões) entre sentimento e retorno do mesmo dia."""
    fig, ax = new_figure()
    color = MODEL_COLORS[MODEL_LABELS["logreg_l2"]]
    widths = {60: (1.2, 0.55), 90: (2.4, 1.0)}
    for window in frame["janela_pregoes"].unique():
        series = frame[frame["janela_pregoes"] == window]
        linewidth, alpha = widths.get(int(window), (1.5, 1.0))
        ax.plot(
            series["dia"],
            series["correlacao"],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=f"Janela de {window} pregões",
        )
    _zero_line(ax)
    period_r = float(frame["r_periodo"].iloc[0])
    ax.axhline(
        period_r,
        color=TEXT_COLOR,
        linewidth=1.4,
        linestyle=":",
        label=f"r de todo o período: {decimal(period_r)}",
    )
    ax.set_ylabel("Correlação de Pearson")
    number_ticks(ax.yaxis, 1)
    year_ticks(ax.xaxis)
    fig.legend(loc="outside upper center", ncols=3)
    return fig


# --------------------------------------------------------------------------- F5


def roc_curves(frame: pd.DataFrame) -> Figure:
    """F5a: curvas ROC fora da amostra com a diagonal do acaso e a AUC [IC 95%]."""
    fig, ax = new_figure()
    ax.plot(
        [0, 1], [0, 1], color=MUTED_COLOR, linestyle="--", linewidth=1.4, label="Acaso (AUC 0,5)"
    )
    for model, color in MODEL_COLORS.items():
        curve = frame[frame["modelo"] == model]
        first = curve.iloc[0]
        ax.plot(
            curve["taxa_falsos_positivos"],
            curve["taxa_verdadeiros_positivos"],
            color=color,
            linewidth=2.2,
            label=f"{model}\nAUC {decimal(first['auc'])} "
            f"{interval(first['auc_ic95_inferior'], first['auc_ic95_superior'])}",
        )
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    number_ticks(ax.xaxis, 1)
    number_ticks(ax.yaxis, 1)
    ax.legend(loc="center left", bbox_to_anchor=(1.04, 0.5), labelspacing=1.0)
    return fig


def auc_by_block(frame: pd.DataFrame) -> Figure:
    """F5b: AUC de cada bloco de teste do walk-forward, com IC 95%."""
    fig, ax = new_figure()
    offsets = dict(zip(MODEL_COLORS, (-0.14, 0.14), strict=True))
    for model, color in MODEL_COLORS.items():
        rows = frame[frame["modelo"] == model]
        x = rows["bloco"] + offsets[model]
        ax.errorbar(
            x,
            rows["auc"],
            yerr=[rows["auc"] - rows["ic95_inferior"], rows["ic95_superior"] - rows["auc"]],
            fmt="o",
            markersize=9,
            capsize=6,
            elinewidth=2,
            color=color,
            label=model,
        )
        side = -1 if offsets[model] < 0 else 1
        for xi, auc in zip(x, rows["auc"], strict=True):
            _value_label(ax, decimal(auc), (xi, auc), side, color)
    _zero_line(ax, 0.5)
    blocks = frame.drop_duplicates("bloco")
    ax.set_xticks(
        blocks["bloco"],
        [
            f"Bloco {block['bloco']}\n{_span_text(block['inicio'], block['fim'])}"
            for block in blocks.to_dict("records")
        ],
    )
    ax.set_xlim(blocks["bloco"].min() - 0.6, blocks["bloco"].max() + 0.6)
    ax.set_ylabel("AUC fora da amostra")
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    number_ticks(ax.yaxis, 2)
    ax.set_ylim(top=frame["ic95_superior"].max() + 0.04)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncols=2)
    return fig


# --------------------------------------------------------------------------- F6


def event_study(frame: pd.DataFrame) -> Figure:
    """F6: CAAR (retorno anormal) por pregão após o evento, com IC 95%, por grupo."""
    fig, ax = new_figure()
    for group, color in EVENT_COLORS.items():
        rows = frame[frame["grupo"] == group]
        n_events = int(rows["n_eventos"].iloc[0])
        ax.fill_between(
            rows["tau_pregoes"],
            rows["ic95_inferior"],
            rows["ic95_superior"],
            color=color,
            alpha=CI_ALPHA,
        )
        ax.plot(
            rows["tau_pregoes"],
            rows["caar"],
            color=color,
            marker="o",
            markersize=8,
            linewidth=2.2,
            label=f"{group} ({n_events} eventos)",
        )
        for _, row in rows.iloc[[0, -1]].iterrows():
            ax.annotate(
                percent(row["caar"]),
                (row["tau_pregoes"], row["caar"]),
                xytext=(0, 12 if row["caar"] >= 0 else -12),
                textcoords="offset points",
                ha="center",
                va="bottom" if row["caar"] >= 0 else "top",
                fontsize=AXIS_PT,
                color=color,
            )
    _zero_line(ax)
    taus = sorted(frame["tau_pregoes"].unique())
    ax.set_xticks(taus)
    ax.set_xlim(min(taus) - 0.3, max(taus) + 0.3)
    ax.set_xlabel("Pregões após o evento (τ)")
    ax.set_ylabel("CAAR (retorno anormal)")
    ax.yaxis.set_major_locator(MultipleLocator(0.002))
    number_ticks(ax.yaxis, 1, scale=100, suffix="%")
    ax.legend(loc="lower left")
    return fig


# --------------------------------------------------------------------------- F7


def equity_limits(*frames: pd.DataFrame) -> tuple[float, float]:
    """Limites comuns do eixo de patrimônio (as duas versões da F7 ficam comparáveis)."""
    values = pd.concat([frame["patrimonio"] for frame in frames])
    spread = values.max() - values.min()
    return float(values.min() - 0.32 * spread), float(values.max() + 0.06 * spread)


def equity_curves(frame: pd.DataFrame, limits: tuple[float, float]) -> Figure:
    """F7: patrimônio das estratégias e do Ibovespa, começando em 1,0."""
    fig, ax = new_figure()
    colors = {**MODEL_COLORS, slide_data.IBOVESPA: IBOVESPA_COLOR}
    for name, color in colors.items():
        curve = frame[frame["serie"] == name]
        first = curve.iloc[0]
        ax.plot(
            curve["data"],
            curve["patrimonio"],
            color=color,
            linewidth=2.6 if name == slide_data.IBOVESPA else 1.8,
            label=f"{name}: CAGR {percent(first['cagr'])}; Sharpe {decimal(first['sharpe'])}",
        )
    _zero_line(ax, 1.0)
    ax.set_ylim(*limits)
    ax.set_ylabel("Patrimônio (início = 1,0)")
    number_ticks(ax.yaxis, 1)
    year_ticks(ax.xaxis)
    ax.legend(loc="lower right")
    return fig


# --------------------------------------------------------------------------- F8


def sharpe_heatmaps(frame: pd.DataFrame) -> Figure:
    """F8: Sharpe das 12 configurações, com os limiares do artigo e sem look-ahead."""
    fig, axes = new_figure(full_width=True, ncols=2, sharey=True)
    benchmark = float(frame["sharpe_ibovespa"].iloc[0])
    rows = [
        (model, quantile)
        for model in MODEL_COLORS
        for quantile in sorted(frame["quantil"].unique())
    ]
    lags = sorted(frame["lag"].unique())
    images = []
    for ax, (threshold, panel) in zip(axes, frame.groupby("limiar", sort=False), strict=True):
        grid = (
            panel.set_index(["modelo", "quantil", "lag"])["sharpe"]
            .unstack("lag")
            .reindex(pd.MultiIndex.from_tuples(rows))
        )
        images.append(
            ax.imshow(
                grid.to_numpy(), cmap="RdBu", vmin=-HEATMAP_LIMIT, vmax=HEATMAP_LIMIT, aspect="auto"
            )
        )
        for (row, column), value in np.ndenumerate(grid.to_numpy()):
            ax.text(
                column,
                row,
                decimal(value),
                ha="center",
                va="center",
                fontsize=AXIS_PT,
                color="white" if abs(value) > 0.3 else TEXT_COLOR,
            )
            if value > benchmark:
                ax.add_patch(
                    Rectangle(
                        (column - 0.5, row - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor=TEXT_COLOR,
                        linewidth=3,
                    )
                )
        ax.set_xticks(range(len(lags)), [f"lag {lag}" for lag in lags])
        label = str(threshold)
        ax.set_xlabel(label[0].upper() + label[1:])
        ax.grid(visible=False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_yticks(
        range(len(rows)), [f"{model}\nq = {decimal(quantile, 2)}" for model, quantile in rows]
    )
    colorbar = fig.colorbar(images[-1], ax=axes, shrink=0.9, pad=0.02)
    colorbar.set_label("Sharpe")
    number_ticks(colorbar.ax.yaxis, 1)
    colorbar.ax.axhline(benchmark, color=TEXT_COLOR, linewidth=3)
    fig.legend(
        [
            Line2D([], [], color=TEXT_COLOR, linewidth=3),
            Patch(facecolor="none", edgecolor=TEXT_COLOR, linewidth=3),
        ],
        [
            f"Ibovespa buy-and-hold no mesmo período: Sharpe {decimal(benchmark)}",
            "Configuração com Sharpe acima do Ibovespa",
        ],
        loc="outside lower center",
        ncols=2,
    )
    return fig


# --------------------------------------------------------------------------- F9


def none_token_auc(frame: pd.DataFrame) -> Figure:
    """F9: AUC com e sem o token espúrio "None", por modelo, com IC 95%."""
    fig, ax = new_figure()
    offsets = dict(zip(VERSION_MARKERS, (-0.12, 0.12), strict=True))
    for position, (model, color) in enumerate(MODEL_COLORS.items()):
        for version, marker in VERSION_MARKERS.items():
            row = frame[(frame["modelo"] == model) & (frame["versao"] == version)].iloc[0]
            x = position + offsets[version]
            ax.errorbar(
                x,
                row["auc"],
                yerr=[[row["auc"] - row["ic95_inferior"]], [row["ic95_superior"] - row["auc"]]],
                fmt=marker,
                markersize=10,
                capsize=6,
                elinewidth=2,
                color=color,
                markerfacecolor=color if version == slide_data.WITH_NONE else "white",
                markeredgewidth=2,
            )
            side = -1 if offsets[version] < 0 else 1
            _value_label(ax, decimal(row["auc"]), (x, row["auc"]), side, color)
    _zero_line(ax, 0.5)
    ax.set_xticks(range(len(MODEL_COLORS)), list(MODEL_COLORS))
    ax.set_xlim(-0.6, len(MODEL_COLORS) - 0.4)
    ax.set_ylabel("AUC fora da amostra")
    number_ticks(ax.yaxis, 2)
    ax.grid(axis="x", visible=False)
    handles = [
        Line2D(
            [],
            [],
            marker=marker,
            linestyle="",
            markersize=10,
            color=MUTED_COLOR,
            markerfacecolor=MUTED_COLOR if version == slide_data.WITH_NONE else "white",
            markeredgewidth=2,
        )
        for version, marker in VERSION_MARKERS.items()
    ]
    ax.legend(handles, list(VERSION_MARKERS), loc="upper center")
    return fig
