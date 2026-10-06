"""Tabelas e figuras do artigo, gravadas em `reports/figures/` (CSV com os dados e PNG).

Os nomes dos arquivos seguem a numeração do artigo: Tabelas 1 a 4 e Figuras 1 a 8; a
Figura 9 é a robustez da correlação móvel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from tcc.config import MODEL_LABELS

DPI = 300
FIGURE_4_MODEL = "logreg_l2"
ROLLING_WINDOWS = (60, 90)
ROBUSTNESS_WINDOWS = (30, 60, 90)
EXTREME_EVENT_QUANTILE = 0.9


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def write_table(frame: pd.DataFrame, output_dir: Path, stem: str, title: str) -> None:
    """Grava a tabela em CSV e uma imagem dela em PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{stem}.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 0.6 + 0.35 * len(frame)), dpi=DPI)
    ax.axis("off")
    table = ax.table(
        cellText=frame.astype(str).to_numpy().tolist(), colLabels=list(frame.columns), loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, output_dir / f"{stem}.png")


def format_decimals(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Formata colunas numéricas com 3 casas (como nas Tabelas 3 e 4 do artigo)."""
    formatted = frame.copy()
    for column in columns:
        formatted[column] = formatted[column].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    return formatted


def _time_axis(ax: Any, fig: Any, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("Data")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()


def figure_1_ibovespa_events(returns: pd.DataFrame, events: pd.DataFrame, output_dir: Path) -> None:
    """Ibovespa com os eventos cujo |CAR| está no percentil 90 ou acima, por polaridade."""
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=DPI)
    ax.plot(returns["day"], returns["close"], color="#1f77b4", label="Ibovespa (close)")
    threshold = events["car_max_abs"].quantile(EXTREME_EVENT_QUANTILE)
    extreme = events[events["car_max_abs"] >= threshold].merge(
        returns[["day", "close"]], left_on="event_day", right_on="day", how="left"
    )
    extreme = extreme.dropna(subset=["close"])
    for polarity, marker, color, label in (
        ("pos", "^", "green", "Sentimento Positivo Extremo"),
        ("neg", "v", "red", "Sentimento Negativo Extremo"),
    ):
        subset = extreme[extreme["polarity"] == polarity]
        if not subset.empty:
            ax.scatter(
                subset["event_day"],
                subset["close"],
                marker=marker,
                color=color,
                label=label,
                zorder=5,
            )
    ax.legend(loc="upper left", framealpha=0.9)
    _time_axis(ax, fig, "Figura 1 – Ibovespa com Eventos", "Pontos do Ibovespa")
    _save(fig, output_dir / "Figura_1_ibov_eventos.png")


def figure_2_daily_sentiment(sentiment: pd.DataFrame, output_dir: Path) -> None:
    """Sentimento diário (2p − 1) de cada modelo."""
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=DPI)
    for model, frame in sentiment.groupby("model"):
        ax.plot(frame["day"], frame["sentiment"], label=MODEL_LABELS.get(str(model), str(model)))
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.legend()
    _time_axis(ax, fig, "Figura 2 – Sentimento diário por modelo", "Sentimento diário (2p − 1)")
    _save(fig, output_dir / "Figura_2_sentimento_medio_diario.png")


def figure_3_model_comparison(summary: pd.DataFrame, output_dir: Path) -> None:
    """Sharpe da regra de limiar fixo por modelo (com o CSV de Sharpe e CAGR)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "Figura_3_comparativo_modelos.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=DPI)
    labels = [MODEL_LABELS.get(model, model) for model in summary["model"]]
    bars = ax.bar(labels, summary["sharpe"], color=["#2ca02c", "#1f77b4"])
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3 if height >= 0 else -12),
            textcoords="offset points",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_ylabel("Sharpe")
    strategy = summary["strategy"].iloc[0]
    ax.set_title(f"Figura 3 – Comparativo de Modelos (Sharpe) | Estratégia: {strategy}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, output_dir / "Figura_3_comparativo_modelos.png")


def _sentiment_and_returns(sentiment: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Sentimento do modelo da Figura 4 ao lado do retorno do mesmo dia (D−1 → D)."""
    model = sentiment[sentiment["model"] == FIGURE_4_MODEL][["day", "sentiment"]]
    return model.merge(returns[["day", "ret"]], on="day", how="inner").sort_values("day")


def figure_4_correlation(sentiment: pd.DataFrame, returns: pd.DataFrame) -> float:
    """R de Pearson entre o sentimento da Regressão Logística e o retorno do mesmo dia."""
    merged = _sentiment_and_returns(sentiment, returns).dropna()
    return float(merged["sentiment"].corr(merged["ret"]))


def figure_4_scatter(sentiment: pd.DataFrame, returns: pd.DataFrame, output_dir: Path) -> None:
    """Dispersão sentimento × retorno do mesmo dia, com o r de Pearson no título."""
    merged = _sentiment_and_returns(sentiment, returns).dropna()
    correlation = figure_4_correlation(sentiment, returns)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=DPI)
    ax.scatter(merged["sentiment"], merged["ret"], alpha=0.35, edgecolor="k", s=24)
    ax.set_title(f"Figura 4 – Dispersão Sentimento × Retorno (r={correlation:.2f})")
    ax.set_xlabel(f"Sentimento ({MODEL_LABELS[FIGURE_4_MODEL]})")
    ax.set_ylabel("Retorno diário do Ibovespa")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, output_dir / "Figura_4_dispersao_sentimento_retorno.png")


def _rolling_correlation_figure(
    sentiment: pd.DataFrame,
    returns: pd.DataFrame,
    windows: tuple[int, ...],
    title: str,
    label: str,
    path: Path,
) -> None:
    merged = _sentiment_and_returns(sentiment, returns)
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=DPI)
    for window in windows:
        rolling = merged["sentiment"].rolling(window).corr(merged["ret"])
        ax.plot(merged["day"], rolling, label=label.format(window=window))
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.legend()
    _time_axis(ax, fig, title, "Correlação de Pearson")
    _save(fig, path)


def figure_5_rolling_correlation(
    sentiment: pd.DataFrame, returns: pd.DataFrame, output_dir: Path
) -> None:
    """Correlação móvel de 60 e 90 dias entre sentimento e retorno do mesmo dia."""
    _rolling_correlation_figure(
        sentiment,
        returns,
        ROLLING_WINDOWS,
        "Figura 5 – Correlação móvel (sentimento x retorno)",
        "Correlação {window}d",
        output_dir / "Figura_5_correlacao_movel_60d_90d.png",
    )


def figure_9_rolling_robustness(
    sentiment: pd.DataFrame, returns: pd.DataFrame, output_dir: Path
) -> None:
    """Correlação móvel com janelas de 30, 60 e 90 dias (robustez da Figura 5)."""
    _rolling_correlation_figure(
        sentiment,
        returns,
        ROBUSTNESS_WINDOWS,
        "Figura 9 – Robustez da correlação móvel (sentimento x retorno)",
        "Corr {window}d",
        output_dir / "Figura_9_robustez_correlacao.png",
    )


def figure_6_distribution(sentiment: pd.DataFrame, output_dir: Path) -> None:
    """Histograma do sentimento diário dos dois modelos."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=DPI)
    ax.hist(sentiment["sentiment"], bins=40, color="#1f77b4", alpha=0.75, edgecolor="white")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Figura 6 – Distribuição do sentimento diário (todos os modelos)")
    ax.set_xlabel("Sentimento")
    ax.set_ylabel("Frequência")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    _save(fig, output_dir / "Figura_6_distribuicao_sentimento.png")


def figure_7a_car_boxplot(events: pd.DataFrame, output_dir: Path) -> None:
    """Distribuição do CAR bruto de D0 a D5 por polaridade."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=DPI)
    groups = [events.loc[events["polarity"] == pol, "car_value"].dropna() for pol in ("pos", "neg")]
    ax.boxplot(
        groups,
        tick_labels=["pos", "neg"],
        showmeans=True,
        meanline=True,
        patch_artist=True,
        boxprops={"facecolor": "#a6cee3"},
        medianprops={"color": "black"},
        meanprops={"color": "red"},
    )
    ax.set_title("Figura 7A – CAR por polaridade (boxplot)")
    ax.set_xlabel("Polaridade")
    ax.set_ylabel("CAR (retorno acumulado bruto D0–D5)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    _save(fig, output_dir / "Figura_7A_latencia_boxplot.png")


def figure_7b_caar(caar: pd.DataFrame, output_dir: Path) -> None:
    """CAAR por τ (dias corridos) com IC 95%, e o CSV com os valores."""
    output_dir.mkdir(parents=True, exist_ok=True)
    caar.to_csv(output_dir / "Figura_7B_event_time_CAAR.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)
    for polarity, color in (("neg", "#d62728"), ("pos", "#2ca02c")):
        ax.plot(caar["tau"], caar[f"caar_{polarity}_mean"], marker="o", label=polarity, color=color)
        ax.fill_between(
            caar["tau"],
            caar[f"caar_{polarity}_ci_low"],
            caar[f"caar_{polarity}_ci_high"],
            alpha=0.2,
            color=color,
        )
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_title("Figura 7B – CAAR por tempo de evento (IC 95%)")
    ax.set_xlabel("Dias corridos após o evento (τ)")
    ax.set_ylabel("CAAR (retorno acumulado bruto, sem ajuste)")
    ax.legend(title="Polaridade")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, output_dir / "Figura_7B_event_time_CAAR.png")


def figure_8_backtest(curves: pd.DataFrame, strategy: str, output_dir: Path) -> None:
    """Curvas da regra de limiar fixo e do Ibovespa, normalizadas em 1, e o CSV delas."""
    output_dir.mkdir(parents=True, exist_ok=True)
    curves.to_csv(output_dir / "Figura_8_backtest_vs_benchmark.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=DPI)
    for model, label in MODEL_LABELS.items():
        ax.plot(curves["date"], curves[f"equity_{model}"], label=f"{label} ({strategy})")
    ax.plot(
        curves["date"], curves["equity_ibov"], label="Ibov buy&hold", color="black", linestyle="--"
    )
    ax.legend()
    _time_axis(
        ax,
        fig,
        "Figura 8 – Curva de backtest vs benchmark (normalizadas em 1.0)",
        "Equity normalizado",
    )
    _save(fig, output_dir / "Figura_8_backtest_vs_benchmark.png")
