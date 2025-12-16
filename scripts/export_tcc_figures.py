"""
Gera figuras e tabelas finais para o TCC:
- Figura_1_ibov_eventos.png
- Figura_7A_latencia_boxplot.png
- Figura_7B_eventstudy_CAR.png
- Figura_8_backtest_vs_benchmark.png
- Tabela_1_metricas.csv / .png
- Tabela_intersecao_periodo.csv / .png
- nota_tabela1.txt

Requisitos: dados em C:\\TCC_USP\\data_processed (não versionados).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DATA = Path(r"C:\TCC_USP\data_processed")
OUTPUT_DIR = Path("reports") / "figures_tcc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OFFICIAL_START = pd.Timestamp("2018-01-02")
OFFICIAL_END = pd.Timestamp("2024-12-31")
PREFERRED_STRATEGY = "long_only_60"
FALLBACK_STRATEGY = "long_only_55"
ANCHOR_MODELS = ["logreg_l2", "rf_200"]


def load_ibov() -> pd.DataFrame:
    df = pd.read_csv(BASE_DATA / "ibovespa_clean.csv")
    date_col = "day" if "day" in df.columns else "date"
    df["day"] = pd.to_datetime(df[date_col])
    if "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]
    df = df.sort_values("day").loc[
        (df["day"] >= OFFICIAL_START) & (df["day"] <= OFFICIAL_END)
    ]
    return df[["day", "close"]].dropna()


def load_events() -> pd.DataFrame:
    df = pd.read_csv(BASE_DATA / "event_study_latency.csv")
    df["event_day"] = pd.to_datetime(df["event_day"])
    df["polarity"] = df["event_name"].str.contains("pos").map({True: "pos", False: "neg"})
    return df


def load_sentiment() -> pd.DataFrame:
    df = pd.read_csv(BASE_DATA / "16_oof_predictions.csv")
    date_col = "day" if "day" in df.columns else "date"
    df[date_col] = pd.to_datetime(df[date_col])
    agg = df.groupby(date_col).agg(sentiment=("proba", lambda s: (s * 2 - 1).mean()))
    agg.index.name = "day"
    return agg.reset_index()


def load_results16() -> pd.DataFrame:
    path = BASE_DATA / "results_16_models_tfidf.json"
    rows = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for model, vals in data.get("models", {}).items():
            rows.append(
                {
                    "model": model,
                    "dataset": "tfidf_daily",
                    "auc": vals["auc"]["value"],
                    "mda": vals["mda"]["value"],
                }
            )
    return pd.DataFrame(rows)


def load_backtest_results() -> pd.DataFrame:
    df = pd.read_csv(BASE_DATA / "18_backtest_results.csv")
    df["dataset"] = "backtest_daily"
    return df


def choose_common_strategy(df: pd.DataFrame) -> Tuple[str | None, set[str]]:
    if df.empty:
        return None, set()
    common: set[str] | None = None
    for m in ANCHOR_MODELS:
        strategies = set(df.loc[df["model"] == m, "strategy"])
        common = strategies if common is None else common & strategies
    common = common or set()
    if not common:
        return None, set()
    if PREFERRED_STRATEGY in common:
        return PREFERRED_STRATEGY, common
    if FALLBACK_STRATEGY in common:
        return FALLBACK_STRATEGY, common
    return sorted(common)[0], common


def figure_ibov_events(ibov: pd.DataFrame, events: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.plot(ibov["day"], ibov["close"], color="#1f77b4", label="Ibovespa (close)")
    if not events.empty:
        ax.vlines(events["event_day"], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], colors="tomato", alpha=0.35, linewidth=1.2, label="Eventos (sentimento)")
        # Scatter nos valores existentes
        merged = events.merge(ibov[["day", "close"]], left_on="event_day", right_on="day", how="left")
        ax.scatter(merged["event_day"], merged["close"], color="tomato", edgecolor="k", zorder=3, s=24)
    ax.set_title("Figura 1 – Ibovespa com Eventos")
    ax.set_xlabel("Data")
    ax.set_ylabel("Pontos do Ibovespa")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "Figura_1_ibov_eventos.png", dpi=300)
    plt.close(fig)


def figure_latency(events: pd.DataFrame):
    events = events.copy()
    events["window"] = events["window"].astype(str)
    # 7A box/violin por polaridade
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    groups = [events.loc[events["polarity"] == pol, "car_value"].dropna() for pol in ["pos", "neg"]]
    ax.boxplot(groups, labels=["pos", "neg"], showmeans=True, meanline=True, patch_artist=True,
               boxprops=dict(facecolor="#a6cee3"), medianprops=dict(color="black"), meanprops=dict(color="red"))
    ax.set_title("Figura 7A – Latência / CAR por polaridade")
    ax.set_xlabel("Polaridade")
    ax.set_ylabel("CAR")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "Figura_7A_latencia_boxplot.png", dpi=300)
    plt.close(fig)

    # 7B CAR médio por window e polaridade
    grp = events.groupby(["window", "polarity"]).agg(
        mean_car=("car_value", "mean"),
        sem_car=("car_value", lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0),
    )
    grp = grp.reset_index()
    # ordenar janela pelo sufixo numérico
    grp["w_order"] = grp["window"].str.extract(r"(\d+)").astype(int)
    grp = grp.sort_values("w_order")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    for pol, df_pol in grp.groupby("polarity"):
        ax.errorbar(df_pol["window"], df_pol["mean_car"], yerr=df_pol["sem_car"], marker="o", label=pol)
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_title("Figura 7B – CAR médio por janela (evento) e polaridade")
    ax.set_xlabel("Janela do evento")
    ax.set_ylabel("CAR médio")
    ax.legend(title="Polaridade")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "Figura_7B_eventstudy_CAR.png", dpi=300)
    plt.close(fig)


def table_metrics(results16: pd.DataFrame, backtest: pd.DataFrame, common_strategy: str | None):
    # tfidf (AUC/MDA)
    tfidf = results16.copy()
    tfidf["cagr"] = "—"
    tfidf["sharpe"] = "—"
    tfidf["strategy"] = "—"

    # backtest (Sharpe/CAGR) na estratégia comum
    back = pd.DataFrame()
    if common_strategy:
        mask = (
            (backtest["dataset"] == "backtest_daily")
            & (backtest["strategy"] == common_strategy)
            & (backtest["model"].isin(ANCHOR_MODELS))
        )
        back = backtest.loc[mask, ["model", "dataset", "strategy", "cagr", "sharpe"]].copy()
        back["auc"] = "—"
        back["mda"] = "—"
    table = pd.concat([tfidf[["model", "dataset", "auc", "mda", "strategy", "cagr", "sharpe"]], back], ignore_index=True)
    table.to_csv(OUTPUT_DIR / "Tabela_1_metricas.csv", index=False)

    # nota
    note = (
        "Nota: AUC/MDA são métricas de classificação em tfidf_daily; backtest_daily reporta métricas econômicas (CAGR/Sharpe). "
        "Sharpe calculado sobre retornos diários (convenção 252); custos/atrito não modelados (implícitos = 0). "
        f"Estratégia comparada nos modelos de backtest: {common_strategy or '—'}."
    )
    (OUTPUT_DIR / "nota_tabela1.txt").write_text(note, encoding="utf-8")

    # PNG simples
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    ax.axis("off")
    tbl = ax.table(cellText=table.values, colLabels=table.columns, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "Tabela_1_metricas.png", dpi=300)
    plt.close(fig)


def figure_backtest_vs_benchmark(curves: pd.DataFrame, ibov: pd.DataFrame, strategy: str | None):
    if curves.empty or strategy is None:
        return
    # normalizar benchmark
    ibov = ibov.copy()
    ibov = ibov.sort_values("day")
    ibov["bench_equity"] = ibov["close"] / ibov["close"].iloc[0]

    # filtrar modelos/estratégia
    date_col = "day" if "day" in curves.columns else "date"
    curves["day"] = pd.to_datetime(curves[date_col])
    mask = (curves["strategy"] == strategy) & (curves["model"].isin(ANCHOR_MODELS))
    sel = curves.loc[mask, ["day", "model", "equity"]].copy()
    if sel.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    for model, dfm in sel.groupby("model"):
        dfm = dfm.sort_values("day")
        ax.plot(dfm["day"], dfm["equity"], label=f"{model} ({strategy})")
    # benchmark limitado ao período do backtest
    start = sel["day"].min()
    bench = ibov.loc[ibov["day"] >= start]
    ax.plot(bench["day"], bench["bench_equity"], label="Ibov buy&hold", color="black", linestyle="--")
    ax.set_title("Figura 8 – Curva de backtest vs benchmark")
    ax.set_ylabel("Equity (normalizado)")
    ax.set_xlabel("Data")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "Figura_8_backtest_vs_benchmark.png", dpi=300)
    plt.close(fig)


def table_intersection(ibov: pd.DataFrame, sent: pd.DataFrame):
    ibov_days = set(ibov["day"])
    sent_days = set(sent["day"])
    inter = ibov_days & sent_days
    data = {
        "Conjunto": ["Pregões Ibov", "Dias com sentimento", "Interseção"],
        "Dias": [len(ibov_days), len(sent_days), len(inter)],
    }
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_DIR / "Tabela_intersecao_periodo.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=300)
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "Tabela_intersecao_periodo.png", dpi=300)
    plt.close(fig)


def main():
    ibov = load_ibov()
    events = load_events()
    sentiment = load_sentiment()
    results16 = load_results16()
    backtest = load_backtest_results()
    backtest_curves = pd.read_csv(BASE_DATA / "18_backtest_daily_curves.csv")

    # Estratégia comum para Sharpe
    common_strategy, strategies_set = choose_common_strategy(
        backtest.loc[backtest["dataset"] == "backtest_daily"]
    )
    print(f"[EXPORT] Estratégias comuns: {strategies_set} | escolhida: {common_strategy}")

    figure_ibov_events(ibov, events)
    figure_latency(events)
    table_metrics(results16, backtest, common_strategy)
    figure_backtest_vs_benchmark(backtest_curves, ibov, common_strategy)
    table_intersection(ibov, sentiment)


if __name__ == "__main__":
    main()
