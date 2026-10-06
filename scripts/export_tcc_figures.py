"""
Pipeline headless para gerar TODAS as figuras/tabelas finais (PNG) do TCC.
- Limpa outputs antigos em reports/figures (PNG/CSV/TXT)
- Usa backend headless Agg
- Recalcula backtest diário mark-to-market a partir de 16_oof_predictions (estratégias long_only_60/55/long_short_sym)
- Valida mtime/size dos PNGs e fail se desatualizado
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io import paths  # noqa: E402

BASE_DATA = paths.DATA_PROCESSED  # respeita TCC_USP_BASE (ver .env.example)
OUTPUT_DIR = (REPO_ROOT / "reports" / "figures").resolve()
FORCE_OVERWRITE = True
OFFICIAL_START = pd.Timestamp("2018-01-02")
OFFICIAL_END = pd.Timestamp("2024-12-31")
ANCHOR_MODELS = ["logreg_l2", "rf_200"]
# Nomes exibidos nas figuras: são os dois classificadores (não há agregação por volume).
MODEL_DISPLAY_NAMES = {
    "logreg_l2": "Regressão Logística",
    "rf_200": "Random Forest",
}
BOOTSTRAP_SEED = 42  # semente do bootstrap da Fig. 7B (no artigo, o bootstrap não tinha semente)
# ordem de preferência; caímos para a próxima se a curva diária ficar peça (nunique<=200)
STRATEGIES_CFG = [
    {
        "name": "long_only_60",
        "long_th": 0.60,
        "short_th": 0.40,
        "allow_short": False,
        "cost": 0.0005,
    },
    {
        "name": "long_only_55",
        "long_th": 0.55,
        "short_th": 0.45,
        "allow_short": False,
        "cost": 0.0005,
    },
    {
        "name": "long_short_sym",
        "long_th": 0.55,
        "short_th": 0.45,
        "allow_short": True,
        "cost": 0.0007,
    },
]
REQUIRED_PNGS = [
    "Figura_1_ibov_eventos.png",
    "Figura_2_sentimento_medio_diario.png",
    "Figura_3_comparativo_modelos.png",
    "Figura_4_dispersao_sentimento_retorno.png",
    "Figura_5_correlacao_movel_60d_90d.png",
    "Figura_6_distribuicao_sentimento.png",
    "Figura_7A_latencia_boxplot.png",
    "Figura_7B_event_time_CAAR.png",
    "Figura_8_backtest_vs_benchmark.png",
    "Tabela_1_metricas.png",
    "Tabela_intersecao_periodo.png",
]


def _capture_prev_mtimes() -> dict[str, float]:
    prev = {}
    for name in REQUIRED_PNGS:
        path = (OUTPUT_DIR / name).resolve()
        if path.exists():
            prev[name] = path.stat().st_mtime
    return prev


def _clamp_period(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df.loc[(df[col] >= OFFICIAL_START) & (df[col] <= OFFICIAL_END)].dropna(subset=[col])


def _clean_outputs() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    removed = 0
    for pattern in ("*.png", "*.csv", "*.txt"):
        for f in OUTPUT_DIR.glob(pattern):
            f.unlink(missing_ok=True)
            if f.exists():
                raise RuntimeError(f"Falha ao apagar antigo: {f}")
            removed += 1
    print(f"[CLEAN] Removidos {removed} arquivos em {OUTPUT_DIR}")
    return removed


def _savefig(fig, path: Path, force: bool | None = None) -> None:
    if force is None:
        force = FORCE_OVERWRITE
    path.parent.mkdir(parents=True, exist_ok=True)
    prev_mtime = path.stat().st_mtime if path.exists() else 0
    prev_size = path.stat().st_size if path.exists() else 0
    if (not force) and path.exists():
        print(f"[SAVEFIG] skip (force=False) {path} | size={prev_size} | mtime={prev_mtime}")
        plt.close(fig)
        return
    if path.exists():
        path.unlink()
        if path.exists():
            raise RuntimeError(f"Falha ao remover destino antes de salvar: {path}")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if (not path.exists()) or path.stat().st_size == 0:
        raise RuntimeError(f"Falha ao salvar figura: {path}")
    stat = path.stat()
    new_size = stat.st_size
    new_mtime = stat.st_mtime
    ts = datetime.fromtimestamp(new_mtime)
    print(
        f"[SAVEFIG] {path} | bytes={new_size} (prev {prev_size}) | mtime={ts} (prev {prev_mtime})"
    )


def _validate_png_mtimes(start_ts: float, prev_mtimes: dict[str, float]) -> None:
    print("\n[VALIDAÇÃO PNGs]")
    ok_count = 0
    rows = []
    for name in REQUIRED_PNGS:
        path = (OUTPUT_DIR / name).resolve()
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        mtime = path.stat().st_mtime if exists else 0
        prev_mt = prev_mtimes.get(name, 0)
        ok = exists and size > 30_000 and mtime >= start_ts - 2 and mtime > prev_mt
        rows.append((path, size, mtime, ok))
        if not ok:
            raise RuntimeError(
                f"PNG inválido/desatualizado: {path} | size={size} | mtime={mtime} | prev_mtime={prev_mt}"
            )
        ok_count += 1
    print(f"[VALIDAÇÃO] {ok_count}/{len(REQUIRED_PNGS)} regravados")
    print("arquivo | bytes | mtime | ok")
    for path, size, mtime, ok in rows:
        ts = datetime.fromtimestamp(mtime)
        print(f"{path} | bytes={size} | mtime={ts} | ok={ok}")


def load_ibov() -> pd.DataFrame:
    df = pd.read_csv(BASE_DATA / "ibovespa_clean.csv")
    date_col = "day" if "day" in df.columns else "date"
    df["day"] = pd.to_datetime(df[date_col])
    if "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]
    df = df.sort_values("day")
    df["ret"] = df["close"].pct_change()
    return _clamp_period(df[["day", "close", "ret"]].dropna(subset=["close"]), "day")


def load_events() -> pd.DataFrame:
    df = pd.read_csv(BASE_DATA / "event_study_latency.csv")
    df["event_day"] = pd.to_datetime(df["event_day"])
    df = _clamp_period(df, "event_day")
    df["polarity"] = df["event_name"].str.contains("pos").map({True: "pos", False: "neg"})
    return df


def load_sentiment() -> pd.DataFrame:
    df = pd.read_csv(BASE_DATA / "16_oof_predictions.csv")
    df["day"] = pd.to_datetime(df["day"])
    df = _clamp_period(df, "day")
    df["sentiment"] = df["proba"] * 2 - 1
    return df


def load_sentiment_daily(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["day", "model"])
        .agg(sentiment=("sentiment", "mean"), n_obs=("sentiment", "size"))
        .reset_index()
    )


def load_results16() -> pd.DataFrame:
    path = BASE_DATA / "results_16_models_tfidf.json"
    rows = []
    if path.exists():
        with open(path, encoding="utf-8") as fh:
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


def load_oof_predictions() -> pd.DataFrame:
    df = pd.read_csv(BASE_DATA / "16_oof_predictions.csv")
    df["day"] = pd.to_datetime(df["day"])
    df = _clamp_period(df, "day")
    return df


def _run_strategy_from_oof(oof: pd.DataFrame, cfg: dict[str, float]) -> pd.DataFrame:
    df = oof.copy().reset_index(drop=True)
    long_th = cfg["long_th"]
    short_th = cfg["short_th"]
    allow_short = cfg.get("allow_short", True)
    cost = cfg.get("cost", 0.0005)

    positions = []
    turnovers = []
    pos_prev = 0
    for proba in df["proba"]:
        if proba >= long_th:
            pos = 1
        elif allow_short and proba <= short_th:
            pos = -1
        elif (not allow_short) and proba <= short_th:
            pos = 0
        else:
            pos = pos_prev  # mantém posição até novo gatilho
        turnovers.append(abs(pos - pos_prev))
        positions.append(pos)
        pos_prev = pos

    df["signal"] = positions
    df["turnover"] = turnovers
    df["cost"] = df["turnover"] * cost
    df["strategy_ret"] = df["signal"].shift(1, fill_value=0) * df["ret_next"].fillna(0) - df["cost"]
    df["equity"] = (1 + df["strategy_ret"]).cumprod()
    return df


def compute_backtest_mark_to_market(
    oof: pd.DataFrame, ibov: pd.DataFrame, strategy_name: str
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], str]:
    cfg = next((c for c in STRATEGIES_CFG if c["name"] == strategy_name), None)
    if cfg is None:
        raise RuntimeError(f"Estratégia {strategy_name} não configurada.")

    records = []
    stats: dict[str, dict[str, float]] = {}
    for model in ANCHOR_MODELS:
        df_model = oof[oof["model"] == model].copy()
        df_model = df_model.sort_values("day")
        if df_model.empty:
            continue
        res = _run_strategy_from_oof(df_model, cfg)
        records.append(res.assign(model=model, strategy=strategy_name))
    if not records:
        raise RuntimeError(f"Nenhum dado para estratégia {strategy_name}.")

    daily = pd.concat(records, ignore_index=True)
    # alinhar datas com Ibov
    common_dates = set(daily["day"]).intersection(set(ibov["day"]))
    if not common_dates:
        raise RuntimeError(f"Sem interseção de datas entre backtest e Ibov para {strategy_name}.")
    common_dates = sorted(common_dates)
    bench = ibov.set_index("day").loc[common_dates, "ret"].fillna(0)
    equity_df = pd.DataFrame({"date": common_dates})
    bench_eq = (1 + bench).cumprod()
    bench_eq = bench_eq / bench_eq.iloc[0]
    equity_df["equity_ibov"] = bench_eq.values

    for model in ANCHOR_MODELS:
        sub = daily[(daily["model"] == model) & (daily["day"].isin(common_dates))].sort_values(
            "day"
        )
        if sub.empty:
            raise RuntimeError(f"Sem dados para {model} em {strategy_name}.")
        eq = (1 + sub["strategy_ret"].fillna(0)).cumprod()
        eq = eq / eq.iloc[0]
        equity_df[f"equity_{model}"] = eq.values
        ret = sub["strategy_ret"].fillna(0)
        cagr = eq.iloc[-1] ** (252 / len(eq)) - 1 if len(eq) > 1 else np.nan
        sharpe = ret.mean() / ret.std(ddof=0) * np.sqrt(252) if ret.std(ddof=0) != 0 else np.nan
        stats[model] = {"cagr": cagr, "sharpe": sharpe, "nunique_equity": eq.nunique()}
        if eq.nunique() <= 200:
            raise RuntimeError(
                f"Curva diária insuficiente para {model} na estratégia {strategy_name}: {eq.nunique()} valores únicos."
            )
    return equity_df, stats, strategy_name


def figure_ibov_events(ibov: pd.DataFrame, events: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    ax.plot(ibov["day"], ibov["close"], color="#1f77b4", label="Ibovespa (close)")
    if not events.empty:
        thresh = events["car_max_abs"].quantile(0.9)
        extreme = events.loc[events["car_max_abs"] >= thresh].copy()
        if not extreme.empty:
            merged = extreme.merge(
                ibov[["day", "close"]], left_on="event_day", right_on="day", how="left"
            ).dropna(subset=["close"])
            pos = merged[merged["polarity"] == "pos"]
            neg = merged[merged["polarity"] == "neg"]
            if not pos.empty:
                ax.scatter(
                    pos["event_day"],
                    pos["close"],
                    marker="^",
                    color="green",
                    label="Sentimento Positivo Extremo",
                    zorder=5,
                )
            if not neg.empty:
                ax.scatter(
                    neg["event_day"],
                    neg["close"],
                    marker="v",
                    color="red",
                    label="Sentimento Negativo Extremo",
                    zorder=5,
                )
    ax.set_title("Figura 1 – Ibovespa com Eventos")
    ax.set_xlabel("Data")
    ax.set_ylabel("Pontos do Ibovespa")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    _savefig(fig, OUTPUT_DIR / "Figura_1_ibov_eventos.png")


def figure_sentiment_daily(sent_daily: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    for model, dfm in sent_daily.groupby("model"):
        ax.plot(dfm["day"], dfm["sentiment"], label=MODEL_DISPLAY_NAMES.get(model, model))
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Figura 2 – Sentimento diário por modelo")
    ax.set_xlabel("Data")
    ax.set_ylabel("Sentimento diário (2p − 1)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    _savefig(fig, OUTPUT_DIR / "Figura_2_sentimento_medio_diario.png")


def figure_scatter(sent_daily: pd.DataFrame, ibov: pd.DataFrame, model: str = "logreg_l2") -> None:
    sent_model = sent_daily[sent_daily["model"] == model].rename(columns={"sentiment": "sent"})
    merged = pd.merge(sent_model, ibov[["day", "ret"]], on="day", how="inner").dropna()
    corr = merged["sent"].corr(merged["ret"]) if not merged.empty else np.nan
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ax.scatter(merged["sent"], merged["ret"], alpha=0.35, edgecolor="k", s=24)
    ax.set_title(f"Figura 4 – Dispersão Sentimento × Retorno (r={corr:.2f})")
    ax.set_xlabel(f"Sentimento ({MODEL_DISPLAY_NAMES.get(model, model)})")
    ax.set_ylabel("Retorno diário do Ibovespa")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _savefig(fig, OUTPUT_DIR / "Figura_4_dispersao_sentimento_retorno.png")


def figure_rolling_corr(
    sent_daily: pd.DataFrame, ibov: pd.DataFrame, model: str = "logreg_l2"
) -> None:
    sent_model = sent_daily[sent_daily["model"] == model][["day", "sentiment"]]
    merged = pd.merge(sent_model, ibov[["day", "ret"]], on="day", how="inner").sort_values("day")
    merged["corr_60"] = merged["sentiment"].rolling(60).corr(merged["ret"])
    merged["corr_90"] = merged["sentiment"].rolling(90).corr(merged["ret"])
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    ax.plot(merged["day"], merged["corr_60"], label="Correlação 60d")
    ax.plot(merged["day"], merged["corr_90"], label="Correlação 90d")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Figura 5 – Correlação móvel (sentimento x retorno)")
    ax.set_ylabel("Correlação de Pearson")
    ax.set_xlabel("Data")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    _savefig(fig, OUTPUT_DIR / "Figura_5_correlacao_movel_60d_90d.png")


def figure_distribution(sent_daily: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ax.hist(sent_daily["sentiment"], bins=40, color="#1f77b4", alpha=0.75, edgecolor="white")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Figura 6 – Distribuição do sentimento diário (todos os modelos)")
    ax.set_xlabel("Sentimento")
    ax.set_ylabel("Frequência")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    _savefig(fig, OUTPUT_DIR / "Figura_6_distribuicao_sentimento.png")


def figure_latency(events: pd.DataFrame) -> None:
    if events.empty:
        raise RuntimeError("Latência: dataset vazio; gere event_study_latency.csv.")
    events = events.copy()
    events["window"] = events["window"].astype(str)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    groups = [events.loc[events["polarity"] == pol, "car_value"].dropna() for pol in ["pos", "neg"]]
    ax.boxplot(
        groups,
        tick_labels=["pos", "neg"],
        showmeans=True,
        meanline=True,
        patch_artist=True,
        boxprops=dict(facecolor="#a6cee3"),
        medianprops=dict(color="black"),
        meanprops=dict(color="red"),
    )
    ax.set_title("Figura 7A – CAR por polaridade (boxplot)")
    ax.set_xlabel("Polaridade")
    ax.set_ylabel("CAR (retorno acumulado bruto D0–D5)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    _savefig(fig, OUTPUT_DIR / "Figura_7A_latencia_boxplot.png")


def _bootstrap_mean_ci(
    values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float, float]:
    """Média e IC percentil da média por bootstrap i.i.d. (semente fixa → resultado reprodutível)."""
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(sample.mean())
    means = np.array(means)
    return (
        float(means.mean()),
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


def _event_cars(
    returns: pd.Series,
    event_day: pd.Timestamp,
    tau_max: int,
    window_unit: str,
    abnormal: bool,
    estimation_window: int,
    first_tau: int,
) -> dict[int, float]:
    """CAR do evento para cada τ disponível (retorno bruto ou anormal, somado a partir de `first_tau`)."""
    position = returns.index.get_indexer([event_day])[0]
    expected = 0.0
    if abnormal:
        if position < estimation_window:
            return {}
        expected = float(returns.iloc[position - estimation_window : position].mean())
    cars = {}
    for tau in range(first_tau, tau_max + 1):
        if window_unit == "calendar_days":
            window = returns.loc[event_day : event_day + pd.Timedelta(days=tau)]
            if len(window) < tau + 1:
                continue
            window = window.iloc[first_tau : tau + 1]
        elif window_unit == "trading_days":
            if position < 0 or position + tau >= len(returns):
                continue
            window = returns.iloc[position + first_tau : position + tau + 1]
        else:
            raise ValueError(f"window_unit inválido: {window_unit!r}")
        cars[tau] = float((window - expected).sum())
    return cars


def compute_caar_by_event_time(
    events: pd.DataFrame,
    returns: pd.Series,
    tau_max: int = 5,
    window_unit: str = "calendar_days",
    abnormal: bool = False,
    estimation_window: int = 60,
    first_tau: int = 0,
    n_boot: int = 1000,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """CAAR por polaridade e τ, com IC 95% por bootstrap i.i.d. dos eventos.

    Padrão (artigo): janela de τ dias corridos a partir do evento e retornos brutos.
    Verificação pós-submissão: `window_unit="trading_days"` (τ pregões, todos os eventos
    entram em todo τ) e `abnormal=True` (retorno anormal = retorno − média dos
    `estimation_window` pregões anteriores ao evento). Com `first_tau=1`, o retorno do
    próprio dia do evento fica de fora da soma.
    """
    rows: list[dict[str, float | int | str]] = []
    for _, row in events.iterrows():
        cars = _event_cars(
            returns,
            row["event_day"],
            tau_max,
            window_unit,
            abnormal,
            estimation_window,
            first_tau,
        )
        rows.extend(
            {"tau": tau, "polarity": row["polarity"], "car": car} for tau, car in cars.items()
        )
    car_df = pd.DataFrame(rows)
    if car_df.empty or car_df["tau"].nunique() <= 1:
        raise RuntimeError("CAAR: número insuficiente de pontos tau (>1) para plotar curva.")
    out_rows: list[dict[str, float | int]] = []
    for tau in sorted(car_df["tau"].unique()):
        sub = car_df[car_df["tau"] == tau]
        entry: dict[str, float | int] = {"tau": tau, "n_boot": n_boot}
        for pol in ["neg", "pos"]:
            pol_vals = sub.loc[sub["polarity"] == pol, "car"].dropna().values
            mean, low, high = _bootstrap_mean_ci(pol_vals, n_boot=n_boot, seed=seed)
            entry[f"caar_{pol}_mean"] = mean
            entry[f"caar_{pol}_ci_low"] = low
            entry[f"caar_{pol}_ci_high"] = high
            entry[f"n_events_{pol}"] = int(len(pol_vals))
        out_rows.append(entry)
    return pd.DataFrame(out_rows)


def figure_caar_event_time(events: pd.DataFrame, ibov: pd.DataFrame, tau_max: int = 5) -> None:
    returns = ibov.set_index("day")["ret"].dropna()
    out_df = compute_caar_by_event_time(events, returns, tau_max)
    if out_df["tau"].nunique() <= 1:
        raise RuntimeError("CAAR CSV: pontos tau insuficientes.")
    out_df.to_csv(OUTPUT_DIR / "Figura_7B_event_time_CAAR.csv", index=False)
    print(
        f"[CAAR] n_events_pos={int(out_df['n_events_pos'].max())} | n_events_neg={int(out_df['n_events_neg'].max())} | taus={list(out_df['tau'])}"
    )
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    for pol, color in [("neg", "#d62728"), ("pos", "#2ca02c")]:
        ax.plot(out_df["tau"], out_df[f"caar_{pol}_mean"], marker="o", label=pol, color=color)
        ax.fill_between(
            out_df["tau"],
            out_df[f"caar_{pol}_ci_low"],
            out_df[f"caar_{pol}_ci_high"],
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
    _savefig(fig, OUTPUT_DIR / "Figura_7B_event_time_CAAR.png")


def figure_backtest_vs_benchmark(
    oof: pd.DataFrame, ibov: pd.DataFrame, strategy_name: str
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], str]:
    equity_df, stats, strategy = compute_backtest_mark_to_market(oof, ibov, strategy_name)
    equity_df.to_csv(OUTPUT_DIR / "Figura_8_backtest_vs_benchmark.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    ax.plot(
        equity_df["date"],
        equity_df["equity_logreg_l2"],
        label=f"{MODEL_DISPLAY_NAMES['logreg_l2']} ({strategy})",
    )
    ax.plot(
        equity_df["date"],
        equity_df["equity_rf_200"],
        label=f"{MODEL_DISPLAY_NAMES['rf_200']} ({strategy})",
    )
    ax.plot(
        equity_df["date"],
        equity_df["equity_ibov"],
        label="Ibov buy&hold",
        color="black",
        linestyle="--",
    )
    ax.set_title("Figura 8 – Curva de backtest vs benchmark (normalizadas em 1.0)")
    ax.set_ylabel("Equity normalizado")
    ax.set_xlabel("Data")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    _savefig(fig, OUTPUT_DIR / "Figura_8_backtest_vs_benchmark.png")
    return equity_df, stats, strategy


def figure_comparativo(
    backtest_stats: dict[str, dict[str, float]], strategy_name: str | None
) -> None:
    if strategy_name is None:
        raise RuntimeError("Não há estratégia comum para Sharpe.")
    rows = []
    for model in ANCHOR_MODELS:
        stats = backtest_stats.get(model, {})
        if stats:
            rows.append({"model": model, "sharpe": stats.get("sharpe")})
    data = pd.DataFrame(rows)
    if len(data["model"].unique()) < 2:
        raise RuntimeError(
            "Figura 3: não foram encontradas duas linhas de Sharpe para a estratégia."
        )
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    x_labels = [MODEL_DISPLAY_NAMES.get(m, m) for m in data["model"]]
    bars = ax.bar(x_labels, data["sharpe"], color=["#2ca02c", "#1f77b4"])
    for bar in bars:
        height = bar.get_height()
        offset = 3 if height >= 0 else -12
        va = "bottom" if height >= 0 else "top"
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=10,
            fontweight="bold",
        )
    ax.set_ylabel("Sharpe")
    ax.set_title(f"Figura 3 – Comparativo de Modelos (Sharpe) | Estratégia: {strategy_name}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _savefig(fig, OUTPUT_DIR / "Figura_3_comparativo_modelos.png")


def table_metrics(
    results16: pd.DataFrame, backtest_stats: dict[str, dict[str, float]], strategy_name: str | None
) -> None:
    tfidf = results16.copy()
    tfidf = tfidf[tfidf["model"].isin(ANCHOR_MODELS)]
    tfidf["cagr"] = "—"
    tfidf["sharpe"] = "—"
    tfidf["strategy"] = "—"
    tfidf["auc"] = tfidf["auc"].apply(lambda x: f"{float(x):.3f}" if pd.notna(x) else "—")
    tfidf["mda"] = tfidf["mda"].apply(lambda x: f"{float(x):.3f}" if pd.notna(x) else "—")
    back_rows = []
    if strategy_name:
        for model in ANCHOR_MODELS:
            stats = backtest_stats.get(model, {})
            back_rows.append(
                {
                    "model": model,
                    "dataset": "backtest_daily",
                    "strategy": strategy_name,
                    "auc": "—",
                    "mda": "—",
                    "cagr": stats.get("cagr"),
                    "sharpe": stats.get("sharpe"),
                }
            )
    back = pd.DataFrame(back_rows)
    if not back.empty:
        back["cagr"] = back["cagr"].apply(lambda x: f"{float(x):+.2%}" if pd.notna(x) else "—")
        back["sharpe"] = back["sharpe"].apply(lambda x: f"{float(x):.3f}" if pd.notna(x) else "—")
    table = pd.concat(
        [tfidf[["model", "dataset", "auc", "mda", "strategy", "cagr", "sharpe"]], back],
        ignore_index=True,
    )
    table.to_csv(OUTPUT_DIR / "Tabela_1_metricas.csv", index=False)
    note = (
        "Nota: AUC/MDA são métricas de classificação em tfidf_daily; backtest_daily reporta métricas econômicas (CAGR/Sharpe). "
        "Sharpe calculado sobre retornos diários (convenção 252), sem taxa livre de risco; "
        "custo de transação de 0,0005 por unidade de turnover. "
        f"Estratégia comparada nos modelos de backtest: {strategy_name or '—'}."
    )
    (OUTPUT_DIR / "nota_tabela1.txt").write_text(note, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    ax.axis("off")
    tbl = ax.table(cellText=table.values, colLabels=table.columns, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.3)
    fig.tight_layout()
    _savefig(fig, OUTPUT_DIR / "Tabela_1_metricas.png")


def table_intersection(ibov: pd.DataFrame, sent_daily: pd.DataFrame) -> None:
    ibov_days = set(ibov["day"])
    sent_days = set(sent_daily["day"])
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
    _savefig(fig, OUTPUT_DIR / "Tabela_intersecao_periodo.png")


def assert_required_outputs() -> None:
    missing = [name for name in REQUIRED_PNGS if not (OUTPUT_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Arquivos obrigatórios ausentes: {missing}")


# ------------------------- ROBUSTEZ / ARTEFATOS EXTRAS ------------------------- #


def _parse_int_list(arg: str, default: list[int]) -> list[int]:
    if not arg:
        return default
    return [int(x.strip()) for x in arg.split(",") if x.strip() != ""]


def _parse_float_list(arg: str, default: list[float]) -> list[float]:
    if not arg:
        return default
    return [float(x.strip()) for x in arg.split(",") if x.strip() != ""]


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    cummax = equity.cummax()
    dd = (equity / cummax - 1).min()
    return float(dd)


def _quantile_thresholds(
    proba: pd.Series, event_q: float, allow_short: bool, threshold_mode: str, threshold_window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Limiares diários de entrada (quantil `event_q`) e de saída/venda.

    - "full_sample" (artigo): quantis de p no período inteiro, o que usa informação futura;
    - "rolling" (verificação pós-submissão): quantis de p nos `threshold_window` pregões
      anteriores; nos primeiros pregões não há limiar e a estratégia fica fora do mercado.
    A saída usa a mediana (só compra) ou o quantil 1 − `event_q` (compra e venda).
    """
    exit_q = 1 - event_q if allow_short else 0.50
    if threshold_mode == "full_sample":
        n_days = len(proba)
        return np.full(n_days, proba.quantile(event_q)), np.full(n_days, proba.quantile(exit_q))
    if threshold_mode == "rolling":
        past = proba.shift(1).rolling(threshold_window, min_periods=threshold_window)
        return past.quantile(event_q).to_numpy(), past.quantile(exit_q).to_numpy()
    raise ValueError(
        f"threshold_mode inválido: {threshold_mode!r} (use 'full_sample' ou 'rolling')."
    )


def _run_strategy_quantile(
    oof_model: pd.DataFrame,
    cfg: dict[str, float],
    event_q: float,
    lag: int,
    threshold_mode: str = "full_sample",
    threshold_window: int = 60,
) -> pd.DataFrame:
    df = oof_model.sort_values("day").copy()
    allow_short = cfg.get("allow_short", True)
    cost = cfg.get("cost", 0.0005)
    long_ths, short_ths = _quantile_thresholds(
        df["proba"], event_q, allow_short, threshold_mode, threshold_window
    )

    positions = []
    turnovers = []
    pos_prev = 0
    for proba, long_th, short_th in zip(df["proba"], long_ths, short_ths, strict=True):
        if proba >= long_th:
            pos = 1
        elif allow_short and proba <= short_th:
            pos = -1
        elif (not allow_short) and proba <= short_th:
            pos = 0
        else:
            pos = pos_prev
        turnovers.append(abs(pos - pos_prev))
        positions.append(pos)
        pos_prev = pos

    df["signal"] = positions
    df["turnover"] = turnovers
    df["cost"] = df["turnover"] * cost
    effective_pos = df["signal"].shift(lag + 1, fill_value=0)
    df["strategy_ret"] = effective_pos * df["ret_next"].fillna(0) - df["cost"]
    df["equity"] = (1 + df["strategy_ret"]).cumprod()
    return df


def _compute_metrics(
    ret: pd.Series, equity: pd.Series, turnover: pd.Series, cost: pd.Series, signal: pd.Series
) -> dict[str, float]:
    ret = ret.fillna(0)
    equity = equity.ffill()
    cagr = equity.iloc[-1] ** (252 / len(equity)) - 1 if len(equity) > 1 else np.nan
    vol = ret.std(ddof=0) * np.sqrt(252) if ret.std(ddof=0) != 0 else np.nan
    sharpe = ret.mean() / ret.std(ddof=0) * np.sqrt(252) if ret.std(ddof=0) != 0 else np.nan
    dd = _max_drawdown(equity)
    n_trades = int((signal.diff().fillna(0) != 0).sum())
    turnover_sum = float(turnover.sum())
    hit_rate = float((ret > 0).mean()) if len(ret) else np.nan
    exposure = float((signal != 0).mean()) if len(signal) else np.nan
    total_cost = float(cost.sum())
    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "vol_anual": vol,
        "max_drawdown": dd,
        "turnover": turnover_sum,
        "n_trades": n_trades,
        "hit_rate": hit_rate,
        "exposure": exposure,
        "total_cost": total_cost,
    }


def _run_robust_backtest_grid(
    oof: pd.DataFrame,
    ibov: pd.DataFrame,
    strategy_name: str,
    lags: list[int],
    event_qs: list[float],
    threshold_mode: str = "full_sample",
    threshold_window: int = 60,
) -> pd.DataFrame:
    cfg = next((c for c in STRATEGIES_CFG if c["name"] == strategy_name), None)
    if cfg is None:
        raise RuntimeError(f"Estratégia {strategy_name} não configurada.")
    rows = []
    for lag in lags:
        for eq in event_qs:
            for model in ANCHOR_MODELS:
                df_model = oof[oof["model"] == model].copy()
                if df_model.empty:
                    continue
                strat = _run_strategy_quantile(
                    df_model,
                    cfg,
                    event_q=eq,
                    lag=lag,
                    threshold_mode=threshold_mode,
                    threshold_window=threshold_window,
                )
                ret = strat["strategy_ret"]
                equity = strat["equity"]
                metrics = _compute_metrics(
                    ret, equity, strat["turnover"], strat["cost"], strat["signal"]
                )
                rows.append(
                    {
                        "model": model,
                        "strategy": strategy_name,
                        "lag": lag,
                        "event_q": eq,
                        **metrics,
                    }
                )
    if not rows:
        raise RuntimeError("Robustez: nenhuma linha gerada.")
    return pd.DataFrame(rows)


def _write_table_png(df: pd.DataFrame, path_csv: Path, path_png: Path, title: str) -> None:
    df.to_csv(path_csv, index=False)
    fig, ax = plt.subplots(figsize=(12, 0.6 + 0.35 * len(df)), dpi=300)
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.2)
    ax.set_title(title)
    fig.tight_layout()
    _savefig(fig, path_png)


def generate_table2_robustez(
    oof: pd.DataFrame,
    ibov: pd.DataFrame,
    strategy_name: str,
    lags: list[int],
    event_qs: list[float],
) -> list[Path]:
    df = _run_robust_backtest_grid(oof, ibov, strategy_name, lags, event_qs)
    for col in [
        "cagr",
        "sharpe",
        "vol_anual",
        "max_drawdown",
        "turnover",
        "hit_rate",
        "exposure",
        "total_cost",
    ]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    out_csv = OUTPUT_DIR / "Tabela_2_robustez_backtest.csv"
    out_png = OUTPUT_DIR / "Tabela_2_robustez_backtest.png"
    _write_table_png(df, out_csv, out_png, title="Tabela 2 – Robustez do backtest (lag x quantil)")
    return [out_csv, out_png]


def generate_table3_metricas_extendidas(
    oof: pd.DataFrame, ibov: pd.DataFrame, strategy_name: str, lag: int, event_q: float
) -> list[Path]:
    cfg = next((c for c in STRATEGIES_CFG if c["name"] == strategy_name), None)
    if cfg is None:
        raise RuntimeError(f"Estratégia {strategy_name} não configurada.")
    rows = []
    for model in ANCHOR_MODELS:
        df_model = oof[oof["model"] == model].copy()
        if df_model.empty:
            continue
        strat = _run_strategy_quantile(df_model, cfg, event_q=event_q, lag=lag)
        metrics = _compute_metrics(
            strat["strategy_ret"],
            strat["equity"],
            strat["turnover"],
            strat["cost"],
            strat["signal"],
        )
        rows.append(
            {"modelo": model, "dataset": "backtest_daily", "strategy": strategy_name, **metrics}
        )
    # benchmark
    bench_ret = ibov.set_index("day")["ret"].dropna()
    bench_eq = (1 + bench_ret).cumprod()
    bench_eq = bench_eq / bench_eq.iloc[0]
    bench_metrics = _compute_metrics(
        bench_ret, bench_eq, pd.Series([0]), pd.Series([0]), pd.Series([1] * len(bench_ret))
    )
    rows.append(
        {"modelo": "ibov_buyhold", "dataset": "benchmark", "strategy": "—", **bench_metrics}
    )
    df = pd.DataFrame(rows)
    for col in [
        "cagr",
        "sharpe",
        "vol_anual",
        "max_drawdown",
        "turnover",
        "hit_rate",
        "exposure",
        "total_cost",
    ]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    out_csv = OUTPUT_DIR / "Tabela_3_metricas_extendidas.csv"
    out_png = OUTPUT_DIR / "Tabela_3_metricas_extendidas.png"
    _write_table_png(df, out_csv, out_png, title="Tabela 3 – Métricas estendidas (cenário padrão)")
    return [out_csv, out_png]


def figure_robust_corr(sent_daily: pd.DataFrame, ibov: pd.DataFrame, windows: list[int]) -> Path:
    sent_model = sent_daily[sent_daily["model"] == "logreg_l2"][["day", "sentiment"]]
    merged = pd.merge(sent_model, ibov[["day", "ret"]], on="day", how="inner").sort_values("day")
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    for w in windows:
        merged[f"corr_{w}"] = merged["sentiment"].rolling(w).corr(merged["ret"])
        ax.plot(merged["day"], merged[f"corr_{w}"], label=f"Corr {w}d")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Figura 9 – Robustez da correlação móvel (sentimento x retorno)")
    ax.set_ylabel("Correlação de Pearson")
    ax.set_xlabel("Data")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_png = OUTPUT_DIR / "Figura_9_robustez_correlacao.png"
    _savefig(fig, out_png)
    return out_png


def _validate_robust_outputs(paths: list[Path], start_ts: float) -> None:
    for p in paths:
        if not p.exists():
            raise RuntimeError(f"Robustez: arquivo ausente {p}")
        stat = p.stat()
        if p.suffix.lower() == ".png":
            if stat.st_size <= 30_000:
                raise RuntimeError(f"Robustez: PNG muito pequeno {p} ({stat.st_size} bytes)")
            if stat.st_mtime < start_ts - 2:
                raise RuntimeError(f"Robustez: mtime antigo {p} ({stat.st_mtime})")
        if stat.st_mtime < start_ts - 2:
            raise RuntimeError(f"Robustez: mtime antigo {p} ({stat.st_mtime})")
    print(f"[ROBUST][OK] {len(paths)} artefatos gerados e validados")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera figuras/tabelas finais (PNG) do TCC.")
    parser.add_argument(
        "--strategy",
        choices=[cfg["name"] for cfg in STRATEGIES_CFG],
        default=STRATEGIES_CFG[0]["name"],
        help="Estratégia determinística para Figura 3/8 e Tabela 1.",
    )
    parser.add_argument(
        "--force",
        dest="force",
        action="store_true",
        default=True,
        help="Sobrescreve sempre PNG/CSV/TXT (default).",
    )
    parser.add_argument(
        "--no-force",
        dest="force",
        action="store_false",
        help="Não sobrescreve arquivos existentes.",
    )
    parser.add_argument(
        "--clean",
        dest="clean",
        action="store_true",
        default=True,
        help="Limpa artefatos de saída antes de gerar (default).",
    )
    parser.add_argument(
        "--no-clean",
        dest="clean",
        action="store_false",
        help="Não limpa artefatos antes de gerar.",
    )
    parser.add_argument(
        "--run_robustness",
        action="store_true",
        default=False,
        help="Gera artefatos extras de robustez (Tabelas 2/3 e Figura 9).",
    )
    parser.add_argument(
        "--robust_lags",
        type=str,
        default="0,1,2",
        help='Lista de lags (dias) separados por vírgula para robustez, ex.: "0,1,2".',
    )
    parser.add_argument(
        "--robust_event_q",
        type=str,
        default="0.90,0.95",
        help='Lista de quantis (0-1) para thresholds do sinal, ex.: "0.90,0.95".',
    )
    parser.add_argument(
        "--robust_corr_windows",
        type=str,
        default="30,60,90",
        help='Janelas de correlação móvel para a Figura 9, ex.: "30,60,90".',
    )
    args = parser.parse_args()
    global FORCE_OVERWRITE
    FORCE_OVERWRITE = args.force
    print(f"[OUTPUT_DIR] {OUTPUT_DIR.resolve()}")
    prev_mtimes = _capture_prev_mtimes()
    start_ts = time.time()
    if args.clean:
        _clean_outputs()
    else:
        print("[CLEAN] Skip (clean=False)")
    print(f"[LOAD] Carregando dados de {BASE_DATA} ...")
    ibov = load_ibov()
    events = load_events()
    sentiment_raw = load_sentiment()
    sentiment_daily = load_sentiment_daily(sentiment_raw)
    results16 = load_results16()
    oof = load_oof_predictions()
    print(f"[EXPORT] Estratégia escolhida (flag --strategy): {args.strategy}")
    equity_df, backtest_stats, strategy_used = figure_backtest_vs_benchmark(
        oof, ibov, args.strategy
    )
    print(f"[BACKTEST] Estratégia utilizada: {strategy_used} (aplicada em Figura 3/8 e Tabela 1)")
    print(
        f"[BACKTEST] nunique equity logreg_l2={equity_df['equity_logreg_l2'].nunique()} | rf_200={equity_df['equity_rf_200'].nunique()}"
    )
    print(
        f"[BACKTEST] Datas: {equity_df['date'].min()} -> {equity_df['date'].max()} | linhas={len(equity_df)}"
    )

    figure_ibov_events(ibov, events)
    figure_sentiment_daily(sentiment_daily)
    figure_scatter(sentiment_daily, ibov, model="logreg_l2")
    figure_rolling_corr(sentiment_daily, ibov, model="logreg_l2")
    figure_distribution(sentiment_daily)
    figure_latency(events)
    figure_caar_event_time(events, ibov, tau_max=5)
    figure_comparativo(backtest_stats, strategy_used)
    table_metrics(results16, backtest_stats, strategy_used)
    table_intersection(ibov, sentiment_daily)
    assert_required_outputs()
    _validate_png_mtimes(start_ts, prev_mtimes)
    print("[OK] Todas as 11 figuras/tabelas geradas em reports/figures/.")
    if args.run_robustness:
        robust_lags = _parse_int_list(args.robust_lags, [0, 1, 2])
        robust_event_q = _parse_float_list(args.robust_event_q, [0.90, 0.95])
        robust_corr = _parse_int_list(args.robust_corr_windows, [30, 60, 90])
        print(
            f"[ROBUST] lags={robust_lags} | event_q={robust_event_q} | corr_windows={robust_corr}"
        )
        extra_paths: list[Path] = []
        extra_paths += generate_table2_robustez(
            oof, ibov, strategy_used, robust_lags, robust_event_q
        )
        extra_paths += generate_table3_metricas_extendidas(
            oof, ibov, strategy_used, robust_lags[0], robust_event_q[0]
        )
        extra_paths.append(figure_robust_corr(sentiment_daily, ibov, robust_corr))
        print("[ROBUST] wrote:", ", ".join(str(p) for p in extra_paths))
        _validate_robust_outputs(extra_paths, start_ts)
        print("[ROBUST][OK] validations passed")


if __name__ == "__main__":
    main()
