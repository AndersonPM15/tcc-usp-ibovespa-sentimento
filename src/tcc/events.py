"""Estudo de eventos: dias de sentimento extremo e retorno acumulado depois deles (Fig. 1, 7A, 7B).

Evento = dia em que a média do sentimento (2p − 1) dos dois modelos fica no percentil 90 ou
acima (positivo) ou no 10 ou abaixo (negativo). No artigo, o CAR soma retornos brutos de D0
até D0 + τ dias corridos. A verificação (d) usa retorno anormal (retorno − média dos 60
pregões anteriores) e janela em pregões.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tcc.stats import N_BOOTSTRAP_ARTICLE, bootstrap_mean_ci

CAR_WINDOW_DAYS = 5  # janela "D0-D5" (dias corridos) dos eventos do artigo
ESTIMATION_WINDOW = 60


def daily_mean_sentiment(oof: pd.DataFrame) -> pd.DataFrame:
    """Média diária do sentimento 2p − 1 entre os modelos e quantos modelos entraram."""
    frame = oof.assign(sentiment=oof["proba"] * 2 - 1)
    return (
        frame.groupby("day")
        .agg(sentiment_mean=("sentiment", "mean"), n_news=("sentiment", "count"))
        .reset_index()
        .sort_values("day")
    )


def detect_events(oof: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Eventos de sentimento extremo e CAR bruto de D0 a D0 + 5 dias corridos.

    `oof` e `returns` (`day`, `ret`) já devem estar recortados no período do artigo.
    """
    market = returns.dropna(subset=["ret"]).sort_values("day")
    merged = daily_mean_sentiment(oof).merge(market[["day", "ret"]], on="day", how="inner")
    low, high = merged["sentiment_mean"].quantile([0.10, 0.90])
    extremes = pd.concat(
        [
            merged[merged["sentiment_mean"] >= high].assign(event_name="sent_pos"),
            merged[merged["sentiment_mean"] <= low].assign(event_name="sent_neg"),
        ]
    ).sort_values("day")
    rows = []
    for event in extremes.itertuples(index=False):
        window = market[
            (market["day"] >= event.day)
            & (market["day"] <= event.day + pd.Timedelta(days=CAR_WINDOW_DAYS))
        ]
        car = float(window["ret"].sum())
        rows.append(
            {
                "event_day": event.day,
                "event_name": event.event_name,
                "polarity": "pos" if event.event_name == "sent_pos" else "neg",
                "n_obs": len(window),
                "car_value": car,
                "car_max_abs": abs(car),
                "n_news": event.n_news,
                "sentiment_mean": event.sentiment_mean,
            }
        )
    return pd.DataFrame(rows).sort_values("event_day").reset_index(drop=True)


def event_cars(
    returns: pd.Series,
    event_day: pd.Timestamp,
    tau_max: int,
    window_unit: str = "calendar_days",
    abnormal: bool = False,
    first_tau: int = 0,
    estimation_window: int = ESTIMATION_WINDOW,
) -> dict[int, float]:
    """CAR do evento em cada τ disponível, somado a partir de `first_tau`.

    `"calendar_days"` (artigo): janela de τ dias corridos; o τ só entra se houver τ + 1
    pregões nela. `"trading_days"`: τ pregões após o evento, então todo evento entra em todo τ.
    Com `abnormal`, subtrai a média dos `estimation_window` pregões anteriores ao evento.
    """
    position = int(returns.index.get_indexer(pd.Index([event_day]))[0])
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


def caar_by_event_time(
    events: pd.DataFrame,
    returns: pd.Series,
    tau_max: int = 5,
    window_unit: str = "calendar_days",
    abnormal: bool = False,
    first_tau: int = 0,
    n_boot: int = N_BOOTSTRAP_ARTICLE,
) -> pd.DataFrame:
    """CAAR por polaridade e τ, com IC 95% por bootstrap i.i.d. dos eventos (Fig. 7B).

    O padrão reproduz o artigo; as opções são as da verificação pós-submissão (d).
    """
    rows = [
        {"tau": tau, "polarity": polarity, "car": car}
        for event_day, polarity in zip(events["event_day"], events["polarity"], strict=True)
        for tau, car in event_cars(
            returns, pd.Timestamp(event_day), tau_max, window_unit, abnormal, first_tau
        ).items()
    ]
    cars = pd.DataFrame(rows)
    if cars.empty or cars["tau"].nunique() <= 1:
        raise RuntimeError("CAAR: menos de dois horizontes τ com eventos.")
    table = []
    for tau in sorted(cars["tau"].unique()):
        entry: dict[str, float] = {"tau": tau, "n_boot": n_boot}
        for polarity in ("neg", "pos"):
            values = cars.loc[(cars["tau"] == tau) & (cars["polarity"] == polarity), "car"].dropna()
            mean, low, high = bootstrap_mean_ci(np.asarray(values), n_boot=n_boot)
            entry[f"caar_{polarity}_mean"] = mean
            entry[f"caar_{polarity}_ci_low"] = low
            entry[f"caar_{polarity}_ci_high"] = high
            entry[f"n_events_{polarity}"] = len(values)
        table.append(entry)
    return pd.DataFrame(table)
