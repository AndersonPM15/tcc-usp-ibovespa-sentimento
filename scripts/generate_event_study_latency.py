#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

START = pd.Timestamp("2018-01-02")
END = pd.Timestamp("2024-12-31")
DATA_DIR = Path(r"C:\TCC_USP\data_processed")


def load_ibov(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Ibovespa não encontrado: {path}")
    df = pd.read_csv(path)
    col = None
    for c in ["day", "date", "Data", "DATE"]:
        if c in df.columns:
            col = c
            break
    if col is None:
        raise ValueError("Coluna de data não encontrada no Ibovespa")
    df["day"] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=["day"])
    df = df[(df["day"] >= START) & (df["day"] <= END)]
    if "return" not in df.columns:
        if "close" not in df.columns:
            raise ValueError("Ibovespa sem colunas return/close para cálculo de retorno")
        df = df.sort_values("day")
        df["return"] = df["close"].pct_change()
    return df.dropna(subset=["return"])[["day", "return"]].sort_values("day")


def load_sentiment(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Sentimento não encontrado: {path}")
    df = pd.read_csv(path)
    if "day" not in df.columns or "proba" not in df.columns:
        raise ValueError("Sentimento precisa de colunas 'day' e 'proba'")
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    df = df.dropna(subset=["day"])
    df = df[(df["day"] >= START) & (df["day"] <= END)]
    df["sentiment"] = df["proba"] * 2 - 1
    agg = df.groupby("day").agg(sentiment_mean=("sentiment", "mean"), n_news=("sentiment", "count")).reset_index()
    return agg.sort_values("day")


def generate_latency(ibov: pd.DataFrame, sent: pd.DataFrame, min_news: int = 1) -> pd.DataFrame:
    merged = sent.merge(ibov, on="day", how="inner").sort_values("day")
    if merged.empty:
        raise ValueError("Merge sentimento x Ibovespa vazio")
    q10 = merged["sentiment_mean"].quantile(0.10)
    q90 = merged["sentiment_mean"].quantile(0.90)
    pos = merged[(merged["sentiment_mean"] >= q90) & (merged["n_news"] >= min_news)].assign(fonte="sent_pos")
    neg = merged[(merged["sentiment_mean"] <= q10) & (merged["n_news"] >= min_news)].assign(fonte="sent_neg")
    events = pd.concat([pos, neg]).sort_values("day")
    if events.empty:
        raise ValueError("Nenhum evento de sentimento encontrado com os filtros atuais")

    rows = []
    for _, ev in events.iterrows():
        start_day = ev["day"]
        window = ibov[(ibov["day"] >= start_day) & (ibov["day"] <= start_day + pd.Timedelta(days=5))]
        if window.empty:
            continue
        car = window["return"].sum()
        rows.append(
            {
                "event_day": start_day.date(),
                "fonte": ev["fonte"],
                "event_name": ev["fonte"],
                "window": "D0-D5",
                "n_obs": len(window),
                "car_value": car,
                "car_max_abs": abs(car),
                "n_news": ev["n_news"],
                "sentiment_mean": ev["sentiment_mean"],
            }
        )
    lat = pd.DataFrame(rows).sort_values("event_day")
    if lat.empty:
        raise ValueError("Nenhum CAR calculado (latência vazia)")
    # clamp safety
    lat = lat[(lat["event_day"] >= START.date()) & (lat["event_day"] <= END.date())]
    return lat.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera event_study_latency.csv a partir de sentimento e Ibovespa")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Diretório com arquivos processados")
    parser.add_argument("--min-news", type=int, default=1, help="Mínimo de notícias por dia para considerar evento")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    ibov = load_ibov(data_dir / "ibovespa_clean.csv")
    sent = load_sentiment(data_dir / "16_oof_predictions.csv")
    lat = generate_latency(ibov, sent, min_news=args.min_news)
    out_path = data_dir / "event_study_latency.csv"
    lat.to_csv(out_path, index=False)
    print(f"Latência gerada: {out_path} rows={len(lat)} min={lat['event_day'].min()} max={lat['event_day'].max()}")


if __name__ == "__main__":
    main()
