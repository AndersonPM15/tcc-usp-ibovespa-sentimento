#!/usr/bin/env python
"""Dash dashboard for TCC USP (Ibovespa x sentimento).

Requisitos atendidos:
- Host/porta configurados via DASH_HOST/DASH_PORT (default 127.0.0.1:8050)
- Healthcheck com socket.connect_ex (1s de espera + 5 tentativas)
- Validacao de integridade (datas, unicidade, intersecoes) e truncamento em 2024-12-31
- Callbacks resilientes: retornos vazios geram figuras anotadas, sem excecoes
- Painel de debug mostra inputs acionadores, filtros e contagens
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path
from threading import Thread
from typing import Dict, List, Tuple

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback_context, dcc, html

from src.config import loader as cfg
from src.io import paths as path_utils


# ---------------------------------------------------------------------------
# Configuracoes basicas
# ---------------------------------------------------------------------------
END_DATE_CAP = pd.Timestamp("2024-12-31")
PERIODO = cfg.get_periodo_estudo()
DATE_MIN = pd.to_datetime(PERIODO.get("start", END_DATE_CAP))
DATE_MAX = min(pd.to_datetime(PERIODO.get("end", END_DATE_CAP)), END_DATE_CAP)


# ---------------------------------------------------------------------------
# Caminhos de dados
# ---------------------------------------------------------------------------
DATA_PATHS = path_utils.get_data_paths(create=True)
PROCESSED = DATA_PATHS["data_processed"]

IBOV_PATH = cfg.get_arquivo("ibov_clean", base_path=DATA_PATHS["base"])
SENTIMENT_PATH = PROCESSED / "16_oof_predictions.csv"
LATENCY_PATH = cfg.get_arquivo("latency_events", base_path=DATA_PATHS["base"])
BACKTEST_PATH = cfg.get_arquivo("backtest_results", base_path=DATA_PATHS["base"])
METRICS_PATH = PROCESSED / "results_16_models_tfidf.json"


# ---------------------------------------------------------------------------
# Funcoes utilitarias de carga e validacao
# ---------------------------------------------------------------------------

def _safe_read_csv(path: Path, parse_dates: List[str]) -> pd.DataFrame:
    """Le CSV retornando DataFrame vazio em caso de ausencia/erro."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception as exc:  # pragma: no cover - log defensivo
        print(f"[app_dashboard] Aviso: falha ao ler {path}: {exc}")
        return pd.DataFrame()


def _safe_read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # pragma: no cover
        print(f"[app_dashboard] Aviso: falha ao ler {path}: {exc}")
        return {}


def _normalize_day(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=[col])
    df[col] = df[col].dt.tz_localize(None)
    df = df[df[col] <= END_DATE_CAP]
    df = df.sort_values(col).drop_duplicates(subset=[col])
    return df


def _validate_range(df: pd.DataFrame, col: str, name: str, warnings: List[str]) -> Tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if df.empty or col not in df.columns:
        warnings.append(f"{name}: dados ausentes ou coluna {col} inexistente")
        return None, None
    min_dt = df[col].min()
    max_dt = df[col].max()
    if max_dt > END_DATE_CAP:
        warnings.append(f"{name}: datas acima de {END_DATE_CAP.date()} foram truncadas; reexecute o pipeline se precisar de extensao")
    dup_count = len(df) - len(df.drop_duplicates(subset=[col]))
    if dup_count > 0:
        warnings.append(f"{name}: {dup_count} linhas duplicadas removidas")
    return min_dt, max_dt


def _filter_by_date(df: pd.DataFrame, col: str, start_date, end_date) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return pd.DataFrame()
    start = pd.to_datetime(start_date) if start_date else df[col].min()
    end = pd.to_datetime(end_date) if end_date else df[col].max()
    mask = (df[col] >= start) & (df[col] <= end)
    return df.loc[mask].copy()


def _annotated_empty_fig(title: str, subtitle: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=subtitle, x=0.5, y=0.5, showarrow=False, font={"size": 14})
    fig.update_layout(title=title, xaxis={"visible": False}, yaxis={"visible": False})
    return fig


def _add_warning_for_missing_intersection(ref: pd.DataFrame, ref_col: str, other: pd.DataFrame, other_col: str, name: str, warnings: List[str]) -> None:
    if ref.empty or other.empty:
        return
    missing = sorted(set(ref[ref_col]) - set(other[other_col]))
    if missing:
        sample = missing[:3]
        warnings.append(
            f"{name}: {len(missing)} dias sem correspondencia (ex.: {', '.join(str(x.date()) for x in sample)}). Reexecute notebooks 16/11/18 se necessario."
        )


# ---------------------------------------------------------------------------
# Carga dos dados
# ---------------------------------------------------------------------------
IBOV_DF = _normalize_day(_safe_read_csv(IBOV_PATH, parse_dates=["day"]), "day")
SENTIMENT_DF = _normalize_day(_safe_read_csv(SENTIMENT_PATH, parse_dates=["day"]), "day")
LATENCY_DF = _normalize_day(_safe_read_csv(LATENCY_PATH, parse_dates=["event_day"]), "event_day")
RESULTS_DF = _safe_read_csv(BACKTEST_PATH, parse_dates=[])
METRICS_JSON = _safe_read_json(METRICS_PATH)

WARNINGS: List[str] = []

ibov_min, ibov_max = _validate_range(IBOV_DF, "day", "Ibovespa", WARNINGS)
sent_min, sent_max = _validate_range(SENTIMENT_DF, "day", "Sentimento (OOF)", WARNINGS)
lat_min, lat_max = _validate_range(LATENCY_DF, "event_day", "Latencia", WARNINGS)

if ibov_min:
    DATE_MIN = max(DATE_MIN, ibov_min)
if ibov_max:
    DATE_MAX = min(DATE_MAX, ibov_max)

_add_warning_for_missing_intersection(IBOV_DF, "day", SENTIMENT_DF, "day", "Sentimento vs Ibov", WARNINGS)
_add_warning_for_missing_intersection(IBOV_DF, "day", LATENCY_DF, "event_day", "Latencia vs Ibov", WARNINGS)

MODEL_OPTIONS = sorted({m for m in SENTIMENT_DF.get("model", []).dropna().unique()} if not SENTIMENT_DF.empty else set())
if not MODEL_OPTIONS and not RESULTS_DF.empty and "model" in RESULTS_DF.columns:
    MODEL_OPTIONS = sorted(RESULTS_DF["model"].dropna().unique())
if not MODEL_OPTIONS:
    MODEL_OPTIONS = ["baseline_auc"]
DEFAULT_MODEL = MODEL_OPTIONS[0]

METRIC_OPTIONS = ["auc", "mda", "accuracy"]


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------
app: Dash = dash.Dash(__name__)
app.title = "TCC USP - Dashboard Ibovespa x Sentimento"


def _build_ibov_fig(start_date, end_date) -> go.Figure:
    df = _filter_by_date(IBOV_DF, "day", start_date, end_date)
    if df.empty:
        return _annotated_empty_fig("Ibovespa", "Nenhum dado de preco encontrado no intervalo")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["day"], y=df["close"], mode="lines", name="Close"))
    fig.update_layout(title="Ibovespa (close)", hovermode="x unified")
    return fig


def _build_sentiment_fig(start_date, end_date, model: str) -> go.Figure:
    df = _filter_by_date(SENTIMENT_DF, "day", start_date, end_date)
    if not df.empty and "model" in df.columns:
        df = df[df["model"] == model]
    if df.empty:
        return _annotated_empty_fig("Sentimento (OOF)", "Sem probabilidades no intervalo")
    daily = df.groupby("day")["proba"].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["proba"], mode="markers+lines", name="Probabilidade media"))
    fig.update_layout(title=f"Probabilidade media por dia ({model})", yaxis={"range": [0, 1]})
    return fig


def _build_metrics_table(model: str):
    records = []
    models_data = METRICS_JSON.get("models", {}) if METRICS_JSON else {}
    model_data = models_data.get(model) or next(iter(models_data.values()), {})
    for metric_name in METRIC_OPTIONS:
        metric_info = model_data.get(metric_name, {}) if isinstance(model_data, dict) else {}
        if metric_info:
            records.append({"metric": metric_name.upper(), "value": metric_info.get("value"), "std": metric_info.get("std")})
    if not records:
        records.append({"metric": "AUC", "value": None, "std": None})
    return records


def update_additional_graphs(start_date, end_date, model: str):
    corr_fig = _build_corr_fig(start_date, end_date, model)
    latency_fig = _build_latency_fig(start_date, end_date)
    backtest_fig = _build_backtest_fig(model)
    return corr_fig, latency_fig, backtest_fig


def _build_corr_fig(start_date, end_date, model: str) -> go.Figure:
    price = _filter_by_date(IBOV_DF, "day", start_date, end_date)
    proba = _filter_by_date(SENTIMENT_DF, "day", start_date, end_date)
    if not proba.empty and "model" in proba.columns:
        proba = proba[proba["model"] == model]
    if price.empty or proba.empty:
        return _annotated_empty_fig("Correlacao prob x retorno", "Requer Ibov e probabilidades no intervalo")
    merged = pd.merge(price[["day", "close"]], proba[["day", "proba"]], on="day", how="inner")
    merged["return"] = merged["close"].pct_change()
    merged = merged.dropna(subset=["return", "proba"])
    if merged.empty:
        return _annotated_empty_fig("Correlacao prob x retorno", "Sem dados suficientes para correlacao")
    corr_val = merged[["return", "proba"]].corr().iloc[0, 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged["proba"], y=merged["return"], mode="markers", name="Obs"))
    fig.update_layout(title=f"Correlacao retorno x proba ({corr_val:.3f})", xaxis_title="Probabilidade", yaxis_title="Retorno diario")
    return fig


def _build_latency_fig(start_date, end_date) -> go.Figure:
    df = _filter_by_date(LATENCY_DF, "event_day", start_date, end_date)
    if df.empty:
        return _annotated_empty_fig("Eventos de latencia", "Nenhum evento encontrado")
    df = df.sort_values("event_day")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["event_day"], y=df.get("T_half_days", df.get("t_half_days", pd.Series([0] * len(df)))), text=df.get("event_name"), name="T_half (dias)"))
    fig.update_layout(title="Latencia de eventos", xaxis_title="Data", yaxis_title="T_half (dias)")
    return fig


def _build_backtest_fig(model: str) -> go.Figure:
    df = RESULTS_DF.copy()
    if df.empty or "model" not in df.columns:
        return _annotated_empty_fig("Backtest", "Sem resultados de backtest (notebook 18)")
    df = df[df["model"] == model] if model in df["model"].unique() else df
    if df.empty:
        return _annotated_empty_fig("Backtest", "Modelo selecionado sem resultados; reexecute notebook 18")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["strategy"], y=df["cagr"], name="CAGR"))
    fig.add_trace(go.Bar(x=df["strategy"], y=df["sharpe"], name="Sharpe"))
    fig.update_layout(title=f"Backtest por estrategia ({model})", barmode="group", xaxis_title="Estrategia")
    return fig


def _healthcheck(host: str, port: int) -> None:
    def _run():
        time.sleep(1)
        for attempt in range(5):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                print(f"[healthcheck] OK - {host}:{port} acessivel (tentativa {attempt + 1})")
                return
            time.sleep(0.5)
        print(f"[healthcheck] Falha ao conectar em {host}:{port} apos 5 tentativas")
        print(f"Processo: pid={os.getpid()} exe={sys.executable}")
        print(f"Sugestao: use 'netstat -ano | findstr {port}' para checar porta em uso")
        os._exit(1)

    Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app.layout = html.Div(
    [
        html.H2("Dashboard: Ibovespa x Sentimento"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Intervalo de datas"),
                        dcc.DatePickerRange(
                            id="date-range",
                            min_date_allowed=DATE_MIN,
                            max_date_allowed=END_DATE_CAP,
                            start_date=DATE_MIN,
                            end_date=DATE_MAX,
                        ),
                    ],
                    style={"marginRight": "24px"},
                ),
                html.Div(
                    [
                        html.Label("Modelo"),
                        dcc.Dropdown(options=[{"label": m, "value": m} for m in MODEL_OPTIONS], value=DEFAULT_MODEL, id="model-dropdown"),
                    ],
                    style={"width": "240px"},
                ),
            ],
            style={"display": "flex", "flexWrap": "wrap", "alignItems": "center", "gap": "16px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Metricas (informativas)"),
                        dcc.Dropdown(
                            options=[{"label": m.upper(), "value": m} for m in METRIC_OPTIONS],
                            value="auc",
                            id="metric-dropdown",
                        ),
                        html.Div(id="metrics-table"),
                    ],
                    style={"minWidth": "240px", "maxWidth": "320px", "marginRight": "24px"},
                ),
                html.Div(
                    [
                        html.Div(
                            [dcc.Markdown("**Avisos de integridade:**"), html.Ul([html.Li(w) for w in WARNINGS] or [html.Li("Sem avisos")])],
                            style={"padding": "8px", "border": "1px solid #ddd"},
                        ),
                        html.Pre(
                            "Host/porta: {host}:{port}\nExecute: python app_dashboard.py".format(
                                host=os.environ.get("DASH_HOST", "127.0.0.1"),
                                port=os.environ.get("DASH_PORT", 8050),
                            ),
                            style={"background": "#f7f7f7", "padding": "8px"},
                        ),
                    ]
                ),
            ],
            style={"display": "flex", "flexWrap": "wrap", "gap": "16px", "marginTop": "12px"},
        ),
        html.Div(
            [
                dcc.Graph(id="ibov-graph"),
                dcc.Graph(id="sentiment-graph"),
            ]
        ),
        html.Div(
            [
                dcc.Graph(id="corr-graph"),
                dcc.Graph(id="latency-graph"),
                dcc.Graph(id="backtest-graph"),
            ],
            style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))", "gap": "12px"},
        ),
        html.Pre(id="debug-panel", style={"background": "#f4f4f4", "padding": "8px", "whiteSpace": "pre-wrap"}),
    ]
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output("ibov-graph", "figure"),
    Output("sentiment-graph", "figure"),
    Output("metrics-table", "children"),
    Output("debug-panel", "children"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("model-dropdown", "value"),
    Input("metric-dropdown", "value"),
)
def update_main_figures(start_date, end_date, model, metric):
    ibov_fig = _build_ibov_fig(start_date, end_date)
    sentiment_fig = _build_sentiment_fig(start_date, end_date, model)
    records = _build_metrics_table(model)
    metric_rows = [html.Div(f"{r['metric']}: {r['value']} +/- {r['std']}") for r in records]

    triggered = callback_context.triggered[0]["prop_id"] if callback_context.triggered else "manual"
    debug_lines = [
        f"Triggered: {triggered}",
        f"Intervalo: {start_date} -> {end_date}",
        f"Modelo: {model}",
        f"Metric: {metric}",
        f"IBOV linhas (filtrado): {len(_filter_by_date(IBOV_DF, 'day', start_date, end_date))}",
        f"Sentimento linhas (filtrado): {len(_filter_by_date(SENTIMENT_DF[SENTIMENT_DF['model'] == model] if not SENTIMENT_DF.empty and 'model' in SENTIMENT_DF.columns else SENTIMENT_DF, 'day', start_date, end_date))}",
    ]
    return ibov_fig, sentiment_fig, metric_rows, "\n".join(debug_lines)


@app.callback(
    Output("corr-graph", "figure"),
    Output("latency-graph", "figure"),
    Output("backtest-graph", "figure"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("model-dropdown", "value"),
)
def _cb_additional_graphs(start_date, end_date, model):
    return update_additional_graphs(start_date, end_date, model)


# ---------------------------------------------------------------------------
# Execucao
# ---------------------------------------------------------------------------
def _run_server():
    host = os.environ.get("DASH_HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("DASH_PORT", 8050))
    except ValueError:
        port = 8050
    print(f"[app_dashboard] Iniciando em http://{host}:{port}")
    _healthcheck(host, port)
    # Dash >=2.16 recomenda app.run em vez de app.run_server
    app.run(host=host, port=port, debug=True, use_reloader=False)


if __name__ == "__main__":
    _run_server()
