"""
Plotly Dash application for the TCC USP sentiment x Ibovespa project.

Run with:
    python app_dashboard.py

The app reads the same artifacts used in `20_final_dashboard_analysis.ipynb`
and exposes interactive filters for period, modelo e métrica.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

from src.config import loader as cfg
from src.config.constants import START_DATE, END_DATE
from src.io import paths

# ------------------------------------------------------------------------------
# Carregamento dos dados
# ------------------------------------------------------------------------------

DATA_PATHS = paths.get_data_paths()
PROJECT_PATHS = paths.get_project_paths()
BASE_PATH = DATA_PATHS["base"]
CONFIG = cfg.load_config()
COL_DATE = cfg.get_colunas_data().get("ibov", "day")
DATE_ALIASES = [COL_DATE, "day", "date", "data", "Data", "DATA"]

# Arquivos principais
IBOV_PATH = cfg.get_arquivo("ibov_clean", BASE_PATH)
OOF_PATH = DATA_PATHS["data_processed"] / "16_oof_predictions.csv"
RESULTS16_PATH = cfg.get_arquivo("tfidf_daily_matrix", BASE_PATH).with_name("results_16_models_tfidf.json")
BACKTEST_PATH = DATA_PATHS["data_processed"] / "18_backtest_results.csv"
LATENCY_PATH = cfg.get_arquivo("latency_events", BASE_PATH)


def _safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        print(f"[aviso] Arquivo não encontrado: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def load_ibov() -> pd.DataFrame:
    df = _safe_read_csv(IBOV_PATH)
    if df.empty:
        return df
    chosen_col = None
    for candidate in DATE_ALIASES:
        if candidate in df.columns:
            chosen_col = candidate
            break
    if chosen_col is None:
        raise KeyError("Arquivo de Ibovespa precisa ter coluna de data.")

    df["day"] = pd.to_datetime(df[chosen_col])
    if "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]
    return df.sort_values("day")


def load_sentiment() -> pd.DataFrame:
    df = _safe_read_csv(OOF_PATH, parse_dates=["day"])
    if df.empty:
        return df
    df["sentiment"] = df["proba"] * 2 - 1
    agg = df.groupby("day").agg(
        sentiment=("sentiment", "mean"),
        prob_mean=("proba", "mean"),
        n_obs=("proba", "count"),
    )
    agg.reset_index(inplace=True)
    return agg


def load_results_table() -> pd.DataFrame:
    rows: List[dict] = []
    if RESULTS16_PATH.exists():
        with open(RESULTS16_PATH, "r", encoding="utf-8") as fh:
            results16 = json.load(fh)
        for model_name, values in results16.get("models", {}).items():
            rows.append(
                {
                    "model": model_name,
                    "dataset": "tfidf_daily",
                    "auc": values["auc"]["value"],
                    "mda": values["mda"]["value"],
                    "strategy": None,
                    "cagr": None,
                    "sharpe": None,
                }
            )
    backtest_df = _safe_read_csv(BACKTEST_PATH)
    if not backtest_df.empty:
        best = (
            backtest_df.sort_values("sharpe", ascending=False)
            .groupby("model")
            .head(1)
        )
        for _, row in best.iterrows():
            rows.append(
                {
                    "model": row["model"],
                    "dataset": "backtest_daily",
                    "auc": None,
                    "mda": None,
                    "strategy": row["strategy"],
                    "cagr": row["cagr"],
                    "sharpe": row["sharpe"],
                }
            )
    return pd.DataFrame(rows)


def load_latency_events() -> pd.DataFrame:
    df = _safe_read_csv(LATENCY_PATH)
    if df.empty:
        return df
    date_col = cfg.get_colunas_data().get("eventos", "event_day")
    if date_col not in df.columns:
        raise KeyError("Arquivo de eventos precisa ter coluna 'event_day'.")
    df["event_day"] = pd.to_datetime(df[date_col])
    return df


IBOV_DF = load_ibov()
SENTIMENT_DF = load_sentiment()
RESULTS_DF = load_results_table()
LATENCY_DF = load_latency_events()

# Usar constantes do plano de pesquisa como limites (2018-01-02 a 2025-12-31)
# FIXO: nunca mais mudar esses valores automaticamente
DATE_MIN = pd.Timestamp(START_DATE)
DATE_MAX = pd.Timestamp(END_DATE)

MODEL_OPTIONS = sorted(RESULTS_DF["model"].dropna().unique()) if not RESULTS_DF.empty and "model" in RESULTS_DF.columns else []
METRIC_OPTIONS = [
    {"label": "AUC", "value": "auc"},
    {"label": "MDA", "value": "mda"},
    {"label": "Sharpe", "value": "sharpe"},
]

# ------------------------------------------------------------------------------
# Dash App
# ------------------------------------------------------------------------------

app = Dash(__name__)
app.title = "Dashboard Sentimento x Ibovespa"


def _build_controls():
    return html.Div(
        className="controls",
        children=[
            html.Div(
                [
                    html.Label("Período"),
                    dcc.DatePickerRange(
                        id="date-range",
                        min_date_allowed=DATE_MIN,
                        max_date_allowed=DATE_MAX,
                        start_date=DATE_MIN,
                        end_date=DATE_MAX,
                        display_format="YYYY-MM-DD",
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label("Modelos"),
                    dcc.Dropdown(
                        id="model-filter",
                        options=[{"label": m, "value": m} for m in MODEL_OPTIONS],
                        value=MODEL_OPTIONS,
                        multi=True,
                        placeholder="Selecione modelos",
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label("Métrica"),
                    dcc.Dropdown(
                        id="metric-filter",
                        options=METRIC_OPTIONS,
                        value="auc",
                        clearable=False,
                    ),
                ]
            ),
        ],
    )


app.layout = html.Div(
    className="container",
    children=[
        html.H1("Sentimento de Notícias x Ibovespa"),
        _build_controls(),
        html.Div(
            className="charts",
            children=[
                dcc.Graph(id="ibov-graph"),
                dcc.Graph(id="sentiment-graph"),
            ],
        ),
        html.H2("Comparativo de Modelos"),
        dcc.Graph(id="model-comparison-graph"),
        dash_table.DataTable(
            id="model-table",
            columns=[
                {"name": "Modelo", "id": "model"},
                {"name": "Dataset", "id": "dataset"},
                {"name": "AUC", "id": "auc", "type": "numeric", "format": {"specifier": ".3f"}},
                {"name": "MDA", "id": "mda", "type": "numeric", "format": {"specifier": ".3f"}},
                {"name": "Estratégia", "id": "strategy"},
                {"name": "CAGR", "id": "cagr", "type": "numeric", "format": {"specifier": ".3%"}},
                {"name": "Sharpe", "id": "sharpe", "type": "numeric", "format": {"specifier": ".2f"}},
            ],
            data=[],
            sort_action="native",
            filter_action="native",
            page_size=10,
            style_table={"overflowX": "auto"},
        ),
    ],
)


# ------------------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------------------


def _filter_by_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if df.empty or start is None or end is None:
        return df
    mask = (df["day"] >= pd.to_datetime(start)) & (df["day"] <= pd.to_datetime(end))
    return df.loc[mask].copy()


@app.callback(
    Output("ibov-graph", "figure"),
    Output("sentiment-graph", "figure"),
    Output("model-comparison-graph", "figure"),
    Output("model-table", "data"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("model-filter", "value"),
    Input("metric-filter", "value"),
)
def update_dashboard(start_date, end_date, selected_models, metric):
    # Gráfico do Ibovespa
    ibov_filtered = _filter_by_period(IBOV_DF, start_date, end_date)

    ibov_fig = go.Figure()
    if not ibov_filtered.empty:
        ibov_fig.add_trace(
            go.Scatter(
                x=ibov_filtered["day"],
                y=ibov_filtered["close"],
                mode="lines",
                name="Ibovespa (close)",
            )
        )
    event_filtered = LATENCY_DF.copy()
    if not event_filtered.empty:
        mask = (event_filtered["event_day"] >= pd.to_datetime(start_date)) & (
            event_filtered["event_day"] <= pd.to_datetime(end_date)
        )
        event_filtered = event_filtered.loc[mask]
        if not event_filtered.empty:
            ibov_fig.add_trace(
                go.Scatter(
                    x=event_filtered["event_day"],
                    y=[ibov_filtered["close"].median()] * len(event_filtered) if not ibov_filtered.empty else event_filtered.index,
                    mode="markers",
                    marker=dict(size=10, color="red", symbol="triangle-up"),
                    name="Eventos",
                    text=event_filtered.get("event_name") or event_filtered[event_filtered.columns[0]],
                    hovertemplate="%{text} - %{x|%Y-%m-%d}",
                )
            )
    ibov_fig.update_layout(
        title="Ibovespa com eventos",
        xaxis_title="Data",
        yaxis_title="Preço",
        hovermode="x unified",
    )

    # Gráfico de sentimento
    sentiment_filtered = _filter_by_period(SENTIMENT_DF, start_date, end_date)
    sentiment_fig = go.Figure()
    if not sentiment_filtered.empty:
        sentiment_fig.add_trace(
            go.Scatter(
                x=sentiment_filtered["day"],
                y=sentiment_filtered["sentiment"],
                mode="lines",
                fill="tozeroy",
                name="Sentimento médio",
            )
        )
        sentiment_fig.add_hline(y=0, line_dash="dash", line_color="gray")
    sentiment_fig.update_layout(
        title="Sentimento médio diário",
        xaxis_title="Data",
        yaxis_title="Sentimento (escala -1/+1)",
        hovermode="x unified",
    )

    # Gráfico de comparação de modelos
    comparison_fig = go.Figure()
    table_df = RESULTS_DF.copy()
    
    # Filtrar por modelos selecionados
    if selected_models:
        table_df = table_df[table_df["model"].isin(selected_models)]
    
    # Ordenar pela métrica selecionada
    if metric in {"auc", "mda", "sharpe"} and metric in table_df.columns:
        table_df_sorted = table_df.dropna(subset=[metric]).sort_values(metric, ascending=False)
        
        if not table_df_sorted.empty:
            # Mapear labels das métricas
            metric_labels = {"auc": "AUC", "mda": "MDA", "sharpe": "Sharpe Ratio"}
            
            comparison_fig.add_trace(
                go.Bar(
                    x=table_df_sorted["model"],
                    y=table_df_sorted[metric],
                    text=table_df_sorted[metric].round(3),
                    textposition="auto",
                    name=metric_labels.get(metric, metric.upper()),
                    marker=dict(
                        color=table_df_sorted[metric],
                        colorscale="Viridis",
                        showscale=True,
                    ),
                )
            )
            
            comparison_fig.update_layout(
                title=f"Comparação de Modelos por {metric_labels.get(metric, metric.upper())}",
                xaxis_title="Modelo",
                yaxis_title=metric_labels.get(metric, metric.upper()),
                hovermode="x unified",
                showlegend=False,
            )
    
    # Ordenar tabela pela métrica
    if metric in {"auc", "mda", "sharpe"} and metric in table_df.columns:
        table_df = table_df.sort_values(metric, ascending=False, na_position="last")
    table_df = table_df.fillna("")

    return ibov_fig, sentiment_fig, comparison_fig, table_df.to_dict("records")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("Iniciando dashboard em http://localhost:8050 ...")
    app.run(debug=True)
