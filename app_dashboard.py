"""
Plotly Dash application for the TCC USP sentiment x Ibovespa project.

Run with:
    python app_dashboard.py

The app reads the same artifacts used in `20_final_dashboard_analysis.ipynb`
and exposes interactive filters for period, modelo e m├®trica.
"""
# cSpell:ignore Ibovespa m├®trica Carregamento colunas ibov Arquivos principais precisa coluna proba cagr eventos modelo arquivo Sentimento Per├¡odo modelos Not├¡cia Not├¡cias Comparativo Estrat├®gia Gr├ífico hovertemplate Pre├ºo hovermode sentiment m├®dio di├írio escala tozeroy hline Selecione Tabela

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
DATE_ALIASES = [COL_DATE, "day", "date", "data", "Data", "DATA", "event_day"]

# Arquivos principais
IBOV_PATH = cfg.get_arquivo("ibov_clean", BASE_PATH)
OOF_PATH = DATA_PATHS["data_processed"] / "16_oof_predictions.csv"
RESULTS16_PATH = cfg.get_arquivo("tfidf_daily_matrix", BASE_PATH).with_name("results_16_models_tfidf.json")
BACKTEST_PATH = DATA_PATHS["data_processed"] / "18_backtest_results.csv"
BACKTEST_CURVES_PATH = DATA_PATHS["data_processed"] / "18_backtest_daily_curves.csv"
LATENCY_PATH = cfg.get_arquivo("latency_events", BASE_PATH)
START_TS = pd.Timestamp(START_DATE)
END_TS = pd.Timestamp(END_DATE)


def _safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        print(f"[aviso] Arquivo não encontrado: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as exc:
        print(f"[aviso] Falha ao ler {path}: {exc}")
        return pd.DataFrame()


def normalize_day(df: pd.DataFrame, candidates: List[str] = DATE_ALIASES, drop_duplicates: bool = True) -> pd.DataFrame:
    if df.empty:
        return df
    chosen = None
    for col in candidates:
        if col in df.columns:
            chosen = col
            break
    if chosen is None:
        return pd.DataFrame()
    df = df.copy()
    df["day"] = pd.to_datetime(df[chosen], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["day"])
    df = df[(df["day"] >= START_TS) & (df["day"] <= END_TS)]
    df = df.sort_values("day")
    if drop_duplicates:
        df = df.drop_duplicates(subset=["day"])
    return df


def load_ibov() -> pd.DataFrame:
    df = normalize_day(_safe_read_csv(IBOV_PATH))
    if df.empty:
        return df
    if "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]
    return df


def load_sentiment() -> pd.DataFrame:
    df = normalize_day(_safe_read_csv(OOF_PATH), drop_duplicates=False)
    if df.empty:
        return df
    df["proba"] = df.get("proba", pd.Series(dtype=float))
    df["sentiment"] = df["proba"] * 2 - 1
    agg = df.groupby(["day", "model"]).agg(
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
    df = normalize_day(_safe_read_csv(LATENCY_PATH), candidates=["event_day", "day", "date"], drop_duplicates=False)
    if df.empty:
        return df
    if "day" in df.columns:
        df["event_day"] = df["day"]
        df = df.drop(columns=["day"])
    return df


IBOV_DF = load_ibov()
SENTIMENT_DF = load_sentiment()
RESULTS_DF = load_results_table()
LATENCY_DF = load_latency_events()
BACKTEST_DF = normalize_day(_safe_read_csv(BACKTEST_CURVES_PATH), drop_duplicates=False)

# Usar constantes do plano de pesquisa como limites (2018-01-02 a 2024-12-31)
# FIXO: nunca mais mudar esses valores automaticamente
DATE_MIN = pd.Timestamp(START_DATE)
DATE_MAX = pd.Timestamp(END_DATE)

MODEL_OPTIONS = sorted(RESULTS_DF["model"].dropna().unique()) if not RESULTS_DF.empty and "model" in RESULTS_DF.columns else []
print(f"[DEBUG] MODEL_OPTIONS carregados: {MODEL_OPTIONS}")
print(f"[DEBUG] RESULTS_DF shape: {RESULTS_DF.shape}")
print(f"[DEBUG] IBOV_DF shape: {IBOV_DF.shape}")
print(f"[DEBUG] SENTIMENT_DF shape: {SENTIMENT_DF.shape}")
METRIC_OPTIONS = [
    {"label": "AUC", "value": "auc"},
    {"label": "MDA", "value": "mda"},
    {"label": "Sharpe", "value": "sharpe"},
]

CARD_STYLE = {
    "backgroundColor": "white",
    "padding": "20px",
    "borderRadius": "10px",
    "marginBottom": "25px",
    "boxShadow": "0 8px 20px rgba(0,0,0,0.06)",
}

GRID_STYLE = {"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(420px, 1fr))", "gap": "20px"}

# ------------------------------------------------------------------------------
# Dash App
# ------------------------------------------------------------------------------

app = Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap",
    ],
)
app.title = "Dashboard Sentimento x Ibovespa"


def _build_controls():
    return html.Div(
        style={
            "backgroundColor": "#f8f9fa",
            "padding": "20px",
            "borderRadius": "8px",
            "marginBottom": "25px",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
        },
        children=[
            html.H3("Controles de An├ílise", style={"marginTop": "0", "marginBottom": "20px", "color": "#2c3e50"}),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(250px, 1fr))", "gap": "20px"},
                children=[
                    html.Div(
                        [
                            html.Label("Per├¡odo de An├ílise", style={"fontWeight": "bold", "marginBottom": "8px", "display": "block"}),
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
                            html.Label("Selecione os Modelos", style={"fontWeight": "bold", "marginBottom": "8px", "display": "block"}),
                            dcc.Dropdown(
                                id="model-filter",
                                options=[{"label": m, "value": m} for m in MODEL_OPTIONS],
                                value=MODEL_OPTIONS,
                                multi=True,
                                placeholder="Escolha um ou mais modelos...",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("M├®trica de Avalia├º├úo", style={"fontWeight": "bold", "marginBottom": "8px", "display": "block"}),
                            dcc.Dropdown(
                                id="metric-filter",
                                options=METRIC_OPTIONS,
                                value="auc",
                                clearable=False,
                                placeholder="Selecione AUC, MDA ou Sharpe",
                            ),
                        ]
                    ),
                ],
            ),
        ],
    )


app.layout = html.Div(
    style={"fontFamily": "'Poppins','Segoe UI','Helvetica Neue',sans-serif", "maxWidth": "1400px", "margin": "0 auto", "padding": "20px"},
    children=[
        # Cabe├ºalho
        html.Div(
            style={"textAlign": "center", "marginBottom": "30px"},
            children=[
                html.H1(
                    "Sentimento de Not├¡cias x Ibovespa",
                    style={"fontSize": "2.5em", "color": "#1a1a1a", "marginBottom": "10px", "fontWeight": "600"}
                ),
                html.P(
                    "An├ílise Preditiva com Modelos de Machine Learning | TCC USP",
                    style={"fontSize": "1.1em", "color": "#666", "marginTop": "0"}
                ),
            ],
        ),
        
        # Card 1: Controles
        _build_controls(),
        
        # Indicadores de Filtros Ativos
        html.Div(
            id="active-filters-indicator",
            style={
                "backgroundColor": "#e3f2fd",
                "padding": "15px 20px",
                "borderRadius": "8px",
                "marginBottom": "25px",
                "borderLeft": "4px solid #2196f3",
            },
        ),
        
        # Card 2: Ibovespa
        html.Div(
            style=CARD_STYLE,
            children=[
                html.Div([
                    html.H3("Ibovespa com Eventos", style={"marginTop": "0", "marginBottom": "5px", "color": "#2c3e50"}),
                    html.P("Evolu├º├úo do ├¡ndice Bovespa com marcadores de eventos relevantes", 
                           style={"fontSize": "0.9em", "color": "#666", "marginTop": "0", "marginBottom": "15px"}),
                ]),
                dcc.Graph(id="ibov-graph", config={"displayModeBar": True}),
            ],
        ),
        
        # Card 3: Sentimento
        html.Div(
            style=CARD_STYLE,
            children=[
                html.Div([
                    html.H3("Sentimento M├®dio Di├írio", style={"marginTop": "0", "marginBottom": "5px", "color": "#2c3e50"}),
                    html.P("Sentimento agregado das not├¡cias (escala -1 a +1, onde valores positivos indicam otimismo)", 
                           style={"fontSize": "0.9em", "color": "#666", "marginTop": "0", "marginBottom": "15px"}),
                ]),
                dcc.Graph(id="sentiment-graph", config={"displayModeBar": True}),
            ],
        ),
        
        # Card 4: Comparativo de Modelos
        html.Div(
            style=CARD_STYLE,
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "20px"},
                    children=[
                        html.H3("Comparativo de Modelos", style={"marginTop": "0", "marginBottom": "0", "color": "#2c3e50"}),
                        html.Span(id="metric-badge", style={
                            "backgroundColor": "#4caf50",
                            "color": "white",
                            "padding": "8px 16px",
                            "borderRadius": "20px",
                            "fontSize": "0.9em",
                            "fontWeight": "bold",
                        }),
                    ],
                ),
                dcc.Graph(id="model-comparison-graph", config={"displayModeBar": True}),
                html.Hr(style={"margin": "30px 0"}),
                html.H4("Detalhamento das M├®tricas", style={"marginBottom": "15px", "color": "#34495e"}),
                dash_table.DataTable(
                    id="model-table",
                    columns=[
                        {"name": "Modelo", "id": "model"},
                        {"name": "Dataset", "id": "dataset"},
                        {"name": "AUC", "id": "auc", "type": "numeric", "format": {"specifier": ".3f"}},
                        {"name": "MDA", "id": "mda", "type": "numeric", "format": {"specifier": ".3f"}},
                        {"name": "Estrat├®gia", "id": "strategy"},
                        {"name": "CAGR", "id": "cagr", "type": "numeric", "format": {"specifier": "+.2%"}},
                        {"name": "Sharpe", "id": "sharpe", "type": "numeric", "format": {"specifier": ".2f"}},
                    ],
                    data=[],
                    sort_action="native",
                    filter_action="native",
                    page_size=10,
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": "#2c3e50",
                        "color": "white",
                        "fontWeight": "bold",
                        "textAlign": "center",
                    },
                    style_cell={
                        "textAlign": "center",
                        "padding": "10px",
                    },
                    style_data_conditional=[
                        {
                            "if": {"row_index": 0},
                            "backgroundColor": "#e8f5e9",
                            "fontWeight": "600",
                        }
                    ],
                ),
            ],
        ),

        # Painel 5: Correlação, Rolling e Distribuição
        html.Div(
            style=GRID_STYLE,
            children=[
                html.Div(
                    style=CARD_STYLE,
                    children=[
                        html.Div([
                            html.H3("Dispersão Sentimento x Retorno", style={"marginTop": "0", "marginBottom": "5px", "color": "#2c3e50"}),
                            html.P("Correlação de Pearson entre sentimento diário e retorno do Ibovespa", 
                                   style={"fontSize": "0.9em", "color": "#666", "marginTop": "0", "marginBottom": "15px"}),
                        ]),
                        dcc.Graph(id="corr-graph", config={"displayModeBar": True}),
                    ],
                ),
                html.Div(
                    style=CARD_STYLE,
                    children=[
                        html.Div([
                            html.H3("Correlação Móvel", style={"marginTop": "0", "marginBottom": "5px", "color": "#2c3e50"}),
                            html.P("Janelas de 60d e 90d entre sentimento agregado e retorno diário", 
                                   style={"fontSize": "0.9em", "color": "#666", "marginTop": "0", "marginBottom": "15px"}),
                        ]),
                        dcc.Graph(id="rolling-corr-graph", config={"displayModeBar": True}),
                    ],
                ),
                html.Div(
                    style=CARD_STYLE,
                    children=[
                        html.Div([
                            html.H3("Distribuição do Sentimento", style={"marginTop": "0", "marginBottom": "5px", "color": "#2c3e50"}),
                            html.P("Histograma e boxplot para avaliar estabilidade e outliers de sentimento", 
                                   style={"fontSize": "0.9em", "color": "#666", "marginTop": "0", "marginBottom": "15px"}),
                        ]),
                        dcc.Graph(id="sentiment-dist-graph", config={"displayModeBar": True}),
                    ],
                ),
            ],
        ),

        # Painel 6: Latência e Backtest
        html.Div(
            style=GRID_STYLE,
            children=[
                html.Div(
                    style=CARD_STYLE,
                    children=[
                        html.Div([
                            html.H3("Latência por Fonte/Daypart", style={"marginTop": "0", "marginBottom": "5px", "color": "#2c3e50"}),
                            html.P("Tempo médio de resposta por fonte e faixa do dia (se disponível)", 
                                   style={"fontSize": "0.9em", "color": "#666", "marginTop": "0", "marginBottom": "15px"}),
                        ]),
                        dcc.Graph(id="latency-graph", config={"displayModeBar": True}),
                    ],
                ),
                html.Div(
                    style=CARD_STYLE,
                    children=[
                        html.Div([
                            html.H3("Curva de Estratégia vs Benchmark", style={"marginTop": "0", "marginBottom": "5px", "color": "#2c3e50"}),
                            html.P("Evolução do patrimônio das estratégias do backtest", 
                                   style={"fontSize": "0.9em", "color": "#666", "marginTop": "0", "marginBottom": "15px"}),
                        ]),
                        dcc.Graph(id="backtest-graph", config={"displayModeBar": True}),
                    ],
                ),
            ],
        ),
    ],
)


# ------------------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------------------


def _filter_by_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if df.empty or start is None or end is None:
        return df
    start_ts = pd.to_datetime(start, errors="coerce")
    end_ts = pd.to_datetime(end, errors="coerce")
    if pd.isna(start_ts) or pd.isna(end_ts):
        return df
    start_ts = max(start_ts.tz_localize(None), START_TS)
    end_ts = min(end_ts.tz_localize(None), END_TS)
    mask = (df["day"] >= start_ts) & (df["day"] <= end_ts)
    return df.loc[mask].copy()


def _empty_fig(title: str, note: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=note, x=0.5, y=0.5, showarrow=False, font=dict(color="#7f8c8d", size=14))
    fig.update_layout(title=title, template="plotly_white", xaxis_visible=False, yaxis_visible=False)
    return fig


def _resolve_period(start_date, end_date) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = pd.to_datetime(start_date, errors="coerce") if start_date else DATE_MIN
    end_ts = pd.to_datetime(end_date, errors="coerce") if end_date else DATE_MAX
    start_ts = max(start_ts, DATE_MIN)
    end_ts = min(end_ts, DATE_MAX)
    return start_ts, end_ts


def build_correlation_fig(ibov_slice: pd.DataFrame, sent_slice: pd.DataFrame) -> go.Figure:
    corr_fig = _empty_fig("Correlacao Sentimento x Retorno", "Sem dados disponiveis no periodo selecionado")
    merged = pd.merge(ibov_slice, sent_slice, on="day", how="inner") if not ibov_slice.empty and not sent_slice.empty else pd.DataFrame()
    if merged.empty or not {"close", "sentiment"}.issubset(merged.columns):
        return corr_fig

    merged = merged.sort_values("day")
    merged["return"] = merged["close"].pct_change()
    merged = merged.dropna(subset=["return", "sentiment"])
    if merged.empty:
        return corr_fig

    corr_val = merged["return"].corr(merged["sentiment"])
    corr_fig = go.Figure(
        data=[
            go.Scatter(
                x=merged["sentiment"],
                y=merged["return"],
                mode="markers",
                marker=dict(color="#1f77b4", opacity=0.75, size=9),
                name="Retorno vs Sentimento",
                hovertemplate="Sentimento=%{x:.3f}<br>Retorno=%{y:.3%}",
            )
        ]
    )
    corr_fig.add_annotation(
        text=f"Corr Pearson: {corr_val:.2f} (n={len(merged)})",
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.99,
        showarrow=False,
        font=dict(color="#2c3e50", size=12),
        bgcolor="rgba(255,255,255,0.8)",
    )
    corr_fig.update_layout(
        title="Dispersao Sentimento x Retorno",
        xaxis_title="Sentimento diario (media)",
        yaxis_title="Retorno diario do Ibovespa",
        template="plotly_white",
    )
    return corr_fig


def build_latency_fig(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> go.Figure:
    latency_fig = _empty_fig("Latencia por fonte/daypart", "Sem eventos de latencia no periodo")
    if LATENCY_DF.empty or "event_day" not in LATENCY_DF.columns:
        return latency_fig
    latency_slice = LATENCY_DF[(LATENCY_DF["event_day"] >= start_ts) & (LATENCY_DF["event_day"] <= end_ts)]
    if latency_slice.empty:
        return latency_fig

    if "daypart" in latency_slice.columns:
        grouped = latency_slice.groupby("daypart").size().reset_index(name="n")
        x_vals = grouped["daypart"]
        y_vals = grouped["n"]
        title = "Eventos de latencia por faixa do dia"
    elif "fonte" in latency_slice.columns:
        grouped = latency_slice.groupby("fonte").size().reset_index(name="n").sort_values("n", ascending=False).head(12)
        x_vals = grouped["fonte"]
        y_vals = grouped["n"]
        title = "Eventos de latencia por fonte"
    else:
        latency_slice["event_day"] = pd.to_datetime(latency_slice["event_day"])
        x_vals = latency_slice["event_day"]
        y_vals = [1] * len(latency_slice)
        title = "Eventos de latencia"

    latency_fig = go.Figure(
        data=[
            go.Bar(
                x=x_vals,
                y=y_vals,
                marker=dict(color="#ff7043"),
                hovertemplate="%{x}<br>Eventos=%{y}",
            )
        ]
    )
    latency_fig.update_layout(
        title=title,
        xaxis_title="Categoria",
        yaxis_title="Quantidade de eventos",
        template="plotly_white",
        showlegend=False,
    )
    return latency_fig


def build_backtest_fig(start_ts: pd.Timestamp, end_ts: pd.Timestamp, selected_models) -> go.Figure:
    backtest_fig = _empty_fig("Curva de estrategia", "Sem resultados de backtest neste periodo")
    if BACKTEST_DF.empty or "equity" not in BACKTEST_DF.columns:
        return backtest_fig

    curve = BACKTEST_DF[(BACKTEST_DF["day"] >= start_ts) & (BACKTEST_DF["day"] <= end_ts)]
    if selected_models and "model" in curve.columns:
        allowed = selected_models if isinstance(selected_models, list) else [selected_models]
        curve = curve[curve["model"].isin(allowed)]
    if curve.empty:
        return backtest_fig

    backtest_fig = go.Figure()
    for name, grp in curve.groupby("strategy"):
        backtest_fig.add_trace(go.Scatter(x=grp["day"], y=grp["equity"], mode="lines", name=name))
    backtest_fig.update_layout(
        title="Curva de estrategia do backtest",
        xaxis_title="Data",
        yaxis_title="Equity normalizado",
        template="plotly_white",
        hovermode="x unified",
    )
    return backtest_fig


def build_rolling_corr_fig(ibov_slice: pd.DataFrame, sent_slice: pd.DataFrame) -> go.Figure:
    fig = _empty_fig("Correlacao movel (retorno x sentimento)", "Sem dados suficientes para rolling")
    merged = pd.merge(ibov_slice, sent_slice, on="day", how="inner") if not ibov_slice.empty and not sent_slice.empty else pd.DataFrame()
    if merged.empty or not {"close", "sentiment"}.issubset(merged.columns):
        return fig

    merged = merged.sort_values("day")
    merged["return"] = merged["close"].pct_change()
    merged = merged.dropna(subset=["return", "sentiment"])
    if merged.empty:
        return fig

    merged = merged.set_index("day")
    roll_60 = merged["return"].rolling(window=60).corr(merged["sentiment"])
    roll_90 = merged["return"].rolling(window=90).corr(merged["sentiment"])

    fig = go.Figure()
    if roll_60.dropna().size > 0:
        fig.add_trace(go.Scatter(x=roll_60.index, y=roll_60, mode="lines", name="Rolling 60d", line=dict(color="#2980b9")))
    if roll_90.dropna().size > 0:
        fig.add_trace(go.Scatter(x=roll_90.index, y=roll_90, mode="lines", name="Rolling 90d", line=dict(color="#e67e22")))

    fig.add_hline(y=0, line_dash="dash", line_color="#95a5a6")
    fig.update_layout(
        title="Correlacao movel (retorno vs sentimento)",
        xaxis_title="Data",
        yaxis_title="Correlacao",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def build_sentiment_dist_fig(sent_slice: pd.DataFrame) -> go.Figure:
    fig = _empty_fig("Distribuicao do sentimento", "Sem dados de sentimento no periodo")
    if sent_slice.empty or "sentiment" not in sent_slice.columns:
        return fig

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(
        go.Histogram(
            x=sent_slice["sentiment"],
            name="Histograma",
            marker=dict(color="#6c5ce7"),
            opacity=0.8,
            nbinsx=40,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Box(
            x=sent_slice["sentiment"],
            name="Boxplot",
            marker_color="#2c3e50",
            boxpoints="outliers",
            orientation="h",
        ),
        row=2,
        col=1,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#95a5a6")
    fig.update_layout(
        title="Distribuicao do sentimento",
        template="plotly_white",
        showlegend=False,
        xaxis_title="Sentimento",
    )
    return fig


def update_additional_graphs(start_date, end_date, selected_model):
    start_ts, end_ts = _resolve_period(start_date, end_date)
    ibov_slice = _filter_by_period(IBOV_DF, start_ts, end_ts)
    sent_slice = _filter_by_period(SENTIMENT_DF, start_ts, end_ts)

    corr_fig = build_correlation_fig(ibov_slice, sent_slice)
    latency_fig = build_latency_fig(start_ts, end_ts)
    backtest_fig = build_backtest_fig(start_ts, end_ts, selected_model)

    return corr_fig, latency_fig, backtest_fig


@app.callback(
    Output("ibov-graph", "figure"),
    Output("sentiment-graph", "figure"),
    Output("model-comparison-graph", "figure"),
    Output("model-table", "data"),
    Output("active-filters-indicator", "children"),
    Output("metric-badge", "children"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("model-filter", "value"),
    Input("metric-filter", "value"),
)
def update_dashboard(start_date, end_date, selected_models, metric):
    print(f"[DEBUG] Callback acionado: start={start_date}, end={end_date}, models={selected_models}, metric={metric}")
    
    start_ts, end_ts = _resolve_period(start_date, end_date)

    # Gr├áfico do Ibovespa
    ibov_filtered = _filter_by_period(IBOV_DF, start_ts, end_ts)

    ibov_fig = go.Figure()
    if not ibov_filtered.empty:
        ibov_fig.add_trace(
            go.Scatter(
                x=ibov_filtered["day"],
                y=ibov_filtered["close"],
                mode="lines",
                name="Ibovespa",
                line=dict(color="#1f77b4", width=2.5),
            )
        )
    event_filtered = LATENCY_DF.copy()
    if not event_filtered.empty:
        mask = (event_filtered["event_day"] >= start_ts) & (event_filtered["event_day"] <= end_ts)
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
        xaxis_title="Data",
        yaxis_title="Pre├ºo do Ibovespa (pontos)",
        hovermode="x unified",
        template="plotly_white",
        font=dict(size=12),
        xaxis=dict(
            tickformat="%b %d\n%Y",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
            tickformat=",.0f",
        ),
        margin=dict(l=60, r=20, t=40, b=60),
    )

    # Gr├ífico de sentimento
    sentiment_filtered = _filter_by_period(SENTIMENT_DF, start_ts, end_ts)
    if selected_models:
        sentiment_filtered = sentiment_filtered[sentiment_filtered["model"].isin(selected_models)]
    sentiment_fig = go.Figure()
    if not sentiment_filtered.empty:
        for model_name, grp in sentiment_filtered.groupby("model"):
            grp = grp.sort_values("day")
            sentiment_fig.add_trace(
                go.Scatter(
                    x=grp["day"],
                    y=grp["sentiment"],
                    mode="lines",
                    name=f"Sentimento {model_name}",
                    line=dict(width=2),
                    fill="tozeroy",
                    fillcolor="rgba(76,175,80,0.15)",
                )
            )
        sentiment_fig.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.3)", line_width=1)
    sentiment_fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Sentimento (escala -1 a +1)",
        hovermode="x unified",
        template="plotly_white",
        font=dict(size=12),
        xaxis=dict(
            tickformat="%b %d\n%Y",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
            zeroline=True,
            zerolinecolor="rgba(0,0,0,0.3)",
            zerolinewidth=2,
        ),
        margin=dict(l=60, r=20, t=40, b=60),
    )

    # Gr├ífico de compara├º├úo de modelos
    comparison_fig = go.Figure()
    table_df = RESULTS_DF.copy()
    
    # Filtrar por modelos selecionados
    if selected_models:
        table_df = table_df[table_df["model"].isin(selected_models)]
    
    # Ordenar pela m├®trica selecionada
    if metric in {"auc", "mda", "sharpe"} and metric in table_df.columns:
        table_df_sorted = table_df.dropna(subset=[metric]).sort_values(metric, ascending=False)
        
        if not table_df_sorted.empty:
            # Mapear labels das m├®tricas
            metric_labels = {"auc": "AUC", "mda": "MDA (%)", "sharpe": "Sharpe Ratio"}
            metric_formats = {"auc": ".3f", "mda": ".3f", "sharpe": ".2f"}
            
            # Identificar melhor modelo (primeira posi├º├úo ap├│s ordena├º├úo)
            best_model = table_df_sorted.iloc[0]["model"]
            colors = ["#2ecc71" if model == best_model else "#3498db" 
                      for model in table_df_sorted["model"]]
            
            # Formatar texto nas barras
            if metric in {"auc", "mda"}:
                text_values = [f"{v:.3f}" for v in table_df_sorted[metric]]
            else:  # sharpe
                text_values = [f"{v:.2f}" for v in table_df_sorted[metric]]
            
            comparison_fig.add_trace(
                go.Bar(
                    x=table_df_sorted["model"],
                    y=table_df_sorted[metric],
                    text=text_values,
                    textposition="outside",
                    name=metric_labels.get(metric, metric.upper()),
                    marker=dict(
                        color=colors,
                        line=dict(
                            color=["#27ae60" if model == best_model else "#2980b9" 
                                   for model in table_df_sorted["model"]],
                            width=2,
                        ),
                    ),
                )
            )
            
            comparison_fig.update_layout(
                xaxis_title="Modelo",
                yaxis_title=metric_labels.get(metric, metric.upper()),
                hovermode="x unified",
                showlegend=False,
                template="plotly_white",
                font=dict(size=12),
                xaxis=dict(
                    showgrid=False,
                    tickangle=-45,
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(0,0,0,0.1)",
                ),
                margin=dict(l=60, r=20, t=40, b=100),
            )
    
    # Ordenar tabela pela m├®trica
    if metric in {"auc", "mda", "sharpe"} and metric in table_df.columns:
        table_df = table_df.sort_values(metric, ascending=False, na_position="last")
    table_df = table_df.fillna("")

    # Criar indicadores visuais de filtros ativos
    metric_labels = {"auc": "AUC", "mda": "MDA", "sharpe": "Sharpe Ratio"}
    
    # Calcular janela temporal
    days_count = (end_ts - start_ts).days + 1
    ibov_days = len(ibov_filtered) if not ibov_filtered.empty else 0
    sentiment_days = len(sentiment_filtered) if not sentiment_filtered.empty else 0
    
    # Criar lista de modelos selecionados
    models_text = ", ".join(selected_models) if selected_models else "Nenhum modelo selecionado"
    if selected_models and len(selected_models) == len(MODEL_OPTIONS):
        models_text = f"Todos os modelos ({len(MODEL_OPTIONS)})"
    
    indicator_content = html.Div(
        style={"display": "flex", "flexWrap": "wrap", "gap": "20px", "alignItems": "center"},
        children=[
            html.Div([
                html.Strong("Periodo: ", style={"color": "#1976d2"}),
                html.Span(f"{start_date} a {end_date} ({days_count} dias)"),
            ]),
            html.Div([
                html.Strong("Dados: ", style={"color": "#1976d2"}),
                html.Span(f"Ibovespa: {ibov_days} dias | Sentimento: {sentiment_days} dias"),
            ]),
            html.Div([
                html.Strong("Modelos: ", style={"color": "#1976d2"}),
                html.Span(models_text),
            ]),
            html.Div([
                html.Strong("Metrica: ", style={"color": "#1976d2"}),
                html.Span(metric_labels.get(metric, metric.upper()), style={"fontWeight": "600", "color": "#2e7d32"}),
            ]),
        ],
    )

    metric_badge_text = f"Metrica: {metric_labels.get(metric, metric.upper())}"
    
    return ibov_fig, sentiment_fig, comparison_fig, table_df.to_dict("records"), indicator_content, metric_badge_text


@app.callback(
    Output("corr-graph", "figure"),
    Output("rolling-corr-graph", "figure"),
    Output("sentiment-dist-graph", "figure"),
    Output("latency-graph", "figure"),
    Output("backtest-graph", "figure"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("model-filter", "value"),
)
def update_secondary_charts(start_date, end_date, selected_models):
    start_ts, end_ts = _resolve_period(start_date, end_date)
    ibov_slice = _filter_by_period(IBOV_DF, start_ts, end_ts)
    sent_slice = _filter_by_period(SENTIMENT_DF, start_ts, end_ts)
    corr_fig, latency_fig, backtest_fig = update_additional_graphs(start_date, end_date, selected_models)
    rolling_fig = build_rolling_corr_fig(ibov_slice, sent_slice)
    dist_fig = build_sentiment_dist_fig(sent_slice)
    return corr_fig, rolling_fig, dist_fig, latency_fig, backtest_fig


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    host = os.getenv("DASH_HOST", "127.0.0.1")
    port = int(os.getenv("DASH_PORT", "8050"))
    print(f"Iniciando dashboard em http://{host}:{port} ...")
    app.run(host=host, port=port, debug=False, use_reloader=False)
