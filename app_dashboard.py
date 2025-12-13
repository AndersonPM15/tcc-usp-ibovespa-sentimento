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
        print(f"[aviso] Arquivo n├úo encontrado: {path}")
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
    df = df[df["day"] <= pd.Timestamp(END_DATE)]
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
    agg = agg[agg["day"] <= pd.Timestamp(END_DATE)]
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
    df = df[df["event_day"] <= pd.Timestamp(END_DATE)]
    return df


IBOV_DF = load_ibov()
SENTIMENT_DF = load_sentiment()
RESULTS_DF = load_results_table()
LATENCY_DF = load_latency_events()
BACKTEST_DF = _safe_read_csv(BACKTEST_PATH)
if not BACKTEST_DF.empty and "day" in BACKTEST_DF.columns:
    BACKTEST_DF["day"] = pd.to_datetime(BACKTEST_DF["day"], errors="coerce")
    BACKTEST_DF = BACKTEST_DF.dropna(subset=["day"])
    BACKTEST_DF = BACKTEST_DF[BACKTEST_DF["day"] <= pd.Timestamp(END_DATE)]

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

# ------------------------------------------------------------------------------
# Dash App
# ------------------------------------------------------------------------------

app = Dash(__name__)
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
    style={"fontFamily": "Arial, sans-serif", "maxWidth": "1400px", "margin": "0 auto", "padding": "20px"},
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
            style={
                "backgroundColor": "white",
                "padding": "20px",
                "borderRadius": "8px",
                "marginBottom": "25px",
                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
            },
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
            style={
                "backgroundColor": "white",
                "padding": "20px",
                "borderRadius": "8px",
                "marginBottom": "25px",
                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
            },
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
            style={
                "backgroundColor": "white",
                "padding": "20px",
                "borderRadius": "8px",
                "marginBottom": "25px",
                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
            },
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


def _empty_fig(title: str, note: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=note, x=0.5, y=0.5, showarrow=False)
    fig.update_layout(title=title, template="plotly_white")
    return fig


def update_additional_graphs(start_date, end_date, selected_model):
    start_ts = pd.to_datetime(start_date) if start_date else DATE_MIN
    end_ts = pd.to_datetime(end_date) if end_date else DATE_MAX

    # Correlation between sentiment and Ibovespa returns
    corr_fig = _empty_fig("Correlação Sentimento x Retorno", "Sem dados disponíveis")
    ibov_slice = _filter_by_period(IBOV_DF, start_ts, end_ts)
    sent_slice = _filter_by_period(SENTIMENT_DF, start_ts, end_ts)
    merged = pd.merge(ibov_slice, sent_slice, on="day", how="inner") if not ibov_slice.empty and not sent_slice.empty else pd.DataFrame()
    if not merged.empty and {"close", "sentiment"}.issubset(merged.columns):
        merged = merged.sort_values("day")
        merged["return"] = merged["close"].pct_change()
        merged = merged.dropna(subset=["return", "sentiment"])
        if not merged.empty:
            corr_val = merged["return"].corr(merged["sentiment"])
            corr_fig = go.Figure(
                data=[
                    go.Scatter(
                        x=merged["sentiment"],
                        y=merged["return"],
                        mode="markers",
                        marker=dict(color="#1f77b4", opacity=0.7),
                        name="Retorno vs Sentimento",
                    )
                ]
            )
            corr_fig.update_layout(
                title=f"Retorno do Ibovespa vs Sentimento (corr={corr_val:.2f})",
                xaxis_title="Sentimento",
                yaxis_title="Retorno diário",
                template="plotly_white",
            )

    # Latency events over time
    latency_fig = _empty_fig("Eventos de Latência", "Sem eventos cadastrados")
    if not LATENCY_DF.empty and "event_day" in LATENCY_DF.columns:
        latency_slice = LATENCY_DF[(LATENCY_DF["event_day"] >= start_ts) & (LATENCY_DF["event_day"] <= end_ts)]
        if not latency_slice.empty:
            latency_fig = go.Figure(
                data=[
                    go.Bar(
                        x=latency_slice["event_day"],
                        y=[1] * len(latency_slice),
                        text=latency_slice.get("event_name", latency_slice.columns[0]),
                        hovertemplate="%{text}<br>%{x|%Y-%m-%d}",
                    )
                ]
            )
            latency_fig.update_layout(
                title="Eventos de Latência",
                xaxis_title="Data",
                yaxis_title="Contagem",
                showlegend=False,
                template="plotly_white",
            )

    # Backtest equity curve
    backtest_fig = _empty_fig("Curva de Estratégia", "Sem resultados de backtest")
    if not BACKTEST_DF.empty and {"day", "equity"}.issubset(BACKTEST_DF.columns):
        curve = BACKTEST_DF[(BACKTEST_DF["day"] >= start_ts) & (BACKTEST_DF["day"] <= end_ts)]
        if selected_model and "model" in curve.columns:
            curve = curve[curve["model"].isin(selected_model if isinstance(selected_model, list) else [selected_model])]
        if not curve.empty:
            backtest_fig = go.Figure(
                data=[go.Scatter(x=curve["day"], y=curve["equity"], mode="lines", name="Equity")]
            )
            backtest_fig.update_layout(
                title="Curva de Estratégia",
                xaxis_title="Data",
                yaxis_title="Equity",
                template="plotly_white",
            )

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
    
    # Gr├ífico do Ibovespa
    ibov_filtered = _filter_by_period(IBOV_DF, start_date, end_date)

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
    sentiment_filtered = _filter_by_period(SENTIMENT_DF, start_date, end_date)
    sentiment_fig = go.Figure()
    if not sentiment_filtered.empty:
        # Criar cores condicionais (positivo vs negativo)
        colors = ["rgba(76, 175, 80, 0.3)" if s >= 0 else "rgba(244, 67, 54, 0.3)" 
                  for s in sentiment_filtered["sentiment"]]
        line_colors = ["#4caf50" if s >= 0 else "#f44336" 
                       for s in sentiment_filtered["sentiment"]]
        
        sentiment_fig.add_trace(
            go.Scatter(
                x=sentiment_filtered["day"],
                y=sentiment_filtered["sentiment"],
                mode="lines",
                fill="tozeroy",
                name="Sentimento",
                line=dict(color="#666", width=2),
                fillcolor="rgba(100, 100, 100, 0.2)",
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
    days_count = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
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
                html.Strong("­ƒôà Per├¡odo: ", style={"color": "#1976d2"}),
                html.Span(f"{start_date} a {end_date} ({days_count} dias)"),
            ]),
            html.Div([
                html.Strong("­ƒôè Dados: ", style={"color": "#1976d2"}),
                html.Span(f"Ibovespa: {ibov_days} dias | Sentimento: {sentiment_days} dias"),
            ]),
            html.Div([
                html.Strong("­ƒñû Modelos: ", style={"color": "#1976d2"}),
                html.Span(models_text),
            ]),
            html.Div([
                html.Strong("­ƒôê M├®trica: ", style={"color": "#1976d2"}),
                html.Span(metric_labels.get(metric, metric.upper()), style={"fontWeight": "600", "color": "#2e7d32"}),
            ]),
        ],
    )

    metric_badge_text = f"­ƒôè M├®trica: {metric_labels.get(metric, metric.upper())}"
    
    return ibov_fig, sentiment_fig, comparison_fig, table_df.to_dict("records"), indicator_content, metric_badge_text


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    host = os.getenv("DASH_HOST", "127.0.0.1")
    port = int(os.getenv("DASH_PORT", "8050"))
    print(f"Iniciando dashboard em http://{host}:{port} ...")
    app.run(host=host, port=port, debug=False)

