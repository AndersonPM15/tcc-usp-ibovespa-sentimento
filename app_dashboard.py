# -*- coding: utf-8 -*-
"""
Dashboard do TCC USP: Sentimento de Notícias x Ibovespa.
Run: python app_dashboard.py
"""
# cSpell:ignore Ibovespa metrica Carregamento colunas ibov proba cagr eventos modelo Sentimento Periodo modelos Noticias Comparativo Estrategia Grafico hovertemplate hovermode tozeroy hline Selecione Tabela

from __future__ import annotations

import argparse
import json
import socket
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html, ctx

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
BACKTEST_CURVES_PATH = DATA_PATHS["data_processed"] / "18_backtest_daily_curves.csv"
LATENCY_PATH = cfg.get_arquivo("latency_events", BASE_PATH)
PREFERRED_STRATEGY = "long_only_60"
FALLBACK_STRATEGY = "long_only_55"
COMPARE_MODELS_ANCHOR = ["logreg_l2", "rf_200"]

PLOTLY_CONFIG = dict(
    displayModeBar=True,
    displaylogo=False,
    responsive=True,
    scrollZoom=True,
    doubleClick="reset",
)
H_NORMAL = 620
H_EXPORT = 900
MARGIN_BASE = dict(l=55, r=20, t=40, b=45)


def _dataset_status_entry(nome: str, df: pd.DataFrame, path: Path, date_col: str | None) -> str:
    if not path.exists():
        return f"{nome}: arquivo ausente em {path}"
    if df.empty:
        return f"{nome}: arquivo vazio em {path}"
    if date_col is None or date_col not in df.columns:
        return f"{nome}: linhas={len(df)} (sem coluna de data para min/max)"
    try:
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    except Exception as exc:  # noqa: BLE001
        return f"{nome}: linhas={len(df)} (erro ao ler datas: {exc.__class__.__name__})"
    if dates.empty:
        return f"{nome}: linhas={len(df)} (datas inválidas ou vazias)"
    min_dt = dates.min()
    max_dt = dates.max()
    return f"{nome}: linhas={len(df)} | min={min_dt.date()} | max={max_dt.date()}"


def normalize_dates(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Garante coluna datetime e dropa inválidos."""
    if df.empty or col not in df.columns:
        return df
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df.dropna(subset=[col])


def filter_period(df: pd.DataFrame, start: str, end: str, col: str) -> pd.DataFrame:
    if df.empty or col not in df.columns or start is None or end is None:
        return df
    df = normalize_dates(df, col)
    mask = (df[col] >= pd.to_datetime(start)) & (df[col] <= pd.to_datetime(end))
    return df.loc[mask].copy()


def check_port(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def check_http(host: str, port: int):
    import urllib.request

    url = f"http://{host}:{port}/"
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return True, r.status, None
    except Exception as exc:  # noqa: BLE001
        return False, None, repr(exc)


def find_free_port(start_port: int) -> int:
    port = start_port
    while port < start_port + 20:
        if not check_port("127.0.0.1", port):
            return port
        port += 1
    return start_port


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
                    "strategy": row.get("strategy"),
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


def load_backtest_curves() -> pd.DataFrame:
    df = _safe_read_csv(BACKTEST_CURVES_PATH)
    if df.empty:
        return df
    date_col = "day" if "day" in df.columns else "date"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if "equity" not in df.columns and "strategy_ret" in df.columns:
        df = df.sort_values(date_col)
        df["equity"] = (1 + df["strategy_ret"].fillna(0)).cumprod()
    return df


IBOV_DF = load_ibov()
SENTIMENT_DF = load_sentiment()
RESULTS_DF = load_results_table()
LATENCY_DF = load_latency_events()
BACKTEST_DF = load_backtest_curves()
BACKTEST_RESULTS_DF = _safe_read_csv(BACKTEST_PATH)
if not BACKTEST_RESULTS_DF.empty and "dataset" not in BACKTEST_RESULTS_DF.columns:
    BACKTEST_RESULTS_DF["dataset"] = "backtest_daily"


def choose_common_strategy(backtest_df: pd.DataFrame, models: list[str], preferred: str = PREFERRED_STRATEGY) -> tuple[str | None, set[str]]:
    if backtest_df.empty or "strategy" not in backtest_df.columns or "model" not in backtest_df.columns:
        return None, set()
    common: set[str] | None = None
    for m in models:
        strategies = set(backtest_df.loc[backtest_df["model"] == m, "strategy"])
        common = strategies if common is None else common & strategies
    common = common or set()
    if not common:
        return None, set()
    if preferred in common:
        return preferred, common
    if FALLBACK_STRATEGY in common:
        return FALLBACK_STRATEGY, common
    return sorted(common)[0], common


COMMON_STRATEGY, COMMON_STRATEGIES_SET = choose_common_strategy(
    BACKTEST_RESULTS_DF.loc[BACKTEST_RESULTS_DF["dataset"] == "backtest_daily"], COMPARE_MODELS_ANCHOR
)
print(f"[DEBUG] Estratégias comuns (logreg_l2 vs rf_200): {COMMON_STRATEGIES_SET}")
if COMMON_STRATEGY:
    print(f"[DEBUG] Estratégia escolhida para comparação Sharpe: {COMMON_STRATEGY}")
else:
    print("[DEBUG] Nenhuma estratégia comum entre logreg_l2 e rf_200 em backtest_daily.")
LATENCY_AVAILABLE = not LATENCY_DF.empty and "fonte" in LATENCY_DF.columns
LATENCY_STATUS = (
    _dataset_status_entry("Latência", LATENCY_DF, LATENCY_PATH, "event_day")
    if LATENCY_AVAILABLE
    else "Latência: 0 linhas (não disponível nesta versão)"
)
DATA_STATUS = [
    _dataset_status_entry("Ibovespa", IBOV_DF, IBOV_PATH, "day"),
    _dataset_status_entry("Sentimento (OOF)", SENTIMENT_DF, OOF_PATH, "day"),
    _dataset_status_entry("Resultados TF-IDF", RESULTS_DF, RESULTS16_PATH, None),
    _dataset_status_entry("Backtest Curvas", BACKTEST_DF, BACKTEST_CURVES_PATH, "day"),
    LATENCY_STATUS,
]

# Usar constantes do plano de pesquisa como limites (2018-01-02 a 2024-12-31)
# FIXO: nunca mais mudar esses valores automaticamente
DATE_MIN = pd.Timestamp(START_DATE)
DATE_MAX = pd.Timestamp(END_DATE)

# Definir range padrão como interseção dos datasets carregados (opção A)
ranges = []
for df, col in [
    (IBOV_DF, "day"),
    (SENTIMENT_DF, "day"),
    (BACKTEST_DF, "day"),
    (LATENCY_DF, "event_day"),
]:
    if not df.empty and col in df.columns:
        dt = pd.to_datetime(df[col], errors="coerce").dropna()
        if not dt.empty:
            ranges.append((dt.min(), dt.max()))
if ranges:
    DEFAULT_START = max(r[0] for r in ranges)
    DEFAULT_END = min(r[1] for r in ranges)
    DEFAULT_START = max(DEFAULT_START, DATE_MIN)
    DEFAULT_END = min(DEFAULT_END, DATE_MAX)
else:
    DEFAULT_START = DATE_MIN
    DEFAULT_END = DATE_MAX
print(f"[DEBUG] DEFAULT_START={DEFAULT_START.date()} DEFAULT_END={DEFAULT_END.date()}")

MODEL_OPTIONS = sorted(RESULTS_DF["model"].dropna().unique()) if not RESULTS_DF.empty and "model" in RESULTS_DF.columns else []
PREFERRED_MODELS = ["logreg_l2", "rf_200"]
MODEL_DEFAULT_SELECTION = PREFERRED_MODELS if set(PREFERRED_MODELS).issubset(set(MODEL_OPTIONS)) else MODEL_OPTIONS.copy()
print(f"[DEBUG] MODEL_OPTIONS carregados: {MODEL_OPTIONS}")
print(f"[DEBUG] RESULTS_DF shape: {RESULTS_DF.shape}")
print(f"[DEBUG] IBOV_DF shape: {IBOV_DF.shape}")
print(f"[DEBUG] SENTIMENT_DF shape: {SENTIMENT_DF.shape}")
METRIC_OPTIONS = [
    {"label": "AUC", "value": "auc"},
    {"label": "MDA", "value": "mda"},
    {"label": "Sharpe", "value": "sharpe"},
]

COMMON_STRATEGY = choose_common_strategy(BACKTEST_DF, MODEL_OPTIONS)
if COMMON_STRATEGY:
    print(f"[DEBUG] Estratégia comum para comparação: {COMMON_STRATEGY}")
else:
    print("[DEBUG] Nenhuma estratégia comum encontrada para todos os modelos; usando conjunto integral.")

# ------------------------------------------------------------------------------
# Dash App
# ------------------------------------------------------------------------------

app = Dash(__name__, meta_tags=[{"charset": "utf-8"}])
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
            html.H3("Controles de Análise", style={"marginTop": "0", "marginBottom": "20px", "color": "#2c3e50"}),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(250px, 1fr))", "gap": "20px"},
                children=[
                    html.Div(
                        [
                            html.Label("Período de Análise", style={"fontWeight": "bold", "marginBottom": "8px", "display": "block"}),
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
                                value=MODEL_DEFAULT_SELECTION,
                                multi=True,
                                placeholder="Escolha um ou mais modelos...",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Métrica de Avaliação", style={"fontWeight": "bold", "marginBottom": "8px", "display": "block"}),
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
    id="page-container",
    className="page-container",
    children=[
        html.Div(
            className="card controls-card",
            children=[
                html.Div(
                    className="header-flex",
                    children=[
                        html.Div(
                            children=[
                                html.H2("Dashboard – Sentimento de Notícias x Ibovespa", className="title-main"),
                                html.P("Período oficial 2018-01-02 a 2024-12-31 • USP", className="title-sub"),
                            ]
                        ),
                        html.Div(
                            className="export-toggle",
                            children=[
                                dcc.Checklist(
                                    id="export-toggle",
                                    options=[{"label": "Modo Exportação", "value": "export"}],
                                    value=[],
                                    inline=True,
                                )
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="controls-bar",
                    children=[
                        html.Div(
                            children=[
                                html.Label("Período de Análise"),
                                dcc.DatePickerRange(
                                    id="date-range",
                                    min_date_allowed=DATE_MIN,
                                    max_date_allowed=DATE_MAX,
                                    start_date=DEFAULT_START,
                                    end_date=DEFAULT_END,
                                    display_format="YYYY-MM-DD",
                                ),
                            ]
                        ),
                        html.Div(
                            children=[
                                html.Label("Modelo"),
                                dcc.Dropdown(
                                    id="model-filter",
                                    options=[{"label": m, "value": m} for m in MODEL_OPTIONS],
                                    value=MODEL_DEFAULT_SELECTION,
                                    multi=True,
                                    placeholder="Escolha um ou mais modelos",
                                    clearable=False,
                                ),
                            ]
                        ),
                        html.Div(
                            children=[
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
                ),
                html.Div(id="overview-kpis", className="kpi-grid"),
                html.Div(id="active-filters-indicator", className="indicator-bar", style={"marginTop": "8px"}),
                html.Div(id="ui-last-trigger", className="last-trigger"),
                html.Div(className="interpret-block", children=[
                    html.Strong("Como interpretar:"),
                    html.Ul(
                        [
                            html.Li("Período padrão = interseção das séries para evitar gráficos vazios."),
                            html.Li("Controles acima afetam todos os 8 gráficos (datas, modelo único, métrica)."),
                            html.Li("Modo Exportação oculta o cabeçalho/controles e amplia os gráficos para recorte."),
                        ]
                    ),
                ]),
            ],
        ),
        html.Div(
            className="grid-two",
            children=[
                html.Div(
                    className="card",
                    children=[
                        html.H3("Figura 1 – Ibovespa com Eventos"),
                        dcc.Graph(id="ibov-graph", config=PLOTLY_CONFIG, className="dash-graph", style={"height": f"{H_NORMAL}px", "flex": "1 1 auto"}),
                        html.Div(id="ibov-meta", className="figure-meta"),
                    ],
                ),
                html.Div(
                    className="card figure-card",
                    children=[
                        html.H3("Figura 2 – Sentimento Médio Diário"),
                        dcc.Graph(id="sentiment-graph", config=PLOTLY_CONFIG, className="dash-graph", style={"height": f"{H_NORMAL}px", "flex": "1 1 auto"}),
                        html.Div(id="sentiment-meta", className="figure-meta"),
                    ],
                ),
            ],
        ),
        html.Div(
            className="card figure-card",
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                    children=[
                        html.H3("Figura 3 – Comparativo de Modelos", style={"margin": 0}),
                        html.Span(id="metric-badge", className="badge"),
                    ],
                ),
                dcc.Graph(id="model-comparison-graph", config=PLOTLY_CONFIG, className="dash-graph", style={"height": f"{H_NORMAL}px", "flex": "1 1 auto"}),
                html.Div(id="comparison-meta", className="figure-meta"),
                html.Hr(),
                html.H4("Tabela de Métricas"),
                dash_table.DataTable(
                    id="model-table",
                    columns=[
                        {"name": "Modelo", "id": "model"},
                        {"name": "Dataset", "id": "dataset"},
                        {"name": "AUC", "id": "auc", "type": "numeric", "format": {"specifier": ".3f"}},
                        {"name": "MDA", "id": "mda", "type": "numeric", "format": {"specifier": ".3f"}},
                        {"name": "Estratégia", "id": "strategy"},
                        {"name": "CAGR", "id": "cagr", "type": "numeric", "format": {"specifier": "+.2%"}},
                        {"name": "Sharpe", "id": "sharpe", "type": "numeric", "format": {"specifier": ".2f"}},
                    ],
                    data=[],
                    sort_action="native",
                    filter_action="native",
                    page_size=10,
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": "#1f2933",
                        "color": "white",
                        "fontWeight": "bold",
                        "textAlign": "center",
                    },
                    style_cell={
                        "textAlign": "center",
                        "padding": "10px",
                    },
                    style_data_conditional=[
                        {"if": {"row_index": 0}, "backgroundColor": "#e8f5e9", "fontWeight": "600"}
                    ],
                ),
            ],
        ),
        html.Div(
            className="grid-two",
            children=[
                html.Div(
                    className="card figure-card",
                    children=[
                        html.H3("Figura 4 – Dispersão Sentimento x Retorno Diário"),
                        dcc.Graph(id="scatter-graph", config=PLOTLY_CONFIG, className="dash-graph", style={"height": f"{H_NORMAL}px", "flex": "1 1 auto"}),
                        html.Div(id="scatter-meta", className="figure-meta"),
                    ],
                ),
                html.Div(
                    className="card figure-card",
                    children=[
                        html.H3("Figura 5 – Correlação Móvel (60d/90d)"),
                        dcc.Graph(id="rolling-corr-graph", config=PLOTLY_CONFIG, className="dash-graph", style={"height": f"{H_NORMAL}px", "flex": "1 1 auto"}),
                        html.Div(id="rolling-meta", className="figure-meta"),
                    ],
                ),
            ],
        ),
        html.Div(
            className="card figure-card",
            children=[
                html.H3("Figura 6 – Distribuição do Sentimento"),
                dcc.Graph(id="sentiment-dist-graph", config=PLOTLY_CONFIG, className="dash-graph", style={"height": f"{H_NORMAL}px", "flex": "1 1 auto"}),
                html.Div(id="dist-meta", className="figure-meta"),
            ],
        ),
        html.Div(
            className="grid-two",
            children=[
                html.Div(
                    className="card figure-card",
                    children=[
                        html.H3("Figura 7 – Latência por Fonte/Daypart"),
                        dcc.Graph(id="latency-graph", config=PLOTLY_CONFIG, className="dash-graph", style={"height": f"{H_NORMAL}px", "flex": "1 1 auto"}),
                        html.Div(id="latency-meta", className="figure-meta"),
                    ],
                ),
                html.Div(
                    className="card figure-card",
                    children=[
                        html.H3("Figura 8 – Curva de Backtest"),
                        dcc.Graph(id="backtest-graph", config=PLOTLY_CONFIG, className="dash-graph", style={"height": f"{H_NORMAL}px", "flex": "1 1 auto"}),
                        html.Div(id="backtest-meta", className="figure-meta"),
                    ],
                ),
            ],
        ),
    ],
)


# ------------------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------------------


def _filter_by_period(df: pd.DataFrame, start: str, end: str, date_col: str = "day") -> pd.DataFrame:
    return filter_period(df, start, end, date_col)


def _log_df(label: str, df: pd.DataFrame, date_col: str | None) -> None:
    if df.empty:
        print(f"[DIAG] {label}: vazio")
        return
    info = {"rows": len(df)}
    if date_col and date_col in df.columns:
        series = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if not series.empty:
            info["min"] = series.min().date()
            info["max"] = series.max().date()
    print(f"[DIAG] {label}: {info}")


def _placeholder_fig(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False)
    fig.update_layout(title=title, template="plotly_white")
    return fig


def _meta_text(source: str, df: pd.DataFrame, date_col: str | None, label: str) -> str:
    if df.empty:
        return f"Fonte: {source} | Período: — | N: 0 ({label})"
    if date_col is None or date_col not in df.columns:
        return f"Fonte: {source} | N: {len(df)} ({label})"
    series = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if series.empty:
        return f"Fonte: {source} | Período: — | N: {len(df)} ({label})"
    return f"Fonte: {source} | Período: {series.min().date()} → {series.max().date()} | N: {len(df)} ({label})"


@app.callback(Output("page-container", "className"), Input("export-toggle", "value"))
def toggle_export_class(toggle_value):
    base = "page-container"
    return f"{base} export-mode" if toggle_value else base


app.clientside_callback(
    """
    function(toggle){
        const isExport = Array.isArray(toggle) && toggle.includes("export");
        const h = isExport ? %d : %d;
        const style = {"height": h + "px", "flex": "1 1 auto"};
        return [style,style,style,style,style,style,style,style];
    }
    """ % (H_EXPORT, H_NORMAL),
    Output("ibov-graph", "style"),
    Output("sentiment-graph", "style"),
    Output("model-comparison-graph", "style"),
    Output("scatter-graph", "style"),
    Output("rolling-corr-graph", "style"),
    Output("sentiment-dist-graph", "style"),
    Output("latency-graph", "style"),
    Output("backtest-graph", "style"),
    Input("export-toggle", "value"),
)


@app.callback(
    Output("ibov-graph", "figure"),
    Output("sentiment-graph", "figure"),
    Output("model-comparison-graph", "figure"),
    Output("model-table", "data"),
    Output("active-filters-indicator", "children"),
    Output("metric-badge", "children"),
    Output("ui-last-trigger", "children"),
    Output("scatter-graph", "figure"),
    Output("rolling-corr-graph", "figure"),
    Output("sentiment-dist-graph", "figure"),
    Output("latency-graph", "figure"),
    Output("backtest-graph", "figure"),
    Output("overview-kpis", "children"),
    Output("ibov-meta", "children"),
    Output("sentiment-meta", "children"),
    Output("comparison-meta", "children"),
    Output("scatter-meta", "children"),
    Output("rolling-meta", "children"),
    Output("dist-meta", "children"),
    Output("latency-meta", "children"),
    Output("backtest-meta", "children"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("model-filter", "value"),
    Input("metric-filter", "value"),
    Input("export-toggle", "value"),
)
def update_dashboard(start_date, end_date, selected_model, metric, export_toggle):
    try:
        selected_models = selected_model if isinstance(selected_model, list) else ([selected_model] if selected_model else [])
        if not selected_models:
            selected_models = MODEL_DEFAULT_SELECTION
        active_model = selected_models[0] if selected_models else None

        print(f"[DEBUG] Callback acionado: start={start_date}, end={end_date}, models={selected_models}, metric={metric}")
        export_mode = bool(export_toggle)
        graph_height = H_EXPORT if export_mode else H_NORMAL

        ibov_filtered = _filter_by_period(IBOV_DF, start_date, end_date, "day")
        sentiment_filtered = _filter_by_period(SENTIMENT_DF, start_date, end_date, "day")
        backtest_filtered = _filter_by_period(BACKTEST_DF, start_date, end_date, "day")
        event_filtered = _filter_by_period(LATENCY_DF, start_date, end_date, "event_day")

        _log_df("IBOV filtrado", ibov_filtered, "day")
        _log_df("Sentimento filtrado", sentiment_filtered, "day")
        _log_df("Backtest filtrado", backtest_filtered, "day")
        _log_df("Latência filtrada", event_filtered, "event_day")

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
        if not event_filtered.empty:
            ibov_fig.add_trace(
                go.Scatter(
                    x=event_filtered["event_day"],
                    y=[ibov_filtered["close"].median()] * len(event_filtered) if not ibov_filtered.empty else event_filtered.index,
                    mode="markers",
                    marker=dict(size=10, color="red", symbol="triangle-up"),
                    name="Eventos",
                    text=event_filtered["event_name"]
                    if "event_name" in event_filtered.columns
                    else event_filtered["fonte"]
                    if "fonte" in event_filtered.columns
                    else event_filtered.index.astype(str),
                    hovertemplate="%{text} - %{x|%Y-%m-%d}",
                )
            )
        if ibov_fig.data:
            ibov_fig.update_layout(
                xaxis_title="Data",
                yaxis_title="Preço do Ibovespa (pontos)",
                hovermode="x unified",
                template="plotly_white",
                font=dict(size=14),
                xaxis=dict(tickformat="%b %d\n%Y", showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)", tickformat=",.0f"),
                margin=MARGIN_BASE,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                height=graph_height,
                title=None,
            )
        else:
            ibov_fig = _placeholder_fig("Ibovespa com Eventos", "Sem dados no intervalo selecionado; ajuste o período.")
            ibov_fig.update_layout(height=graph_height)

        sentiment_fig = go.Figure()
        if not sentiment_filtered.empty:
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
        if sentiment_fig.data:
            sentiment_fig.update_layout(
                xaxis_title="Data",
                yaxis_title="Sentimento (escala -1 a +1)",
                hovermode="x unified",
                template="plotly_white",
                font=dict(size=14),
                xaxis=dict(tickformat="%b %d\n%Y", showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)", zeroline=True, zerolinecolor="rgba(0,0,0,0.3)", zerolinewidth=2),
                margin=MARGIN_BASE,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                height=graph_height,
                title=None,
            )
        else:
            sentiment_fig = _placeholder_fig("Sentimento Médio Diário", "Sem dados no intervalo selecionado; ajuste o período.")
            sentiment_fig.update_layout(height=graph_height)

        comparison_fig = go.Figure()
        base_df = RESULTS_DF.copy()
        if selected_models:
            base_df = base_df[base_df["model"].isin(selected_models)]
        comparison_meta = _meta_text("results_16_models_tfidf.json + 18_backtest_results.csv", base_df, None, "Modelos")
        backtest_common_df = pd.DataFrame()
        display_df_for_bars = base_df
        if metric == "sharpe":
            # força comparação justa logreg_l2 vs rf_200
            anchor_models = [m for m in COMPARE_MODELS_ANCHOR if m in MODEL_OPTIONS]
            if COMMON_STRATEGY:
                mask_common = (
                    (BACKTEST_RESULTS_DF["model"].isin(anchor_models))
                    & (BACKTEST_RESULTS_DF["strategy"] == COMMON_STRATEGY)
                    & (BACKTEST_RESULTS_DF.get("dataset", "backtest_daily") == "backtest_daily")
                )
                backtest_common_df = BACKTEST_RESULTS_DF.loc[mask_common].copy()
                print(f"[DEBUG] Estratégia comum usada em Sharpe: {COMMON_STRATEGY}")
                print(f"[DEBUG] common_strategies_set={COMMON_STRATEGIES_SET}")
                if not backtest_common_df.empty:
                    print(backtest_common_df[["model", "strategy", "cagr", "sharpe"]].to_string(index=False))
                display_df_for_bars = backtest_common_df
            else:
                comparison_fig = _placeholder_fig(
                    "Comparativo de Modelos",
                    "Sem estratégia comum entre logreg_l2 e rf_200 em backtest_daily; recalcule o backtest.",
                )
                comparison_fig.update_layout(height=graph_height)
                display_df_for_bars = pd.DataFrame()
        table_df_display = base_df if metric != "sharpe" else pd.concat(
            [base_df.loc[base_df["dataset"] != "backtest_daily"], backtest_common_df], ignore_index=True
        )
        best_model_name = None
        best_metric_val = None
        metric_labels = {"auc": "AUC", "mda": "MDA", "sharpe": "Sharpe Ratio"}
        if metric in {"auc", "mda", "sharpe"} and not display_df_for_bars.empty and metric in display_df_for_bars.columns:
            table_df_sorted = display_df_for_bars.dropna(subset=[metric]).sort_values(metric, ascending=False)
            if not table_df_sorted.empty:
                best_model_name = table_df_sorted.iloc[0]["model"]
                best_metric_val = table_df_sorted.iloc[0][metric]
                colors = []
                text_values = [f"{v:.3f}" if metric in {"auc", "mda"} else f"{v:.2f}" for v in table_df_sorted[metric]]
                outlines = []
                for model in table_df_sorted["model"]:
                    base_color = "#3498db"
                    outline_color = "#2980b9"
                    if model == best_model_name:
                        base_color = "#2ecc71"
                        outline_color = "#27ae60"
                    if active_model and model == active_model:
                        outline_color = "#fcb421"
                    colors.append(base_color)
                    outlines.append(outline_color)
                comparison_fig.add_trace(
                    go.Bar(
                        x=table_df_sorted["model"],
                        y=table_df_sorted[metric],
                        text=text_values,
                        textposition="outside",
                        name=metric_labels.get(metric, metric.upper()),
                        marker=dict(
                            color=colors,
                            line=dict(color=outlines, width=2.5),
                        ),
                    )
                )
        if comparison_fig.data:
            comparison_fig.update_layout(
                xaxis_title="Modelos",
                yaxis_title=metric.upper(),
                template="plotly_white",
                hovermode="x",
                uniformtext_minsize=10,
                uniformtext_mode="hide",
                margin=MARGIN_BASE,
                font=dict(size=14),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                height=graph_height,
                title=None,
            )
        else:
            comparison_fig = _placeholder_fig("Comparativo de Modelos", "Sem dados no intervalo selecionado; ajuste o período ou modelo.")
            comparison_fig.update_layout(height=graph_height)
        if metric in {"auc", "mda", "sharpe"} and metric in table_df_display.columns:
            table_df_display = table_df_display.sort_values(metric, ascending=False, na_position="last")
        table_df_display = table_df_display.fillna("—")
        if metric == "sharpe":
            if COMMON_STRATEGY:
                comparison_meta = f"{comparison_meta} | Estratégia comparada: {COMMON_STRATEGY}"
            else:
                comparison_meta = f"{comparison_meta} | Estratégia comparada: (nenhuma disponível)"

        scatter_fig = go.Figure()
        merged_sr = (
            sentiment_filtered.merge(ibov_filtered[["day", "return"]], on="day", how="left")
            if not sentiment_filtered.empty and not ibov_filtered.empty
            else pd.DataFrame()
        )
        merged_sr = merged_sr.dropna(subset=["return"]) if not merged_sr.empty else merged_sr
        if not merged_sr.empty:
            corr = merged_sr["sentiment"].corr(merged_sr["return"])
            scatter_fig.add_trace(
                go.Scatter(
                    x=merged_sr["sentiment"],
                    y=merged_sr["return"],
                    mode="markers",
                    marker=dict(color="#1f77b4", size=6, opacity=0.6),
                    name="Sentimento x Retorno",
                )
            )
            scatter_fig.add_annotation(text=f"Corr(Pearson)={corr:.3f} | N={len(merged_sr)}", xref="paper", yref="paper", x=0, y=1.1, showarrow=False)
        if scatter_fig.data:
            scatter_fig.update_layout(
                title=None,
                xaxis_title="Sentimento",
                yaxis_title="Retorno diário",
                template="plotly_white",
                height=graph_height,
                margin=MARGIN_BASE,
                font=dict(size=14),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
        else:
            scatter_fig = _placeholder_fig("Dispersão Sentimento x Retorno diário", "Sem dados no intervalo selecionado; ajuste o período.")
            scatter_fig.update_layout(height=graph_height)

        rolling_fig = go.Figure()
        if not merged_sr.empty:
            merged_sr = merged_sr.sort_values("day")
            for window in [60, 90]:
                merged_sr[f"corr_{window}d"] = merged_sr["sentiment"].rolling(window).corr(merged_sr["return"])
                rolling_fig.add_trace(
                    go.Scatter(
                        x=merged_sr["day"],
                        y=merged_sr[f"corr_{window}d"],
                        mode="lines",
                        name=f"Corr {window}d",
                    )
                )
        if rolling_fig.data:
            rolling_fig.update_layout(
                title=None,
                xaxis_title="Data",
                yaxis_title="Correlação",
                template="plotly_white",
                height=graph_height,
                margin=MARGIN_BASE,
                font=dict(size=14),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
        else:
            rolling_fig = _placeholder_fig("Correlação móvel", "Sem dados no intervalo selecionado; ajuste o período.")
            rolling_fig.update_layout(height=graph_height)

        dist_fig = go.Figure()
        if not sentiment_filtered.empty:
            dist_fig.add_trace(go.Histogram(x=sentiment_filtered["sentiment"], nbinsx=40, name="Histograma", opacity=0.6))
            dist_fig.add_trace(go.Box(x=sentiment_filtered["sentiment"], name="Boxplot", boxpoints="outliers", marker_color="#e74c3c"))
        if dist_fig.data:
            dist_fig.update_layout(
                title=None,
                xaxis_title="Sentimento",
                template="plotly_white",
                barmode="overlay",
                height=graph_height,
                margin=MARGIN_BASE,
                font=dict(size=14),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
        else:
            dist_fig = _placeholder_fig("Distribuição do Sentimento", "Sem dados de sentimento para distribuir")
            dist_fig.update_layout(height=graph_height)

        if not event_filtered.empty and "fonte" in event_filtered.columns:
            latency_fig = go.Figure()
            latency_fig.add_trace(go.Bar(x=event_filtered["fonte"], y=event_filtered.get("car_max_abs", 0), name="Latência por fonte"))
            latency_fig.update_layout(
                title=None,
                xaxis_title="Fonte",
                yaxis_title="Latência / CAR",
                template="plotly_white",
                height=graph_height,
                margin=MARGIN_BASE,
                font=dict(size=14),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
        else:
            latency_fig = _placeholder_fig(
                "Latência",
                "Sem eventos de latência no período (arquivo vazio ou não gerado). Para gerar: executar notebook 11 / pipeline de latência.",
            )
            latency_fig.update_layout(height=graph_height)

        backtest_fig = go.Figure()
        if selected_models and "model" in backtest_filtered.columns:
            backtest_filtered = backtest_filtered[backtest_filtered["model"].isin(selected_models)]
        if not backtest_filtered.empty:
            for (model, strategy), grp in backtest_filtered.groupby(["model", "strategy"]):
                backtest_fig.add_trace(
                    go.Scatter(
                        x=grp["day"],
                        y=grp["equity"],
                        mode="lines",
                        name=f"{model} | {strategy}",
                    )
                )
        if backtest_fig.data:
            backtest_fig.update_layout(
                title=None,
                xaxis_title="Data",
                yaxis_title="Equity",
                template="plotly_white",
                height=graph_height,
                margin=MARGIN_BASE,
                font=dict(size=14),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
        else:
            backtest_fig = _placeholder_fig("Curva de Backtest", "Sem curva de backtest disponível")
            backtest_fig.update_layout(height=graph_height)

        metric_labels = {"auc": "AUC", "mda": "MDA", "sharpe": "Sharpe Ratio"}
        days_count = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
        ibov_days = len(ibov_filtered) if not ibov_filtered.empty else 0
        sentiment_days = len(sentiment_filtered) if not sentiment_filtered.empty else 0
        inter_days = len(
            pd.merge(
                ibov_filtered[["day"]] if not ibov_filtered.empty else pd.DataFrame(columns=["day"]),
                sentiment_filtered[["day"]] if not sentiment_filtered.empty else pd.DataFrame(columns=["day"]),
                on="day",
                how="inner",
            )
        )
        models_text = ", ".join(selected_models) if selected_models else "Nenhum modelo selecionado"

        kpis_cards = [
            html.Div(className="kpi-card", children=[html.Div("Dias IBOV", className="kpi-label"), html.Div(f"{ibov_days}", className="kpi-value")]),
            html.Div(className="kpi-card", children=[html.Div("Dias Sentimento", className="kpi-label"), html.Div(f"{sentiment_days}", className="kpi-value")]),
            html.Div(className="kpi-card", children=[html.Div("Interseção IBOV ∩ Sent", className="kpi-label"), html.Div(f"{inter_days}", className="kpi-value")]),
            html.Div(
                className="kpi-card",
                children=[
                    html.Div("Melhor modelo", className="kpi-label"),
                    html.Div(f"{best_model_name} ({best_metric_val:.3f})" if best_model_name else "—", className="kpi-value"),
                ],
            ),
        ]

        indicator_content = [
            html.Div([html.Strong("Período: "), html.Span(f"{start_date} a {end_date} ({days_count} dias)")]),
            html.Div([html.Strong("Modelo: "), html.Span(models_text)]),
            html.Div([html.Strong("Métrica: "), html.Span(metric_labels.get(metric, metric.upper()))]),
        ]
        metric_badge_text = f"Métrica: {metric_labels.get(metric, metric.upper())}"
        ui_status = f"Última interação: {ctx.triggered_id or 'init'} @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        ibov_meta = _meta_text("ibovespa_clean.csv + event_study_latency.csv", ibov_filtered, "day", "Ibovespa")
        sentiment_meta = _meta_text("16_oof_predictions.csv", sentiment_filtered, "day", "Sentimento")
        comparison_meta = _meta_text("results_16_models_tfidf.json + 18_backtest_results.csv", table_df, None, "Modelos")
        if COMMON_STRATEGY:
            comparison_meta = f"{comparison_meta} | Estratégia (comparação): {COMMON_STRATEGY}"
        scatter_meta = _meta_text("16_oof_predictions.csv ∩ ibovespa_clean.csv", merged_sr, "day", "Dispersão")
        rolling_meta = _meta_text("16_oof_predictions.csv ∩ ibovespa_clean.csv", merged_sr, "day", "Correlação móvel")
        dist_meta = _meta_text("16_oof_predictions.csv", sentiment_filtered, "day", "Distribuição")
        latency_meta = _meta_text("event_study_latency.csv", event_filtered, "event_day", "Latência")
        backtest_meta = _meta_text("18_backtest_daily_curves.csv", backtest_filtered, "day", "Backtest")

        print(f"[DEBUG] Filtered IBOV rows={len(ibov_filtered)}, SENT rows={len(sentiment_filtered)}, BACKTEST rows={len(backtest_filtered)}, INTER={inter_days}")
        return (
            ibov_fig,
            sentiment_fig,
            comparison_fig,
            table_df.to_dict("records"),
            indicator_content,
            metric_badge_text,
            ui_status,
            scatter_fig,
            rolling_fig,
            dist_fig,
            latency_fig,
            backtest_fig,
            kpis_cards,
            ibov_meta,
            sentiment_meta,
            comparison_meta,
            scatter_meta,
            rolling_meta,
            dist_meta,
            latency_meta,
            backtest_meta,
        )
    except Exception as exc:  # noqa: BLE001
        import traceback

        err_msg = f"Erro no callback: {exc}"
        print("[ERRO CALLBACK]", err_msg, file=sys.stderr)
        traceback.print_exc()
        placeholder = _placeholder_fig("Erro", "Verifique logs no servidor.")
        return (
            placeholder,
            placeholder,
            placeholder,
            [],
            html.Div(f"Erro no callback: {exc}", style={"color": "red"}),
            "Erro",
            f"Erro: {exc}",
            placeholder,
            placeholder,
            placeholder,
            placeholder,
            placeholder,
            [],
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
        )
# ------------------------------------------------------------------------------
# Helpers usados em smoke tests (pytest)
# ------------------------------------------------------------------------------
def update_additional_graphs(start_date=None, end_date=None, selected_model=None):
    """Retorna 3 figuras válidas (correlação, latência, backtest) mesmo sem dados."""
    corr_fig = go.Figure()
    corr_fig.add_annotation(text="Sem dados de correlação", x=0.5, y=0.5, showarrow=False)
    corr_fig.update_layout(title="Dispersão Sentimento x Retorno", template="plotly_white")

    latency_fig = go.Figure()
    latency_fig.add_annotation(text="Sem eventos de latência", x=0.5, y=0.5, showarrow=False)
    latency_fig.update_layout(title="Eventos de Latência", template="plotly_white")

    backtest_fig = go.Figure()
    backtest_fig.add_annotation(text="Sem curva de backtest", x=0.5, y=0.5, showarrow=False)
    backtest_fig.update_layout(title="Curva de Estratégia", template="plotly_white")

    return corr_fig, latency_fig, backtest_fig


# ------------------------------------------------------------------------------
# Main / CLI
# ------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dashboard Sentimento x Ibovespa")
    parser.add_argument("--host", default="127.0.0.1", help="Host para o Dash (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8050, help="Porta para o Dash (default: 8050)")
    parser.add_argument("--debug", action="store_true", help="Ativa debug do Dash (default: False)")
    parser.add_argument("--open", action="store_true", help="Abre o navegador automaticamente")
    parser.add_argument("--probe", action="store_true", help="Apenas testa porta/HTTP e sai")
    parser.add_argument("--find-port", action="store_true", help="Se porta ocupada, tenta próxima livre")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    host = args.host
    port = args.port

    if args.find_port and check_port(host, port):
        new_port = find_free_port(port)
        if new_port != port:
            print(f"[info] Porta {port} ocupada. Usando porta livre {new_port}.")
            port = new_port

    if args.probe:
        open_flag = check_port(host, port)
        ok, status, err = check_http(host, port)
        print(f"PORT {host}:{port} OPEN={open_flag}")
        if ok:
            print(f"HTTP {host}:{port} STATUS={status}")
            sys.exit(0)
        else:
            print(f"HTTP {host}:{port} FAIL={err}")
            sys.exit(1)

    url = f"http://{host}:{port}/"
    print(f"Dashboard: {url}")
    print(f"Python: {sys.executable}")
    print(f"CWD: {Path.cwd()}")
    print(f"Debug: {args.debug}")

    if args.open:
        webbrowser.open(url)

    app.run(host=host, port=port, debug=args.debug, use_reloader=False)
