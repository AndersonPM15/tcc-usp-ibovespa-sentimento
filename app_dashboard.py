"""
Plotly Dash application for the TCC USP sentiment x Ibovespa project.

Run with:
    python app_dashboard.py

The app reads the same artifacts used in `20_final_dashboard_analysis.ipynb`
and exposes interactive filters for period, modelo e métrica.
"""
# cSpell:ignore Ibovespa métrica Carregamento colunas ibov Arquivos principais precisa coluna proba cagr eventos modelo arquivo Sentimento Período modelos Notícia Notícias Comparativo Estratégia Gráfico hovertemplate Preço hovermode sentiment médio diário escala tozeroy hline Selecione Tabela

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html, callback_context

from src.config import loader as cfg
from src.config.constants import START_DATE, END_DATE

# Porta configurável via variável de ambiente
DASH_PORT = int(os.getenv("DASH_PORT", "8050"))
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




def build_corr_sentiment_return_fig(start_date: str, end_date: str) -> go.Figure:
    """Scatter de correlação entre sentimento diário e retorno do Ibovespa."""
    if IBOV_DF.empty:
        return _empty_figure_with_message("Dados do Ibovespa não disponíveis para correlação")

    ibov_df = IBOV_DF.copy()
    
    # Garantir coluna close
    if "close" not in ibov_df.columns and "adj_close" in ibov_df.columns:
        ibov_df["close"] = ibov_df["adj_close"]
    if "close" not in ibov_df.columns:
        return _empty_figure_with_message("Preço de fechamento ausente no Ibovespa")
    
    # Garantir coluna day
    if "day" not in ibov_df.columns and "date" in ibov_df.columns:
        ibov_df["day"] = pd.to_datetime(ibov_df["date"], errors="coerce")
    if "day" not in ibov_df.columns:
        return _empty_figure_with_message("Coluna de data não encontrada no Ibovespa")
    
    ibov_df["day"] = pd.to_datetime(ibov_df["day"], errors="coerce")
    ibov_df = ibov_df.dropna(subset=["day"])
    
    # Filtrar por período com proteção anti-2025
    start_dt = pd.to_datetime(start_date) if start_date else pd.Timestamp(START_DATE)
    end_dt = pd.to_datetime(end_date) if end_date else pd.Timestamp(END_DATE)
    hard_cap = pd.Timestamp(END_DATE)
    if end_dt > hard_cap:
        end_dt = hard_cap
    
    ibov_df = ibov_df[(ibov_df["day"] >= start_dt) & (ibov_df["day"] <= end_dt)]
    ibov_df = ibov_df.sort_values("day")
    
    if ibov_df.empty:
        return _empty_figure_with_message("Sem dados do Ibovespa no período selecionado")

    # Calcular retorno diário
    ibov_df["return_1d"] = ibov_df["close"].pct_change()
    ibov_ret = ibov_df[["day", "return_1d"]].dropna()
    
    if ibov_ret.empty:
        return _empty_figure_with_message("Dados insuficientes para calcular retornos")

    # Carregar sentimento
    if SENTIMENT_DF.empty:
        return _empty_figure_with_message("Dados de sentimento não disponíveis para correlação")
    
    sentiment_df = SENTIMENT_DF.copy()
    sentiment_df["day"] = pd.to_datetime(sentiment_df["day"], errors="coerce")
    sentiment_df = sentiment_df[(sentiment_df["day"] >= start_dt) & (sentiment_df["day"] <= end_dt)]
    
    if sentiment_df.empty:
        return _empty_figure_with_message("Sem dados de sentimento no período selecionado")

    # Merge por dia
    merged = pd.merge(ibov_ret, sentiment_df[["day", "sentiment"]], on="day", how="inner").dropna()
    
    if merged.empty or len(merged) < 3:
        return _empty_figure_with_message("Dados insuficientes para calcular a correlação (mínimo 3 observações)")

    corr_value = merged["return_1d"].corr(merged["sentiment"])
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=merged["sentiment"],
            y=merged["return_1d"],
            mode="markers",
            marker=dict(color="#1f77b4", size=8, opacity=0.7),
            name="Observações diárias",
            hovertemplate="Sentimento=%{x:.3f}<br>Retorno=%{y:.3%}<extra></extra>",
        )
    )
    
    # Linha de tendência simples (regressão linear)
    if len(merged) >= 5:
        import numpy as np
        z = np.polyfit(merged["sentiment"], merged["return_1d"], 1)
        p = np.poly1d(z)
        x_line = [merged["sentiment"].min(), merged["sentiment"].max()]
        y_line = [p(x) for x in x_line]
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(color="#e74c3c", width=2, dash="dash"),
                name="Tendência",
            )
        )
    
    corr_text = f"r = {corr_value:.3f}" if pd.notna(corr_value) else "r = N/A"
    n_obs = len(merged)
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        text=f"Correlação de Pearson: {corr_text}<br>N = {n_obs} observações",
        font=dict(color="#2c3e50", size=12),
        bgcolor="rgba(255,255,255,0.8)",
        borderpad=4,
        align="left",
    )
    
    fig.update_layout(
        title="Correlação entre Sentimento Diário e Retorno do Ibovespa",
        xaxis_title="Sentimento Médio Diário",
        yaxis_title="Retorno Diário do Ibovespa",
        template="plotly_white",
        margin=dict(l=60, r=20, t=70, b=60),
        showlegend=True,
        legend=dict(orientation="h", y=-0.15, x=0),
    )
    return fig



def build_latency_fig(start_date: str, end_date: str) -> go.Figure:
    """Barra de latência informacional por fonte/daypart."""
    df = LATENCY_DF.copy()
    if df.empty:
        return _empty_figure_with_message("Arquivo de latência não disponível.\nRode o notebook 11 ou o pipeline para gerar os dados.")

    # Identificar coluna de data e aplicar filtro anti-2025
    date_col = next((c for c in ["event_day", "date", "day"] if c in df.columns), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        
        start_dt = pd.to_datetime(start_date) if start_date else pd.Timestamp(START_DATE)
        end_dt = pd.to_datetime(end_date) if end_date else pd.Timestamp(END_DATE)
        hard_cap = pd.Timestamp(END_DATE)
        if end_dt > hard_cap:
            end_dt = hard_cap
        
        df = df[(df[date_col] >= start_dt) & (df[date_col] <= end_dt)]

    if df.empty:
        return _empty_figure_with_message("Sem dados de latência no período selecionado")

    # Identificar métrica de latência
    metric_candidates = ["t_half_mediana", "t_half_media", "t_half", "latency", "car", "car_mean", "impact"]
    metric_col = next((c for c in metric_candidates if c in df.columns), None)
    if metric_col is None:
        return _empty_figure_with_message("Métrica de latência não encontrada no arquivo")

    # Identificar colunas de agrupamento
    group_cols: list[str] = []
    if "fonte" in df.columns:
        group_cols.append("fonte")
    elif "source" in df.columns:
        df["fonte"] = df["source"]
        group_cols.append("fonte")
    if "daypart" in df.columns:
        group_cols.append("daypart")
    
    if not group_cols:
        return _empty_figure_with_message("Colunas de agrupamento (fonte/daypart) não encontradas")

    agg = df.groupby(group_cols, dropna=False)[metric_col].mean().reset_index()
    if agg.empty:
        return _empty_figure_with_message("Sem dados agregados de latência")

    fig = go.Figure()
    metric_label = metric_col.replace("_", " ").title()
    
    if len(group_cols) == 1:
        # Ordenar por valor para melhor visualização
        agg_sorted = agg.sort_values(metric_col, ascending=False)
        fig.add_trace(
            go.Bar(
                x=agg_sorted[group_cols[0]],
                y=agg_sorted[metric_col],
                marker_color="#3498db",
                name=metric_label,
                text=[f"{v:.2f}" for v in agg_sorted[metric_col]],
                textposition="outside",
            )
        )
    else:
        # Múltiplos grupos (fonte + daypart)
        for daypart, subdf in agg.groupby("daypart"):
            subdf_sorted = subdf.sort_values(metric_col, ascending=False)
            fig.add_trace(
                go.Bar(
                    x=subdf_sorted["fonte"],
                    y=subdf_sorted[metric_col],
                    name=str(daypart),
                    text=[f"{v:.2f}" for v in subdf_sorted[metric_col]],
                    textposition="outside",
                )
            )
        fig.update_layout(barmode="group")

    fig.update_layout(
        title="Latência Informacional por Fonte de Notícia",
        xaxis_title="Fonte",
        yaxis_title=f"Média de {metric_label}",
        template="plotly_white",
        margin=dict(l=60, r=20, t=60, b=80),
        xaxis=dict(tickangle=-45),
        showlegend=len(group_cols) > 1,
    )
    return fig

def _pick_equity_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    strategy_candidates = [
        "equity_strategy",
        "strategy_equity",
        "strategy",
        "equity",
        "portfolio",
    ]
    benchmark_candidates = [
        "benchmark_equity",
        "benchmark",
        "buy_hold",
        "bh",
        "ibov_equity",
    ]
    strat_col = next((c for c in strategy_candidates if c in df.columns), None)
    bench_col = next((c for c in benchmark_candidates if c in df.columns), None)
    if strat_col and bench_col:
        return strat_col, bench_col
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for drop_col in ["cagr", "sharpe", "mdd", "vol"]:
        if drop_col in numeric_cols:
            numeric_cols.remove(drop_col)
    if len(numeric_cols) >= 2:
        return numeric_cols[0], numeric_cols[1]
    return None, None




def build_backtest_fig(model_value: str | None, start_date: str = None, end_date: str = None) -> go.Figure:
    """Curva de patrimônio da estratégia de sentimento versus benchmark."""
    df = _safe_read_csv(BACKTEST_PATH)
    if df.empty:
        return _empty_figure_with_message("Backtest não disponível.\nRode o notebook 18 para gerar os dados.")

    # Identificar coluna de data
    date_col = next((c for c in ["date", "day", "Data", "data"] if c in df.columns), None)
    if date_col is None:
        # Arquivo existe mas não tem série temporal (apenas métricas agregadas)
        return _empty_figure_with_message(
            "Arquivo de backtest contém apenas métricas agregadas.\n"
            "Para visualizar curvas de patrimônio, gere um arquivo\n"
            "com colunas: date, equity_strategy, benchmark."
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    
    # Aplicar filtro anti-2025
    start_dt = pd.to_datetime(start_date) if start_date else pd.Timestamp(START_DATE)
    end_dt = pd.to_datetime(end_date) if end_date else pd.Timestamp(END_DATE)
    hard_cap = pd.Timestamp(END_DATE)
    if end_dt > hard_cap:
        end_dt = hard_cap
    
    df = df[(df[date_col] >= start_dt) & (df[date_col] <= end_dt)]
    
    if df.empty:
        return _empty_figure_with_message("Sem dados de backtest no período selecionado")

    # Filtrar por modelo se especificado
    chosen_model = None
    if "model" in df.columns:
        available_models = df["model"].dropna().unique().tolist()
        if model_value and model_value in available_models:
            chosen_model = model_value
        elif available_models:
            chosen_model = available_models[0]
        
        if chosen_model:
            df = df[df["model"] == chosen_model]
            if df.empty:
                return _empty_figure_with_message(f"Sem dados para o modelo '{chosen_model}'")

    strat_col, bench_col = _pick_equity_columns(df)
    if strat_col is None or bench_col is None:
        return _empty_figure_with_message("Colunas de patrimônio não encontradas no backtest")

    curves = df[[date_col, strat_col, bench_col]].dropna(how="all").sort_values(date_col)
    if curves.empty:
        return _empty_figure_with_message("Sem dados de curva de patrimônio")

    fig = go.Figure()
    
    # Normalizar para base 100 para comparação
    if len(curves) > 0:
        strat_base = curves[strat_col].iloc[0]
        bench_base = curves[bench_col].iloc[0]
        
        if strat_base != 0 and bench_base != 0:
            curves["strat_norm"] = (curves[strat_col] / strat_base) * 100
            curves["bench_norm"] = (curves[bench_col] / bench_base) * 100
            strat_col_plot = "strat_norm"
            bench_col_plot = "bench_norm"
            y_title = "Valor da Carteira (base 100)"
        else:
            strat_col_plot = strat_col
            bench_col_plot = bench_col
            y_title = "Valor da Carteira"
    else:
        strat_col_plot = strat_col
        bench_col_plot = bench_col
        y_title = "Valor da Carteira"
    
    fig.add_trace(
        go.Scatter(
            x=curves[date_col],
            y=curves[strat_col_plot],
            mode="lines",
            name="Estratégia (Sentimento)",
            line=dict(color="#2c3e50", width=2.5),
            hovertemplate="%{x|%Y-%m-%d}<br>Estratégia: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=curves[date_col],
            y=curves[bench_col_plot],
            mode="lines",
            name="Benchmark (Buy & Hold)",
            line=dict(color="#e67e22", width=2.5, dash="dash"),
            hovertemplate="%{x|%Y-%m-%d}<br>Benchmark: %{y:.2f}<extra></extra>",
        )
    )
    
    # Calcular retorno acumulado final
    if len(curves) > 1:
        strat_final = curves[strat_col_plot].iloc[-1]
        bench_final = curves[bench_col_plot].iloc[-1]
        strat_ret = ((strat_final / 100) - 1) * 100 if strat_col_plot == "strat_norm" else 0
        bench_ret = ((bench_final / 100) - 1) * 100 if bench_col_plot == "bench_norm" else 0
        
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            text=f"Retorno Acumulado:<br>Estratégia: {strat_ret:+.1f}%<br>Benchmark: {bench_ret:+.1f}%",
            font=dict(color="#2c3e50", size=11),
            bgcolor="rgba(255,255,255,0.9)",
            borderpad=4,
            align="left",
        )
    
    title = "Curva de Patrimônio - Estratégia de Sentimento vs. Benchmark"
    if chosen_model:
        title += f" ({chosen_model})"
    
    fig.update_layout(
        title=title,
        xaxis_title="Data",
        yaxis_title=y_title,
        template="plotly_white",
        legend=dict(orientation="h", y=-0.15, x=0),
        margin=dict(l=60, r=20, t=70, b=80),
        hovermode="x unified",
    )
    return fig

IBOV_DF = load_ibov()
SENTIMENT_DF = load_sentiment()
RESULTS_DF = load_results_table()
LATENCY_DF = load_latency_events()
backtest_df_for_models = _safe_read_csv(BACKTEST_PATH)
BACKTEST_MODELS = (
    sorted(backtest_df_for_models["model"].dropna().unique())
    if not backtest_df_for_models.empty and "model" in backtest_df_for_models.columns
    else []
)

DATE_MIN = pd.Timestamp(START_DATE)
DATE_MAX = pd.Timestamp(END_DATE)
MODEL_OPTIONS = (
    sorted(RESULTS_DF["model"].dropna().unique()) if not RESULTS_DF.empty and "model" in RESULTS_DF.columns else []
)
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
                                value=MODEL_OPTIONS,
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
    style={"fontFamily": "Arial, sans-serif", "maxWidth": "1400px", "margin": "0 auto", "padding": "20px"},
    children=[
        # Cabeçalho
        html.Div(
            style={"textAlign": "center", "marginBottom": "30px"},
            children=[
                html.H1(
                    "Sentimento de Notícias x Ibovespa",
                    style={"fontSize": "2.5em", "color": "#1a1a1a", "marginBottom": "10px", "fontWeight": "600"}
                ),
                html.P(
                    "Análise Preditiva com Modelos de Machine Learning | TCC USP",
                    style={"fontSize": "1.1em", "color": "#666", "marginTop": "0"}
                ),
            ],
        ),
        
        # Painel de Depuração (visível para diagnóstico)
        html.Details(
            open=False,
            style={
                "backgroundColor": "#fff3cd",
                "padding": "15px 20px",
                "borderRadius": "8px",
                "marginBottom": "20px",
                "border": "1px solid #ffc107",
            },
            children=[
                html.Summary(
                    "🔧 Painel de Depuração (clique para expandir)",
                    style={"cursor": "pointer", "fontWeight": "bold", "color": "#856404"}
                ),
                html.Div(
                    id="debug-panel",
                    style={"marginTop": "15px", "fontFamily": "monospace", "fontSize": "0.85em"},
                    children=[html.Span("Aguardando primeiro callback...", style={"color": "#666"})]
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
                    html.P("Evolução do índice Bovespa com marcadores de eventos relevantes", 
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
                    html.H3("Sentimento Médio Diário", style={"marginTop": "0", "marginBottom": "5px", "color": "#2c3e50"}),
                    html.P("Sentimento agregado das notícias (escala -1 a +1, onde valores positivos indicam otimismo)", 
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
                html.H4("Detalhamento das Métricas", style={"marginBottom": "15px", "color": "#34495e"}),
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

        # Card 5: Análises adicionais
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
                    html.H3("Correlação Sentimento × Retorno Diário", style={"marginTop": "0", "marginBottom": "5px", "color": "#2c3e50"}),
                    html.P("Relação entre o sentimento médio diário das notícias e o retorno diário do Ibovespa.",
                           style={"fontSize": "0.9em", "color": "#666", "marginTop": "0", "marginBottom": "15px"}),
                ]),
                dcc.Graph(id="scatter-sentiment-return", config={"displayModeBar": True}),
                html.Hr(style={"margin": "30px 0"}),
                html.Div([
                    html.H3("Latência Informacional", style={"marginTop": "0", "marginBottom": "5px", "color": "#2c3e50"}),
                    html.P("Métrica de latência/impacto agregada por fonte (e faixa de horário, se disponível).",
                           style={"fontSize": "0.9em", "color": "#666", "marginTop": "0", "marginBottom": "15px"}),
                ]),
                dcc.Graph(id="bar-latency-source", config={"displayModeBar": True}),
            ],
        ),

        # Card 6: Backtest
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
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "15px"},
                    children=[
                        html.H3("Backtest – Curva de Patrimônio", style={"marginTop": "0", "marginBottom": "0", "color": "#2c3e50"}),
                        dcc.Dropdown(
                            id="backtest-model-dropdown",
                            options=[{"label": m, "value": m} for m in BACKTEST_MODELS],
                            value=BACKTEST_MODELS[0] if BACKTEST_MODELS else None,
                            placeholder="Selecione o modelo do backtest",
                            style={"width": "280px"},
                            clearable=True,
                        ),
                    ],
                ),
                dcc.Graph(id="line-backtest-equity", config={"displayModeBar": True}),
            ],
        ),
    ],
)


# ------------------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------------------


def _filter_by_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Filtra DataFrame por período temporal com proteção anti-2025.
    
    - Usa coluna 'day' se existir, senão 'date'
    - Aplica hard cap em END_DATE (2024-12-31) independente do DatePicker
    - Retorna cópia do DataFrame filtrado
    """
    if df.empty:
        return df
    
    # Identificar coluna de data
    date_col = None
    if "day" in df.columns:
        date_col = "day"
    elif "date" in df.columns:
        date_col = "date"
    else:
        # Sem coluna de data, retornar sem filtro
        return df
    
    # Garantir que a coluna é datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    
    # Converter limites
    start_dt = pd.to_datetime(start) if start else pd.Timestamp(START_DATE)
    end_dt = pd.to_datetime(end) if end else pd.Timestamp(END_DATE)
    
    # HARD CAP: nunca passar de END_DATE (proteção anti-2025)
    hard_cap = pd.Timestamp(END_DATE)
    if end_dt > hard_cap:
        end_dt = hard_cap
    
    # Aplicar filtro
    mask = (df[date_col] >= start_dt) & (df[date_col] <= end_dt)
    return df.loc[mask].copy()


def _empty_figure_with_message(msg: str) -> go.Figure:
    """Retorna figura vazia com mensagem centralizada (evita gráficos em branco)."""
    fig = go.Figure()
    fig.add_annotation(
        text=msg,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="#666"),
        align="center",
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


print("[DEBUG] Registrando callbacks...")

@app.callback(
    Output("ibov-graph", "figure"),
    Output("sentiment-graph", "figure"),
    Output("model-comparison-graph", "figure"),
    Output("model-table", "data"),
    Output("active-filters-indicator", "children"),
    Output("metric-badge", "children"),
    Output("debug-panel", "children"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("model-filter", "value"),
    Input("metric-filter", "value"),
)
def update_dashboard(start_date, end_date, selected_models, metric):
    # Capturar informações de debug
    ctx = callback_context
    triggered_by = ctx.triggered[0]["prop_id"] if ctx.triggered else "Nenhum"
    
    print(f"[DEBUG] Callback acionado: start={start_date}, end={end_date}, models={selected_models}, metric={metric}")
    print(f"[DEBUG] IBOV_DF shape: {IBOV_DF.shape if not IBOV_DF.empty else 'VAZIO'}")
    print(f"[DEBUG] SENTIMENT_DF shape: {SENTIMENT_DF.shape if not SENTIMENT_DF.empty else 'VAZIO'}")
    metric_map = {"auc": "AUC", "mda": "MDA (%)", "sharpe": "Sharpe Ratio"}
    
    # Gráfico do Ibovespa
    ibov_filtered = _filter_by_period(IBOV_DF, start_date, end_date)
    print(f"[DEBUG] ibov_filtered shape: {ibov_filtered.shape if not ibov_filtered.empty else 'VAZIO'}")

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
        yaxis_title="Pre?o do Ibovespa (pontos)",
        hovermode="x unified",
        template="plotly_white",
        font=dict(size=12),
        legend=dict(orientation="h", y=-0.2, x=0),
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
    print(f"[DEBUG] ibov_fig traces: {len(ibov_fig.data)}")

    # Gráfico de sentimento
    sentiment_filtered = _filter_by_period(SENTIMENT_DF, start_date, end_date)
    print(f"[DEBUG] sentiment_filtered shape: {sentiment_filtered.shape if not sentiment_filtered.empty else 'VAZIO'}")
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
        legend=dict(orientation="h", y=-0.2, x=0),
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

    # Gráfico de comparação de modelos
    comparison_fig = go.Figure()
    table_df = RESULTS_DF.copy()
    metric_value = metric if metric in metric_map and metric in table_df.columns else "auc"
    
    # Filtrar por modelos selecionados
    if selected_models:
        table_df = table_df[table_df["model"].isin(selected_models)]
    
    # Ordenar pela métrica selecionada
    if metric_value in {"auc", "mda", "sharpe"} and metric_value in table_df.columns:
        table_df_sorted = table_df.dropna(subset=[metric_value]).sort_values(metric_value, ascending=False)
        
        if not table_df_sorted.empty:
            # Identificar melhor modelo (primeira posição após ordenação)
            best_model = table_df_sorted.iloc[0]["model"]
            colors = ["#2ecc71" if model == best_model else "#3498db" 
                      for model in table_df_sorted["model"]]
            
            # Formatar texto nas barras
            if metric_value in {"auc", "mda"}:
                text_values = [f"{v:.3f}" for v in table_df_sorted[metric_value]]
            else:  # sharpe
                text_values = [f"{v:.2f}" for v in table_df_sorted[metric_value]]
            
            comparison_fig.add_trace(
                go.Bar(
                    x=table_df_sorted["model"],
                    y=table_df_sorted[metric_value],
                    text=text_values,
                    textposition="outside",
                    name=metric_map.get(metric_value, metric_value.upper()),
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
                yaxis_title=metric_map.get(metric_value, metric_value.upper()),
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
    
    # Ordenar tabela pela métrica
    if metric_value in {"auc", "mda", "sharpe"} and metric_value in table_df.columns:
        table_df = table_df.sort_values(metric_value, ascending=False, na_position="last")
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
                html.Strong("📅 Período: ", style={"color": "#1976d2"}),
                html.Span(f"{start_date} a {end_date} ({days_count} dias)"),
            ]),
            html.Div([
                html.Strong("📊 Dados: ", style={"color": "#1976d2"}),
                html.Span(f"Ibovespa: {ibov_days} dias | Sentimento: {sentiment_days} dias"),
            ]),
            html.Div([
                html.Strong("🤖 Modelos: ", style={"color": "#1976d2"}),
                html.Span(models_text),
            ]),
            html.Div([
                html.Strong("📈 Métrica: ", style={"color": "#1976d2"}),
                html.Span(metric_labels.get(metric_value, metric_value.upper()), style={"fontWeight": "600", "color": "#2e7d32"}),
            ]),
        ],
    )

    metric_badge_text = f"📊 Métrica: {metric_labels.get(metric_value, metric_value.upper())}"
    
    # Painel de depuração
    table_rows_count = len(table_df) if not table_df.empty else 0
    debug_content = html.Div([
        html.Div([
            html.Strong("🔔 Último trigger: "),
            html.Code(triggered_by, style={"backgroundColor": "#f5f5f5", "padding": "2px 6px", "borderRadius": "4px"}),
        ], style={"marginBottom": "8px"}),
        html.Div([
            html.Strong("📅 Valores atuais: "),
            html.Span(f"start_date={start_date}, end_date={end_date}"),
        ], style={"marginBottom": "8px"}),
        html.Div([
            html.Strong("🤖 Modelos selecionados: "),
            html.Span(str(selected_models) if selected_models else "[]"),
        ], style={"marginBottom": "8px"}),
        html.Div([
            html.Strong("📈 Métrica: "),
            html.Span(f"{metric} → {metric_value}"),
        ], style={"marginBottom": "8px"}),
        html.Div([
            html.Strong("📊 Contagens pós-filtro: "),
            html.Span(f"ibov_filtered={ibov_days}, sentiment_filtered={sentiment_days}, table_df={table_rows_count}"),
        ], style={"marginBottom": "8px"}),
        html.Div([
            html.Strong("⚠️ Avisos: "),
            html.Span(
                "Nenhum" if ibov_days > 0 and sentiment_days > 0 else 
                f"{'IBOV vazio! ' if ibov_days == 0 else ''}{'Sentimento vazio!' if sentiment_days == 0 else ''}"
            , style={"color": "#d32f2f" if ibov_days == 0 or sentiment_days == 0 else "#388e3c"}),
        ]),
    ])
    
    return ibov_fig, sentiment_fig, comparison_fig, table_df.to_dict("records"), indicator_content, metric_badge_text, debug_content


@app.callback(
    Output("scatter-sentiment-return", "figure"),
    Output("bar-latency-source", "figure"),
    Output("line-backtest-equity", "figure"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("backtest-model-dropdown", "value"),
)
def update_additional_graphs(start_date, end_date, backtest_model):
    """Callback para gráficos adicionais (correlação, latência, backtest)."""
    corr_fig = build_corr_sentiment_return_fig(start_date, end_date)
    latency_fig = build_latency_fig(start_date, end_date)
    backtest_fig = build_backtest_fig(backtest_model, start_date, end_date)
    return corr_fig, latency_fig, backtest_fig


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Iniciando dashboard em http://localhost:{DASH_PORT} ...")
    # use_reloader=False evita double-loading que causa "Duplicate callback outputs"
    app.run(port=DASH_PORT, debug=True, use_reloader=False)
