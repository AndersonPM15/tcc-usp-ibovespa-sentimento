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
import socket
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html, callback_context

from src.config import loader as cfg
from src.config.constants import START_DATE, END_DATE
from src.io import paths

# ------------------------------------------------------------------------------
# Configuração de Porta/Host (TAREFA 1)
# ------------------------------------------------------------------------------
DASH_HOST = os.getenv("DASH_HOST", "127.0.0.1")
DASH_PORT = int(os.getenv("DASH_PORT", "8050"))

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
LATENCY_PATH = DATA_PATHS["data_processed"] / "latency_events.parquet"
LATENCY_DAYPART_PATH = DATA_PATHS["data_processed"] / "latency_by_daypart.csv"

# Período oficial do TCC
END_DATE_CAP = pd.Timestamp(END_DATE)


def _safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Lê CSV com tratamento de arquivo não encontrado."""
    if not path.exists():
        print(f"[aviso] Arquivo não encontrado: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    """Lê Parquet com tratamento de arquivo não encontrado."""
    if not path.exists():
        print(f"[aviso] Arquivo não encontrado: {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_ibov() -> pd.DataFrame:
    """Carrega dados do Ibovespa."""
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


def load_oof_raw() -> pd.DataFrame:
    """Carrega OOF predictions bruto (antes da agregação)."""
    df = _safe_read_csv(OOF_PATH, parse_dates=["day"])
    return df


def load_sentiment() -> pd.DataFrame:
    """Carrega sentimento agregado por dia."""
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
    """Carrega tabela de resultados dos modelos."""
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
    """Carrega eventos de latência."""
    # Tenta parquet primeiro, depois csv
    if LATENCY_PATH.exists():
        df = _safe_read_parquet(LATENCY_PATH)
    else:
        csv_path = LATENCY_PATH.with_suffix(".csv")
        df = _safe_read_csv(csv_path)
    if df.empty:
        return df
    
    # Detectar coluna de data
    date_candidates = ["event_day", "day", "date", "data"]
    for col in date_candidates:
        if col in df.columns:
            df["event_day"] = pd.to_datetime(df[col])
            break
    return df


def load_latency_daypart() -> pd.DataFrame:
    """Carrega latência por daypart."""
    return _safe_read_csv(LATENCY_DAYPART_PATH)


def load_backtest_curves() -> pd.DataFrame:
    """Carrega curvas diárias do backtest."""
    return _safe_read_csv(BACKTEST_CURVES_PATH, parse_dates=["day"])


# Carregar dados globais
print("\n" + "=" * 70)
print("CARREGANDO DADOS DO DASHBOARD")
print("=" * 70)

IBOV_DF = load_ibov()
OOF_RAW_DF = load_oof_raw()
SENTIMENT_DF = load_sentiment()
RESULTS_DF = load_results_table()
LATENCY_DF = load_latency_events()
LATENCY_DAYPART_DF = load_latency_daypart()
BACKTEST_CURVES_DF = load_backtest_curves()


# ------------------------------------------------------------------------------
# TAREFA 2: Validação de Integridade dos Dados
# ------------------------------------------------------------------------------

def validate_data_integrity() -> Dict[str, Any]:
    """
    Valida integridade dos dados carregados.
    Retorna dicionário com resumo completo para exibição.
    """
    report = {
        "datasets": {},
        "cross_validation": {},
        "alerts": [],
        "summary_text": [],
    }
    
    # Helper para extrair info de cada dataset
    def dataset_info(name: str, df: pd.DataFrame, path: Path, date_col: str = "day") -> dict:
        info = {
            "path": str(path),
            "exists": path.exists() if path else False,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": list(df.columns) if not df.empty else [],
        }
        if not df.empty and date_col in df.columns:
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
            valid_dates = df_copy[date_col].dropna()
            if len(valid_dates) > 0:
                info["min_date"] = valid_dates.min().strftime("%Y-%m-%d")
                info["max_date"] = valid_dates.max().strftime("%Y-%m-%d")
                info["n_unique_days"] = valid_dates.dt.date.nunique()
                
                # Verificar violações anti-2025
                violations = (valid_dates > END_DATE_CAP).sum()
                info["n_after_2024"] = int(violations)
                info["rows_after_filter"] = len(df_copy[valid_dates <= END_DATE_CAP])
            else:
                info["min_date"] = None
                info["max_date"] = None
                info["n_unique_days"] = 0
                info["n_after_2024"] = 0
                info["rows_after_filter"] = 0
        return info
    
    # Validar cada dataset
    report["datasets"]["IBOV"] = dataset_info("IBOV", IBOV_DF, IBOV_PATH, "day")
    report["datasets"]["OOF_RAW"] = dataset_info("OOF_RAW", OOF_RAW_DF, OOF_PATH, "day")
    report["datasets"]["SENTIMENT"] = dataset_info("SENTIMENT", SENTIMENT_DF, OOF_PATH, "day")
    report["datasets"]["RESULTS"] = {
        "path": str(RESULTS16_PATH),
        "exists": RESULTS16_PATH.exists(),
        "n_rows": len(RESULTS_DF),
        "columns": list(RESULTS_DF.columns) if not RESULTS_DF.empty else [],
    }
    
    # Validações cruzadas
    if not IBOV_DF.empty and not SENTIMENT_DF.empty:
        ibov_days = set(pd.to_datetime(IBOV_DF["day"]).dt.date)
        sent_days = set(pd.to_datetime(SENTIMENT_DF["day"]).dt.date)
        
        intersection = ibov_days & sent_days
        only_ibov = ibov_days - sent_days
        only_sent = sent_days - ibov_days
        
        report["cross_validation"] = {
            "ibov_unique_days": len(ibov_days),
            "sentiment_unique_days": len(sent_days),
            "intersection_days": len(intersection),
            "days_only_ibov": len(only_ibov),
            "days_only_sentiment": len(only_sent),
            "sample_only_ibov": sorted([d.isoformat() for d in list(only_ibov)[:5]]),
            "sample_only_sentiment": sorted([d.isoformat() for d in list(only_sent)[:5]]),
        }
    
    # Alertas
    for name, info in report["datasets"].items():
        if isinstance(info, dict) and info.get("n_after_2024", 0) > 0:
            report["alerts"].append(
                f"⚠️ {name} contém {info['n_after_2024']} registros com data > 2024-12-31 "
                f"— necessário rerodar notebook 16/pipeline para regenerar!"
            )
    
    # Gerar texto resumo
    lines = []
    lines.append("📊 RESUMO DE VALIDAÇÃO DE DADOS")
    lines.append("-" * 40)
    
    for name, info in report["datasets"].items():
        if isinstance(info, dict):
            lines.append(f"\n{name}:")
            lines.append(f"  • Arquivo: {info.get('path', 'N/A')}")
            lines.append(f"  • Linhas: {info.get('n_rows', 0)}")
            if info.get("min_date"):
                lines.append(f"  • Período: {info.get('min_date')} a {info.get('max_date')}")
                lines.append(f"  • Dias únicos: {info.get('n_unique_days', 0)}")
            if info.get("n_after_2024", 0) > 0:
                lines.append(f"  • ⚠️ Registros pós-2024: {info.get('n_after_2024')}")
    
    if report["cross_validation"]:
        cv = report["cross_validation"]
        lines.append(f"\n📈 VALIDAÇÃO CRUZADA:")
        lines.append(f"  • Dias IBOV: {cv['ibov_unique_days']}")
        lines.append(f"  • Dias Sentimento: {cv['sentiment_unique_days']}")
        lines.append(f"  • Interseção: {cv['intersection_days']} dias")
        lines.append(f"  • Apenas IBOV: {cv['days_only_ibov']} dias")
        lines.append(f"  • Apenas Sentimento: {cv['days_only_sentiment']} dias")
        if cv['sample_only_ibov']:
            lines.append(f"  • Exemplos só IBOV: {', '.join(cv['sample_only_ibov'][:3])}")
        if cv['sample_only_sentiment']:
            lines.append(f"  • Exemplos só Sent: {', '.join(cv['sample_only_sentiment'][:3])}")
    
    if report["alerts"]:
        lines.append(f"\n🚨 ALERTAS:")
        for alert in report["alerts"]:
            lines.append(f"  {alert}")
    
    report["summary_text"] = lines
    return report


# Executar validação ao iniciar
VALIDATION_REPORT = validate_data_integrity()
print("\n" + "\n".join(VALIDATION_REPORT["summary_text"]))
print("=" * 70 + "\n")


# ------------------------------------------------------------------------------
# Helpers para gráficos
# ------------------------------------------------------------------------------

def _filter_by_period(df: pd.DataFrame, start_date, end_date, date_col: str = "day") -> pd.DataFrame:
    """Filtra DataFrame por período com hard cap anti-2025."""
    if df.empty or date_col not in df.columns:
        return df
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    
    start_dt = pd.to_datetime(start_date) if start_date else pd.Timestamp(START_DATE)
    end_dt = pd.to_datetime(end_date) if end_date else END_DATE_CAP
    
    # HARD CAP: nunca permitir datas após 2024-12-31
    if end_dt > END_DATE_CAP:
        end_dt = END_DATE_CAP
    
    mask = (df[date_col] >= start_dt) & (df[date_col] <= end_dt)
    return df[mask].copy()


def _empty_figure_with_message(message: str) -> go.Figure:
    """Cria figura vazia com mensagem centralizada."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="#666"),
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
    )
    return fig


def build_ibov_figure(df: pd.DataFrame) -> go.Figure:
    """Constrói gráfico de linha do Ibovespa."""
    if df.empty:
        return _empty_figure_with_message("Dados do Ibovespa não disponíveis")
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["day"],
            y=df["close"],
            mode="lines",
            line=dict(color="#2c3e50", width=2),
            name="Ibovespa",
            hovertemplate="Data: %{x|%Y-%m-%d}<br>Preço: %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Ibovespa - Preço de Fechamento",
        xaxis_title="Data",
        yaxis_title="Pontos",
        hovermode="x unified",
    )
    return fig


def build_sentiment_figure(df: pd.DataFrame) -> go.Figure:
    """Constrói gráfico de linha do sentimento."""
    if df.empty:
        return _empty_figure_with_message("Dados de sentimento não disponíveis")
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["day"],
            y=df["sentiment"],
            mode="lines",
            line=dict(color="#27ae60", width=2),
            name="Sentimento",
            hovertemplate="Data: %{x|%Y-%m-%d}<br>Sentimento: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title="Sentimento Médio Diário",
        xaxis_title="Data",
        yaxis_title="Sentimento",
        yaxis=dict(rangemode="tozero" if df["sentiment"].min() >= 0 else None),
        hovermode="x unified",
    )
    return fig


def build_corr_sentiment_return_fig(start_date: str, end_date: str) -> go.Figure:
    """Scatter de correlação entre sentimento diário e retorno do Ibovespa."""
    if IBOV_DF.empty:
        return _empty_figure_with_message("Dados do Ibovespa não disponíveis para correlação")

    ibov_df = _filter_by_period(IBOV_DF, start_date, end_date)
    if ibov_df.empty:
        return _empty_figure_with_message("Sem dados do Ibovespa no período selecionado")

    # Calcular retorno diário
    ibov_df = ibov_df.sort_values("day")
    ibov_df["return_1d"] = ibov_df["close"].pct_change()
    ibov_ret = ibov_df[["day", "return_1d"]].dropna()
    
    if ibov_ret.empty:
        return _empty_figure_with_message("Dados insuficientes para calcular retornos")

    # Carregar sentimento
    if SENTIMENT_DF.empty:
        return _empty_figure_with_message("Dados de sentimento não disponíveis para correlação")
    
    sentiment_df = _filter_by_period(SENTIMENT_DF, start_date, end_date)
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
    
    # Linha de tendência
    if len(merged) >= 5:
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
        yaxis_title="Retorno Diário (%)",
        yaxis_tickformat=".2%",
    )
    return fig


def build_latency_fig(start_date: str, end_date: str) -> go.Figure:
    """Gráfico de barras de latência por fonte/daypart."""
    # Tentar usar arquivo de latência por daypart
    if not LATENCY_DAYPART_DF.empty:
        df = LATENCY_DAYPART_DF.copy()
        
        # Verificar colunas esperadas
        if "daypart" in df.columns and "mean_latency_hours" in df.columns:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=df["daypart"],
                    y=df["mean_latency_hours"],
                    marker_color="#3498db",
                    name="Latência média",
                )
            )
            fig.update_layout(
                title="Latência Informacional por Período do Dia",
                xaxis_title="Período do Dia",
                yaxis_title="Latência Média (horas)",
            )
            return fig
    
    # Fallback: usar eventos de latência
    if not LATENCY_DF.empty:
        df = _filter_by_period(LATENCY_DF, start_date, end_date, "event_day")
        if not df.empty and "latency_hours" in df.columns:
            # Agrupar por fonte se existir
            group_col = "source" if "source" in df.columns else None
            if group_col:
                agg = df.groupby(group_col)["latency_hours"].mean().reset_index()
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=agg[group_col],
                        y=agg["latency_hours"],
                        marker_color="#9b59b6",
                    )
                )
                fig.update_layout(
                    title="Latência Informacional por Fonte",
                    xaxis_title="Fonte",
                    yaxis_title="Latência Média (horas)",
                )
                return fig
    
    return _empty_figure_with_message(
        "Dados de latência não disponíveis.\n"
        "Execute o notebook 11_event_study_latency.ipynb para gerar os dados."
    )


def build_backtest_fig(start_date: str, end_date: str) -> go.Figure:
    """Gráfico de curva de patrimônio do backtest."""
    if not BACKTEST_CURVES_DF.empty:
        df = _filter_by_period(BACKTEST_CURVES_DF, start_date, end_date)
        
        if not df.empty:
            fig = go.Figure()
            
            # Verificar colunas disponíveis
            if "equity_strategy" in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["day"],
                        y=df["equity_strategy"],
                        mode="lines",
                        name="Estratégia",
                        line=dict(color="#27ae60", width=2),
                    )
                )
            
            if "benchmark" in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["day"],
                        y=df["benchmark"],
                        mode="lines",
                        name="Benchmark (B&H)",
                        line=dict(color="#95a5a6", width=2, dash="dash"),
                    )
                )
            
            if fig.data:
                fig.update_layout(
                    title="Backtest: Curva de Patrimônio",
                    xaxis_title="Data",
                    yaxis_title="Valor do Patrimônio",
                    legend=dict(x=0.02, y=0.98),
                )
                return fig
    
    # Fallback: mostrar métricas agregadas como tabela visual
    if not RESULTS_DF.empty:
        backtest_rows = RESULTS_DF[RESULTS_DF["dataset"] == "backtest_daily"]
        if not backtest_rows.empty:
            return _empty_figure_with_message(
                "Curvas diárias não disponíveis.\n"
                f"Arquivo 18_backtest_daily_curves.csv não encontrado.\n"
                f"Métricas agregadas: {len(backtest_rows)} modelos avaliados."
            )
    
    return _empty_figure_with_message(
        "Dados de backtest não disponíveis.\n"
        "Execute o notebook 18_backtest_simulation.ipynb para gerar os dados."
    )


def build_model_comparison_fig(df: pd.DataFrame, metric: str) -> go.Figure:
    """Gráfico de barras comparando modelos por métrica."""
    if df.empty:
        return _empty_figure_with_message("Dados de modelos não disponíveis")
    
    # Filtrar por métrica disponível
    if metric not in df.columns or df[metric].isna().all():
        available = [c for c in ["auc", "mda", "sharpe", "cagr"] if c in df.columns and df[c].notna().any()]
        if available:
            metric = available[0]
        else:
            return _empty_figure_with_message("Nenhuma métrica disponível")
    
    df_valid = df[df[metric].notna()].copy()
    if df_valid.empty:
        return _empty_figure_with_message(f"Sem dados válidos para métrica '{metric}'")
    
    df_valid = df_valid.sort_values(metric, ascending=False)
    
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df_valid["model"],
            y=df_valid[metric],
            marker_color="#e74c3c",
            name=metric.upper(),
        )
    )
    fig.update_layout(
        title=f"Comparativo de Modelos por {metric.upper()}",
        xaxis_title="Modelo",
        yaxis_title=metric.upper(),
        xaxis_tickangle=-45,
    )
    return fig


# ------------------------------------------------------------------------------
# App Dash
# ------------------------------------------------------------------------------

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "TCC USP - Sentimento x Ibovespa"

# Layout
app.layout = html.Div([
    html.H1("📈 TCC USP: Análise de Sentimento e Ibovespa", style={"textAlign": "center"}),
    html.Hr(),
    
    # Painel de Depuração (TAREFA 2)
    html.Details([
        html.Summary("🔧 Painel de Depuração", style={"cursor": "pointer", "fontWeight": "bold"}),
        html.Div(id="debug-panel", style={
            "backgroundColor": "#f8f9fa",
            "padding": "15px",
            "borderRadius": "5px",
            "marginTop": "10px",
            "fontFamily": "monospace",
            "fontSize": "12px",
        }),
    ], open=False, style={"marginBottom": "20px", "padding": "10px", "border": "1px solid #ddd", "borderRadius": "5px"}),
    
    # Indicadores
    html.Div(id="indicators", style={
        "display": "flex",
        "justifyContent": "space-around",
        "flexWrap": "wrap",
        "marginBottom": "20px",
        "padding": "15px",
        "backgroundColor": "#ecf0f1",
        "borderRadius": "5px",
    }),
    
    # Filtros
    html.Div([
        html.Div([
            html.Label("Período:"),
            dcc.DatePickerRange(
                id="date-range",
                start_date=START_DATE,
                end_date=END_DATE,
                display_format="DD/MM/YYYY",
            ),
        ], style={"flex": "1", "marginRight": "20px"}),
        
        html.Div([
            html.Label("Modelos:"),
            dcc.Dropdown(
                id="models-dropdown",
                options=[{"label": m, "value": m} for m in RESULTS_DF["model"].unique()] if not RESULTS_DF.empty else [],
                value=[],
                multi=True,
                placeholder="Todos os modelos",
            ),
        ], style={"flex": "1", "marginRight": "20px"}),
        
        html.Div([
            html.Label("Métrica:"),
            dcc.Dropdown(
                id="metric-dropdown",
                options=[
                    {"label": "AUC", "value": "auc"},
                    {"label": "MDA", "value": "mda"},
                    {"label": "Sharpe", "value": "sharpe"},
                    {"label": "CAGR", "value": "cagr"},
                ],
                value="auc",
            ),
        ], style={"flex": "1"}),
    ], style={"display": "flex", "marginBottom": "30px", "padding": "20px", "backgroundColor": "#fff", "borderRadius": "5px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),
    
    # Gráficos principais
    html.Div([
        html.Div([dcc.Graph(id="ibov-graph")], style={"flex": "1", "marginRight": "10px"}),
        html.Div([dcc.Graph(id="sentiment-graph")], style={"flex": "1", "marginLeft": "10px"}),
    ], style={"display": "flex", "marginBottom": "20px"}),
    
    # Gráficos adicionais
    html.Div([
        html.Div([dcc.Graph(id="corr-graph")], style={"flex": "1", "marginRight": "10px"}),
        html.Div([dcc.Graph(id="model-comparison-graph")], style={"flex": "1", "marginLeft": "10px"}),
    ], style={"display": "flex", "marginBottom": "20px"}),
    
    html.Div([
        html.Div([dcc.Graph(id="latency-graph")], style={"flex": "1", "marginRight": "10px"}),
        html.Div([dcc.Graph(id="backtest-graph")], style={"flex": "1", "marginLeft": "10px"}),
    ], style={"display": "flex", "marginBottom": "20px"}),
    
    # Tabela de resultados
    html.H3("📊 Tabela de Resultados dos Modelos"),
    html.Div(id="results-table-container"),
    
    # Rodapé
    html.Hr(),
    html.Div([
        html.P(f"Período oficial: {START_DATE} a {END_DATE}", style={"color": "#666"}),
        html.P("TCC USP - Análise de Sentimento e Ibovespa", style={"color": "#999", "fontSize": "12px"}),
    ], style={"textAlign": "center", "marginTop": "30px"}),
])


@app.callback(
    [
        Output("indicators", "children"),
        Output("ibov-graph", "figure"),
        Output("sentiment-graph", "figure"),
        Output("corr-graph", "figure"),
        Output("model-comparison-graph", "figure"),
        Output("latency-graph", "figure"),
        Output("backtest-graph", "figure"),
        Output("results-table-container", "children"),
        Output("debug-panel", "children"),
    ],
    [
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("models-dropdown", "value"),
        Input("metric-dropdown", "value"),
    ],
)
def update_dashboard(start_date, end_date, selected_models, metric):
    """Callback principal para atualizar todos os componentes."""
    
    # Debug info
    ctx = callback_context
    trigger = ctx.triggered[0]["prop_id"] if ctx.triggered else "Inicialização"
    
    # Aplicar filtros com hard cap anti-2025
    ibov_filtered = _filter_by_period(IBOV_DF, start_date, end_date)
    sentiment_filtered = _filter_by_period(SENTIMENT_DF, start_date, end_date)
    
    # Filtrar modelos
    results_filtered = RESULTS_DF.copy()
    if selected_models and len(selected_models) > 0:
        results_filtered = results_filtered[results_filtered["model"].isin(selected_models)]
    
    # Calcular período efetivo
    eff_start = pd.to_datetime(start_date) if start_date else pd.Timestamp(START_DATE)
    eff_end = min(pd.to_datetime(end_date) if end_date else END_DATE_CAP, END_DATE_CAP)
    n_days = (eff_end - eff_start).days
    
    # Indicadores
    indicators = [
        html.Div([
            html.Strong("📅 Período: "),
            html.Span(f"{eff_start.strftime('%Y-%m-%d')} a {eff_end.strftime('%Y-%m-%d')} ({n_days} dias)"),
        ]),
        html.Div([
            html.Strong("📊 Dados: "),
            html.Span(f"Ibovespa: {len(ibov_filtered)} | Sentimento: {len(sentiment_filtered)}"),
        ]),
        html.Div([
            html.Strong("🤖 Modelos: "),
            html.Span(f"{', '.join(selected_models) if selected_models else 'Todos'} ({len(results_filtered)})"),
        ]),
        html.Div([
            html.Strong("📈 Métrica: "),
            html.Span(metric.upper() if metric else "N/A"),
        ]),
    ]
    
    # Gráficos
    ibov_fig = build_ibov_figure(ibov_filtered)
    sentiment_fig = build_sentiment_figure(sentiment_filtered)
    corr_fig = build_corr_sentiment_return_fig(start_date, end_date)
    model_fig = build_model_comparison_fig(results_filtered, metric or "auc")
    latency_fig = build_latency_fig(start_date, end_date)
    backtest_fig = build_backtest_fig(start_date, end_date)
    
    # Tabela de resultados
    if not results_filtered.empty:
        table = dash_table.DataTable(
            data=results_filtered.to_dict("records"),
            columns=[{"name": c, "id": c} for c in results_filtered.columns],
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "10px"},
            style_header={"backgroundColor": "#2c3e50", "color": "white", "fontWeight": "bold"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"},
            ],
        )
    else:
        table = html.P("Nenhum resultado disponível", style={"color": "#999"})
    
    # Debug panel
    debug_items = [
        html.Li(f"Último trigger: {trigger}"),
        html.Li(f"start_date: {start_date}"),
        html.Li(f"end_date: {end_date} (cap: {END_DATE})"),
        html.Li(f"selected_models: {selected_models}"),
        html.Li(f"metric: {metric}"),
        html.Li(f"Contagem pós-filtro:"),
        html.Ul([
            html.Li(f"IBOV: {len(ibov_filtered)} linhas"),
            html.Li(f"Sentimento: {len(sentiment_filtered)} linhas"),
            html.Li(f"Resultados: {len(results_filtered)} linhas"),
        ]),
    ]
    
    # Adicionar validação ao debug
    if VALIDATION_REPORT.get("alerts"):
        debug_items.append(html.Li("🚨 ALERTAS:"))
        debug_items.append(html.Ul([html.Li(a) for a in VALIDATION_REPORT["alerts"]]))
    
    cv = VALIDATION_REPORT.get("cross_validation", {})
    if cv:
        debug_items.append(html.Li(f"📈 Validação cruzada:"))
        debug_items.append(html.Ul([
            html.Li(f"Interseção IBOV∩Sent: {cv.get('intersection_days', 0)} dias"),
            html.Li(f"Só IBOV: {cv.get('days_only_ibov', 0)} | Só Sent: {cv.get('days_only_sentiment', 0)}"),
        ]))
    
    debug_panel = html.Ul(debug_items, style={"listStyleType": "none", "paddingLeft": "0"})
    
    return (
        indicators,
        ibov_fig,
        sentiment_fig,
        corr_fig,
        model_fig,
        latency_fig,
        backtest_fig,
        table,
        debug_panel,
    )


# ------------------------------------------------------------------------------
# TAREFA 1: Inicialização robusta com fallback de porta
# ------------------------------------------------------------------------------

def find_free_port(start_port: int, max_attempts: int = 10) -> int:
    """Encontra uma porta livre a partir de start_port."""
    for i in range(max_attempts):
        port = start_port + i
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((DASH_HOST, port))
                return port
            except OSError:
                print(f"[aviso] Porta {port} ocupada, tentando próxima...")
                continue
    raise RuntimeError(f"Não foi possível encontrar porta livre após {max_attempts} tentativas a partir de {start_port}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("INICIANDO DASHBOARD")
    print("=" * 70)
    
    # Encontrar porta livre
    try:
        port = find_free_port(DASH_PORT)
        if port != DASH_PORT:
            print(f"[info] Porta original {DASH_PORT} ocupada, usando {port}")
    except RuntimeError as e:
        print(f"[erro] {e}")
        exit(1)
    
    # Imprimir URL de acesso
    url = f"http://{DASH_HOST}:{port}"
    print(f"\n🚀 Dashboard em: {url}")
    print(f"   Host: {DASH_HOST} (env: DASH_HOST)")
    print(f"   Porta: {port} (env: DASH_PORT)")
    print(f"   Período oficial: {START_DATE} a {END_DATE}")
    print("\n" + "-" * 70)
    print("Pressione Ctrl+C para encerrar")
    print("-" * 70 + "\n")
    
    # Rodar servidor
    app.run(
        host=DASH_HOST,
        port=port,
        debug=True,
        use_reloader=False,  # Evita double-load no VS Code
    )
