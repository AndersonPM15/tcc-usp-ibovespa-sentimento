<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 1a36a252b9058167036afa797e4ea731cbfd170b
>>>>>>> d840ce4f55f46272373765ca9c5f288e418fd15e
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
<<<<<<< HEAD
=======
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
<<<<<<< HEAD
    
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
    
=======
=======
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




def build_corr_sentiment_return_fig(start_date: str, end_date: str) -> go.Figure:
    """Scatter de correlacao entre sentimento diario e retorno do Ibovespa."""
    if IBOV_DF.empty:
        return go.Figure().add_annotation(text="Dados insuficientes para calcular a correlacao", showarrow=False)

    ibov_df = IBOV_DF.copy()
    if "date" not in ibov_df.columns:
        return go.Figure().add_annotation(text="Coluna 'date' nao encontrada no Ibovespa", showarrow=False)
    if "close" not in ibov_df.columns and "adj_close" not in ibov_df.columns:
        return go.Figure().add_annotation(text="Preco de fechamento ausente no Ibovespa", showarrow=False)
    if "close" not in ibov_df.columns and "adj_close" in ibov_df.columns:
        ibov_df["close"] = ibov_df["adj_close"]

    ibov_df["date"] = pd.to_datetime(ibov_df["date"], errors="coerce")
    ibov_df = ibov_df.dropna(subset=["date"])
    if start_date and end_date:
        ibov_df = ibov_df[(ibov_df["date"] >= pd.to_datetime(start_date)) & (ibov_df["date"] <= pd.to_datetime(end_date))]
    ibov_df = ibov_df.sort_values("date")
    if ibov_df.empty:
        return go.Figure().add_annotation(text="Dados insuficientes para calcular a correlacao", showarrow=False)

    if "return_1d" not in ibov_df.columns:
        ibov_df["return_1d"] = ibov_df["close"].pct_change()
    ibov_df["day"] = ibov_df["date"].dt.normalize()
    ibov_df = ibov_df[["day", "return_1d"]].dropna()
    if ibov_df.empty:
        return go.Figure().add_annotation(text="Dados insuficientes para calcular a correlacao", showarrow=False)

    oof_df = _safe_read_csv(OOF_PATH)
    if oof_df.empty:
        return go.Figure().add_annotation(text="Dados insuficientes para calcular a correlacao", showarrow=False)

    if "day" in oof_df.columns:
        oof_df["day"] = pd.to_datetime(oof_df["day"], errors="coerce").dt.normalize()
        date_col = "day"
    elif "date" in oof_df.columns:
        oof_df["day"] = pd.to_datetime(oof_df["date"], errors="coerce").dt.normalize()
        date_col = "day"
    else:
        return go.Figure().add_annotation(text="Coluna de data nao encontrada no OOF de sentimento", showarrow=False)

    prob_candidates = [c for c in ["proba", "prob", "score"] if c in oof_df.columns]
    prob_col = prob_candidates[0] if prob_candidates else None
    if prob_col is None:
        prob_col = next(
            (c for c in oof_df.columns if c not in {"day", "model", "fold", "date"} and pd.api.types.is_numeric_dtype(oof_df[c])),
            None,
        )
    if prob_col is None:
        return go.Figure().add_annotation(text="Colunas de probabilidade de sentimento nao encontradas", showarrow=False)

    oof_df = oof_df.dropna(subset=[date_col])
    if start_date and end_date:
        oof_df = oof_df[(oof_df[date_col] >= pd.to_datetime(start_date)) & (oof_df[date_col] <= pd.to_datetime(end_date))]
    if oof_df.empty:
        return go.Figure().add_annotation(text="Dados insuficientes para calcular a correlacao", showarrow=False)

    sentiment_daily = (
        oof_df.groupby("day")[prob_col]
        .mean()
        .reset_index()
        .rename(columns={prob_col: "sentiment_score"})
    )

    merged = pd.merge(ibov_df, sentiment_daily, on="day", how="inner").dropna()
    if merged.empty or len(merged) < 3:
        return go.Figure().add_annotation(text="Dados insuficientes para calcular a correlacao", showarrow=False)

    corr_value = merged["return_1d"].corr(merged["sentiment_score"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=merged["sentiment_score"],
            y=merged["return_1d"],
            mode="markers",
            marker=dict(color="#1f77b4", size=8, opacity=0.7),
            name="Observacoes diarias",
            hovertemplate="Sentimento=%{x:.3f}<br>Retorno=%{y:.3%}<extra></extra>",
        )
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0,
        y=1.05,
        showarrow=False,
        text=f"Correlacao de Pearson: {corr_value:.3f}" if pd.notna(corr_value) else "Correlacao nao disponivel",
        font=dict(color="#444", size=12),
    )
    fig.update_layout(
        title="Correlacao entre sentimento diario e retorno do Ibovespa",
        xaxis_title="Sentimento medio diario",
        yaxis_title="Retorno diario do Ibovespa",
        template="plotly_white",
        margin=dict(l=60, r=20, t=70, b=60),
    )
    return fig



def build_latency_fig(start_date: str, end_date: str) -> go.Figure:
    """Barra de latencia informacional por fonte/daypart."""
    df = LATENCY_DF.copy()
    if df.empty:
        return go.Figure().add_annotation(text="Dados de latencia nao disponiveis", showarrow=False)

    date_col = next((c for c in ["event_day", "date", "day"] if c in df.columns), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        if start_date and end_date:
            df = df[(df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))]
    else:
        # Dataset nao tem coluna de data clara; nenhum filtro temporal aplicado.
        pass

    if df.empty:
        return go.Figure().add_annotation(text="Dados de latencia nao disponiveis", showarrow=False)

    metric_candidates = ["t_half_mediana", "t_half_media", "t_half", "latency", "car", "car_mean", "impact"]
    metric = next((c for c in metric_candidates if c in df.columns), None)
    if metric is None:
        return go.Figure().add_annotation(text="Dados de latencia nao disponiveis", showarrow=False)

    group_cols: list[str] = []
    if "fonte" in df.columns:
        group_cols.append("fonte")
    if "daypart" in df.columns:
        group_cols.append("daypart")
    if not group_cols:
        return go.Figure().add_annotation(text="Dados de latencia nao disponiveis", showarrow=False)

    agg = df.groupby(group_cols, dropna=False)[metric].mean().reset_index()
    if agg.empty:
        return go.Figure().add_annotation(text="Dados de latencia nao disponiveis", showarrow=False)

    fig = go.Figure()
    label_y = f"Media de {metric}"
    if len(group_cols) == 1:
        fig.add_trace(
            go.Bar(
                x=agg[group_cols[0]],
                y=agg[metric],
                marker_color="#3498db",
                name=label_y,
            )
        )
    else:
        for daypart, subdf in agg.groupby("daypart"):
            fig.add_trace(
                go.Bar(
                    x=subdf["fonte"],
                    y=subdf[metric],
                    name=daypart,
                )
            )
        fig.update_layout(barmode="group")

    fig.update_layout(
        title="Latencia informacional por fonte de noticia",
        xaxis_title="Fonte",
        yaxis_title=label_y,
        template="plotly_white",
        margin=dict(l=60, r=20, t=60, b=60),
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




def build_backtest_fig(model_value: str | None) -> go.Figure:
    """Curva de patrimonio da estrategia de sentimento versus benchmark."""
    df = _safe_read_csv(BACKTEST_PATH)
    if df.empty:
        return go.Figure().add_annotation(text="Resultados de backtest nao disponiveis", showarrow=False)

    date_col = next((c for c in ["date", "day", "Data", "data"] if c in df.columns), None)
    if date_col is None:
        return go.Figure().add_annotation(text="Resultados de backtest nao disponiveis", showarrow=False)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if df.empty:
        return go.Figure().add_annotation(text="Resultados de backtest nao disponiveis", showarrow=False)

    if "model" in df.columns:
        chosen_model = model_value or (df["model"].dropna().iloc[0] if not df["model"].dropna().empty else None)
        if chosen_model is not None:
            df = df[df["model"] == chosen_model]
            if df.empty:
                return go.Figure().add_annotation(text="Resultados de backtest nao disponiveis", showarrow=False)

    strat_col, bench_col = _pick_equity_columns(df)
    if strat_col is None or bench_col is None:
        return go.Figure().add_annotation(text="Resultados de backtest nao disponiveis", showarrow=False)

    curves = df[[date_col, strat_col, bench_col]].dropna(how="all")
    if curves.empty:
        return go.Figure().add_annotation(text="Resultados de backtest nao disponiveis", showarrow=False)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curves[date_col],
            y=curves[strat_col],
            mode="lines",
            name="Estrategia (sentimento)",
            line=dict(color="#2c3e50", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=curves[date_col],
            y=curves[bench_col],
            mode="lines",
            name="Benchmark",
            line=dict(color="#e67e22", width=2, dash="dash"),
        )
    )
>>>>>>> 0bec322af875e1eb1f81db1d0ff06b12a116d373
>>>>>>> 1a36a252b9058167036afa797e4ea731cbfd170b
>>>>>>> d840ce4f55f46272373765ca9c5f288e418fd15e
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

<<<<<<< HEAD
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "TCC USP - Sentimento x Ibovespa"
=======
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 1a36a252b9058167036afa797e4ea731cbfd170b
app = Dash(__name__)
app.title = "Dashboard Sentimento x Ibovespa"
>>>>>>> d840ce4f55f46272373765ca9c5f288e418fd15e

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
<<<<<<< HEAD
=======


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
<<<<<<< HEAD
=======
=======
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
    if df.empty or start is None or end is None:
        return df
    mask = (df["day"] >= pd.to_datetime(start)) & (df["day"] <= pd.to_datetime(end))
    return df.loc[mask].copy()


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
>>>>>>> 0bec322af875e1eb1f81db1d0ff06b12a116d373
>>>>>>> 1a36a252b9058167036afa797e4ea731cbfd170b
>>>>>>> d840ce4f55f46272373765ca9c5f288e418fd15e
def update_dashboard(start_date, end_date, selected_models, metric):
    """Callback principal para atualizar todos os componentes."""
    
    # Debug info
    ctx = callback_context
    trigger = ctx.triggered[0]["prop_id"] if ctx.triggered else "Inicialização"
    
    # Aplicar filtros com hard cap anti-2025
    ibov_filtered = _filter_by_period(IBOV_DF, start_date, end_date)
<<<<<<< HEAD
    sentiment_filtered = _filter_by_period(SENTIMENT_DF, start_date, end_date)
=======
<<<<<<< HEAD
    print(f"[DEBUG] ibov_filtered shape: {ibov_filtered.shape if not ibov_filtered.empty else 'VAZIO'}")
=======
<<<<<<< HEAD
>>>>>>> 1a36a252b9058167036afa797e4ea731cbfd170b

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
<<<<<<< HEAD
=======
=======

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

    # Gráfico de sentimento
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
>>>>>>> 0bec322af875e1eb1f81db1d0ff06b12a116d373
>>>>>>> 1a36a252b9058167036afa797e4ea731cbfd170b
    comparison_fig = go.Figure()
    table_df = RESULTS_DF.copy()
    metric_value = metric if metric in metric_map and metric in table_df.columns else "auc"
>>>>>>> d840ce4f55f46272373765ca9c5f288e418fd15e
    
    # Filtrar modelos
    results_filtered = RESULTS_DF.copy()
    if selected_models and len(selected_models) > 0:
        results_filtered = results_filtered[results_filtered["model"].isin(selected_models)]
    
<<<<<<< HEAD
    # Calcular período efetivo
    eff_start = pd.to_datetime(start_date) if start_date else pd.Timestamp(START_DATE)
    eff_end = min(pd.to_datetime(end_date) if end_date else END_DATE_CAP, END_DATE_CAP)
    n_days = (eff_end - eff_start).days
    
    # Indicadores
    indicators = [
=======
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
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 1a36a252b9058167036afa797e4ea731cbfd170b
                            color=["#27ae60" if model == best_model else "#2980b9" 
                                   for model in table_df_sorted["model"]],
                            width=2,
                        ),
                    ),
                )
<<<<<<< HEAD
=======
=======
                            color=["#27ae60" if model == best_model else "#2980b9" 
                                   for model in table_df_sorted["model"]],
                            width=2,
                        ),
                    ),
                )
>>>>>>> 0bec322af875e1eb1f81db1d0ff06b12a116d373
>>>>>>> 1a36a252b9058167036afa797e4ea731cbfd170b
                )
            
            comparison_fig.update_layout(
                xaxis_title="Modelo",
                yaxis_title=metric_map.get(metric_value, metric_value.upper()),
                hovermode="x unified",
                showlegend=False,
                template="plotly_white",
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 1a36a252b9058167036afa797e4ea731cbfd170b
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
<<<<<<< HEAD
=======
=======
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
>>>>>>> 0bec322af875e1eb1f81db1d0ff06b12a116d373
>>>>>>> 1a36a252b9058167036afa797e4ea731cbfd170b
    if metric_value in {"auc", "mda", "sharpe"} and metric_value in table_df.columns:
        table_df = table_df.sort_values(metric_value, ascending=False, na_position="last")
    table_df = table_df.fillna("")

    # Criar indicadores visuais de filtros ativos
    metric_labels = {"auc": "AUC", "mda": "MDA", "sharpe": "Sharpe Ratio"}
    
    # Calcular janela temporal
    days_count = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 1a36a252b9058167036afa797e4ea731cbfd170b
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
<<<<<<< HEAD
=======
=======
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
>>>>>>> 0bec322af875e1eb1f81db1d0ff06b12a116d373
>>>>>>> 1a36a252b9058167036afa797e4ea731cbfd170b
            html.Div([
                html.Strong("📈 Métrica: ", style={"color": "#1976d2"}),
                html.Span(metric_labels.get(metric_value, metric_value.upper()), style={"fontWeight": "600", "color": "#2e7d32"}),
            ]),
        ],
    )

    metric_badge_text = f"📊 Métrica: {metric_labels.get(metric_value, metric_value.upper())}"
    
<<<<<<< HEAD
    # Painel de depuração
    table_rows_count = len(table_df) if not table_df.empty else 0
    debug_content = html.Div([
>>>>>>> d840ce4f55f46272373765ca9c5f288e418fd15e
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
    
<<<<<<< HEAD
    # Gráficos
    ibov_fig = build_ibov_figure(ibov_filtered)
    sentiment_fig = build_sentiment_figure(sentiment_filtered)
=======
    return ibov_fig, sentiment_fig, comparison_fig, table_df.to_dict("records"), indicator_content, metric_badge_text, debug_content
=======
    return ibov_fig, sentiment_fig, comparison_fig, table_df.to_dict("records"), indicator_content, metric_badge_text
<<<<<<< HEAD
>>>>>>> 1a36a252b9058167036afa797e4ea731cbfd170b


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
>>>>>>> d840ce4f55f46272373765ca9c5f288e418fd15e
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
<<<<<<< HEAD
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
=======
<<<<<<< HEAD
    print(f"Iniciando dashboard em http://localhost:{DASH_PORT} ...")
    # use_reloader=False evita double-loading que causa "Duplicate callback outputs"
    app.run(port=DASH_PORT, debug=True, use_reloader=False)
=======
    print("Iniciando dashboard em http://localhost:8050 ...")
    app.run(debug=True)
=======


@app.callback(
    Output("scatter-sentiment-return", "figure"),
    Output("bar-latency-source", "figure"),
    Output("line-backtest-equity", "figure"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("backtest-model-dropdown", "value"),
)
def update_additional_graphs(start_date, end_date, backtest_model):
    corr_fig = build_corr_sentiment_return_fig(start_date, end_date)
    latency_fig = build_latency_fig(start_date, end_date)
    backtest_fig = build_backtest_fig(backtest_model)
    return corr_fig, latency_fig, backtest_fig


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("Iniciando dashboard em http://localhost:8050 ...")
    app.run(debug=True)
>>>>>>> 0bec322af875e1eb1f81db1d0ff06b12a116d373
>>>>>>> 1a36a252b9058167036afa797e4ea731cbfd170b
>>>>>>> d840ce4f55f46272373765ca9c5f288e418fd15e
