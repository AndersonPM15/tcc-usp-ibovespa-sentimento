"""Gera as figuras da apresentação em `reports/apresentacao/` (`python -m tcc presentation-figures`).

Para cada figura: PNG (300 dpi), SVG e o CSV com os dados plotados, além de um README com a
lista. Arquivos com resultados pós-submissão (não constam do artigo) levam `pos-submissao`
no nome. F1 e F9 precisam também da base limpa de notícias
(`data_interim/news_clean_multisource.parquet`); sem ela, as outras figuras são geradas.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from tcc import datasets, reproduce
from tcc.config import Settings
from tcc.figures import write_csv
from tcc.presentation import charts, style
from tcc.presentation import data as slide_data

POST_SUBMISSION = "pos-submissao"
COVERAGE_STEM = "F1_cobertura_por_ano"
COVERAGE_DESCRIPTION = (
    "Tabela de apoio à F1 (só CSV): dias com manchete, média, mediana e total de manchetes "
    "por dia em cada ano da base limpa (2025 até 19/11)"
)
NEEDS_NEWS = ("F1", "F9")


@dataclass(frozen=True)
class Slide:
    """Uma figura: nome do arquivo, o que ela mostra, dados e função que a desenha."""

    stem: str
    description: str
    data: pd.DataFrame
    draw: Callable[[pd.DataFrame], Figure]


def slides(
    results: reproduce.ArticleResults,
    news: pd.DataFrame | None = None,
    ibovespa: pd.DataFrame | None = None,
) -> list[Slide]:
    """As figuras F1–F9, na ordem da apresentação (F1 e F9 só com a base de notícias)."""
    schedule = slide_data.walk_forward_schedule(results)
    article_equity = slide_data.equity_curves(results, "full_sample")
    rolling_equity = slide_data.equity_curves(results, "rolling")
    limits = charts.equity_limits(article_equity, rolling_equity)
    items = [
        Slide(
            "F2_sentimento_diario",
            "Sentimento diário (2p − 1) dos dois modelos, fronteiras dos blocos do walk-forward "
            "e histograma",
            slide_data.daily_sentiment(results),
            charts.daily_sentiment,
        ),
        Slide(
            "F3_walk_forward",
            "As 4 etapas do walk-forward, com treino e teste em datas reais",
            schedule,
            charts.walk_forward,
        ),
        Slide(
            f"F4a_dispersao_sentimento_retorno_{POST_SUBMISSION}",
            "Sentimento da Regressão Logística × retorno do mesmo dia e do dia seguinte, com reta "
            "de MQO, r e IC 95% por bootstrap em blocos (verificação b)",
            slide_data.sentiment_scatter(results),
            charts.sentiment_scatter,
        ),
        Slide(
            "F4b_correlacao_movel",
            "Correlação móvel de 60 e 90 pregões entre sentimento e retorno do mesmo dia "
            "(Figura 5 do artigo)",
            slide_data.rolling_correlations(results),
            charts.rolling_correlation,
        ),
        Slide(
            "F5a_curvas_roc",
            "Curvas ROC fora da amostra, com a AUC e o IC 95% da Tabela 2",
            slide_data.roc_curves(results),
            charts.roc_curves,
        ),
        Slide(
            f"F5b_auc_por_bloco_{POST_SUBMISSION}",
            "AUC de cada bloco de teste do walk-forward, com IC 95% (bootstrap i.i.d.)",
            slide_data.auc_by_block(results, schedule),
            charts.auc_by_block,
        ),
        Slide(
            f"F6_caar_retorno_anormal_{POST_SUBMISSION}",
            "CAAR com retorno anormal, τ = 0 a 4 pregões e IC 95% (verificação d)",
            slide_data.event_study(results),
            charts.event_study,
        ),
        Slide(
            f"F7a_patrimonio_limiares_artigo_{POST_SUBMISSION}",
            "Patrimônio (início = 1,0) das estratégias da Tabela 3 e do Ibovespa nas mesmas "
            "linhas (verificação a); sinais de 05/08/2019 a 30/12/2024, cada ponto datado pelo "
            "pregão em que o retorno se realiza",
            article_equity,
            partial(charts.equity_curves, limits=limits),
        ),
        Slide(
            f"F7b_patrimonio_sem_lookahead_{POST_SUBMISSION}",
            "O mesmo, com limiares sem look-ahead (verificação c)",
            rolling_equity,
            partial(charts.equity_curves, limits=limits),
        ),
        Slide(
            f"F8_sharpe_12_configuracoes_{POST_SUBMISSION}",
            "Sharpe das 12 configurações da Tabela 4, com os limiares do artigo e sem "
            "look-ahead, e o Sharpe do Ibovespa no mesmo período (verificações a e c)",
            slide_data.sharpe_grid(results),
            charts.sharpe_heatmaps,
        ),
    ]
    if news is None or ibovespa is None:
        return items
    first = Slide(
        "F1_ibovespa_periodos_manchetes",
        "Ibovespa de 2018 a 2025 com as faixas do estudo e manchetes por dia (base limpa)",
        slide_data.ibovespa_and_headlines(results, news, schedule),
        charts.ibovespa_and_headlines,
    )
    last = Slide(
        f"F9_auc_token_none_{POST_SUBMISSION}",
        'AUC com e sem o token espúrio "None" (bug 12), com IC 95%',
        slide_data.none_token_auc(results, news, ibovespa),
        charts.none_token_auc,
    )
    return [first, *items, last]


def readme(items: list[Slide], tables: dict[str, str]) -> str:
    """Lista das figuras e das tabelas de apoio para `reports/apresentacao/README.md`."""
    lines = [
        "# Figuras da apresentação",
        "",
        "Geradas por `python -m tcc presentation-figures` (código em `src/tcc/presentation/`).",
        "Cada figura tem PNG (300 dpi), SVG e o CSV com os dados plotados (ponto decimal no",
        "CSV; vírgula decimal nas figuras). Tamanho de slide 11,3 × 5,3 pol. (largura total:",
        "17,7 × 5,3 pol.), sem título dentro da figura.",
        "",
        f"> Arquivos com `{POST_SUBMISSION}` no nome trazem resultados calculados depois da",
        "> submissão (verificações em `reports/verificacao/`); não constam do artigo.",
        "",
        "| Arquivo | O que mostra |",
        "|---|---|",
    ]
    lines += [f"| `{item.stem}` | {item.description} |" for item in items]
    lines += [f"| `{stem}` | {description} |" for stem, description in tables.items()]
    return "\n".join(lines) + "\n"


def write_figures(
    settings: Settings, results: reproduce.ArticleResults | None = None
) -> tuple[list[Path], list[str]]:
    """Grava as figuras e devolve os arquivos gravados e as figuras puladas (sem notícias)."""
    results = results if results is not None else reproduce.compute_article(settings)
    news_path = settings.interim_dir / datasets.NEWS_CLEAN_FILE
    news = datasets.read_clean_news(settings) if news_path.exists() else None
    ibovespa = datasets.read_ibovespa(settings)
    output = settings.presentation_dir
    written: list[Path] = []
    with style.slide_style():
        items = slides(results, news, ibovespa)
        for item in items:
            written += style.save(item.draw(item.data), item.data, output, item.stem)
    tables: dict[str, str] = {}
    if news is not None:
        coverage_path = output / f"{COVERAGE_STEM}.csv"
        write_csv(slide_data.coverage_by_year(news), coverage_path)
        written.append(coverage_path)
        tables[COVERAGE_STEM] = COVERAGE_DESCRIPTION
    readme_path = output / "README.md"
    readme_path.write_text(readme(items, tables), encoding="utf-8")
    skipped = [] if news is not None else list(NEEDS_NEWS)
    return [*written, readme_path], skipped
