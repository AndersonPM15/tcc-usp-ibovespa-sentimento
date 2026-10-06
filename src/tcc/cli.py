"""Linha de comando: `python -m tcc <comando>` (ou `tcc <comando>` após `pip install -e .`).

Pipeline completo, na ordem: `collect-news` → `clean-news` → `build-tfidf` e
`download-ibovespa` → `reproduce`. Para reproduzir o artigo bastam as três entradas
descritas em `data/MANIFEST.md` e o comando `reproduce`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from scipy.sparse import save_npz

from tcc import market, reproduce
from tcc.config import Settings, load_settings
from tcc.datasets import (
    IBOVESPA_FILE,
    NEWS_CLEAN_FILE,
    NEWS_RAW_FILE,
    TFIDF_INDEX_FILE,
    TFIDF_MATRIX_FILE,
)
from tcc.news import etl, gdelt, text

COLLECTION_START = date(2018, 1, 2)
COLLECTION_END = date(2025, 11, 19)
CHECKPOINT_DAYS = 30


def _collect_news(settings: Settings, args: argparse.Namespace) -> None:
    """Coleta o GDELT em blocos de 30 dias (com checkpoint) e grava a base de notícias."""
    import requests  # dependência opcional (extra `pipeline`)

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = settings.raw_dir / "gdelt_checkpoint.parquet"
    chunks: list[pd.DataFrame] = [pd.read_parquet(checkpoint)] if checkpoint.exists() else []
    start = chunks[0]["date"].max().date() + timedelta(days=1) if chunks else args.start
    failed: list[date] = []
    with requests.Session() as session:
        while start <= args.end:
            end = min(start + timedelta(days=CHECKPOINT_DAYS - 1), args.end)
            articles, failed_days = gdelt.collect(start, end, session)
            chunks.append(articles)
            failed.extend(failed_days)
            pd.concat(chunks, ignore_index=True).to_parquet(checkpoint, index=False)
            print(f"{start} a {end}: {len(articles)} artigos; {len(failed_days)} dias com falha")
            start = end + timedelta(days=1)
    base = gdelt.consolidate(chunks)
    base.to_parquet(settings.raw_dir / NEWS_RAW_FILE, index=False)
    print(
        f"{settings.raw_dir / NEWS_RAW_FILE}: {len(base)} manchetes, {base['date'].nunique()} dias"
    )
    if failed:
        print(f"Dias sem resposta após as novas tentativas ({len(failed)}): {failed}")


def _clean_news(settings: Settings, _args: argparse.Namespace) -> None:
    """Deduplica a base de notícias e grava `data_interim/news_clean_multisource.parquet`."""
    clean = etl.deduplicate(pd.read_parquet(settings.raw_dir / NEWS_RAW_FILE))
    settings.interim_dir.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(settings.interim_dir / NEWS_CLEAN_FILE, index=False)
    print(f"{settings.interim_dir / NEWS_CLEAN_FILE}: {len(clean)} manchetes")


def _build_tfidf(settings: Settings, _args: argparse.Namespace) -> None:
    """Gera a matriz TF-IDF diária e o seu índice em `data_processed/`."""
    news = pd.read_parquet(settings.interim_dir / NEWS_CLEAN_FILE)
    matrix, index, vocabulary = text.tfidf_matrix(text.daily_documents(news))
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    save_npz(settings.processed_dir / TFIDF_MATRIX_FILE, matrix)
    index.to_csv(settings.processed_dir / TFIDF_INDEX_FILE, index=False)
    print(f"Matriz TF-IDF: {matrix.shape[0]} dias × {len(vocabulary)} termos")


def _download_ibovespa(settings: Settings, _args: argparse.Namespace) -> None:
    """Baixa o Ibovespa do período do artigo e grava `data_processed/ibovespa_clean.csv`."""
    prices = market.prepare_ibovespa(market.download_ibovespa())
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    target = settings.processed_dir / IBOVESPA_FILE
    prices.to_csv(target, index=False, encoding="utf-8")
    print(
        f"{target}: {len(prices)} pregões, {prices['date'].min():%d/%m/%Y} a "
        f"{prices['date'].max():%d/%m/%Y}"
    )


def _reproduce(settings: Settings, args: argparse.Namespace) -> None:
    """Reproduz tabelas, figuras e verificações e compara com os números do artigo."""
    checks = reproduce.run(settings, with_verifications=not args.skip_verifications)
    for check in checks:
        print(f"[{'OK' if check.ok else 'DIFERENTE'}] {check.item}: {check.detail}")
    print(f"Tabelas e figuras: {settings.figures_dir}")
    if not args.skip_verifications:
        print(f"Verificações pós-submissão: {settings.verification_dir}")
    if not all(check.ok for check in checks):
        raise SystemExit("Algum número difere do artigo (ver linhas DIFERENTE acima).")


Command = Callable[[Settings, argparse.Namespace], None]
COMMANDS: dict[str, tuple[str, Command]] = {
    "collect-news": ("coleta as manchetes no GDELT (horas; resultado pode variar)", _collect_news),
    "clean-news": ("deduplica a base de notícias", _clean_news),
    "build-tfidf": ("gera a matriz TF-IDF diária a partir das notícias limpas", _build_tfidf),
    "download-ibovespa": ("baixa o Ibovespa diário (02/01/2018 a 18/11/2025)", _download_ibovespa),
    "reproduce": ("reproduz as Tabelas 1–4, as figuras e as verificações a–e", _reproduce),
}


def build_parser() -> argparse.ArgumentParser:
    """Argumentos comuns (pastas de dados e de saída) e um subcomando por etapa."""
    parser = argparse.ArgumentParser(prog="tcc", description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, help="pasta com data_raw/, data_interim/ e data_processed/"
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=Path("reports"), help="pasta de saída (padrão: reports)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    commands = {
        name: subparsers.add_parser(name, help=help_text)
        for name, (help_text, _) in COMMANDS.items()
    }
    commands["reproduce"].add_argument(
        "--skip-verifications",
        action="store_true",
        help="só o artigo, sem as verificações pós-submissão (mais rápido)",
    )
    commands["collect-news"].add_argument(
        "--start", type=date.fromisoformat, default=COLLECTION_START
    )
    commands["collect-news"].add_argument("--end", type=date.fromisoformat, default=COLLECTION_END)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Executa o subcomando pedido."""
    args = build_parser().parse_args(argv)
    settings = load_settings(args.data_dir, args.reports_dir)
    _, command = COMMANDS[args.command]
    command(settings, args)
