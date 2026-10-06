"""Linha de comando: `python -m tcc <comando>` (ou `tcc <comando>` após `pip install -e .`)."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from tcc import market
from tcc.config import Settings, load_settings
from tcc.datasets import IBOVESPA_FILE


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


COMMANDS: dict[str, tuple[str, Callable[[Settings, argparse.Namespace], None]]] = {
    "download-ibovespa": ("baixa o Ibovespa diário (02/01/2018 a 18/11/2025)", _download_ibovespa),
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
    for name, (help_text, _) in COMMANDS.items():
        subparsers.add_parser(name, help=help_text)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Executa o subcomando pedido."""
    args = build_parser().parse_args(argv)
    settings = load_settings(args.data_dir, args.reports_dir)
    _, command = COMMANDS[args.command]
    command(settings, args)
