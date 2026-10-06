"""Linha de comando (tcc.cli)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tcc import cli
from tcc.config import BASE_DIR_ENV, Settings

WORDS = ["alta", "queda", "bancos", "petróleo", "dólar", "juros", "varejo"]


def test_parser_knows_every_command() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["reproduce", "--skip-verifications"])
    assert args.command == "reproduce" and args.skip_verifications
    assert parser.parse_args(["download-ibovespa"]).command == "download-ibovespa"


def test_missing_data_folder_is_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(BASE_DIR_ENV, raising=False)
    monkeypatch.chdir(tmp_path)  # sem .env
    with pytest.raises(RuntimeError, match=BASE_DIR_ENV):
        cli.main(["reproduce"])


def test_reproduce_writes_article_outputs(settings: Settings, tmp_path: Path) -> None:
    cli.main(
        [
            "--data-dir",
            str(settings.base_dir),
            "--reports-dir",
            str(tmp_path),
            "reproduce",
            "--skip-verifications",
        ]
    )
    produced = {path.name for path in (tmp_path / "figures").iterdir()}
    for table in range(1, 5):
        assert any(name.startswith(f"Tabela_{table}_") for name in produced), table
    for figure in ("1", "2", "3", "4", "5", "6", "7A", "7B", "8", "9"):
        assert any(name.startswith(f"Figura_{figure}_") for name in produced), figure


def test_news_commands_build_tfidf_from_raw_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tcc.news import text

    monkeypatch.setattr(text, "portuguese_stopwords", lambda: frozenset({"de", "a"}))
    raw_dir = tmp_path / "data_raw"
    raw_dir.mkdir()
    days = pd.date_range("2024-01-01", periods=220)
    pd.DataFrame(
        {
            "date": days.repeat(2),
            "source": "site.com",
            "title": [
                f"Bolsa fecha em {WORDS[i % 7]} com {WORDS[(i * 3) % 7]}" for i in range(440)
            ],
            "url": [f"https://site.com/{i}" for i in range(440)],
            "text_full": "",
        }
    ).to_parquet(raw_dir / "news_multisource.parquet", index=False)
    cli.main(["--data-dir", str(tmp_path), "clean-news"])
    cli.main(["--data-dir", str(tmp_path), "build-tfidf"])
    index = pd.read_csv(tmp_path / "data_processed" / "tfidf_daily_index.csv")
    assert len(index) == 220
    assert (tmp_path / "data_processed" / "tfidf_daily_matrix.npz").exists()
