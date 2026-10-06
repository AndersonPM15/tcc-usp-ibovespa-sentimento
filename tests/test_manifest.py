"""Manifesto dos dados (tcc.manifest)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, save_npz

from tcc import manifest
from tcc.config import Settings, load_settings

COMMITTED = Path(__file__).resolve().parents[1] / "data" / "MANIFEST.json"


def test_table_hash_ignores_line_endings(tmp_path: Path) -> None:
    frame = pd.DataFrame({"day": ["2024-01-02", "2024-01-03"], "row_id": [0, 1]})
    frame.to_csv(tmp_path / "lf.csv", index=False, lineterminator="\n")
    frame.to_csv(tmp_path / "crlf.csv", index=False, lineterminator="\r\n")
    assert manifest.file_sha256(tmp_path / "lf.csv") != manifest.file_sha256(tmp_path / "crlf.csv")
    assert manifest.content_sha256(tmp_path / "lf.csv") == manifest.content_sha256(
        tmp_path / "crlf.csv"
    )


def test_matrix_hash_depends_only_on_values(tmp_path: Path) -> None:
    matrix = csr_matrix(np.array([[0.0, 1.5], [2.0, 0.0]]))
    save_npz(tmp_path / "a.npz", matrix)
    save_npz(tmp_path / "b.npz", matrix.tocoo().tocsr())
    changed = matrix.copy()
    changed[0, 1] = 1.6
    save_npz(tmp_path / "c.npz", changed)
    assert manifest.content_sha256(tmp_path / "a.npz") == manifest.content_sha256(
        tmp_path / "b.npz"
    )
    assert manifest.content_sha256(tmp_path / "a.npz") != manifest.content_sha256(
        tmp_path / "c.npz"
    )


def _data_dir_with_ibovespa(base: Path) -> Settings:
    (base / "data_processed").mkdir(parents=True)
    pd.DataFrame({"date": ["2018-01-02", "2018-01-03"], "close": [1.0, 2.0]}).to_csv(
        base / "data_processed" / "ibovespa_clean.csv", index=False
    )
    return load_settings(base)


def test_build_and_check_manifest(tmp_path: Path) -> None:
    settings = _data_dir_with_ibovespa(tmp_path)
    entries = manifest.build_manifest(settings)
    assert [entry["path"] for entry in entries] == ["data_processed/ibovespa_clean.csv"]
    assert entries[0]["rows"] == 2
    assert (entries[0]["start"], entries[0]["end"]) == ("2018-01-02", "2018-01-03")
    assert manifest.check_manifest(settings, entries) == []

    pd.DataFrame({"date": ["2018-01-02"], "close": [1.0]}).to_csv(
        settings.processed_dir / "ibovespa_clean.csv", index=False
    )
    assert manifest.check_manifest(settings, entries) == [
        "conteúdo diferente: data_processed/ibovespa_clean.csv"
    ]
    missing = [{"path": "data_raw/x.parquet", "used_by": "clean-news", "content_sha256": None}]
    assert manifest.check_manifest(settings, missing)[0].startswith("ausente")


def test_file_hash_is_checked_when_content_hash_is_missing(tmp_path: Path) -> None:
    # Ex.: base bruta registrada pelo `certutil` do Windows, numa máquina sem Python.
    settings = _data_dir_with_ibovespa(tmp_path)
    path = settings.processed_dir / "ibovespa_clean.csv"
    entry = {
        "path": "data_processed/ibovespa_clean.csv",
        "used_by": "reproduce",
        "sha256": manifest.file_sha256(path),
        "content_sha256": None,
    }
    assert manifest.check_manifest(settings, [entry]) == []
    path.write_text("date,close\n2018-01-02,1.0\n", encoding="utf-8")
    assert manifest.check_manifest(settings, [entry]) == [
        "arquivo diferente: data_processed/ibovespa_clean.csv"
    ]
    no_hash = {**entry, "sha256": None}
    assert manifest.check_manifest(settings, [no_hash])[0].startswith("sem SHA-256")
    markdown = manifest.manifest_markdown([{**entry, "origin": "GDELT", "observacao": "certutil"}])
    assert f"`{entry['sha256']}` (do arquivo: certutil)" in markdown


def test_write_and_read_round_trip(tmp_path: Path) -> None:
    entries = manifest.build_manifest(_data_dir_with_ibovespa(tmp_path / "dados"))
    manifest.write_manifest(entries, tmp_path / "MANIFEST.json", tmp_path / "MANIFEST.md")
    assert manifest.read_manifest(tmp_path / "MANIFEST.json") == entries
    assert "ibovespa_clean.csv" in (tmp_path / "MANIFEST.md").read_text(encoding="utf-8")


def test_reproduction_inputs_match_the_committed_manifest(settings: Settings) -> None:
    inputs = [e for e in manifest.read_manifest(COMMITTED) if e["used_by"] == "reproduce"]
    assert len(inputs) == 3
    problems = manifest.check_manifest(settings, inputs)
    if problems:
        pytest.fail(f"entradas diferentes das do artigo: {problems}")
