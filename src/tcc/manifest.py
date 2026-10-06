"""Manifesto dos arquivos de dados: nome, origem, período, linhas e SHA-256.

O manifesto versionado (`data/MANIFEST.json`, com a versão legível `data/MANIFEST.md`)
identifica os arquivos usados no artigo. `python -m tcc manifest` confere a pasta de dados
contra ele pelo SHA-256 do conteúdo, que não depende de como o arquivo foi gravado: tabelas em
CSV canônico (o mesmo arquivo gravado no Windows tem quebras de linha CRLF) e a matriz TF-IDF
pelos valores (o .npz guarda a data de gravação). O SHA-256 do arquivo também é registrado.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from tcc.config import Settings
from tcc.datasets import (
    IBOVESPA_FILE,
    NEWS_CLEAN_FILE,
    NEWS_RAW_FILE,
    TFIDF_INDEX_FILE,
    TFIDF_MATRIX_FILE,
)

CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class DataFile:
    """Arquivo de entrada do pipeline e de onde ele vem."""

    path: str  # relativo à pasta dos dados (TCC_USP_BASE)
    origin: str
    used_by: str
    date_column: str | None


DATA_FILES = (
    DataFile(
        f"data_processed/{IBOVESPA_FILE}",
        "Yahoo Finance, ^BVSP sem ajuste (`python -m tcc download-ibovespa`)",
        "reproduce",
        "date",
    ),
    DataFile(
        f"data_processed/{TFIDF_MATRIX_FILE}",
        "TF-IDF diário das notícias limpas (`python -m tcc build-tfidf`)",
        "reproduce",
        None,
    ),
    DataFile(
        f"data_processed/{TFIDF_INDEX_FILE}",
        "dia de cada linha da matriz TF-IDF (`python -m tcc build-tfidf`)",
        "reproduce",
        "day",
    ),
    DataFile(
        f"data_interim/{NEWS_CLEAN_FILE}",
        "manchetes do GDELT deduplicadas (`python -m tcc clean-news`)",
        "build-tfidf",
        "date",
    ),
    DataFile(
        f"data_raw/{NEWS_RAW_FILE}",
        "coleta do GDELT DOC 2.0 (`python -m tcc collect-news`)",
        "clean-news",
        "date",
    ),
)


def file_sha256(path: Path) -> str:
    """SHA-256 do arquivo, lido em blocos."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_content_sha256(path: Path) -> str:
    """SHA-256 do conteúdo de uma matriz esparsa (forma, valores e posições)."""
    matrix = load_npz(path).tocsr()
    matrix.sort_indices()
    digest = hashlib.sha256(np.asarray(matrix.shape, dtype=np.int64).tobytes())
    for array in (matrix.data, matrix.indices, matrix.indptr):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def table_content_sha256(frame: pd.DataFrame) -> str:
    """SHA-256 da tabela em CSV canônico (UTF-8, quebra de linha LF, sem índice)."""
    text = frame.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def content_sha256(path: Path) -> str:
    """SHA-256 do conteúdo de uma tabela (.csv/.parquet) ou matriz (.npz)."""
    if path.suffix == ".npz":
        return matrix_content_sha256(path)
    return table_content_sha256(_read_table(path))


def describe(base_dir: Path, spec: DataFile) -> dict[str, Any]:
    """Linhas, período, tamanho e SHA-256 de um arquivo presente."""
    path = base_dir / spec.path
    entry: dict[str, Any] = {
        "path": spec.path,
        "origin": spec.origin,
        "used_by": spec.used_by,
        "date_column": spec.date_column,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "content_sha256": content_sha256(path),
    }
    if path.suffix == ".npz":
        matrix = load_npz(path)
        entry |= {"rows": matrix.shape[0], "columns": matrix.shape[1]}
        return entry
    frame = _read_table(path)
    entry["rows"] = len(frame)
    if spec.date_column:
        dates = pd.to_datetime(frame[spec.date_column])
        entry["start"], entry["end"] = f"{dates.min():%Y-%m-%d}", f"{dates.max():%Y-%m-%d}"
    return entry


def build_manifest(settings: Settings) -> list[dict[str, Any]]:
    """Descrição dos arquivos de entrada encontrados na pasta dos dados."""
    return [
        describe(settings.base_dir, spec)
        for spec in DATA_FILES
        if (settings.base_dir / spec.path).exists()
    ]


def _mismatch(path: Path, expected: dict[str, Any]) -> str | None:
    """Diferença entre um arquivo presente e o manifesto (None = confere).

    Compara o SHA-256 do conteúdo; sem ele, o do arquivo (registrado, por exemplo, pelo
    `certutil` do Windows numa máquina sem Python).
    """
    if expected.get("content_sha256"):
        return None if content_sha256(path) == expected["content_sha256"] else "conteúdo diferente"
    if expected.get("sha256"):
        return None if file_sha256(path) == expected["sha256"] else "arquivo diferente"
    return "sem SHA-256 no manifesto (rode --write)"


def check_manifest(settings: Settings, manifest: list[dict[str, Any]]) -> list[str]:
    """Diferenças entre a pasta dos dados e o manifesto (vazio = tudo confere)."""
    problems = []
    for expected in manifest:
        path = settings.base_dir / expected["path"]
        if not path.exists():
            problems.append(f"ausente: {expected['path']} (usado por {expected['used_by']})")
        elif mismatch := _mismatch(path, expected):
            problems.append(f"{mismatch}: {expected['path']}")
    return problems


def read_manifest(path: Path) -> list[dict[str, Any]]:
    """Lê o manifesto JSON."""
    entries: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))["files"]
    return entries


def write_manifest(entries: list[dict[str, Any]], json_path: Path, markdown_path: Path) -> None:
    """Grava o manifesto em JSON (para o código) e em Markdown (para leitura)."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps({"files": entries}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(manifest_markdown(entries), encoding="utf-8")


def manifest_markdown(entries: list[dict[str, Any]]) -> str:
    """Tabela legível do manifesto."""
    lines = [
        "# Manifesto dos dados",
        "",
        "Arquivos de entrada do pipeline, na pasta apontada por `TCC_USP_BASE`. Os dados não são",
        "versionados (licenças das fontes de notícias). Gerado por `python -m tcc manifest --write`;",
        "`python -m tcc manifest` confere a sua pasta. O SHA-256 abaixo é o do conteúdo (tabela em",
        "CSV canônico ou valores da matriz), que não depende do sistema em que o arquivo foi",
        "gravado; o do arquivo está em `MANIFEST.json`. Sem o do conteúdo, a tabela mostra o do",
        "arquivo, e é ele que `python -m tcc manifest` confere.",
        "",
        "| Arquivo | Origem | Usado por | Período | Linhas | SHA-256 |",
        "|---|---|---|---|---|---|",
    ]
    for entry in entries:
        period = f"{entry['start']} a {entry['end']}" if "start" in entry else "—"
        rows = f"{entry['rows']:,}".replace(",", ".") if "rows" in entry else "—"
        if "columns" in entry:
            rows += f" × {entry['columns']:,}".replace(",", ".")
        if entry.get("content_sha256"):
            digest = f"`{entry['content_sha256']}`"
        elif entry.get("sha256"):
            digest = f"`{entry['sha256']}` (do arquivo: {entry.get('observacao', '')})"
        else:
            digest = f"pendente: {entry.get('observacao', '')}"
        lines.append(
            f"| `{entry['path']}` | {entry['origin']} | `{entry['used_by']}` | {period} | {rows} | {digest} |"
        )
    return "\n".join(lines) + "\n"
