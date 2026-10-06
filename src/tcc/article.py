"""Números publicados no artigo e comparação de uma reprodução com eles.

As referências ficam em `tcc/reference/`: os valores completos dos arquivos de resultado da
rodada do artigo (`artigo.json`) e as tabelas e séries publicadas (CSV).

Tolerâncias: 1e-9 nos valores calculados (diferenças de arredondamento de ponto flutuante);
igualdade de texto nas Tabelas 3 e 4, publicadas com 3 casas. Exceção documentada: a Fig. 7B
do artigo veio de um bootstrap sem semente; em 200 repetições ele se afastou do artigo até
0,0019 nas médias e 0,0068 nos ICs, por isso as tolerâncias são 0,0025 e 0,01.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any

import numpy as np
import pandas as pd

from tcc.config import MODEL_LABELS

TOLERANCE = 1e-9
FIGURE_7B_MEAN_TOLERANCE = 2.5e-3
FIGURE_7B_CI_TOLERANCE = 1e-2


@dataclass(frozen=True)
class Check:
    """Resultado da comparação de um item do artigo."""

    item: str
    ok: bool
    detail: str


@cache
def published_numbers() -> dict[str, Any]:
    """Valores publicados (Tabelas 1 e 2, Figuras 3, 4 e 8)."""
    text = resources.files("tcc.reference").joinpath("artigo.json").read_text(encoding="utf-8")
    numbers: dict[str, Any] = json.loads(text)
    return numbers


def published_table(name: str, as_text: bool = False) -> pd.DataFrame:
    """Tabela ou série publicada (`tabela_3`, `tabela_4`, `figura_7b_caar`, …).

    `as_text` mantém cada célula como está no arquivo (ex.: "-0.080", e não -0.08).
    """
    with resources.files("tcc.reference").joinpath(f"{name}.csv").open(encoding="utf-8") as file:
        return pd.read_csv(file, dtype=str if as_text else None)


def _close(value: Any, expected: float, tolerance: float = TOLERANCE) -> bool:
    return bool(abs(float(value) - expected) <= tolerance)


def check_table1(counts: dict[str, int]) -> Check:
    """Tabela 1: pregões do Ibovespa, dias com sentimento e interseção."""
    expected = published_numbers()["tabela_1"]
    return Check("Tabela 1", counts == expected, f"obtido {counts}; artigo {expected}")


def check_table2(summary: pd.DataFrame) -> Check:
    """Tabela 2: AUC, MDA, IC da AUC e nº de dias de cada modelo."""
    expected = published_numbers()["tabela_2"]
    produced = summary.set_index("model")
    gaps = {
        f"{model}.{column}": float(produced.loc[model, column]) - value
        for model, values in expected.items()
        for column, value in values.items()
    }
    ok = all(abs(gap) <= TOLERANCE for gap in gaps.values())
    worst = max(gaps, key=lambda key: abs(gaps[key]))
    return Check("Tabela 2", ok, f"maior diferença {worst} = {gaps[worst]:.2e}")


def check_formatted_table(item: str, produced: pd.DataFrame, reference: str) -> Check:
    """Tabelas 3 e 4: mesmo texto, célula a célula (3 casas decimais)."""
    expected = published_table(reference, as_text=True)
    produced = produced.astype(str).reset_index(drop=True)
    same = produced.shape == expected.shape and bool(
        (produced.to_numpy() == expected.to_numpy()).all()
    )
    return Check(item, same, f"{len(produced)} linhas comparadas")


def check_figure4(correlation: float) -> Check:
    """Figura 4: r de Pearson entre o sentimento da LR e o retorno do mesmo dia."""
    expected = published_numbers()["figura_4_pearson"]
    return Check("Figura 4", _close(correlation, expected), f"r = {correlation:.6f}")


def check_figures_3_and_8(curves: pd.DataFrame, summary: pd.DataFrame) -> Check:
    """Figuras 3 e 8: curvas diárias, Sharpe e CAGR da regra de limiar fixo."""
    numbers = published_numbers()
    reference = published_table("figura_8_curvas")
    columns = ["equity_ibov", *(f"equity_{model}" for model in MODEL_LABELS)]
    same_dates = bool(
        (
            pd.to_datetime(curves["date"]).dt.strftime("%Y-%m-%d").to_numpy() == reference["date"]
        ).all()
    )
    curve_gap = float(np.max(np.abs(curves[columns].to_numpy() - reference[columns].to_numpy())))
    by_model = summary.set_index("model")
    stats_ok = all(
        _close(by_model.at[model, "sharpe"], numbers["figura_3_sharpe"][model])
        and _close(by_model.at[model, "cagr"], numbers["figura_8_cagr"][model])
        for model in MODEL_LABELS
    )
    ok = same_dates and curve_gap <= TOLERANCE and stats_ok
    return Check("Figuras 3 e 8", ok, f"maior diferença nas curvas {curve_gap:.2e}")


def check_events(events: pd.DataFrame) -> Check:
    """Estudo de eventos: os 270 eventos e o CAR bruto de cada um."""
    reference = published_table("eventos")
    same_events = len(events) == len(reference) and bool(
        (events["event_day"].dt.strftime("%Y-%m-%d").to_numpy() == reference["event_day"]).all()
        and (events["event_name"].to_numpy() == reference["event_name"]).all()
    )
    gap = float(np.max(np.abs(events["car_value"].to_numpy() - reference["car_value"].to_numpy())))
    return Check(
        "Eventos", same_events and gap <= TOLERANCE, f"{len(events)} eventos; CAR {gap:.2e}"
    )


def check_figure7b(caar: pd.DataFrame) -> Check:
    """Figura 7B: nº de eventos por τ exato; médias e ICs dentro das tolerâncias documentadas."""
    reference = published_table("figura_7b_caar")
    ok = True
    for polarity in ("neg", "pos"):
        ok &= bool((caar[f"n_events_{polarity}"] == reference[f"n_events_{polarity}"]).all())
        mean = f"caar_{polarity}_mean"
        ok &= bool(np.allclose(caar[mean], reference[mean], rtol=0, atol=FIGURE_7B_MEAN_TOLERANCE))
        for bound in ("ci_low", "ci_high"):
            column = f"caar_{polarity}_{bound}"
            ok &= bool(
                np.allclose(caar[column], reference[column], rtol=0, atol=FIGURE_7B_CI_TOLERANCE)
            )
    return Check("Figura 7B", ok, "bootstrap com semente 42 (o do artigo não tinha semente)")
