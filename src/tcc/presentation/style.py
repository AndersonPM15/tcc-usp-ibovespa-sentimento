"""Padrão visual dos slides (20 × 11,25 pol.): tamanhos, fonte, cores e números em português.

- tamanho padrão 11,3 × 5,3 pol. e largura total 17,7 × 5,3 pol., sem título dentro da figura;
- Montserrat (incluída no pacote, licença OFL em `fonts/OFL.txt`), com DejaVu Sans para os
  símbolos que ela não tem (como τ);
- corpo 18 pt, eixos e legenda 16 pt, grade cinza-clara;
- vírgula decimal e ponto de milhar em todos os números (sem depender do locale da máquina).

Cada figura é gravada em PNG (300 dpi) e SVG (texto convertido em curvas, para abrir igual em
qualquer editor de slides), com o tamanho exato do slide, e o CSV com os dados plotados.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from matplotlib.typing import RcKeyType

from tcc.config import MODEL_LABELS
from tcc.figures import DPI, write_csv

FIGURE_SIZE = (11.3, 5.3)
FULL_WIDTH_SIZE = (17.7, 5.3)
BODY_PT = 18
AXIS_PT = 16

MODEL_COLORS = {MODEL_LABELS["logreg_l2"]: "#004AAD", MODEL_LABELS["rf_200"]: "#F08217"}
IBOVESPA_COLOR = "#1F2A37"
TEXT_COLOR = "#1F2A37"
MUTED_COLOR = "#6B7280"
GRID_COLOR = "#E5E7EB"
CI_ALPHA = 0.15
# Polaridade do sentimento extremo (F6): fora da paleta dos modelos e distinguíveis com
# daltonismo (pior caso: ΔE 8,2 em OKLab na deuteranopia)
POLARITY_COLORS = {"pos": "#0F766E", "neg": "#9F1239"}

FONT_FILE = Path(__file__).parent / "fonts" / "Montserrat-Regular.ttf"
FONT = "Montserrat"
FALLBACK_FONT = "DejaVu Sans"
SVG_HASH_SALT = "tcc-semead-2026"  # ids fixos no SVG: o arquivo sai igual a cada execução
MINUS = "−"
MONTHS = ("jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez")


@cache
def font_family() -> tuple[str, ...]:
    """Montserrat, se o arquivo da fonte estiver no pacote, com DejaVu Sans de reserva."""
    if not FONT_FILE.exists():
        return (FALLBACK_FONT,)
    font_manager.fontManager.addfont(str(FONT_FILE))
    return (FONT, FALLBACK_FONT)


def rc_params() -> dict[RcKeyType, Any]:
    """Parâmetros do matplotlib para o padrão dos slides."""
    return {
        "font.family": list(font_family()),
        "font.size": BODY_PT,
        "axes.labelsize": AXIS_PT,
        "xtick.labelsize": AXIS_PT,
        "ytick.labelsize": AXIS_PT,
        "legend.fontsize": AXIS_PT,
        "legend.title_fontsize": AXIS_PT,
        "legend.frameon": False,
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.edgecolor": MUTED_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": True,
        "axes.formatter.use_locale": False,
        "savefig.dpi": DPI,
        "svg.fonttype": "path",
        "svg.hashsalt": SVG_HASH_SALT,
    }


@contextmanager
def slide_style() -> Iterator[None]:
    """Aplica o padrão dos slides enquanto as figuras são criadas e gravadas."""
    with plt.rc_context(rc_params()):
        yield


def new_figure(full_width: bool = False, **subplots: Any) -> tuple[Figure, Any]:
    """Figura no tamanho do slide (margens automáticas, sem alterar o tamanho)."""
    size = FULL_WIDTH_SIZE if full_width else FIGURE_SIZE
    return plt.subplots(figsize=size, layout="constrained", **subplots)


def save(fig: Figure, data: pd.DataFrame, output_dir: Path, stem: str) -> list[Path]:
    """Grava `stem`.png (300 dpi), `stem`.svg e `stem`.csv (dados plotados).

    Sem `bbox_inches="tight"`, para que a imagem tenha exatamente o tamanho definido; o SVG
    sai sem data, para ser idêntico a cada execução.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    png, svg, csv = (output_dir / f"{stem}.{suffix}" for suffix in ("png", "svg", "csv"))
    fig.savefig(png, dpi=DPI)
    fig.savefig(svg, metadata={"Date": None})
    plt.close(fig)
    write_csv(data, csv)
    return [png, svg, csv]


# --------------------------------------------------------------------------- números e datas


def decimal(value: float, places: int = 3) -> str:
    """Número em português: vírgula decimal, ponto de milhar e sinal de menos (−0,127)."""
    text = f"{value:,.{places}f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return text.replace("-", MINUS)


def percent(value: float, places: int = 2) -> str:
    """Fração como porcentagem em português (−0,0316 → −3,16%)."""
    return f"{decimal(value * 100, places)}%"


def interval(
    low: float, high: float, places: int = 3, formatter: Callable[..., str] = decimal
) -> str:
    """Intervalo de confiança no formato [−0,177; −0,084]."""
    return f"[{formatter(low, places)}; {formatter(high, places)}]"


def date_br(day: pd.Timestamp) -> str:
    """Data no formato 05/08/2019."""
    return f"{day:%d/%m/%Y}"


def month_year(day: pd.Timestamp) -> str:
    """Mês abreviado em português e ano (ago/2019)."""
    return f"{MONTHS[day.month - 1]}/{day.year}"


def number_ticks(axis: Axis, places: int, scale: float = 1.0, suffix: str = "") -> None:
    """Rótulos do eixo com vírgula decimal (`scale=100` e `suffix="%"` para porcentagens)."""
    axis.set_major_formatter(
        FuncFormatter(lambda value, _position: decimal(value * scale, places) + suffix)
    )


def year_ticks(axis: Axis) -> None:
    """Um rótulo por ano no eixo de datas."""
    axis.set_major_locator(mdates.YearLocator())
    axis.set_major_formatter(mdates.DateFormatter("%Y"))
