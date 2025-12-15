#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

from plotly.graph_objs import Figure

# Garantir que o root do projeto está no sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reutiliza carregamento/callbacks do app, com ctx dummy para evitar erro fora de callback
import app_dashboard as dash_app  # type: ignore


class _DummyCtx:
    @property
    def triggered_id(self):
        return "export_script"


dash_app.ctx = _DummyCtx()
DATE_MIN = dash_app.DATE_MIN
DATE_MAX = dash_app.DATE_MAX
MODEL_OPTIONS = dash_app.MODEL_OPTIONS
update_dashboard = dash_app.update_dashboard

OUTPUT_DIR = Path("reports/release_pack/FIGURES")


def try_png(fig: Figure, path: Path) -> bool:
    """Tenta salvar PNG se kaleido estiver disponível; retorna True/False."""
    try:
        import kaleido  # type: ignore  # noqa: F401

        fig.write_image(str(path))
        return True
    except Exception:  # noqa: BLE001
        return False


def save_figure(fig: Figure, stem: str, save_png: bool) -> Dict[str, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = OUTPUT_DIR / f"{stem}.html"
    png_path = OUTPUT_DIR / f"{stem}.png"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    saved = {"html": str(html_path)}
    if save_png and try_png(fig, png_path):
        saved["png"] = str(png_path)
    return saved


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start = str(DATE_MIN.date())
    end = str(DATE_MAX.date())
    selected_models = list(MODEL_OPTIONS) if MODEL_OPTIONS is not None else []
    metric = "auc"

    (
        ibov_fig,
        sentiment_fig,
        comparison_fig,
        table_data,
        indicator_content,
        metric_badge,
        ui_status,
        scatter_fig,
        rolling_fig,
        dist_fig,
        latency_fig,
        backtest_fig,
    ) = update_dashboard(start, end, selected_models, metric)

    figures = {
        "01_ibovespa": ibov_fig,
        "02_sentimento_medio": sentiment_fig,
        "03_comparativo_modelos": comparison_fig,
        "04_dispersao_sentimento_retorno": scatter_fig,
        "05_correlacao_movel": rolling_fig,
        "06_distribuicao_sentimento": dist_fig,
        "07_latencia": latency_fig,
        "08_backtest": backtest_fig,
    }

    save_png = True  # tenta PNG; se falhar, segue com HTML
    saved_paths = []
    for stem, fig in figures.items():
        paths = save_figure(fig, stem, save_png=save_png)
        saved_paths.append(f"{stem}: " + ", ".join(f"{k}={v}" for k, v in paths.items()))

    print("Figuras exportadas:")
    for line in saved_paths:
        print(f" - {line}")


if __name__ == "__main__":
    main()
