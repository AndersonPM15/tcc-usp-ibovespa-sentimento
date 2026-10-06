"""
Verificações pós-submissão (não constam do artigo), gravadas em reports/verificacao/.

a) benchmark buy-and-hold no mesmo período e convenção das estratégias;
b) H1: correlações com IC por bootstrap em blocos, regressão com erros Newey-West, ADF/KPSS;
c) backtest sem look-ahead (limiares nos 60 pregões anteriores);
d) estudo de eventos com retorno anormal (média de t−60 a t−1) e janela em pregões;
e) modelo técnico e modelo texto + técnico, comparados ao modelo só-texto.

Uso:  TCC_USP_BASE=/caminho/TCC_USP python scripts/run_post_submission_checks.py
O Ibovespa até 18/11/2025 vem do histórico do git (blob 1e9e2ec de ibovespa_clean.csv).
"""

from __future__ import annotations

import subprocess
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, load_npz
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import export_tcc_figures as figures  # noqa: E402  (mesma pasta: scripts/)

from src.analysis import h1_tests  # noqa: E402
from src.features.technical import technical_features  # noqa: E402
from src.io import paths  # noqa: E402
from src.models import walk_forward  # noqa: E402

LABEL = "pós-submissão; não consta do artigo"
OUTPUT_DIR = REPO_ROOT / "reports" / "verificacao"
IBOV_GIT_BLOB = "1e9e2ec"
ARTICLE_END = pd.Timestamp("2024-12-31")
MODEL_NAMES = {"logreg_l2": "Regressão Logística", "rf_200": "Random Forest"}
BLOCK_SIZES_EXTRA = (5, 10, 20)
NEWEY_WEST_EXTRA_LAGS = (5, 10)
RETURN_TIMINGS = {"mesmo dia (D−1→D)": "ret_same_day", "dia seguinte (D→D+1)": "ret_next_day"}
DEMEANED = "menos a média do seu bloco do walk-forward"
TEXT_ONLY = "só texto (TF-IDF)"
TECHNICAL_ONLY = "técnico (5 retornos defasados + volatilidade de 5 e 20 dias)"
TEXT_AND_TECHNICAL = "texto + técnico"
SAME_PERIOD_BENCHMARK = "Ibovespa buy-and-hold — mesmas linhas e retorno das estratégias (D→D+1)"


# --------------------------------------------------------------------------- dados


def load_extended_ibov() -> pd.DataFrame:
    """Ibovespa diário até 18/11/2025, lido do histórico do git (blob 1e9e2ec)."""
    csv_text = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "cat-file", "-p", IBOV_GIT_BLOB],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    ibov = pd.read_csv(StringIO(csv_text), parse_dates=["date"]).rename(columns={"date": "day"})
    ibov = ibov.sort_values("day").reset_index(drop=True)
    ibov["ret"] = ibov["close"].pct_change()
    return ibov[["day", "close", "ret"]]


def load_text_model_inputs(data_dir: Path) -> pd.DataFrame:
    """Dias com rótulo (alinhados à matriz TF-IDF), como no notebook 16."""
    index = pd.read_csv(data_dir / "tfidf_daily_index.csv", parse_dates=["day"])
    labels = pd.read_csv(data_dir / "labels_y_daily.csv", parse_dates=["day"])
    base = index.merge(labels[["day", "y", "ret_next"]], how="left", on="day")
    return base.assign(row=np.arange(len(base))).loc[base["y"].notna()].reset_index(drop=True)


def load_text_matrix(data_dir: Path, rows: pd.DataFrame):
    """Linhas da matriz TF-IDF diária correspondentes aos dias com rótulo."""
    return load_npz(data_dir / "tfidf_daily_matrix.npz").tocsr()[rows["row"].to_numpy()]


def text_only_predictions(rows: pd.DataFrame, text_matrix) -> dict[str, np.ndarray]:
    """Refaz as probabilidades fora da amostra do modelo só-texto (Tabela 2, 1.552 dias)."""
    y = rows["y"].astype(int).to_numpy()
    return walk_forward.walk_forward_predictions(text_matrix, y, walk_forward.baseline_models())


def _long_oof(rows: pd.DataFrame, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    folds = walk_forward.fold_ids(len(rows))
    frames = [
        rows[["day", "y", "ret_next"]].assign(model=name, proba=proba, fold=folds)
        for name, proba in predictions.items()
    ]
    return pd.concat(frames, ignore_index=True).dropna(subset=["proba"])


def daily_sentiment(oof: pd.DataFrame, ibov: pd.DataFrame) -> pd.DataFrame:
    """Sentimento 2p − 1 por modelo, com o retorno do mesmo dia (D−1→D), o de D→D+1 e o
    bloco do walk-forward de cada dia."""
    wide = oof.pivot(index="day", columns="model", values="proba").mul(2).sub(1)
    per_day = oof.drop_duplicates("day").set_index("day")
    per_day = per_day[["ret_next", "fold"]].rename(columns={"ret_next": "ret_next_day"})
    same_day = ibov.set_index("day")["ret"].rename("ret_same_day")
    return wide.join(per_day).join(same_day).reset_index()


def with_label(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(rotulo=LABEL)


# --------------------------------------------------------------------------- (a)


def check_benchmark(oof_article: pd.DataFrame, ibov_article: pd.DataFrame) -> pd.DataFrame:
    """(a) Ibovespa buy-and-hold no período da Tabela 3 e no mesmo período das estratégias.

    As estratégias têm 1.341 linhas: os 1.346 pregões de 05/08/2019 a 30/12/2024, menos 5
    dias sem notícias. Por isso há três versões do benchmark no mesmo período:
    - mesmas linhas e retorno das estratégias (D→D+1): a comparação direta com a Tabela 3;
    - todos os pregões: o desempenho real do índice;
    - convenção da Figura 8: retorno D−1→D nas datas das estratégias (pula outros 5 retornos).
    """
    market = ibov_article.set_index("day")["ret"]
    full_ret = market.dropna()
    strategy_days = oof_article[oof_article["model"] == "logreg_l2"].sort_values("day")
    first_day, last_day = strategy_days["day"].min(), strategy_days["day"].max()
    same_rows = strategy_days["ret_next"].fillna(0).reset_index(drop=True)
    all_days = market.loc[first_day:last_day].copy()
    all_days.iloc[0] = 0.0  # a curva começa em 1 no fechamento de 05/08/2019
    figure8 = market.reindex(strategy_days["day"]).reset_index(drop=True)
    figure8.iloc[0] = 0.0

    rows = [
        _metrics_row(
            "Ibovespa buy-and-hold — período da Tabela 3 do artigo",
            full_ret,
            normalize_to_first=True,
        ),
        _metrics_row(SAME_PERIOD_BENCHMARK, same_rows, strategy_days["day"]),
        _metrics_row(
            "Ibovespa buy-and-hold — todos os pregões de 05/08/2019 a 30/12/2024",
            all_days,
        ),
        _metrics_row(
            "Ibovespa buy-and-hold — convenção da Figura 8 (D−1→D nas datas das estratégias)",
            figure8,
            strategy_days["day"],
        ),
    ]
    cfg = next(c for c in figures.STRATEGIES_CFG if c["name"] == "long_only_60")
    for model, name in MODEL_NAMES.items():
        strat = figures._run_strategy_quantile(
            oof_article[oof_article["model"] == model], cfg, event_q=0.90, lag=0
        )
        metrics = figures._compute_metrics(
            strat["strategy_ret"],
            strat["equity"],
            strat["turnover"],
            strat["cost"],
            strat["signal"],
        )
        rows.append(
            {
                "cenario": f"{name} — Tabela 3 do artigo (lag 0, quantil 0,90)",
                "inicio": strat["day"].min().date(),
                "fim": strat["day"].max().date(),
                "n": len(strat),
                **metrics,
            }
        )
    return with_label(pd.DataFrame(rows))


def _metrics_row(
    name: str, ret: pd.Series, days: pd.Series | None = None, normalize_to_first: bool = False
) -> dict[str, object]:
    """Métricas de buy-and-hold (posição sempre comprada, sem custos) com a função do artigo.

    `normalize_to_first` repete a Tabela 3 do artigo, que divide a curva pelo seu 1º valor.
    """
    days = pd.Series(ret.index) if days is None else days
    equity = (1 + ret).cumprod()
    if normalize_to_first:
        equity = equity / equity.iloc[0]
    ones = pd.Series(np.ones(len(ret)))
    zeros = pd.Series([0.0])
    metrics = figures._compute_metrics(
        ret.reset_index(drop=True), equity.reset_index(drop=True), zeros, zeros, ones
    )
    return {
        "cenario": name,
        "inicio": pd.Timestamp(days.min()).date(),
        "fim": pd.Timestamp(days.max()).date(),
        "n": len(ret),
        **metrics,
    }


# --------------------------------------------------------------------------- (b)


def check_h1(sentiment: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """(b) Correlações, IC por bootstrap em blocos, Newey-West e estacionariedade."""
    correlations, bootstrap, newey_west, stationarity = [], [], [], []
    for period, frame in _periods(sentiment).items():
        stationarity.extend(_stationarity_rows(period, frame))
        for model, name in MODEL_NAMES.items():
            for timing, column in RETURN_TIMINGS.items():
                pair = frame[[model, column, "fold"]].dropna()
                x, y = pair[model].to_numpy(), pair[column].to_numpy()
                keys = {"periodo": period, "modelo": name, "retorno": timing}
                correlations.append({**keys, **h1_tests.correlations(x, y)})
                bootstrap.extend(_block_bootstrap_rows(keys, x, y))
                newey_west.extend(_newey_west_rows(keys, x, y, pair["fold"].to_numpy()))
    return {
        "b_h1_correlacoes": with_label(pd.DataFrame(correlations)),
        "b_h1_bootstrap_blocos": with_label(pd.DataFrame(bootstrap)),
        "b_h1_newey_west": with_label(pd.DataFrame(newey_west)),
        "b_h1_estacionariedade": with_label(pd.DataFrame(stationarity)),
    }


def _periods(sentiment: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "artigo (05/08/2019–30/12/2024)": sentiment[sentiment["day"] <= ARTICLE_END],
        "completo (05/08/2019–17/11/2025)": sentiment,
    }


def _stationarity_rows(period: str, frame: pd.DataFrame) -> list[dict]:
    """ADF/KPSS do sentimento (bruto e sem a média de cada bloco) e do retorno diário."""
    folds = frame["fold"].to_numpy()
    rows = []
    for model, name in MODEL_NAMES.items():
        series = frame[model].to_numpy()
        variants = {
            f"sentimento — {name}": series,
            f"sentimento — {name}, {DEMEANED}": h1_tests.demean_by_group(series, folds),
        }
        for label, values in variants.items():
            rows.append({"periodo": period, "serie": label, **h1_tests.stationarity_tests(values)})
    returns = frame["ret_same_day"].dropna().to_numpy()
    rows.append(
        {
            "periodo": period,
            "serie": "retorno diário do Ibovespa",
            **h1_tests.stationarity_tests(returns),
        }
    )
    return rows


def _newey_west_rows(keys: dict, x: np.ndarray, y: np.ndarray, folds: np.ndarray) -> list[dict]:
    """Regressão com HAC: lags pela regra usual e de sensibilidade; e, com os lags usuais,
    o sentimento sem a média do bloco (equivale a um efeito fixo por bloco)."""
    default_lags = h1_tests.newey_west_default_lags(len(x))
    rows = [
        {**keys, "sentimento": "bruto", **h1_tests.newey_west_regression(y, x, lags)}
        for lags in (default_lags, *NEWEY_WEST_EXTRA_LAGS)
    ]
    demeaned = h1_tests.demean_by_group(x, folds)
    rows.append(
        {
            **keys,
            "sentimento": DEMEANED,
            **h1_tests.newey_west_regression(y, demeaned, default_lags),
        }
    )
    return rows


def _block_bootstrap_rows(keys: dict, x: np.ndarray, y: np.ndarray) -> list[dict]:
    main_block = h1_tests.cube_root_block_size(len(x))
    rows = []
    for block in (main_block, *BLOCK_SIZES_EXTRA):
        for stat_name, statistic in (
            ("pearson", h1_tests.pearson),
            ("spearman", h1_tests.spearman),
        ):
            low, high = h1_tests.moving_block_bootstrap_ci(x, y, statistic, block)
            rows.append(
                {
                    **keys,
                    "estatistica": stat_name,
                    "estimativa": statistic(x, y),
                    "bloco": block,
                    "regra_bloco": "n^(1/3) (principal)"
                    if block == main_block
                    else "sensibilidade",
                    "ic95_inferior": low,
                    "ic95_superior": high,
                    "ic_inclui_zero": low <= 0 <= high,
                    "reamostragens": h1_tests.N_BOOTSTRAP,
                    "semente": h1_tests.RANDOM_SEED,
                }
            )
    return rows


# --------------------------------------------------------------------------- (c)


def check_backtest(
    oof_article: pd.DataFrame, ibov_article: pd.DataFrame, benchmark: pd.DataFrame
) -> pd.DataFrame:
    """(c) Grade da Tabela 4 com limiares do período inteiro (artigo) e dos 60 pregões anteriores."""
    grids = []
    for mode, description in (
        ("full_sample", "artigo: quantis do período inteiro"),
        ("rolling", "sem look-ahead: quantis dos 60 pregões anteriores"),
    ):
        grid = figures._run_robust_backtest_grid(
            oof_article, ibov_article, "long_only_60", [0, 1, 2], [0.90, 0.95], threshold_mode=mode
        )
        grids.append(grid.assign(limiar=description, modelo=grid["model"].map(MODEL_NAMES)))
    result = pd.concat(grids, ignore_index=True).drop(columns=["model", "strategy"])
    same_period = benchmark[benchmark["cenario"] == SAME_PERIOD_BENCHMARK]
    same_period = same_period.rename(columns={"cenario": "limiar"})
    result = pd.concat([result, same_period.drop(columns=["inicio", "fim", "n", "rotulo"])])
    return with_label(result.reset_index(drop=True))


# --------------------------------------------------------------------------- (d)


def check_event_study(events: pd.DataFrame, ibov_extended: pd.DataFrame) -> pd.DataFrame:
    """(d) CAAR com retorno anormal e janela em pregões, ao lado da replicação do artigo."""
    returns = ibov_extended.set_index("day")["ret"].dropna()
    scenarios = {
        "artigo: retorno bruto, dias corridos, [0, τ]": {"tau_max": 5},
        "anormal (média t−60..t−1), pregões, [0, τ]": {
            "tau_max": 4,
            "window_unit": "trading_days",
            "abnormal": True,
            "n_boot": h1_tests.N_BOOTSTRAP,
        },
        "anormal (média t−60..t−1), pregões, [1, τ] (sem o dia do evento)": {
            "tau_max": 4,
            "window_unit": "trading_days",
            "abnormal": True,
            "first_tau": 1,
            "n_boot": h1_tests.N_BOOTSTRAP,
        },
    }
    tables = [
        figures.compute_caar_by_event_time(events, returns, **options).assign(cenario=name)
        for name, options in scenarios.items()
    ]
    return with_label(pd.concat(tables, ignore_index=True))


# --------------------------------------------------------------------------- (e)


def check_models(
    rows: pd.DataFrame,
    text_matrix,
    text_predictions: dict[str, np.ndarray],
    ibov_extended: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """(e) Modelos técnico e texto + técnico, com o mesmo walk-forward da Tabela 2, ao lado
    do modelo só-texto (cujas previsões já foram refeitas e conferidas com o artigo)."""
    y = rows["y"].astype(int).to_numpy()
    technical = (
        technical_features(ibov_extended.set_index("day")["close"])
        .reindex(rows["day"])
        .fillna(0.0)
        .to_numpy()
    )
    n_text, n_tech = text_matrix.shape[1], technical.shape[1]
    feature_sets = {
        TECHNICAL_ONLY: (technical, range(n_tech)),
        TEXT_AND_TECHNICAL: (
            hstack([text_matrix, technical]).tocsr(),
            range(n_text, n_text + n_tech),
        ),
    }
    oofs = {TEXT_ONLY: text_predictions}
    for set_name, (features, technical_columns) in feature_sets.items():
        models = _with_technical_scaling(walk_forward.baseline_models(), technical_columns)
        oofs[set_name] = walk_forward.walk_forward_predictions(features, y, models)

    article_window = (rows["day"] <= ARTICLE_END).to_numpy()
    summaries = [
        _summarize_with_article_window(y, predictions, article_window).assign(variaveis=name)
        for name, predictions in oofs.items()
    ]
    result = pd.concat(summaries, ignore_index=True)
    result["modelo"] = result["model"].map(MODEL_NAMES)
    valid = ~np.isnan(text_predictions["logreg_l2"])
    result["mda_sempre_alta"] = float(y[valid].mean())
    return {
        "e_modelos_texto_tecnico": with_label(result.drop(columns=["model"])),
        "e_diferencas_auc": with_label(_auc_differences(y, oofs, valid)),
    }


def _summarize_with_article_window(
    y: np.ndarray, predictions: dict[str, np.ndarray], article_window: np.ndarray
) -> pd.DataFrame:
    """AUC/MDA em 1.552 dias (como a Tabela 2) e só nos 1.341 dias até 30/12/2024."""
    summary = walk_forward.summarize_predictions(y, predictions)
    for model, proba in predictions.items():
        valid = ~np.isnan(proba) & article_window
        is_model = summary["model"] == model
        summary.loc[is_model, "auc_artigo_1341d"] = roc_auc_score(y[valid], proba[valid])
        summary.loc[is_model, "mda_artigo_1341d"] = walk_forward.mean_directional_accuracy(
            y[valid], proba[valid]
        )
    return summary


def _with_technical_scaling(models: dict, technical_columns: range | None) -> dict:
    """Padroniza só as colunas técnicas, dentro de cada bloco de treino (sem vazamento)."""
    if technical_columns is None:
        return models
    scaler = ColumnTransformer(
        [("technical", StandardScaler(with_mean=False), list(technical_columns))],
        remainder="passthrough",
        sparse_threshold=1.0,
    )
    return {name: Pipeline([("scale", scaler), ("model", model)]) for name, model in models.items()}


def auc_gap(target: np.ndarray, pair: np.ndarray) -> float:
    """AUC da coluna 1 menos AUC da coluna 0 (NaN se a amostra tiver uma só classe)."""
    if np.unique(target).size < 2:
        return np.nan
    return roc_auc_score(target, pair[:, 1]) - roc_auc_score(target, pair[:, 0])


def _auc_differences(y: np.ndarray, oofs: dict, valid: np.ndarray) -> pd.DataFrame:
    """Diferença de AUC em relação ao modelo só-texto, com IC por bootstrap em blocos pareado."""
    y_valid = y[valid]
    block = h1_tests.cube_root_block_size(len(y_valid))
    rows = []
    for set_name, predictions in oofs.items():
        if set_name == TEXT_ONLY:
            continue
        for model, name in MODEL_NAMES.items():
            # Coluna 0 = só texto, coluna 1 = candidato; os blocos sorteiam os mesmos dias.
            pairs = np.column_stack([oofs[TEXT_ONLY][model][valid], predictions[model][valid]])
            low, high = h1_tests.moving_block_bootstrap_ci(y_valid, pairs, auc_gap, block)
            rows.append(
                {
                    "modelo": name,
                    "comparacao": f"{set_name} − {TEXT_ONLY}",
                    "diferenca_auc": auc_gap(y_valid, pairs),
                    "ic95_inferior": low,
                    "ic95_superior": high,
                    "ic_inclui_zero": low <= 0 <= high,
                    "bloco": block,
                    "n": int(valid.sum()),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- execução


def main() -> None:
    data_dir = paths.DATA_PROCESSED
    figures.BASE_DATA = data_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ibov_article = figures.load_ibov()
    ibov_extended = load_extended_ibov()
    oof_article = figures.load_oof_predictions()
    rows = load_text_model_inputs(data_dir)
    text_matrix = load_text_matrix(data_dir, rows)
    text_predictions = text_only_predictions(rows, text_matrix)
    oof_full = _long_oof(rows, text_predictions)
    _assert_matches_article(oof_full, oof_article)

    tables = {"a_benchmark_mesmo_periodo": check_benchmark(oof_article, ibov_article)}
    tables.update(check_h1(daily_sentiment(oof_full, ibov_extended)))
    tables["c_backtest_sem_lookahead"] = check_backtest(
        oof_article, ibov_article, tables["a_benchmark_mesmo_periodo"]
    )
    tables["d_eventos_retorno_anormal"] = check_event_study(figures.load_events(), ibov_extended)
    tables.update(check_models(rows, text_matrix, text_predictions, ibov_extended))

    for name, table in tables.items():
        table.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
        print(f"[OK] {OUTPUT_DIR / name}.csv ({len(table)} linhas)")


def _assert_matches_article(oof_full: pd.DataFrame, oof_article: pd.DataFrame) -> None:
    """Garante que o OOF refeito é o mesmo do artigo no período em comum."""
    merged = oof_article.merge(oof_full, on=["model", "day"], suffixes=("_artigo", "_refeito"))
    gap = (merged["proba_artigo"] - merged["proba_refeito"]).abs().max()
    if len(merged) != len(oof_article) or gap > 1e-9:
        raise RuntimeError(f"OOF refeito difere do artigo (linhas={len(merged)}, diferença={gap}).")


if __name__ == "__main__":
    main()
