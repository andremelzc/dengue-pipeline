"""Etapas 5 y 6. Backtesting de ventana expansiva, en dos condiciones (serie final y datos disponibles).

Protocolo (propuesta; ver reports/protocolo_evaluacion.md):
  - En la semana de origen t solo se usa lo conocido en t. Nada aleatorio: todo en orden temporal.
  - El modelo se reentrena cada `refit_every_weeks` semanas con los orígenes t' tales que su objetivo terminó
    hace al menos `train_label_guard_weeks` (t' + h <= t_fit - guarda). Las ETIQUETAS de entrenamiento se toman
    con lo conocido en t_fit, no con el valor final.
  - La verdad de la evaluación es la serie final (casos acumulados de 4 semanas que terminan en t + h).
  - Hiperparámetros: se eligen una sola vez con datos hasta `tuning_until_year`, con pliegues ordenados y una
    separación (purga) entre entrenamiento y validación, y se congelan.
  - Un solo modelo por tipo, entrenado con `train_units`; `transfer_units` (Lima Norte) solo se evalúan.
La función `forecast_at` es pura: depende únicamente del tensor de conocimiento hasta t_fit. La prueba
tests/test_backtest.py altera todo lo posterior a t_fit y exige que las predicciones no cambien.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .epiweek import abs_week, weeks_in_year
from .features import _week_of_year, cols_for, endemic_channel, level_features, target_cases
from .models import DEFAULT_PARAMS, MODEL_NAMES, REFERENCE, NegBin, TreeModel, tune
from .vintage import Vintage, from_final, from_triangle


def build_feats(v: Vintage, origins: np.ndarray, lags: int, exog=None) -> pd.DataFrame:
    uu, tt = np.meshgrid(np.arange(v.U), origins, indexing="ij")
    u, t = uu.ravel(), tt.ravel()
    F = level_features(v, u, t, lags)
    F["u"], F["t"] = u, t
    if exog is not None:                                  # ablación: clima / ICEN con rezagos (exogenous.py)
        F = pd.concat([F, exog.features(u, t)], axis=1)
    return F


def final_y4(v: Vintage, u: np.ndarray, t_end: np.ndarray) -> np.ndarray:
    big = np.full_like(np.asarray(t_end), v.w_last + v.D + 1)
    return target_cases(v, u, big, t_end)


def forecast_at(v: Vintage, F: pd.DataFrame, bt: dict, params: dict, train_idx: list[int], t_fit: int,
                test_origins: np.ndarray, h: int, models: list[str], first_origin: int,
                fitinfo: dict | None = None, extra: dict | None = None) -> pd.DataFrame:
    """Entrena con lo conocido en t_fit y pronostica, para los orígenes `test_origins`, el acumulado de 4 semanas en t+h."""
    guard, lags, seed = bt["train_label_guard_weeks"], bt["lags_weeks"], bt["seed"]
    tr = F[F["u"].isin(train_idx) & (F["t"] >= first_origin) & (F["t"] + h <= t_fit - guard)]
    te = F[F["t"].isin(test_origins)]
    out = {"u": te["u"].to_numpy(), "t": te["t"].to_numpy()}
    res = {}
    res[REFERENCE] = te["c4_raw"].to_numpy()
    if "endemic" in models:
        e = endemic_channel(v, te["u"].to_numpy(), te["t"].to_numpy(), te["t"].to_numpy() + h, bt["endemic_channel_years"],
                           bt.get("endemic_channel_center", "median"), bt.get("endemic_channel_shift", 0.001))
        res["endemic"] = np.where(np.isfinite(e), e, te["c4_raw"].to_numpy())
    if len(tr) >= 30:
        assert int(tr["t"].max()) + h <= t_fit - guard, "fuga: un objetivo de entrenamiento termina después de t_fit - guarda"
        y4 = target_cases(v, tr["u"].to_numpy(), np.full(len(tr), t_fit), tr["t"].to_numpy() + h)
        z = np.log1p(y4) - tr["c4"].to_numpy()
        for name in [m for m in models if m in ("rf", "gb")]:
            cols = cols_for(name, lags, extra)
            m = TreeModel(name, params[name], seed).fit(tr[cols], z)
            res[name] = m.predict(te[cols], te["c4"].to_numpy())
        if "negbin" in models:
            cols = cols_for("negbin", lags, extra)
            ok = tr[cols].notna().all(axis=1).to_numpy()          # la binomial negativa no admite NaN: se omiten esas filas al entrenar
            if te[cols].isna().any().any():
                raise ValueError("faltan variables exógenas en un origen de prueba (binomial negativa); completar los datos, no se imputa")
            nb = NegBin().fit(tr.loc[ok, cols], y4[ok], tr["c4"].to_numpy()[ok])
            res["negbin"] = nb.predict(te[cols], te["c4"].to_numpy())
            if fitinfo is not None and nb.fallback:
                fitinfo.setdefault("negbin_fallbacks", []).append((t_fit, nb.fallback))
        if fitinfo is not None:
            fitinfo.setdefault("train_sizes", []).append(int(len(tr)))
    rows = [pd.DataFrame({**out, "model": name, "pred": p}) for name, p in res.items()]
    return pd.concat(rows, ignore_index=True)


def train_origin(bt: dict, first_origin: int) -> int:
    """Primer origen que entra al entrenamiento y a la elección de hiperparámetros.

    `train_first_year` limita de verdad el entrenamiento: con el valor por defecto (2010) coincide con `first_origin`
    (inicio de la serie más los rezagos) y no cambia nada; con 2014 se descartan los orígenes anteriores.
    """
    return max(int(first_origin), abs_week(bt["train_first_year"], 1))


def cv_edges(bt: dict) -> list[int]:
    """Bordes de los 3 pliegues ordenados de la elección de hiperparámetros, anclados a `tuning_until_year`.

    Con tuning_until_year = 2019: 2014, 2016, 2018 y 2020 (los mismos de siempre). No dependen de `train_first_year`: si el
    entrenamiento empieza después del primer borde, ese pliegue se queda sin entrenamiento y se omite (con 2014 quedan 2 pliegues).
    """
    end = bt["tuning_until_year"] + 1
    return [abs_week(end - k, 1) for k in (6, 4, 2, 0)]


def tune_hyperparameters(v: Vintage, F: pd.DataFrame, bt: dict, train_idx: list[int], h: int, first_origin: int) -> dict:
    """Elige hiperparámetros con datos hasta tuning_until_year (condición serie final), pliegues ordenados y con purga.

    `first_origin` es el primer origen de entrenamiento (ya considera `train_first_year`, ver `train_origin`).
    """
    lags = bt["lags_weeks"]
    tune_end = abs_week(bt["tuning_until_year"], weeks_in_year(bt["tuning_until_year"]))
    edges = cv_edges(bt)
    Ft = F[F["u"].isin(train_idx) & (F["t"] >= first_origin) & (F["t"] + h <= tune_end)].reset_index(drop=True)
    y4 = final_y4(v, Ft["u"].to_numpy(), Ft["t"].to_numpy() + h)
    z = np.log1p(y4) - Ft["c4"].to_numpy()
    cols = cols_for("rf", lags)
    folds = []
    for a, b in zip(edges[:-1], edges[1:]):
        tr = np.flatnonzero((Ft["t"] + h + 3 < a).to_numpy())          # purga: ningún objetivo de entrenamiento llega a la validación
        va = np.flatnonzero(((Ft["t"] >= a) & (Ft["t"] + h < b)).to_numpy())
        if len(tr) > 100 and len(va) > 20:
            folds.append((tr, va))
    best = tune(Ft[cols], z, folds, bt["seed"])
    best["n_folds"], best["rows"], best["horizon"] = len(folds), int(len(Ft)), h
    return best


def load_frozen_params(path: str | Path) -> dict:
    """Hiperparámetros congelados de hyperparameters.json -> {horizonte: {"rf": {...}, "gb": {...}}}."""
    j = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(k[1:]): {"rf": v["rf"], "gb": v["gb"]} for k, v in j.items()}


def run_backtest(weekly: pd.DataFrame, tri: pd.DataFrame, cfg: dict, models: list[str] | None = None,
                 conditions: tuple[str, ...] = ("final", "disponibles"), out_dir: str | Path | None = None,
                 log=print, features: str = "historial", exog=None, params_by_h: dict | None = None,
                 out_name: str = "predictions.csv.gz") -> tuple[pd.DataFrame, dict]:
    bt, D = cfg["backtest"], cfg["delay"]["max_delay_weeks"]
    units = [u["id"] for u in cfg["units"]]
    models = models or MODEL_NAMES
    L = bt["lags_weeks"]
    vints = {"final": from_final(weekly, units, D), "disponibles": from_triangle(tri, weekly, units, D)}
    v_ref = vints["final"]
    w_last = v_ref.w_last
    first_origin = v_ref.w0 + L - 1
    t_origin = train_origin(bt, first_origin)          # primer origen de entrenamiento (train_first_year)
    log(f"[backtest] entrenamiento desde la semana {t_origin} (train_first_year={bt['train_first_year']})")
    test_start = abs_week(bt["test_first_year"], 1)
    train_idx = [units.index(u) for u in bt["train_units"]]
    origins_all = np.arange(first_origin, w_last + 1)
    info = {"hyperparameters": {}, "fits": 0}
    wk_of = _week_of_year

    # Hiperparámetros: una vez, antes del periodo de prueba, y congelados. En la ablación (clima, ICEN) se reutilizan los
    # de `historial` (params_by_h) para que la diferencia entre condiciones sea solo de variables.
    extra = {"tree": exog.tree_cols, "negbin": exog.negbin_cols} if exog is not None else None
    F_final = build_feats(vints["final"], origins_all, L, exog)
    frozen = params_by_h is not None
    params_by_h = params_by_h or {}
    for h in ([] if frozen else bt["horizons_weeks"]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best = tune_hyperparameters(vints["final"], build_feats(vints["final"], origins_all, L), bt, train_idx, h, t_origin)
        params_by_h[h] = {k: {kk: vv for kk, vv in best[k].items()} for k in ("rf", "gb")}
        info["hyperparameters"][f"h{h}"] = {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in best.items()}
        log(f"[tune] h={h}: rf={best['rf']} gb={best['gb']}")

    parts = []
    for cond in conditions:
        v = vints[cond]
        F = F_final if cond == "final" else build_feats(v, origins_all, L, exog)
        for h in bt["horizons_weeks"]:
            last_origin = w_last - h
            t_fits = list(range(test_start, last_origin + 1, bt["refit_every_weeks"]))
            for t_fit in t_fits:
                block = np.arange(t_fit, min(t_fit + bt["refit_every_weeks"] - 1, last_origin) + 1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p = forecast_at(v, F, bt, params_by_h[h], train_idx, t_fit, block, h, models, t_origin, info, extra)
                p["condition"], p["horizon"], p["fit_week"] = cond, h, t_fit
                parts.append(p)
                info["fits"] += 1
            log(f"[backtest] {cond} h={h}: {len(t_fits)} reentrenamientos")
    pred = pd.concat(parts, ignore_index=True)
    pred["unit"] = pred["u"].map(dict(enumerate(units)))
    pred["target_week"] = pred["t"] + pred["horizon"]
    pred["obs"] = final_y4(v_ref, pred["u"].to_numpy(), pred["target_week"].to_numpy())
    pred = pred.rename(columns={"t": "origin_week"})
    pred["year"] = wk_of(pred["target_week"].to_numpy())[1]
    pred["features"] = features
    pred["unit_group"] = np.where(pred["unit"].isin(bt["transfer_units"]), "transferencia", "entrenadas")
    pred = pred.drop(columns=["u"])
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pred.to_csv(out_dir / out_name, index=False)
        if not frozen:
            (out_dir / "hyperparameters.json").write_text(json.dumps(info["hyperparameters"], indent=2, default=str), encoding="utf-8")
    return pred, info
