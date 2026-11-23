"""Métricas, niveles de alerta y bootstrap por bloques.

Regla de la evaluación: los brotes son pocos y no independientes, así que nunca se promedian como si
lo fueran. Se informan métricas por año y los intervalos salen de un bootstrap por bloques de semanas
consecutivas (conserva la autocorrelación), no de remuestrear semanas sueltas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_abs_error(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    return np.abs(np.log1p(pred) - np.log1p(obs))


def log_bias(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    return np.log1p(pred) - np.log1p(obs)


def block_bootstrap_ci(values: np.ndarray, block: int, n: int, ci: float, seed: int, stat=np.mean) -> tuple[float, float, float]:
    """Estimador, y límites del intervalo por bootstrap de bloques móviles (valores ordenados en el tiempo)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    m = len(v)
    if m == 0:
        return (np.nan, np.nan, np.nan)
    block = max(1, min(block, m))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(m / block))
    starts = rng.integers(0, m - block + 1, size=(n, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :m]
    est = stat(v[idx], axis=1)
    lo, hi = np.quantile(est, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return (float(stat(v)), float(lo), float(hi))


def incidence_per_100k(cases4w: np.ndarray, population: np.ndarray) -> np.ndarray:
    return np.asarray(cases4w, dtype=float) / np.asarray(population, dtype=float) * 1e5


def alert_level(incidence: np.ndarray, yellow: float, red: float) -> np.ndarray:
    """0 = verde (< yellow), 1 = amarillo (yellow a < red), 2 = rojo (>= red). La asignación en tres niveles es [INFERENCIA]."""
    inc = np.asarray(incidence, dtype=float)
    return np.where(inc >= red, 2, np.where(inc >= yellow, 1, 0))


def alert_metrics(pred_level: np.ndarray, obs_level: np.ndarray, threshold_level: int) -> dict:
    """Sensibilidad, falsas alarmas y valor predictivo para 'alerta si nivel >= threshold_level'."""
    p, o = np.asarray(pred_level) >= threshold_level, np.asarray(obs_level) >= threshold_level
    tp, fp, fn, tn = int((p & o).sum()), int((p & ~o).sum()), int((~p & o).sum()), int((~p & ~o).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensibilidad": tp / (tp + fn) if tp + fn else np.nan,
            "falsas_alarmas_sobre_alertas": fp / (tp + fp) if tp + fp else np.nan,
            "tasa_falsa_alarma": fp / (fp + tn) if fp + tn else np.nan}


def episode_lead_weeks(pred_alert: np.ndarray, obs_alert: np.ndarray, max_back: int = 8) -> list[float]:
    """Semanas de anticipación por episodio observado.

    `pred_alert[i]` / `obs_alert[i]` son booleanos por SEMANA OBJETIVO consecutiva. Un episodio es una racha de
    semanas observadas en alerta. Para cada episodio que empieza en s, la anticipación es el largo de la racha
    ininterrumpida de alertas pronosticadas que termina en s (0 = solo se alertó en s; NaN = no se alertó en s,
    episodio no detectado). Se cuenta hasta `max_back` semanas antes.
    """
    p, o = np.asarray(pred_alert, bool), np.asarray(obs_alert, bool)
    leads = []
    for s in np.flatnonzero(o & ~np.r_[False, o[:-1]]):
        if not p[s]:
            leads.append(np.nan)
            continue
        k = 0
        while k < max_back and s - k - 1 >= 0 and p[s - k - 1]:
            k += 1
        leads.append(float(k))
    return leads


def summarize(pred: pd.DataFrame, block: int, n: int, ci: float, seed: int) -> pd.DataFrame:
    """Resumen por (condition, model, horizon, group, year): MAE en casos, error absoluto en escala log y sesgo, con IC.

    `pred` necesita columnas condition, model, horizon, unit, origin_week, year, pred, obs.
    Las semanas se ordenan en el tiempo antes del bootstrap; las unidades se promedian por semana.
    """
    rows = []
    for (cond, model, h), g in pred.groupby(["condition", "model", "horizon"]):
        for label, sub in _groups(g):
            sub = sub.assign(ae=np.abs(sub["pred"] - sub["obs"]),
                             lae=log_abs_error(sub["pred"].to_numpy(), sub["obs"].to_numpy()),
                             bias=log_bias(sub["pred"].to_numpy(), sub["obs"].to_numpy()))
            wk = sub.groupby("origin_week")[["ae", "lae", "bias"]].mean().sort_index()
            row = {"condition": cond, "model": model, "horizon": h, "group": label, "weeks": len(wk)}
            for col, name in (("ae", "mae_casos"), ("lae", "error_abs_log"), ("bias", "sesgo_log")):
                est, lo, hi = block_bootstrap_ci(wk[col].to_numpy(), block, n, ci, seed)
                row[name], row[name + "_lo"], row[name + "_hi"] = est, lo, hi
            rows.append(row)
    return pd.DataFrame(rows)


def _groups(g: pd.DataFrame):
    yield "todo", g
    for y, s in g.groupby("year"):
        yield str(int(y)), s
    yield "2023-2024", g[g["year"].isin([2023, 2024])]


def paired_difference(pred: pd.DataFrame, block: int, n: int, ci: float, seed: int) -> pd.DataFrame:
    """Diferencia de error (datos disponibles - serie final), semana a semana, con IC por bloques.

    Es la prueba de la hipótesis central: evaluar con los datos disponibles cambia el desempeño.
    Positivo = peor con datos disponibles.
    """
    a = pred[pred["condition"] == "disponibles"]
    f = pred[pred["condition"] == "final"]
    keys = ["model", "horizon", "unit", "origin_week"]
    m = a.merge(f, on=keys, suffixes=("_a", "_f"))
    m["d_lae"] = log_abs_error(m["pred_a"].to_numpy(), m["obs_a"].to_numpy()) - log_abs_error(m["pred_f"].to_numpy(), m["obs_f"].to_numpy())
    m["d_ae"] = np.abs(m["pred_a"] - m["obs_a"]) - np.abs(m["pred_f"] - m["obs_f"])
    rows = []
    for (model, h), g in m.groupby(["model", "horizon"]):
        wk = g.groupby("origin_week")[["d_lae", "d_ae"]].mean().sort_index()
        row = {"model": model, "horizon": h, "weeks": len(wk)}
        for col, name in (("d_lae", "dif_error_abs_log"), ("d_ae", "dif_mae_casos")):
            est, lo, hi = block_bootstrap_ci(wk[col].to_numpy(), block, n, ci, seed)
            row[name], row[name + "_lo"], row[name + "_hi"] = est, lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def block_bootstrap_ratio(num: np.ndarray, den: np.ndarray, block: int, n: int, ci: float, seed: int) -> tuple[float, float, float]:
    """Razón de sumas sum(num)/sum(den) con IC por bootstrap de bloques de semanas consecutivas.

    `num` y `den` son conteos por semana (ordenados en el tiempo). Si en un remuestreo den suma 0, ese
    remuestreo queda fuera del intervalo (se cuenta en la razón como NaN).
    """
    a, b = np.asarray(num, float), np.asarray(den, float)
    m = len(a)
    if m == 0 or b.sum() == 0:
        return (np.nan, np.nan, np.nan)
    block = max(1, min(block, m))
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(m / block))
    starts = rng.integers(0, m - block + 1, size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :m]
    d = b[idx].sum(1)
    with np.errstate(all="ignore"):
        est = np.where(d > 0, a[idx].sum(1) / d, np.nan)
    est = est[np.isfinite(est)]
    if len(est) == 0:
        return (float(a.sum() / b.sum()), np.nan, np.nan)
    lo, hi = np.quantile(est, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return (float(a.sum() / b.sum()), float(lo), float(hi))
