"""Variables y objetivos, siempre calculados con lo conocido en la semana de origen t.

Variables en el origen t (historial de casos; clima e ICEN entran solo en la ablación, ver exogenous.py):
  x_k = casos de la semana t-k conocidos en t (k = 0..L-1)  -> log1p(x_k)
  c4  = suma de x_0..x_3, c8 = suma de x_0..x_7, crecimiento = log1p(c4) - log1p(suma x_4..x_7)
  estacionalidad anual: sin y cos de la semana epidemiológica.
Objetivo para el horizonte h: casos acumulados de 4 semanas que terminan en t+h (y4).
  Árboles: z = log1p(y4) - log1p(c4) (cambio relativo respecto del nivel actual: es independiente de la
  escala, que cambia unas cuatro veces entre años; un random forest no extrapola por encima de lo visto).
  Binomial negativa: y4 con desplazamiento (offset) log1p(c4).
Las etiquetas de entrenamiento se toman con lo conocido en el momento de entrenar (no con el valor final):
esa es la regla de "datos disponibles" aplicada también a las etiquetas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .epiweek import abs_week, week_table, weeks_in_year
from .vintage import Vintage

_TAB_CACHE: dict = {}


def _week_of_year(abs_idx: np.ndarray) -> np.ndarray:
    key = "tab"
    if key not in _TAB_CACHE:
        t = week_table(1998, 2040).set_index("abs_week")
        _TAB_CACHE[key] = (t.index.min(), t["week"].to_numpy(), t["year"].to_numpy())
    lo, wk, yr = _TAB_CACHE[key]
    return wk[np.asarray(abs_idx) - lo], yr[np.asarray(abs_idx) - lo]


def level_features(v: Vintage, u: np.ndarray, t: np.ndarray, lags: int) -> pd.DataFrame:
    """Variables de historial en el origen t para cada par (unidad, origen)."""
    u, t = np.asarray(u), np.asarray(t)
    X = np.stack([v.known(u, t, t - k) for k in range(lags)], axis=1)  # (n, lags) conteos conocidos en t
    c4, c8 = X[:, :4].sum(1), X[:, :8].sum(1)
    out = {f"l{k}": np.log1p(X[:, k]) for k in range(lags)}
    out["c4"], out["c8"] = np.log1p(c4), np.log1p(c8)
    out["growth"] = np.log1p(c4) - np.log1p(c8 - c4)
    wk, _ = _week_of_year(t)
    out["sin"], out["cos"] = np.sin(2 * np.pi * wk / 52.18), np.cos(2 * np.pi * wk / 52.18)
    out["c4_raw"] = c4
    return pd.DataFrame(out)


FEATURE_COLS_TREES = None  # se fija en cols_for


def cols_for(model: str, lags: int, extra: dict | None = None) -> list[str]:
    """Columnas del modelo. `extra` = {"tree": [...], "negbin": [...]} con las variables de clima/ICEN (ablación)."""
    base = [f"l{k}" for k in range(lags)] + ["c4", "c8", "growth", "sin", "cos"]
    if model == "negbin":  # pocas variables poco colineales: l0..l3, resto agregado, estacionalidad
        return ["l0", "l1", "l2", "l3", "growth", "sin", "cos"] + list((extra or {}).get("negbin", []))
    return base + list((extra or {}).get("tree", []))


def target_cases(v: Vintage, u: np.ndarray, t_known: np.ndarray, t_end: np.ndarray) -> np.ndarray:
    """Casos acumulados de las 4 semanas que terminan en t_end, conocidos en la semana t_known."""
    u, t_known, t_end = np.broadcast_arrays(np.asarray(u), np.asarray(t_known), np.asarray(t_end))
    return sum(v.known(u, t_known, t_end - j) for j in range(4))


def endemic_channel(v: Vintage, u: np.ndarray, t: np.ndarray, t_end: np.ndarray, years: int,
                    center: str = "median", shift: float = 0.001) -> np.ndarray:
    """Línea base: mediana de los acumulados de 4 semanas de la misma semana epidemiológica en los años previos.

    Usa solo años anteriores al del objetivo y solo lo conocido en t. [SIN FUENTE: agregar referencia]
    (Hussain-Alkhateeb 2018 es la referencia opcional del presupuesto).

    center="median" (por defecto, no cambia resultados previos) o "geometric": media geométrica con desplazamiento `shift`
    (exp(mean(log(x+shift)))-shift). Umaña et al. 2026 (JMIR Public Health Surveill, e79914) reporta mejor desempeño sumando
    0.001 que sumando 1 al manejar ceros; con la mediana el desplazamiento no tiene efecto. [PROPUESTA: no aprobado]
    """
    if center not in ("median", "geometric"):
        raise ValueError("center debe ser 'median' o 'geometric'")
    u, t, t_end = np.broadcast_arrays(np.asarray(u), np.asarray(t), np.asarray(t_end))
    wk, yr = _week_of_year(t_end)
    preds = np.full(len(u), np.nan)
    cols = []
    for k in range(1, years + 1):
        py = yr - k
        pw = np.minimum(wk, np.array([weeks_in_year(int(y)) for y in py]))
        a = np.array([abs_week(int(y), int(w)) for y, w in zip(py, pw)])
        ok = a - 3 >= v.w0
        vals = np.where(ok, sum(v.known(u, t, a - j) for j in range(4)), np.nan)
        cols.append(vals)
    arr = np.stack(cols, axis=1)
    with np.errstate(all="ignore"):
        if center == "median":
            preds = np.nanmedian(arr, axis=1)
        else:
            preds = np.exp(np.nanmean(np.log(arr + shift), axis=1)) - shift
    return preds
