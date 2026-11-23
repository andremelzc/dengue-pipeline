"""Clima local (NASA POWER) e ICEN para la ablación de tres condiciones.

Condiciones (decididas el 19-nov-2026): `historial`; `clima` (historial + clima local); `clima_icen` (+ ICEN).

Reglas contra la fuga de información:
- Clima: la variable de la semana t-k se usa en el origen t solo con k >= 1 (la semana t aún no termina de medirse).
- ICEN: se usa el valor que EXISTÍA en cada fecha (ICEN temporal si el definitivo aún no se publicaba): ver `icen_weekly`.
- Estas variables NO se reentrenan ni se afinan aparte: la ablación usa los hiperparámetros congelados de `historial`.

Estado: la descarga de NASA POWER (`fetch_all`) NO se ha ejecutado contra la red [NO VERIFICADO]; las coordenadas de cada
unidad son [FALTA] (config/coordenadas_plantilla.csv) y no hay serie de ICEN temporal [FALTA] (config/icen_plantilla.csv).
`make_example_*` genera datos SINTÉTICOS solo para probar la interfaz: están marcados y nunca son resultados.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import integrate
from .epiweek import abs_week_of_dates

CONDITIONS = ("historial", "clima", "clima_icen")
CLIMATE_VARS = ["temp_mean", "hum_mean", "rain_sum"]
EXAMPLE_FLAG = "EJEMPLO_NO_REAL"


# ---------- coordenadas ----------
def read_coordinates(path: str | Path, units: list[str]) -> dict[str, tuple[float, float]]:
    """CSV con columnas unit, lat, lon, source. Falla si falta una unidad o un valor: nunca se completa de memoria."""
    c = pd.read_csv(path)
    for col in ("unit", "lat", "lon", "source"):
        if col not in c.columns:
            raise ValueError(f"coordenadas: falta la columna {col!r} [FALTA]")
    c = c.set_index("unit")
    missing = [u for u in units if u not in c.index or c.loc[u, ["lat", "lon"]].isna().any() or pd.isna(c.loc[u, "source"])]
    if missing:
        raise ValueError(f"[FALTA] coordenadas (lat, lon y fuente) de: {missing}")
    if not c.loc[units, "lat"].between(-19, 1).all() or not c.loc[units, "lon"].between(-82, -68).all():
        raise ValueError("coordenadas fuera del rango de Perú: ¿lat y lon invertidas?")
    return {u: (float(c.loc[u, "lat"]), float(c.loc[u, "lon"])) for u in units}


# ---------- NASA POWER ----------
def parse_nasa_power(raw: bytes) -> pd.DataFrame:
    """JSON de la API diaria de POWER -> date, T2M, RH2M, PRECTOTCORR. El valor -999 es dato faltante (NaN).

    [NO VERIFICADO] estructura `properties.parameter.<VAR>.<AAAAMMDD>` tomada de la documentación; no se ejecutó contra la red.
    """
    j = json.loads(raw)
    par = j["properties"]["parameter"]
    df = pd.DataFrame({k: pd.Series(v) for k, v in par.items()})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.replace(-999, np.nan).rename_axis("date").reset_index()
    return df


def fetch_all(cfg: dict, coords: dict, start: str, end: str, raw_dir: str | Path, downloaded_on: str,
              fetcher=integrate.fetch_nasa_power, retries: int = 3, pause: float = 2.0, sleep=time.sleep) -> list[dict]:
    """Descarga el clima diario de cada unidad, guarda la respuesta CRUDA y devuelve entradas de manifiesto (SHA-256).

    `downloaded_on` lo declara quien descarga. Reintentos con pausa creciente. Incluir en el artículo el agradecimiento que pide NASA.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    params = cfg["nasa_power"]["parameters"]
    out = []
    for u, (lat, lon) in coords.items():
        for attempt in range(1, retries + 1):
            try:
                raw = fetcher(lat, lon, start, end, params)
                break
            except RuntimeError:
                if attempt == retries:
                    raise
                sleep(pause * attempt)
        p = raw_dir / f"nasa_power_{u}_{start}_{end}.json"
        p.write_bytes(raw)
        out.append({"unit": u, "file": str(p), "sha256": hashlib.sha256(raw).hexdigest(), "lat": lat, "lon": lon,
                    "start": start, "end": end, "parameters": params, "downloaded_on": downloaded_on})
    return out


def climate_weekly(daily_by_unit: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Clima semanal por unidad (temp_mean, hum_mean, rain_sum, complete) con integrate.daily_to_weekly."""
    parts = []
    for u, d in daily_by_unit.items():
        w = integrate.daily_to_weekly(d.rename(columns={"T2M": "T2M"}), "date", "T2M", "RH2M", "PRECTOTCORR")
        parts.append(w.assign(unit=u))
    return pd.concat(parts, ignore_index=True)[["unit", "abs_week"] + CLIMATE_VARS + ["complete"]]


# ---------- ICEN ----------
def read_icen_csv(path: str | Path) -> pd.DataFrame:
    """CSV con columnas month (AAAA-MM), value, published_on (AAAA-MM-DD), kind (definitivo|temporal)."""
    c = pd.read_csv(path)
    for col in ("month", "value", "published_on", "kind"):
        if col not in c.columns:
            raise ValueError(f"ICEN: falta la columna {col!r} [FALTA]")
    c["published_on"] = pd.to_datetime(c["published_on"])
    c["month"] = pd.PeriodIndex(c["month"], freq="M")
    if c["value"].isna().any():
        raise ValueError("ICEN: valores vacíos")
    if not set(c["kind"]) <= {"definitivo", "temporal"}:
        raise ValueError("ICEN: kind debe ser 'definitivo' o 'temporal'")
    return c


def icen_weekly(weeks: pd.DataFrame, icen: pd.DataFrame) -> pd.DataFrame:
    """ICEN conocido al final de cada semana: valor del mes de la semana, en su versión más reciente publicada hasta esa fecha.

    Si aún no se había publicado nada de ese mes, el del mes anterior más reciente conocido; si no hay nada, NaN.
    `weeks` necesita abs_week y week_end. Nunca usa una versión publicada después del fin de la semana.
    """
    rows = []
    for aw, we in zip(weeks["abs_week"], pd.to_datetime(weeks["week_end"])):
        known = icen[icen["published_on"] <= we]
        m = we.to_period("M")
        cand = known[known["month"] == m]
        if cand.empty and not known.empty:
            prev = known[known["month"] < m]
            cand = prev[prev["month"] == prev["month"].max()] if not prev.empty else prev
        rows.append({"abs_week": aw, "icen": float(cand.sort_values("published_on")["value"].iloc[-1]) if len(cand) else np.nan})
    return pd.DataFrame(rows)


# ---------- variables para el modelo ----------
class Exog:
    """Cubo (unidad, semana, variable) con lo exógeno y la forma de convertirlo en variables en el origen t."""

    def __init__(self, units: list[str], w0: int, w1: int, cols: list[str], lags: dict[str, list[int]], data: dict[str, np.ndarray]):
        self.units, self.w0, self.w1, self.cols, self.lags = list(units), int(w0), int(w1), list(cols), lags
        self.cube = np.full((len(units), w1 - w0 + 1, len(cols)), np.nan)
        for j, c in enumerate(cols):
            self.cube[:, :, j] = data[c]
        for c in cols:
            if c in CLIMATE_VARS and min(lags[c]) < 1:
                raise ValueError(f"{c}: el clima solo entra con rezago >= 1 semana (la semana de origen no está completa)")
            if min(lags[c]) < 0:
                raise ValueError(f"{c}: rezagos negativos = usar el futuro")

    @property
    def tree_cols(self) -> list[str]:
        return [f"{c}_l{k}" for c in self.cols for k in self.lags[c]]

    @property
    def negbin_cols(self) -> list[str]:
        return [f"{c}_l{min(self.lags[c])}" for c in self.cols]

    def features(self, u: np.ndarray, t: np.ndarray) -> pd.DataFrame:
        out = {}
        for j, c in enumerate(self.cols):
            for k in self.lags[c]:
                idx = np.asarray(t) - k - self.w0
                val = np.full(len(idx), np.nan)
                ok = (idx >= 0) & (idx < self.cube.shape[1])
                val[ok] = self.cube[np.asarray(u)[ok], idx[ok], j]
                out[f"{c}_l{k}"] = val
        return pd.DataFrame(out)


def build_exog(condition: str, weekly: pd.DataFrame, units: list[str], cfg: dict, climate: pd.DataFrame | None,
               icen: pd.DataFrame | None) -> Exog | None:
    """None para `historial`. `climate`: salida de climate_weekly; `icen`: salida de icen_weekly (abs_week, icen)."""
    if condition not in CONDITIONS:
        raise ValueError(f"condición desconocida: {condition}")
    if condition == "historial":
        return None
    ex = cfg.get("exogenous", {})
    w0, w1 = int(weekly["abs_week"].min()), int(weekly["abs_week"].max())
    idx = pd.RangeIndex(w0, w1 + 1)
    if climate is None:
        raise ValueError("[FALTA] clima semanal (NASA POWER): sin datos no hay condición 'clima'")
    data, lags = {}, {}
    for c in CLIMATE_VARS:
        data[c] = np.stack([climate[climate["unit"] == u].set_index("abs_week")[c].reindex(idx).to_numpy() for u in units])
        lags[c] = list(ex.get("climate_lags_weeks", [1, 2, 4, 8]))
    cols = list(CLIMATE_VARS)
    if condition == "clima_icen":
        if icen is None:
            raise ValueError("[FALTA] serie del ICEN temporal: sin datos no hay condición 'clima_icen'")
        s = icen.set_index("abs_week")["icen"].reindex(idx).to_numpy()
        data["icen"] = np.tile(s, (len(units), 1))
        lags["icen"] = list(ex.get("icen_lags_weeks", [0, 4, 8]))
        cols.append("icen")
    return Exog(units, w0, w1, cols, lags, data)


# ---------- datos de EJEMPLO (sintéticos; solo prueban la interfaz) ----------
def make_example_climate(weekly: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for u, g in weekly.groupby("unit"):
        wk = g["week"].to_numpy()
        rows.append(pd.DataFrame({
            "unit": u, "abs_week": g["abs_week"].to_numpy(),
            "temp_mean": 24 + 4 * np.sin(2 * np.pi * wk / 52) + rng.normal(0, .5, len(g)),
            "hum_mean": 70 + 8 * np.cos(2 * np.pi * wk / 52) + rng.normal(0, 1, len(g)),
            "rain_sum": np.maximum(0, 5 + 5 * np.sin(2 * np.pi * wk / 52) + rng.normal(0, 2, len(g))),
            "complete": True, "fuente": f"{EXAMPLE_FLAG}: sintético, NO es clima real"}))
    return pd.concat(rows, ignore_index=True)


def make_example_icen(weekly: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = weekly[weekly["unit"] == weekly["unit"].iloc[0]]
    return pd.DataFrame({"abs_week": w["abs_week"].to_numpy(), "icen": np.cumsum(rng.normal(0, .05, len(w))),
                         "fuente": f"{EXAMPLE_FLAG}: sintético, NO es ICEN real"})
