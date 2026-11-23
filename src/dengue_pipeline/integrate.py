"""Etapa 3. Tabla unidad x semana epidemiológica, clima semanal y variables con rezago.

Regla de fondo: nunca usar el futuro. Una variable con rezago L > 0 en la semana t sale de la
semana t - L de la misma unidad.

Importante: en el CSV nacional `semana` es la semana de INICIO DE SÍNTOMAS, no la de
notificación (datos.md 2.1). Esta tabla es, por tanto, la "serie final" por inicio de síntomas.
La población (INEI) y el ICEN aún no se integran: [PENDIENTE DE IMPLEMENTAR].
"""
from __future__ import annotations

import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from .config import assign_unit
from .epiweek import abs_week_of_dates, week_table


def weekly_table(nat: pd.DataFrame, cfg: dict, first_year: int | None = None,
                 last_year: int | None = None, case_types: list[str] | None = None) -> pd.DataFrame:
    """Casos por unidad x semana epidemiológica (inicio de síntomas); semanas sin registros valen 0.

    Columnas: unit, year, week, abs_week, week_start, week_end, cases_C, cases_P, cases.
    `cases` suma los tipos de `case_types` (por defecto cfg['case_definition']).
    Los casos con (año, semana) que no existen se descartan y se informan en `df.attrs`.
    """
    first_year = first_year or cfg["study_window"]["first_year"]
    last_year = last_year or cfg["study_window"]["last_year"]
    case_types = case_types or cfg["case_definition"]

    d = nat.copy()
    d["unit"] = assign_unit(d["departamento"], d["ubigeo"], cfg)
    d = d[d["unit"].notna()]
    d["year"] = pd.to_numeric(d["ano"], errors="coerce")
    d["week"] = pd.to_numeric(d["semana"], errors="coerce")
    d = d.dropna(subset=["year", "week"])
    d["year"], d["week"] = d["year"].astype(int), d["week"].astype(int)

    counts = d.groupby(["unit", "year", "week", "tipo_dx"]).size().unstack("tipo_dx", fill_value=0)
    for t in ("C", "P"):
        if t not in counts:
            counts[t] = 0
    counts = counts[["C", "P"]].rename(columns={"C": "cases_C", "P": "cases_P"}).reset_index()

    grid = week_table(first_year, last_year)
    units = [u["id"] for u in cfg["units"]]
    full = pd.MultiIndex.from_product([units, grid["abs_week"]], names=["unit", "abs_week"]).to_frame(index=False)
    full = full.merge(grid, on="abs_week")
    out = full.merge(counts, on=["unit", "year", "week"], how="left")
    out[["cases_C", "cases_P"]] = out[["cases_C", "cases_P"]].fillna(0).astype(int)
    out["cases"] = sum(out[f"cases_{t}"] for t in case_types)
    out = out.sort_values(["unit", "abs_week"]).reset_index(drop=True)

    in_window = counts[(counts["year"] >= first_year) & (counts["year"] <= last_year)]
    kept = int(out[["cases_C", "cases_P"]].to_numpy().sum())
    out.attrs["dropped_cases_invalid_week"] = int(in_window[["cases_C", "cases_P"]].to_numpy().sum()) - kept
    return out


def daily_to_weekly(daily: pd.DataFrame, date_col: str = "date", temp_col: str = "T2M",
                    hum_col: str = "RH2M", rain_col: str = "PRECTOTCORR") -> pd.DataFrame:
    """Clima semanal: media de temperatura y humedad y suma de lluvia por semana epidemiológica.

    Una semana con menos de 7 días con dato queda con `complete = False` y NaN en las tres
    variables (no se rellena en silencio).
    """
    d = daily.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d["abs_week"] = abs_week_of_dates(d[date_col])
    g = d.groupby("abs_week")
    out = pd.DataFrame({
        "temp_mean": g[temp_col].mean(), "hum_mean": g[hum_col].mean(),
        "rain_sum": g[rain_col].sum(min_count=1), "n_days": g[date_col].nunique(),
    }).reset_index()
    out["complete"] = out["n_days"] == 7
    out.loc[~out["complete"], ["temp_mean", "hum_mean", "rain_sum"]] = np.nan
    return out


def add_lags(df: pd.DataFrame, cols: list[str], lags: list[int], unit_col: str = "unit",
             order_col: str = "abs_week") -> pd.DataFrame:
    """Agrega columnas `<col>_lag<L>` (L > 0 semanas). Exige semanas consecutivas por unidad."""
    if any(L <= 0 for L in lags):
        raise ValueError("los rezagos deben ser positivos: usar el futuro está prohibido")
    d = df.sort_values([unit_col, order_col]).copy()
    gaps = d.groupby(unit_col)[order_col].diff().dropna()
    if (gaps != 1).any():
        raise ValueError("las semanas no son consecutivas dentro de cada unidad; no se pueden calcular rezagos")
    for c in cols:
        for L in lags:
            d[f"{c}_lag{L}"] = d.groupby(unit_col)[c].shift(L)
    return d


def fetch_nasa_power(lat: float, lon: float, start: str, end: str,
                     parameters: list[str], timeout: int = 60) -> bytes:
    """Descarga cruda diaria de NASA POWER para un punto (AAAAMMDD). Devuelve los bytes sin procesar.

    [NO VERIFICADO] Escrita a partir de la forma documentada de la API; no se ha ejecutado contra la red
    (el entorno donde se armó no tiene acceso). Guardar la respuesta cruda con fecha y hash en el manifiesto,
    porque pueden reemplazarse por versiones mejoradas (datos.md 1.2), y citar el agradecimiento que pide NASA.
    """
    url = ("https://power.larc.nasa.gov/api/temporal/daily/point"
           f"?parameters={','.join(parameters)}&community=AG&longitude={lon}&latitude={lat}"
           f"&start={start}&end={end}&format=JSON")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310
            return r.read()
    except urllib.error.URLError as e:  # reintentos y pausas: [PENDIENTE DE IMPLEMENTAR]
        raise RuntimeError(f"no se pudo descargar NASA POWER: {e}") from e
