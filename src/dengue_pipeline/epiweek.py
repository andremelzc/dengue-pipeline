"""Semana epidemiológica según la regla MMWR.

Las semanas van de domingo a sábado y la semana 1 es la primera que tiene al menos
cuatro días del año. Algunos años tienen 53 semanas (entre 1995 y 2035, según este código:
1997, 2003, 2008, 2014, 2020, 2025 y 2031; 2014, 2020 y 2025 coinciden con el máximo de `semana`
del CSV nacional). Es un error que falla en silencio, por eso hay pruebas automáticas
(tests/test_epiweek.py).

[PENDIENTE DE VERIFICAR] que esta numeración coincida con la del CDC Perú en todos los
años; en datos.md se verificó que el corte de 2026 del archivo regional coincide al 100 %.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

REF_SUNDAY = date(1999, 12, 26)  # un domingo; el índice absoluto de semana cuenta desde aquí


def _as_date(d) -> date:
    if hasattr(d, "date") and callable(d.date):  # datetime o Timestamp
        return d.date()
    return d


def week1_start(year: int) -> date:
    """Domingo con que empieza la semana epidemiológica 1 de `year`."""
    jan1 = date(year, 1, 1)
    days_since_sunday = (jan1.weekday() + 1) % 7
    start = jan1 - timedelta(days=days_since_sunday)
    # Si el 1 de enero cae de domingo a miércoles, esa semana tiene >= 4 días del año.
    return start if days_since_sunday <= 3 else start + timedelta(days=7)


def weeks_in_year(year: int) -> int:
    return (week1_start(year + 1) - week1_start(year)).days // 7


def epiweek(d) -> tuple[int, int]:
    """(año epidemiológico, semana) de una fecha."""
    d = _as_date(d)
    for y in (d.year + 1, d.year, d.year - 1):
        start = week1_start(y)
        if d >= start:
            return y, (d - start).days // 7 + 1
    raise ValueError(f"fecha fuera de rango: {d}")


def week_start(year: int, week: int) -> date:
    return week1_start(year) + timedelta(days=7 * (week - 1))


def week_end(year: int, week: int) -> date:
    return week_start(year, week) + timedelta(days=6)


def abs_week(year: int, week: int) -> int:
    """Índice absoluto de semana (consecutivo, sin saltos entre años)."""
    return (week_start(year, week) - REF_SUNDAY).days // 7


def abs_week_start(idx: int) -> date:
    return REF_SUNDAY + timedelta(days=7 * int(idx))


def abs_week_end(idx: int) -> date:
    return abs_week_start(idx) + timedelta(days=6)


def abs_week_of_date(d) -> int:
    d = _as_date(d)
    sunday = d - timedelta(days=(d.weekday() + 1) % 7)
    return (sunday - REF_SUNDAY).days // 7


def abs_week_of_dates(s: pd.Series) -> pd.Series:
    """Versión vectorizada de abs_week_of_date para una serie datetime64 (NaT -> NaN)."""
    days = (s - pd.Timestamp(REF_SUNDAY)).dt.days
    offset = (s.dt.day_of_week + 1) % 7
    return (days - offset) // 7


def epiweek_of_dates(s: pd.Series) -> pd.DataFrame:
    """Año y semana epidemiológicos de una serie datetime64 (vectorizado, vía tabla)."""
    idx = abs_week_of_dates(s)
    valid = idx.dropna()
    if valid.empty:  # todas las fechas inválidas (NaT): no hay semana que asignar
        return pd.DataFrame({"epi_year": np.nan, "epi_week": np.nan}, index=s.index)
    lo, hi = int(valid.min()), int(valid.max())
    years = range(abs_week_start(lo).year - 1, abs_week_start(hi).year + 2)
    tab = week_table(min(years), max(years)).set_index("abs_week")
    return pd.DataFrame({"epi_year": idx.map(tab["year"]), "epi_week": idx.map(tab["week"])})


def abs_week_vec(year: pd.Series, week: pd.Series) -> pd.Series:
    """Índice absoluto de semana a partir de columnas (año, semana) numéricas."""
    base = {int(y): (week1_start(int(y)) - REF_SUNDAY).days // 7 for y in year.dropna().unique()}
    return year.map(base) + week - 1


def week_table(first_year: int, last_year: int) -> pd.DataFrame:
    """Una fila por semana epidemiológica de first_year a last_year."""
    rows = []
    for y in range(first_year, last_year + 1):
        for w in range(1, weeks_in_year(y) + 1):
            rows.append((abs_week(y, w), y, w, pd.Timestamp(week_start(y, w)), pd.Timestamp(week_end(y, w))))
    return pd.DataFrame(rows, columns=["abs_week", "year", "week", "week_start", "week_end"])
