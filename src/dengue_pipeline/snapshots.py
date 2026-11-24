"""Etapa 4. Instantáneas de datos disponibles y retraso de notificación.

Definición (datos.md 3.6-3.7, pipeline.md 3): un caso está DISPONIBLE en la semana t si su fecha de
notificación es anterior o igual al fin de la semana t. La instantánea de la semana t es el conteo
de casos disponibles en t, agrupados por semana de inicio de síntomas.

Representación: un "triángulo" con columnas
    unit, onset_week, avail_week, n
donde avail_week es la semana (índice absoluto) desde la que el caso está disponible. Para casos reales
es max(semana de inicio, semana de notificación): un caso notificado antes de su propio inicio de
síntomas (defecto observado) no puede contarse antes de que existan los síntomas.
La instantánea en t es la suma de n con avail_week <= t. Por construcción no contiene nada posterior a t.

Límites que se declaran:
- Las instantáneas reales se reconstruyen filtrando el archivo de un corte por fecha de notificación. Eso
  supone que no hubo reclasificaciones ni bajas posteriores que el archivo no registre.
- El retraso medido solo incluye casos que ya llegaron al archivo, así que aparece subestimado (datos.md 3.6).
- Los parámetros marcados [PROVISIONAL] en config.yaml no están decididos.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from .config import assign_unit
from .epiweek import abs_week_of_date, abs_week_of_dates, abs_week_start
from .io import parse_yyyymmdd

TRI_COLS = ["unit", "onset_week", "avail_week", "n"]


def regional_triangle(reg: pd.DataFrame, cut: str, unit: str = "PIURA", cfg: dict | None = None) -> pd.DataFrame:
    """Triángulo real de un corte del archivo regional (solo filas del departamento de la unidad)."""
    g = reg[reg["FECHA_CORTE"] == cut].copy()
    if cfg is not None:
        g = g[assign_unit(g["DEPARTAMENTO"], g["UBIGEO"], cfg) == unit]
    else:
        g = g[g["DEPARTAMENTO"] == unit]
    onset = abs_week_of_dates(parse_yyyymmdd(g["FECHA_INICIO_SINTOMAS"]))
    notif = abs_week_of_dates(parse_yyyymmdd(g["FECHA_NOTIFICACION"]))
    tri = pd.DataFrame({"unit": unit, "onset_week": onset, "avail_week": np.maximum(onset, notif)})
    tri = tri.dropna().astype({"onset_week": int, "avail_week": int})
    return tri.groupby(TRI_COLS[:3]).size().rename("n").reset_index()


def snapshot(tri: pd.DataFrame, t: int, max_history_weeks: int | None = None) -> pd.DataFrame:
    """Instantánea de la semana t (índice absoluto): casos por unidad y semana de inicio, solo lo disponible en t."""
    s = tri[tri["avail_week"] <= t]
    if max_history_weeks is not None:
        s = s[s["onset_week"] > t - max_history_weeks]
    return s.groupby(["unit", "onset_week"], as_index=False)["n"].sum()


def assert_no_future(tri: pd.DataFrame, snap: pd.DataFrame, t: int) -> None:
    """Falla si la instantánea contiene información posterior a la semana t."""
    if len(snap) and int(snap["onset_week"].max()) > t:
        raise AssertionError(f"la instantánea de la semana {t} contiene semanas de inicio posteriores")
    expected = int(tri.loc[tri["avail_week"] <= t, "n"].sum())
    if int(snap["n"].sum()) != expected:
        raise AssertionError("la instantánea no coincide con los casos disponibles en esa semana")


def delay_pmf(reg: pd.DataFrame, cut: str, onset_year: int, exclude_last_weeks: int,
              max_delay_weeks: int, unit: str = "PIURA") -> dict:
    """Distribución del retraso (en semanas epidemiológicas) entre inicio de síntomas y notificación.

    Se usan casos de `unit` del corte `cut`, con inicio en `onset_year`, excluyendo las últimas
    `exclude_last_weeks` semanas del corte (incompletas). Se descartan los retrasos negativos y los
    mayores que `max_delay_weeks`; ambos conteos se informan. El resultado es una pmf normalizada
    sobre 0..max_delay_weeks.
    """
    g = reg[(reg["FECHA_CORTE"] == cut) & (reg["DEPARTAMENTO"] == unit)]
    on_d, nt_d = parse_yyyymmdd(g["FECHA_INICIO_SINTOMAS"]), parse_yyyymmdd(g["FECHA_NOTIFICACION"])
    cut_week = abs_week_of_date(pd.to_datetime(cut, format="%Y%m%d"))
    on_w, nt_w = abs_week_of_dates(on_d), abs_week_of_dates(nt_d)
    sel = (on_d.dt.year == onset_year) & (on_w <= cut_week - exclude_last_weeks)
    delay = (nt_w - on_w)[sel].dropna().astype(int)
    negative, over = int((delay < 0).sum()), int((delay > max_delay_weeks).sum())
    kept = delay[(delay >= 0) & (delay <= max_delay_weeks)]
    counts = np.bincount(kept, minlength=max_delay_weeks + 1).astype(float)
    pmf = counts / counts.sum() if counts.sum() else counts
    return {"pmf": pmf, "n_used": int(len(kept)), "dropped_negative": negative, "dropped_over_max": over,
            "cut": cut, "onset_year": onset_year, "exclude_last_weeks": exclude_last_weeks,
            "max_delay_weeks": max_delay_weeks,
            "cum": np.cumsum(pmf), "mean_delay_weeks": float((pmf * np.arange(len(pmf))).sum())}


def simulate_triangle(final_weekly: pd.DataFrame, pmf: np.ndarray, seed: int, delay_scale: int = 1) -> pd.DataFrame:
    """Triángulo simulado: cada caso de la serie final recibe un retraso en semanas sorteado de `pmf`.

    `final_weekly` necesita columnas unit, abs_week y cases (casos por semana de inicio, serie final).
    Se sortea de forma multinomial por celda (unidad, semana de inicio): el total por celda se conserva. Cada unidad usa su propia semilla derivada de (semilla, nombre de la unidad).
    Con la misma semilla el resultado es idéntico. El retraso simulado es un SUPUESTO que se declara
    (observado solo en Piura; se usa la misma distribución en las demás unidades).

    `delay_scale` (entero >= 1, por defecto 1 = sin cambio) multiplica el retraso sorteado, en semanas, DESPUÉS de sortearlo: la
    secuencia de números aleatorios es la misma que con 1. Es una prueba de estrés de robustez, no una estimación del retraso de otra región.
    """
    if int(delay_scale) != delay_scale or delay_scale < 1:
        raise ValueError("delay_scale debe ser un entero >= 1")
    pmf = np.asarray(pmf, dtype=float)
    pmf = pmf / pmf.sum()
    cells_all = final_weekly.loc[final_weekly["cases"] > 0, ["unit", "abs_week", "cases"]].reset_index(drop=True)
    parts = []
    for unit, cells in cells_all.groupby("unit", sort=True):
        # semilla propia de cada unidad: lo que le pase a otra unidad (p. ej. cambiar la lista de distritos de Lima Norte) no altera sus sorteos
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), zlib.crc32(str(unit).encode("utf-8"))]))
        draws = rng.multinomial(cells["cases"].to_numpy(), pmf)  # (celdas, retrasos)
        cell_idx, delay = np.nonzero(draws)
        aw = cells["abs_week"].to_numpy()
        parts.append(pd.DataFrame({
            "unit": unit,
            "onset_week": aw[cell_idx],
            "avail_week": aw[cell_idx] + delay * int(delay_scale),
            "n": draws[cell_idx, delay],
        }))
    tri = pd.concat(parts, ignore_index=True)
    return tri.groupby(TRI_COLS[:3], as_index=False)["n"].sum()


def cut_vs_final(reg: pd.DataFrame, cut: str, final_weekly: pd.DataFrame, unit: str, weeks: list[int]) -> pd.DataFrame:
    """Par real de versiones (datos.md 3.7): conteo del corte temprano frente a la serie final.

    `weeks` son índices absolutos de semana de inicio de síntomas. Devuelve corte, final y razón.
    """
    tri = regional_triangle(reg, cut, unit)
    early = tri.groupby("onset_week")["n"].sum()
    final = final_weekly[final_weekly["unit"] == unit].set_index("abs_week")["cases"]
    out = pd.DataFrame({"abs_week": weeks})
    out["week_start"] = out["abs_week"].map(abs_week_start)
    out["corte"] = out["abs_week"].map(early).fillna(0).astype(int)
    out["final"] = out["abs_week"].map(final).fillna(0).astype(int)
    out["razon"] = (out["corte"] / out["final"]).round(2)
    return out
