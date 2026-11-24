"""Escenarios del retraso de notificación y semillas múltiples (análisis de sensibilidad).

Base: el patrón de Piura 2025 (`delay.estimation` de config.yaml). Sensibilidad: `delay.scenarios` (Piura 2023 y
Piura 2026). Cada escenario se simula con varias semillas (`delay.seeds`). La elección del escenario base se fijó
antes de ver resultados; los otros dos escenarios lo encierran (2023 con la cola más larga, 2026 con la más corta).

Límites que se declaran: el retraso se mide solo en Piura y se aplica a las demás unidades como supuesto; los tres
patrones subestiman el retraso real porque solo cuentan casos que ya llegaron al archivo (los años 2023 y 2026 están
incompletos en su corte). No se explica por qué difieren 2023 y 2026 mientras no se compruebe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import snapshots

BASE = "piura2025"


def scenarios(cfg: dict) -> dict[str, dict]:
    """{nombre: {"cut", "onset_year"}}; el escenario base va primero."""
    d = cfg["delay"]
    out = {BASE: dict(d["estimation"])}
    for name, s in (d.get("scenarios") or {}).items():
        if name == BASE:
            raise ValueError(f"el escenario '{BASE}' es el base (delay.estimation); no se repite en delay.scenarios")
        out[name] = dict(s)
    return out


def seeds(cfg: dict) -> list[int]:
    d = cfg["delay"]
    lst = [int(s) for s in d.get("seeds", [d["seed"]])]
    if len(set(lst)) != len(lst):
        raise ValueError("delay.seeds tiene semillas repetidas")
    return lst


def tag(name: str, seed: int, scale: int = 1) -> str:
    """Etiqueta de archivo; con escala 1 es la de siempre (`piura2025_s20261120`), con otra agrega `_x<escala>`."""
    return f"{name}_s{seed}" if scale == 1 else f"{name}_x{scale}_s{seed}"


def share_beyond_max(tri: pd.DataFrame, max_delay_weeks: int) -> float:
    """Proporción de casos del triángulo cuyo retraso supera `max_delay_weeks` (el tensor de datos disponibles los trunca en ese valor)."""
    total = int(tri["n"].sum())
    if total == 0:
        return 0.0
    over = int(tri.loc[(tri["avail_week"] - tri["onset_week"]) > max_delay_weeks, "n"].sum())
    return over / total


def estimate(reg: pd.DataFrame, cfg: dict, name: str = BASE) -> dict:
    sc = scenarios(cfg)
    if name not in sc:
        raise ValueError(f"escenario desconocido: {name!r}; opciones: {list(sc)}")
    d = cfg["delay"]
    return snapshots.delay_pmf(reg, sc[name]["cut"], sc[name]["onset_year"], d["exclude_last_weeks"], d["max_delay_weeks"])


def summary(reg: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Una fila por escenario: casos usados, descartados, media y percentiles del retraso (en semanas)."""
    rows = []
    for name in scenarios(cfg):
        r = estimate(reg, cfg, name)
        pct = {f"p{int(p * 100)}_semanas": int(np.searchsorted(r["cum"], p)) for p in (0.5, 0.9, 0.99)}
        rows.append({"escenario": name, "corte": r["cut"], "anio_inicio": r["onset_year"], "casos_usados": r["n_used"],
                     "descartados_negativos": r["dropped_negative"], "descartados_sobre_maximo": r["dropped_over_max"],
                     "media_semanas": round(r["mean_delay_weeks"], 3), **pct})
    return pd.DataFrame(rows)
