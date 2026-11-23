"""Lectura de la configuración y asignación de casos a unidades de estudio."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def assign_unit(departamento: pd.Series, ubigeo: pd.Series, cfg: dict) -> pd.Series:
    """Unidad de estudio de cada caso; <NA> si no pertenece a ninguna.

    Los departamentos se asignan por nombre del departamento; Lima Norte por la lista de
    `ubigeo` (se une por ubigeo, no por nombres: hay errores de texto en los nombres).
    """
    unit = pd.Series([None] * len(departamento), index=departamento.index, dtype="object")
    for u in cfg["units"]:
        if u["type"] == "departamento":
            mask = departamento == u["departamento"]
        elif u["type"] == "ubigeo_group":
            mask = ubigeo.isin(u["ubigeos"])
        else:
            raise ValueError(f"tipo de unidad desconocido: {u['type']}")
        unit[mask] = u["id"]
    return unit
