"""Constructores de datos sintéticos para las pruebas (no dependen de los CSV reales)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from dengue_pipeline.io import NATIONAL_COLS, REGIONAL_COLS


def national_rows(rows: list[dict]) -> pd.DataFrame:
    base = dict(departamento="PIURA", provincia="PIURA", distrito="PIURA", localidad="X", enfermedad="DENGUE",
                ano="2023", semana="10", diagnostic="A97.0", tipo_dx="C", diresa="20", ubigeo="200101",
                localcod="", edad="30", tipo_edad="A", sexo="M")
    return pd.DataFrame([{**base, **{k: str(v) for k, v in r.items()}} for r in rows], columns=NATIONAL_COLS)


def regional_rows(rows: list[dict]) -> pd.DataFrame:
    base = dict(FECHA_CORTE="20260702", DEPARTAMENTO="PIURA", PROVINCIA="PIURA", DISTRITO="PIURA", UBIGEO="200101",
                FECHA_INGRESO="20250110", ANO="2025", LOCALIDAD="X", CENTRO_SALUD="E.S.", EDAD="30", TIPO_EDAD="A",
                SEXO="Masculino", FECHA_INICIO_SINTOMAS="20250105", FECHA_NOTIFICACION="20250108",
                SEMANA_NOTIFICACION="2")
    return pd.DataFrame([{**base, **{k: str(v) for k, v in r.items()}} for r in rows], columns=REGIONAL_COLS)


def tiny_cfg() -> dict:
    return {
        "study_window": {"first_year": 2023, "last_year": 2024, "core_first_year": 2023},
        "case_definition": ["C", "P"],
        "units": [
            {"id": "PIURA", "type": "departamento", "departamento": "PIURA"},
            {"id": "TUMBES", "type": "departamento", "departamento": "TUMBES"},
            {"id": "LIMA_NORTE", "type": "ubigeo_group", "departamento": "LIMA", "ubigeos": ["150106", "150110"]},
        ],
        "validation": {"last_year_ratio_threshold": 0.25, "recent_week_ratio_threshold": 0.7},
    }


def fmt(d) -> str:
    return pd.Timestamp(d).strftime("%Y%m%d")
