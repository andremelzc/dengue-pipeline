"""Lectura estricta de los CSV. Todo se lee como texto (ubigeo incluido) y se tipa después."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

NATIONAL_COLS = [
    "departamento", "provincia", "distrito", "localidad", "enfermedad", "ano", "semana",
    "diagnostic", "tipo_dx", "diresa", "ubigeo", "localcod", "edad", "tipo_edad", "sexo",
]
REGIONAL_COLS = [
    "FECHA_CORTE", "DEPARTAMENTO", "PROVINCIA", "DISTRITO", "UBIGEO", "FECHA_INGRESO", "ANO",
    "LOCALIDAD", "CENTRO_SALUD", "EDAD", "TIPO_EDAD", "SEXO", "FECHA_INICIO_SINTOMAS",
    "FECHA_NOTIFICACION", "SEMANA_NOTIFICACION",
]


def read_strict(path: str | Path) -> pd.DataFrame:
    """Lector estricto: todo como texto, sin convertir vacíos a NaN. Falla ante filas mal formadas.

    Un lector tolerante (p. ej. DuckDB por defecto) perdió 85 filas del CSV nacional sin avisar
    (datos.md 3.1); aquí se prefiere fallar.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8")


def physical_lines(path: str | Path) -> int:
    """Líneas físicas del archivo (equivale a `wc -l`, contando una última línea sin salto)."""
    n = 0
    last = b""
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            n += chunk.count(b"\n")
            last = chunk[-1:]
    return n + (1 if last not in (b"\n", b"") else 0)


def parse_yyyymmdd(s: pd.Series) -> pd.Series:
    """Fechas AAAAMMDD a datetime64; las inválidas quedan NaT (se cuentan en la validación)."""
    return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
