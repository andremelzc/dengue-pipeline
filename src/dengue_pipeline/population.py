"""Población (INEI), incidencia acumulada de 4 semanas y niveles de alerta.

Fuente leída: cuadro "Población estimada al 30 de junio, por años calendario y sexo, según departamento,
2000-2026" del INEI (proy_03_4.xlsx; la nota al pie del propio archivo cita el Boletín especial N.° 25,
estimaciones y proyecciones departamentales 1995-2030). El archivo lo descarga y declara quien lo adjunta.

Reglas (todas PROPUESTA hasta que el equipo las apruebe):
- A cada semana epidemiológica se le asigna la población de su año epidemiológico (30 de junio de ese año).
- Incidencia semanal = casos acumulados de las últimas 4 semanas (la semana t y las 3 anteriores) por 100 000.
- Lima Norte: suma de sus distritos (ubigeos de config.yaml) en el cuadro distrital del INEI "Población total
  proyectada al 30 de junio de cada año, según departamento, provincia y distrito, 2018-2026" (gob.pe, 31-dic-2025;
  misma base, censo 2017, que el cuadro departamental). Solo hay 2018-2026: antes de 2018 Lima Norte queda sin tasa (NaN),
  y nunca se sustituye por la población de todo el departamento de Lima.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import alert_level, incidence_per_100k


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.upper().split())


def read_inei_departmental(path: str | Path) -> pd.DataFrame:
    """Lee todas las hojas del cuadro del INEI. Devuelve departamento (normalizado), ubigeo, year, population.

    Comprueba que Hombre + Mujer = Total en cada celda leída; si no, falla (no corrige en silencio).
    """
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True)
    rows = []
    for ws in wb:
        data = [r for r in ws.iter_rows(values_only=True)]
        hdr = next((i for i, r in enumerate(data) if r and r[0] == "Ubigeo"), None)
        if hdr is None:
            continue
        year_cols = {j: int(v) for j, v in enumerate(data[hdr]) if isinstance(v, (int, float)) and 1900 < v < 2100}
        for r in data[hdr + 2:]:
            ubigeo = r[0]
            if not (isinstance(ubigeo, str) and ubigeo.isdigit() and len(ubigeo) == 6):
                continue
            for j, year in year_cols.items():
                tot, hom, muj = r[j], r[j + 1], r[j + 2]
                if not all(isinstance(x, (int, float)) for x in (tot, hom, muj)):
                    raise ValueError(f"hoja {ws.title}: celda no numérica en {r[1]!r}, {year}")
                if hom + muj != tot:
                    raise ValueError(f"hoja {ws.title}: Hombre + Mujer != Total en {r[1]!r}, {year}")
                rows.append({"ubigeo": ubigeo, "departamento": _norm(r[1]), "year": year, "population": int(tot)})
    if not rows:
        raise ValueError("no se encontró ninguna tabla con encabezado 'Ubigeo' en el archivo")
    out = pd.DataFrame(rows)
    if out.duplicated(["ubigeo", "year"]).any():
        raise ValueError("años repetidos para un mismo ubigeo en el archivo")
    return out


def population_by_unit(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Columnas unit, year, population para las unidades de tipo departamento dentro de la ventana de estudio.

    Las unidades de tipo ubigeo_group (Lima Norte) no se incluyen: no hay cuadro distrital [FALTA].
    Falla si una unidad no se encuentra o si falta algún año de la ventana.
    """
    w = cfg["study_window"]
    years = range(w["first_year"], w["last_year"] + 1)
    parts = []
    for u in cfg["units"]:
        if u["type"] != "departamento":
            continue
        sub = raw[raw["departamento"] == _norm(u["departamento"])]
        if sub.empty:
            raise ValueError(f"departamento {u['departamento']!r} (unidad {u['id']}) no está en el cuadro")
        sub = sub[sub["year"].isin(years)]
        missing = sorted(set(years) - set(sub["year"]))
        if missing:
            raise ValueError(f"{u['id']}: faltan años {missing}")
        parts.append(sub.assign(unit=u["id"])[["unit", "year", "population"]])
    return pd.concat(parts).sort_values(["unit", "year"]).reset_index(drop=True)


def read_inei_districts(path: str | Path) -> pd.DataFrame:
    """Cuadro distrital 2018-2026 del INEI -> columnas ubigeo, year, population (solo filas con ubigeo de 6 dígitos)."""
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    rows = list(wb.worksheets[0].iter_rows(values_only=True))
    hdr = next((i for i, r in enumerate(rows) if r and r[0] is None and sum(isinstance(c, int) and 2000 < c < 2100 for c in r) >= 3), None)
    if hdr is None:
        raise ValueError("cuadro distrital: no se encontró la fila de años")
    years = {j: int(c) for j, c in enumerate(rows[hdr]) if isinstance(c, int) and 2000 < c < 2100}
    out = []
    for r in rows[hdr + 1:]:
        if r and r[0] is not None and str(r[0]).strip().isdigit() and len(str(r[0]).strip()) == 6:
            for j, y in years.items():
                if isinstance(r[j], (int, float)):          # hay notas de texto (p. ej. "Distrito creado en 2021"): se ignoran
                    out.append((str(r[0]).strip(), y, float(r[j])))
    d = pd.DataFrame(out, columns=["ubigeo", "year", "population"])
    if d.empty:
        raise ValueError("cuadro distrital vacío")
    return d


def population_ubigeo_groups(dist: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Población de las unidades ubigeo_group = suma de sus distritos, por año disponible dentro de la ventana de estudio.

    Falla si falta algún ubigeo de la unidad (en cualquier año) o si un distrito no tiene los mismos años que los demás.
    """
    w = cfg["study_window"]
    parts = []
    for u in cfg["units"]:
        if u["type"] != "ubigeo_group":
            continue
        want = list(u["ubigeos"])
        sub = dist[dist["ubigeo"].isin(want)]
        absent = sorted(set(want) - set(sub["ubigeo"]))
        if absent:
            raise ValueError(f"{u['id']}: ubigeos {absent} no están en el cuadro distrital")
        sub = sub[(sub["year"] >= w["first_year"]) & (sub["year"] <= w["last_year"])]
        n = sub.groupby("year")["ubigeo"].nunique()
        bad = n[n != len(want)]
        if len(bad):
            raise ValueError(f"{u['id']}: años con distritos faltantes: {bad.to_dict()}")
        tot = sub.groupby("year", as_index=False)["population"].sum()
        parts.append(tot.assign(unit=u["id"])[["unit", "year", "population"]])
    return pd.concat(parts).sort_values(["unit", "year"]).reset_index(drop=True) if parts else pd.DataFrame(columns=["unit", "year", "population"])


def read_population_csv(path: str | Path) -> pd.DataFrame:
    p = pd.read_csv(path)
    if list(p.columns[:3]) != ["unit", "year", "population"]:
        raise ValueError("el CSV de población debe tener las columnas unit, year, population")
    if (p["population"] <= 0).any() or p["population"].isna().any():
        raise ValueError("población no positiva o vacía")
    return p


def add_incidence(weekly: pd.DataFrame, pop: pd.DataFrame, cfg: dict, window: int = 4) -> pd.DataFrame:
    """Agrega population, cases4w, inc4w_per100k, level y level_sens100 / level_sens300.

    cases4w usa solo la semana t y las window-1 anteriores de la misma unidad (nada del futuro); las primeras
    window-1 semanas de cada unidad quedan en NaN. Una unidad sin población queda con NaN, no con cero.
    Niveles: 0 verde, 1 amarillo, 2 rojo; -1 donde no hay tasa.
    """
    d = weekly.sort_values(["unit", "abs_week"]).copy()
    if (d.groupby("unit")["abs_week"].diff().dropna() != 1).any():
        raise ValueError("las semanas no son consecutivas dentro de cada unidad")
    d = d.merge(pop[["unit", "year", "population"]], on=["unit", "year"], how="left")
    d = d.sort_values(["unit", "abs_week"]).reset_index(drop=True)
    d["cases4w"] = d.groupby("unit")["cases"].transform(lambda s: s.rolling(window, min_periods=window).sum())
    d["inc4w_per100k"] = incidence_per_100k(d["cases4w"].to_numpy(), d["population"].to_numpy())
    a = cfg["alerts"]
    cuts = {"level": (a["yellow_per_100k"], a["red_per_100k"])}
    sens = a["sensitivity_cuts_per_100k"]       # [100, 300]: amarillo, rojo de la sensibilidad
    cuts["level_sens"] = (sens[0], sens[1])
    ok = d["inc4w_per100k"].notna()
    for col, (y, r) in cuts.items():
        d[col] = np.where(ok, alert_level(d["inc4w_per100k"].fillna(0).to_numpy(), y, r), -1)
    return d
