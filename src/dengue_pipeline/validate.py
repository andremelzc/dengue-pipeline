"""Etapa 2. Validación de los datos crudos.

Cada regla corresponde a un defecto real observado (datos.md): filas perdidas por un lector
tolerante, deriva de significado de SEMANA_NOTIFICACION y FECHA_INGRESO entre cortes, régimen
cambiante de tipo_dx y el último año aparentemente incompleto en la costa. No se afirma la causa
de ningún defecto que datos.md no explique: aquí solo se detectan y se informan.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import assign_unit
from .epiweek import abs_week_of_dates, epiweek_of_dates, weeks_in_year
from .io import NATIONAL_COLS, REGIONAL_COLS, parse_yyyymmdd, physical_lines

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    data: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- CSV nacional
def validate_national(df: pd.DataFrame, cfg: dict, path: str | Path | None = None) -> list[Check]:
    out: list[Check] = []
    ok_cols = list(df.columns) == NATIONAL_COLS
    out.append(Check("esquema", PASS if ok_cols else FAIL,
                     "15 columnas esperadas" if ok_cols else f"columnas distintas: {list(df.columns)}"))
    if not ok_cols:
        return out

    if path is not None:
        lines = physical_lines(path)
        same = len(df) == lines - 1
        out.append(Check(
            "filas_vs_lineas_fisicas", PASS if same else WARN,
            f"{len(df)} filas leídas con lector estricto; {lines - 1} líneas de datos."
            + ("" if same else " Difieren: puede haber campos con saltos de línea o comillas mal cerradas."),
            {"rows": int(len(df)), "physical_data_lines": int(lines - 1)}))

    bad_ubigeo = int((~df["ubigeo"].str.fullmatch(r"\d{6}")).sum())
    out.append(Check("ubigeo_texto_6_digitos", PASS if bad_ubigeo == 0 else FAIL,
                     f"{bad_ubigeo} ubigeos que no son 6 dígitos (se leen como texto)."))

    dx = df["tipo_dx"].value_counts().to_dict()
    bad_dx = [k for k in dx if k not in ("C", "P")]
    out.append(Check("tipo_dx_dominio", PASS if not bad_dx else FAIL,
                     f"tipo_dx: {dx}" if not bad_dx else f"valores fuera de C/P: {bad_dx}", {"counts": dx}))

    ano = pd.to_numeric(df["ano"], errors="coerce")
    sem = pd.to_numeric(df["semana"], errors="coerce")
    non_num = int(ano.isna().sum() + sem.isna().sum())
    max_w = ano.dropna().astype(int).map(weeks_in_year)
    invalid = (sem < 1) | (sem > max_w.reindex(sem.index))
    by_year_week = {f"{int(a)}-S{int(s)}": int(n) for (a, s), n in
                    pd.DataFrame({"a": ano[invalid], "s": sem[invalid]}).groupby(["a", "s"]).size().items()}
    win = cfg["study_window"]
    in_window = int((invalid & ano.between(win["first_year"], win["last_year"])).sum())
    if non_num or in_window:
        st = FAIL
    elif invalid.any():
        st = WARN  # fuera de la ventana de estudio: se informa, no bloquea
    else:
        st = PASS
    out.append(Check(
        "semana_valida_para_el_año", st,
        f"{non_num} valores no numéricos; {int(invalid.sum())} filas con semana fuera de 1..semanas MMWR del año "
        f"({by_year_week}); {in_window} dentro de la ventana {win['first_year']}-{win['last_year']}. "
        "La causa no se conoce: no se corrigen ni se reasignan.", {"invalid": by_year_week, "in_window": in_window}))

    out.append(Check("rango_de_años", INFO, f"{int(ano.min())} a {int(ano.max())}",
                     {"min": int(ano.min()), "max": int(ano.max())}))

    share = (df.assign(ano=ano, p=(df["tipo_dx"] == "P")).groupby("ano")["p"].mean() * 100).round(1)
    out.append(Check("tipo_dx_probables_por_año", INFO,
                     "Porcentaje de probables por año (el régimen cambia por período; la causa no se conoce).",
                     {"pct_probables": {int(k): float(v) for k, v in share.items()}}))

    text_cols = ["departamento", "provincia", "distrito", "localidad"]
    mangled = int(df[text_cols].apply(lambda c: c.str.contains("�", regex=False)).any(axis=1).sum())
    out.append(Check("texto_con_caracteres_dañados", WARN if mangled else PASS,
                     f"{mangled} filas con el carácter de reemplazo U+FFFD en nombres; unir por ubigeo, no por nombre."))

    dup = int(df.duplicated().sum())
    out.append(Check("filas_duplicadas_exactas", INFO,
                     f"{dup} filas idénticas en todas las columnas (en vigilancia por caso pueden ser casos distintos; "
                     "no se eliminan)."))

    # Último año aparentemente incompleto en las unidades de estudio (caso 2025 en la costa).
    tmp = pd.DataFrame({"unit": assign_unit(df["departamento"], df["ubigeo"], cfg), "ano": ano})
    tmp = tmp.dropna()
    yearly = tmp.groupby(["unit", "ano"]).size().unstack("ano", fill_value=0)
    last = int(ano.max())
    thr = cfg["validation"]["last_year_ratio_threshold"]
    flagged, ratios = [], {}
    for unit, row in yearly.iterrows():
        prev = row.get(last - 1, 0)
        ratios[unit] = round(float(row.get(last, 0) / prev), 3) if prev else None
        if prev >= 500 and row.get(last, 0) / prev < thr:
            flagged.append(unit)
    out.append(Check(
        "ultimo_año_posiblemente_incompleto", WARN if flagged else PASS,
        f"Año {last} frente a {last - 1}: razón por unidad {ratios}. "
        + (f"Por debajo de {thr}: {flagged}. Hipótesis sin comprobar: notificación tardía o reclasificación; "
           "no usar ese año para evaluar falsas alarmas hasta contrastar con otra fuente." if flagged else ""),
        {"ratios": ratios, "flagged": flagged, "year": last}))
    return out


# ----------------------------------------------------------------------------- archivo regional
def _semantic_meaning(onset_match: float, notif_match: float) -> str:
    if onset_match > 0.99 and notif_match < 0.9:
        return "inicio_de_sintomas"
    if notif_match > 0.99:
        return "notificacion"
    return "indeterminado"


def validate_regional(df: pd.DataFrame, cfg: dict) -> list[Check]:
    out: list[Check] = []
    ok_cols = list(df.columns) == REGIONAL_COLS
    out.append(Check("esquema", PASS if ok_cols else FAIL,
                     "15 columnas esperadas (distintas del nacional)" if ok_cols else f"columnas: {list(df.columns)}"))
    if not ok_cols:
        return out

    d = df.copy()
    date_cols = ["FECHA_CORTE", "FECHA_INGRESO", "FECHA_INICIO_SINTOMAS", "FECHA_NOTIFICACION"]
    bad = {c: int(parse_yyyymmdd(d[c]).isna().sum()) for c in date_cols}
    for c in date_cols:
        d[c + "_d"] = parse_yyyymmdd(d[c])
    out.append(Check("fechas_validas", PASS if sum(bad.values()) == 0 else FAIL,
                     f"fechas AAAAMMDD inválidas por columna: {bad}", bad))
    cuts = sorted(d["FECHA_CORTE"].unique())
    out.append(Check("cortes", INFO, f"cortes presentes (FECHA_CORTE): {cuts}",
                     {"cuts": {c: int((d['FECHA_CORTE'] == c).sum()) for c in cuts}}))
    out.append(Check("filas_fuera_de_piura", INFO,
                     f"{int((d['DEPARTAMENTO'] != 'PIURA').sum())} filas de otros departamentos; "
                     "no se usan para el retraso de Piura.",
                     d.loc[d["DEPARTAMENTO"] != "PIURA", "DEPARTAMENTO"].value_counts().to_dict()))

    meanings = {}
    for cut, g in d.groupby("FECHA_CORTE"):
        on_w = epiweek_of_dates(g["FECHA_INICIO_SINTOMAS_d"])["epi_week"]
        nt_w = epiweek_of_dates(g["FECHA_NOTIFICACION_d"])["epi_week"]
        sn = pd.to_numeric(g["SEMANA_NOTIFICACION"], errors="coerce")
        onset_match = float((sn == on_w).mean())
        notif_match = float((sn == nt_w).mean())
        before = float((g["FECHA_INGRESO_d"] < g["FECHA_NOTIFICACION_d"]).mean())
        after = float((g["FECHA_INGRESO_d"] > g["FECHA_NOTIFICACION_d"]).mean())
        meanings[cut] = {
            "semana_notificacion_coincide_con_semana_de_inicio_sintomas": round(onset_match, 4),
            "semana_notificacion_coincide_con_semana_de_notificacion": round(notif_match, 4),
            "ingreso_anterior_a_notificacion": round(before, 4),
            "ingreso_posterior_a_notificacion": round(after, 4),
            "significado_inferido_de_SEMANA_NOTIFICACION": _semantic_meaning(onset_match, notif_match),
        }
    kinds = {m["significado_inferido_de_SEMANA_NOTIFICACION"] for m in meanings.values()}
    drift = len(meanings) > 1 and len(kinds) > 1
    out.append(Check(
        "deriva_de_significado_entre_cortes", WARN if drift else PASS,
        "SEMANA_NOTIFICACION y FECHA_INGRESO cambian de significado entre cortes: recalcular siempre desde "
        "las fechas." if drift else "Sin deriva detectada entre los cortes presentes.", meanings))

    neg = {}
    for cut, g in d.groupby("FECHA_CORTE"):
        neg[cut] = int((g["FECHA_NOTIFICACION_d"] < g["FECHA_INICIO_SINTOMAS_d"]).sum())
    out.append(Check("notificacion_anterior_al_inicio_de_sintomas", WARN if sum(neg.values()) else PASS,
                     f"filas con notificación antes del inicio de síntomas por corte: {neg}. "
                     "Se excluyen al estimar el retraso y se consideran disponibles desde el inicio de síntomas.", neg))

    after_cut, other_week = {}, {}
    for cut, g in d.groupby("FECHA_CORTE"):
        cut_d = pd.to_datetime(cut, format="%Y%m%d")
        after_cut[cut] = int((g["FECHA_NOTIFICACION_d"] > cut_d).sum())
        other_week[cut] = int((abs_week_of_dates(g["FECHA_NOTIFICACION_d"]) > abs_week_of_dates(pd.Series([cut_d]))[0]).sum())
    st = FAIL if sum(other_week.values()) else (WARN if sum(after_cut.values()) else PASS)
    out.append(Check(
        "notificacion_posterior_al_corte", st,
        f"filas con notificación posterior a la fecha de su corte: {after_cut}; de ellas, en una semana epidemiológica "
        f"posterior a la del corte: {other_week}. Si solo caen en la misma semana, la fecha de corte y el último dato "
        "difieren por días y no afectan a los conteos semanales; si caen en semanas posteriores, la disponibilidad "
        "semanal sería incorrecta.", {"after_cut": after_cut, "other_week": other_week}))

    # Últimas semanas del corte (datos.md 3.6): suelen estar incompletas.
    thr = cfg["validation"]["recent_week_ratio_threshold"]
    recent = {}
    for cut, g in d[d["DEPARTAMENTO"] == "PIURA"].groupby("FECHA_CORTE"):
        wk = abs_week_of_dates(g["FECHA_INICIO_SINTOMAS_d"]).value_counts().sort_index()
        wk = wk.reindex(range(int(wk.index.min()), int(wk.index.max()) + 1), fill_value=0)
        if len(wk) >= 6:
            ref = float(wk.iloc[-6:-2].median())
            recent[cut] = {"ultima_semana": int(wk.iloc[-1]), "penultima": int(wk.iloc[-2]),
                           "mediana_4_previas": ref,
                           "razon_penultima": round(float(wk.iloc[-2]) / ref, 2) if ref else None,
                           "razon_ultima": round(float(wk.iloc[-1]) / ref, 2) if ref else None}
    flag = [c for c, r in recent.items() if r["razon_ultima"] is not None and r["razon_ultima"] < thr]
    out.append(Check("ultimas_semanas_del_corte_incompletas", WARN if flag else PASS,
                     f"Casos por semana de inicio de síntomas, Piura, al final de cada corte: {recent}. "
                     f"Cortes con la última semana < {thr} de la mediana previa: {flag}.", recent))
    return out


# ----------------------------------------------------------------------------- informe
def write_report(checks: dict[str, list[Check]], out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath, mpath = out_dir / "validation_report.json", out_dir / "validation_report.md"
    jpath.write_text(json.dumps({k: [asdict(c) for c in v] for k, v in checks.items()},
                                indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    lines = ["# Informe de validación", ""]
    for source, cs in checks.items():
        lines += [f"## {source}", "", "| Regla | Estado | Detalle |", "|---|---|---|"]
        for c in cs:
            detail = re.sub(r"\s+", " ", c.detail).replace("|", "/")
            lines.append(f"| {c.name} | {c.status} | {detail} |")
        lines.append("")
    mpath.write_text("\n".join(lines), encoding="utf-8")
    return jpath, mpath


def has_failures(checks: dict[str, list[Check]]) -> bool:
    return any(c.status == FAIL for v in checks.values() for c in v)
