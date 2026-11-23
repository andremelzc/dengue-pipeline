"""Arma la tabla de ICEN por versión publicada (month, value, published_on, kind) a partir de los informes técnicos mensuales del IGP.

Cada informe "Informe Técnico PpR/El Niño-IGP/AAAA-MM" trae el ICEN de los meses recientes (tabla "Año Mes ICEN ...") y los
valores temporales de los meses siguientes (tabla "Año Mes ICENtmp ..."), con la fecha del informe en la primera página.
Cada fila de cada informe es una versión publicada: la misma mes puede aparecer en varios informes, a veces con valores distintos
(eso es justo lo que se necesita para "el valor que existía en cada fecha").

Qué hace el lector (heurístico: [NO VERIFICADO] fuera de los formatos vistos):
- toma solo filas "AAAA Mes valor" bajo un encabezado "Año Mes ICEN..." (kind = temporal si el encabezado dice tmp/temporal, si no definitivo);
- ignora las tablas de ONI / RONI y el ICENr (solo la primera cifra después del mes, que es el ICEN clásico);
- NO corrige nada: lo dudoso va a la tabla de revisión (`icen_reports_review.csv`) y NO entra en `icen.csv`. Hay que mirarlo a mano.
Probado con un informe real (2014-02) y con texto sintético con la forma de informes de 2024-2025; los otros formatos pueden fallar.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

MONTHS = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
          "setiembre": 9, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12}
_ROW = re.compile(r"^\s*(\d{4})\s+([A-Za-zñÑáéíóú]+)\s+(.*)$")
_NUM = re.compile(r"(?<![\d/])[–−-]?\d+[.,]\d+(?![\d/])")
_SKIP = {"categoría", "categoria", "condiciones", "condición", "condicion", "(ºc)", "(°c)", "fuente"}
_HDR = re.compile(r"^\s*A[ñn]o\s+Mes\s+(.*)$", re.I)
_REPORT = re.compile(r"PpR/El\s*Ni[ñn]o-IGP/(\d{4})-(\d{1,2})", re.I)
_DATE = re.compile(r"\b(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{4})\b")
COLS = ["month", "value", "published_on", "kind", "serie", "report", "file"]


def _num(s: str) -> float:
    return float(s.replace("–", "-").replace("−", "-").replace(",", "."))


def parse_report_text(text: str, file: str = "") -> tuple[list[dict], list[dict], dict]:
    """Devuelve (filas válidas, filas a revisar, metadatos) de un informe ya convertido a texto."""
    head = text[:4000]
    m_rep, m_date = _REPORT.search(head), _DATE.search(head)
    meta = {"file": file, "report": f"{m_rep.group(1)}-{int(m_rep.group(2)):02d}" if m_rep else None,
            "published_on": f"{m_date.group(3)}-{int(m_date.group(2)):02d}-{int(m_date.group(1)):02d}" if m_date else None}
    rows, review = [], []
    kind, serie, col = None, None, 0
    lines = text.splitlines()
    for n, line in enumerate(lines):
        h = _HDR.match(line)
        if h:
            # solo la primera columna de valores, y solo si es el ICEN clásico (ICEN, ICENtmp, ICENv3...); ICENOI, ONI, etc. se ignoran
            toks = [t for t in h.group(1).split() if t.lower() not in _SKIP]
            cand = [(k, t) for k, t in enumerate(toks) if re.match(r"^ICEN(?!OI)\w*$", t, re.I)]
            if cand:
                col, tok = cand[0]                       # posición (entre las columnas de valores) del primer ICEN clásico
                prev = " ".join(lines[max(0, n - 3):n]).lower()
                kind = "temporal" if ("tmp" in tok.lower() or "temporal" in prev) else "definitivo"
                serie = tok
            else:
                kind = serie = None
            continue
        r = _ROW.match(line)
        if not r or kind is None:
            continue
        year, mon = int(r.group(1)), r.group(2).lower()
        nums = _NUM.findall(r.group(3))
        if mon not in MONTHS or len(nums) <= col:
            continue
        val = _num(nums[col])
        rows.append({"month": f"{year}-{MONTHS[mon]:02d}", "value": val, "published_on": meta["published_on"], "kind": kind, "serie": serie,
                     "report": meta["report"], "file": file, "line": line.strip()[:90]})
    # comprobaciones de coherencia (no se corrige nada: lo dudoso sale de la tabla válida)
    defin = [r["month"] for r in rows if r["kind"] == "definitivo"]
    last_def = max(defin) if defin else None
    good = []
    for r in rows:
        why = None
        if meta["published_on"] is None:
            why = "sin_fecha_de_publicacion"
        elif not -5 <= r["value"] <= 5:
            why = "valor_fuera_de_rango"
        elif r["kind"] == "temporal" and last_def and r["month"] <= last_def:
            why = "temporal_de_un_mes_ya_definitivo (¿año mal escrito en el informe?)"
        elif r["month"] > meta["published_on"][:7]:
            if r["kind"] == "definitivo":
                why = "definitivo_de_un_mes_posterior_a_la_fecha_del_informe"
        if why:
            review.append({**r, "motivo": why})
        else:
            good.append(r)
    return good, review, meta


def pdf_text(path: str | Path, max_pages: int = 8) -> str:
    """Texto de las primeras `max_pages` páginas (las tablas ICEN están al inicio; así se lee mucho más rápido)."""
    import pdfplumber                                   # dependencia opcional: pip install -e ".[icen]"
    with pdfplumber.open(path) as p:
        return "\n".join((pg.extract_text() or "") for pg in p.pages[:max_pages])


def build_table(folder: str | Path, text_fn=pdf_text) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lee todos los PDF de `folder`. Devuelve (icen, revisión, resumen por archivo)."""
    good, review, summ = [], [], []
    for f in sorted(Path(folder).glob("*.pdf")):
        try:
            g, rv, meta = parse_report_text(text_fn(f), f.name)
        except Exception as e:  # noqa: BLE001
            summ.append({"file": f.name, "report": None, "published_on": None, "filas_validas": 0, "filas_a_revisar": 0, "error": repr(e)})
            continue
        good += g
        review += rv
        summ.append({**meta, "filas_validas": len(g), "filas_a_revisar": len(rv), "error": None})
    icen = pd.DataFrame(good, columns=COLS + ["line"]).drop(columns="line") if good else pd.DataFrame(columns=COLS)
    conflicts = icen.duplicated(["month", "kind", "published_on"], keep=False) if len(icen) else pd.Series(dtype=bool)
    if len(icen) and conflicts.any():                   # mismo mes, tipo y fecha con valores distintos: a revisar, no se elige uno
        dif = icen[conflicts].groupby(["month", "kind", "published_on"])["value"].transform("nunique") > 1
        bad = icen[conflicts][dif]
        review += [{**r, "motivo": "valores_distintos_para_el_mismo_mes_y_fecha"} for r in bad.to_dict("records")]
        icen = icen.drop(index=bad.index)
        icen = icen.drop_duplicates(["month", "kind", "published_on", "value"])
    return icen.sort_values(["month", "published_on"]).reset_index(drop=True), pd.DataFrame(review), pd.DataFrame(summ)
