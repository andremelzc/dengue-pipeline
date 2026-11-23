"""Compara dos corridas del pipeline (dos carpetas `data/processed`) para comprobar que son reproducibles.

Para cada archivo .csv, .csv.gz o .json de las dos carpetas dice si:
  OK         coincide (en los números, con tolerancia `tol`; el orden de las filas no importa)
  DIFERENTE  existe en las dos pero no coincide (con la máxima diferencia numérica)
  FALTA      solo existe en una de las dos carpetas
Los campos que dependen de quién y cuándo corre (fechas de descarga, rutas, tiempos) se ignoran.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

IGNORE_KEYS = {"downloaded_on", "file", "registered_at_utc", "downloaded_on_note"}
SKIP_NAMES = {"reproducir_log.json"}


def _files(root: Path) -> dict[str, Path]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name not in SKIP_NAMES and (p.suffix in (".csv", ".json") or p.name.endswith(".csv.gz")):
            out[p.relative_to(root).as_posix()] = p
    return out


def compare_frames(a: pd.DataFrame, b: pd.DataFrame, tol: float) -> tuple[bool, str]:
    if list(a.columns) != list(b.columns):
        return False, "columnas distintas"
    if len(a) != len(b):
        return False, f"número de filas distinto ({len(a)} frente a {len(b)})"
    num = [c for c in a.columns if pd.api.types.is_float_dtype(a[c]) or pd.api.types.is_float_dtype(b[c])]
    key = [c for c in a.columns if c not in num]
    if key:
        a, b = a.sort_values(key, kind="mergesort").reset_index(drop=True), b.sort_values(key, kind="mergesort").reset_index(drop=True)
        if not a[key].equals(b[key]):
            return False, "las filas (claves) no coinciden"
    else:
        a, b = a.sort_values(list(a.columns), kind="mergesort").reset_index(drop=True), b.sort_values(list(b.columns), kind="mergesort").reset_index(drop=True)
    worst = 0.0
    for c in num:
        x, y = a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)
        if not np.array_equal(np.isnan(x), np.isnan(y)):
            return False, f"valores faltantes distintos en '{c}'"
        ok = ~np.isnan(x)
        if ok.any():
            worst = max(worst, float(np.max(np.abs(x[ok] - y[ok]))))
    return worst <= tol, f"máxima diferencia {worst:.3g}"


def _json_diff(a, b, tol: float, path: str = "") -> str | None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted((set(a) | set(b)) - IGNORE_KEYS):
            if k not in a or k not in b:
                return f"falta la clave {path}/{k}"
            d = _json_diff(a[k], b[k], tol, f"{path}/{k}")
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"largo distinto en {path}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _json_diff(x, y, tol, f"{path}[{i}]")
            if d:
                return d
        return None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return None if abs(a - b) <= tol else f"{path}: {a} frente a {b}"
    return None if a == b else f"{path}: {a!r} frente a {b!r}"


def compare_dirs(dir_a: str | Path, dir_b: str | Path, tol: float = 1e-6) -> list[dict]:
    fa, fb = _files(Path(dir_a)), _files(Path(dir_b))
    rows = []
    for name in sorted(set(fa) | set(fb)):
        if name not in fa or name not in fb:
            rows.append({"archivo": name, "estado": "FALTA", "detalle": "solo en " + ("B" if name not in fa else "A")})
            continue
        if name.endswith(".json"):
            d = _json_diff(json.loads(fa[name].read_text(encoding="utf-8")), json.loads(fb[name].read_text(encoding="utf-8")), tol)
            rows.append({"archivo": name, "estado": "OK" if d is None else "DIFERENTE", "detalle": d or "coincide"})
        else:
            ok, detail = compare_frames(pd.read_csv(fa[name]), pd.read_csv(fb[name]), tol)
            rows.append({"archivo": name, "estado": "OK" if ok else "DIFERENTE", "detalle": detail})
    return rows


def format_report(rows: list[dict]) -> str:
    lines = [f"{r['estado']:<10} {r['archivo']}  ({r['detalle']})" for r in rows]
    n_ok = sum(r["estado"] == "OK" for r in rows)
    lines.append(f"\n{n_ok} de {len(rows)} archivos coinciden." + ("" if n_ok == len(rows) else " HAY DIFERENCIAS."))
    return "\n".join(lines)
