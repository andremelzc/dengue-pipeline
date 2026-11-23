"""Etapa 1. Adquisición manual con manifiesto (fecha, hash SHA-256, filas).

La descarga es manual: el portal solo guarda el archivo vigente y puede bloquear clientes que
no son navegadores; no se evade (pipeline.md 1). Lo que sigue a la descarga es automático.
La fecha de descarga la declara quien descarga: el sistema de archivos no la conoce.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .io import NATIONAL_COLS, REGIONAL_COLS, physical_lines, read_strict

KINDS = {"national": NATIONAL_COLS, "regional": REGIONAL_COLS}


def sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def register_file(path: str | Path, kind: str, source_url: str, downloaded_on: str | None,
                  manifest_path: str | Path) -> dict:
    """Registra un archivo crudo en el manifiesto. `downloaded_on` (AAAA-MM-DD) lo declara el usuario."""
    if kind not in KINDS:
        raise ValueError(f"kind debe ser uno de {list(KINDS)}")
    path = Path(path)
    df = read_strict(path)
    entry = {
        "file": path.name,
        "kind": kind,
        "source_url": source_url,
        "downloaded_on": downloaded_on or "[FALTA: fecha de descarga]",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_of(path),
        "physical_lines": physical_lines(path),
        "rows_parsed": int(len(df)),
        "columns_ok": list(df.columns) == KINDS[kind],
    }
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    manifest = [m for m in manifest if m["sha256"] != entry["sha256"]]  # mismo contenido: reemplaza
    manifest.append(entry)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return entry
