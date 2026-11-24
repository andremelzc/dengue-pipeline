"""Reproducción completa del pipeline con un solo comando (`python -m dengue_pipeline reproduce ...`).

Pensado para que varias personas corran exactamente lo mismo en su computador y comparen los resultados con
`compare-runs`. Antes de correr nada se comprueba que los CSV crudos tengan el mismo hash SHA-256 que
`data/manifest.json`: si los datos son distintos, la comparación no tendría sentido.

Los CSV del portal de datos abiertos pueden actualizarse; hay que usar la misma copia que registra el manifiesto.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

CONFIRMED_DIR = "data/processed/solo_confirmados"
CLIMATE_START, CLIMATE_END = "20091101", "20241231"
PACKAGES = ("pandas", "numpy", "scikit-learn", "statsmodels", "matplotlib", "scipy", "PyYAML", "openpyxl")


@dataclass
class Step:
    name: str
    argv: list[str]
    help: str = ""
    needs_network: bool = False
    tags: set[str] = field(default_factory=set)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_inputs(inputs: dict[str, str | Path], manifest_path: str | Path) -> list[dict]:
    """Compara el hash de cada CSV crudo (`national`, `regional`) con el de `data/manifest.json`."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    by_kind = {e["kind"]: e for e in manifest}
    out = []
    for kind, path in inputs.items():
        got = sha256_file(path)
        exp = by_kind.get(kind, {}).get("sha256")
        out.append({"kind": kind, "file": str(path), "sha256": got, "esperado": exp, "ok": exp is not None and got == exp})
    return out


def build_steps(national: str, regional: str, population_xlsx: str, districts: str | None = None,
                climate_downloaded_on: str | None = None, skip_climate: bool = False, ablation: bool = True,
                sensitivity: bool = True, icen_csv: str = "config/icen.csv", stress: bool = False,
                confirmed: bool = False) -> list[Step]:
    """Secuencia completa, en orden de dependencia.

    `stress` y `confirmed` agregan pasos opcionales que NO forman parte de la secuencia de 14 pasos: la prueba de estrés del retraso
    (x2 y x4) y la variante de solo confirmados (tipo_dx C), que guarda todo en data/processed/solo_confirmados.
    """
    pop = ["population", population_xlsx] + (["--districts", districts] if districts else [])
    steps = [
        Step("validar", ["validate", "--national", national, "--regional", regional], "etapa 2: validación de los CSV"),
        Step("integrar", ["integrate", "--national", national], "etapa 3: serie final unidad x semana"),
        Step("retraso", ["delay", "--regional", regional, "--all-scenarios"], "etapa 4: retraso de notificación (3 escenarios)"),
        Step("simular", ["simulate", "--national", national, "--regional", regional], "etapa 4: triángulo simulado (escenario base)"),
        Step("poblacion", pop, "población INEI"),
        Step("incidencia", ["incidence"], "incidencia y niveles de alerta"),
        Step("icen", ["icen", icen_csv], "ICEN conocido en cada semana"),
    ]
    if not skip_climate:
        date = climate_downloaded_on or dt.date.today().isoformat()
        steps.append(Step("clima", ["climate", "--coordinates", "config/coordenadas.csv", "--start", CLIMATE_START,
                                    "--end", CLIMATE_END, "--downloaded-on", date], "clima NASA POWER (necesita red)", True))
    steps += [
        Step("principal", ["backtest"], "corrida principal (entrenamiento desde 2010)"),
        Step("alertas", ["alerts"], "métricas de alerta de la corrida principal"),
    ]
    if sensitivity:
        steps.append(Step("sensibilidad", ["delay-sensitivity", "--regional", regional],
                          "sensibilidad al retraso: 3 escenarios x 5 semillas", tags={"pesado"}))
    if stress:
        steps.append(Step("estres_retraso", ["delay-sensitivity", "--regional", regional, "--delay-scale", "2", "4"],
                          "prueba de estrés: retraso x2 y x4, escenario base x 5 semillas", tags={"pesado"}))
    if ablation:
        steps += [
            Step("hist2014", ["backtest", "--train-first-year", "2014"], "ablación desde 2014: historial", tags={"pesado"}),
            Step("clima2014", ["backtest", "--train-first-year", "2014", "--features", "clima"],
                 "ablación desde 2014: + clima", tags={"pesado"}),
            Step("clima_icen2014", ["backtest", "--train-first-year", "2014", "--features", "clima_icen"],
                 "ablación desde 2014: + clima + ICEN", tags={"pesado"}),
        ]
    if confirmed:
        cdir = f"{CONFIRMED_DIR}/backtest"
        steps += [
            Step("integrar_confirmados", ["integrate", "--national", national, "--case-definition", "confirmed"],
                 "variante solo confirmados: serie final"),
            Step("simular_confirmados", ["simulate", "--national", national, "--regional", regional, "--case-definition", "confirmed"],
                 "variante solo confirmados: triángulo simulado"),
            Step("principal_confirmados", ["backtest", "--case-definition", "confirmed"],
                 "variante solo confirmados: corrida principal", tags={"pesado"}),
            Step("alertas_confirmados", ["alerts", "--predictions", f"{cdir}/predictions.csv.gz", "--out", cdir],
                 "variante solo confirmados: métricas de alerta"),
        ]
    return steps


def select(steps: list[Step], start: str | None = None, only: list[str] | None = None) -> list[Step]:
    names = [s.name for s in steps]
    for n in ([start] if start else []) + list(only or []):
        if n not in names:
            raise ValueError(f"paso desconocido: {n!r}; opciones: {names}")
    if only:
        return [s for s in steps if s.name in only]
    if start:
        return steps[names.index(start):]
    return steps


def environment() -> dict:
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    return {"python": platform.python_version(), "sistema": platform.platform(), "paquetes": versions}


def run(steps: list[Step], dry_run: bool = False, log=print, log_path: str | Path | None = None,
        python: str = sys.executable) -> int:
    """Ejecuta los pasos en orden y se detiene en el primer fallo. Devuelve 0 si todo salió bien."""
    records, code = [], 0
    for i, s in enumerate(steps, 1):
        cmd = [python, "-m", "dengue_pipeline", *s.argv]
        log(f"[{i}/{len(steps)}] {s.name}: {s.help}\n    {subprocess.list2cmdline(['python', '-m', 'dengue_pipeline', *s.argv])}")
        if dry_run:
            continue
        t0, started = time.time(), dt.datetime.now().isoformat(timespec="seconds")
        rc = subprocess.run(cmd).returncode
        records.append({"paso": s.name, "comando": s.argv, "inicio": started, "segundos": round(time.time() - t0, 1), "codigo": rc})
        log(f"    -> código {rc} ({records[-1]['segundos']} s)")
        if rc != 0:
            code = rc
            log(f"ABORTA en '{s.name}'. Corrige el problema y continúa con --from {s.name}")
            break
    if log_path and not dry_run:
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"entorno": environment(), "pasos": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    return code
