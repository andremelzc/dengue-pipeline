"""Línea de comandos. Uso: python -m dengue_pipeline <comando> --help"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import acquire, alerts_eval, eda, exogenous, icen_reports, integrate, population, snapshots, validate
from .config import load_config
from .epiweek import abs_week
from .io import read_strict


def _cmd_register(a, cfg):
    e = acquire.register_file(a.file, a.kind, a.source_url, a.downloaded_on, a.manifest)
    print(json.dumps(e, indent=2, ensure_ascii=False))
    return 0


def _cmd_validate(a, cfg):
    checks = {}
    if a.national:
        checks["CSV nacional"] = validate.validate_national(read_strict(a.national), cfg, a.national)
    if a.regional:
        checks["Archivo regional de Piura"] = validate.validate_regional(read_strict(a.regional), cfg)
    j, m = validate.write_report(checks, a.out)
    for src, cs in checks.items():
        for c in cs:
            print(f"[{c.status}] {src}: {c.name}")
    print(f"Informe: {m}")
    return 1 if validate.has_failures(checks) else 0


def _cmd_integrate(a, cfg):
    tab = integrate.weekly_table(read_strict(a.national), cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "weekly_final.csv", index=False)
    print(f"{len(tab)} filas ({tab['unit'].nunique()} unidades); casos descartados por semana inválida: "
          f"{tab.attrs.get('dropped_cases_invalid_week')}")
    return 0


def _estimate(a, cfg):
    d = cfg["delay"]
    return snapshots.delay_pmf(read_strict(a.regional), d["estimation"]["cut"], d["estimation"]["onset_year"],
                               d["exclude_last_weeks"], d["max_delay_weeks"])


def _cmd_delay(a, cfg):
    r = _estimate(a, cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"delay_weeks": range(len(r["pmf"])), "pmf": r["pmf"], "cum": r["cum"]}).to_csv(
        out / "delay_pmf.csv", index=False)
    print({k: v for k, v in r.items() if k not in ("pmf", "cum")})
    return 0


def _cmd_simulate(a, cfg):
    r = _estimate(a, cfg)
    final = integrate.weekly_table(read_strict(a.national), cfg)
    tri = snapshots.simulate_triangle(final, r["pmf"], cfg["delay"]["seed"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tri.to_csv(out / "triangle_simulated.csv.gz", index=False)
    print(f"{len(tri)} filas del triángulo simulado (semilla {cfg['delay']['seed']})")
    return 0


def _cmd_pair(a, cfg):
    reg, final = read_strict(a.regional), integrate.weekly_table(read_strict(a.national), cfg)
    weeks = [abs_week(2023, w) for w in range(13, 21)]
    print(snapshots.cut_vs_final(reg, "20230517", final, "PIURA", weeks).to_string(index=False))
    return 0


def _cmd_population(a, cfg):
    raw = population.read_inei_departmental(a.xlsx)
    pop = population.population_by_unit(raw, cfg)
    if a.districts:
        pop = pd.concat([pop, population.population_ubigeo_groups(population.read_inei_districts(a.districts), cfg)], ignore_index=True)
        msg = "Lima Norte = suma de distritos (solo desde 2018)"
    else:
        msg = "LIMA_NORTE sin población [FALTA]"
    pop = pop.sort_values(["unit", "year"]).reset_index(drop=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pop.to_csv(out / "population.csv", index=False)
    print(f"{len(pop)} filas ({pop['unit'].nunique()} unidades); {msg}")
    return 0


def _cmd_incidence(a, cfg):
    weekly = pd.read_csv(a.weekly)
    inc = population.add_incidence(weekly, population.read_population_csv(a.population), cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    inc.to_csv(out / "weekly_incidence.csv", index=False)
    print(f"{len(inc)} filas; sin tasa (sin población o primeras semanas): {int((inc['level'] < 0).sum())}")
    return 0


def _cmd_alerts(a, cfg):
    pred = pd.read_csv(a.predictions)
    pop = population.read_population_csv(a.population)
    d = alerts_eval.attach_levels(pred, pop, cfg)
    b, seed = cfg["backtest"]["bootstrap"], cfg["backtest"]["seed"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    alerts_eval.alert_metrics_table(d, b["block_weeks"], b["n_resamples"], b["ci"], seed).to_csv(out / "alert_metrics.csv", index=False)
    alerts_eval.alert_paired_difference(d, b["block_weeks"], b["n_resamples"], b["ci"], seed).to_csv(out / "alert_paired_difference.csv", index=False)
    ep = alerts_eval.episodes_table(d)
    ep.to_csv(out / "alert_episodes.csv", index=False)
    alerts_eval.episode_summary(ep).to_csv(out / "alert_episode_summary.csv", index=False)
    print(f"{len(d)} predicciones con tasa ({d['unit'].nunique()} unidades); {len(ep)} episodios; "
          "AVISO: corrida de prueba con retraso PROVISIONAL, no citar como resultado")
    return 0


def _cmd_climate(a, cfg):
    units = [u["id"] for u in cfg["units"]]
    coords = exogenous.read_coordinates(a.coordinates, units)           # falla con [FALTA] si hay una vacía
    entries = exogenous.fetch_all(cfg, coords, a.start, a.end, a.raw_dir, a.downloaded_on)
    daily = {e["unit"]: exogenous.parse_nasa_power(Path(e["file"]).read_bytes()) for e in entries}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    exogenous.climate_weekly(daily).to_csv(out / "climate_weekly.csv", index=False)
    (out / "climate_manifest.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(entries)} unidades descargadas; manifiesto en {out / 'climate_manifest.json'}")
    return 0


def _cmd_icen_reports(a, cfg):
    icen, review, summ = icen_reports.build_table(a.folder)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    icen.to_csv(out / "icen.csv", index=False)
    review.to_csv(out / "icen_reports_review.csv", index=False)
    summ.to_csv(out / "icen_reports_summary.csv", index=False)
    meses = sorted(set(icen.loc[icen["kind"] == "definitivo", "month"])) if len(icen) else []
    print(f"{len(summ)} PDF leídos; {len(icen)} filas válidas ({len(meses)} meses con valor definitivo); "
          f"{len(review)} filas A REVISAR A MANO (icen_reports_review.csv); archivos con error: {int(summ['error'].notna().sum()) if len(summ) else 0}")
    return 0


def _cmd_icen(a, cfg):
    weekly = pd.read_csv(a.weekly)
    weeks = weekly[weekly["unit"] == weekly["unit"].iloc[0]][["abs_week", "week_end"]]
    res = exogenous.icen_weekly(weeks, exogenous.read_icen_csv(a.icen))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "icen_weekly.csv", index=False)
    print(f"{len(res)} semanas; sin valor conocido: {int(res['icen'].isna().sum())}")
    return 0


def _cmd_eda(a, cfg):
    r = eda.run_eda(pd.read_csv(a.weekly), pd.read_csv(a.incidence), pd.read_csv(a.delay), a.out)
    print(r["stats"].round(3).to_string())
    return 0


def _cmd_backtest(a, cfg):
    from . import backtest, metrics
    weekly = pd.read_csv(a.weekly)
    tri = pd.read_csv(a.triangle)
    out = Path(a.out)
    models = a.models.split(",") if a.models else None
    units = [u["id"] for u in cfg["units"]]
    if a.train_first_year:                       # ablación con ICEN desde 2014: carpeta aparte para no pisar la corrida principal
        cfg["backtest"]["train_first_year"] = a.train_first_year
        if a.out == "data/processed/backtest":
            out = Path(f"data/processed/backtest_desde{a.train_first_year}")
    exog, tag = None, ""
    if a.features != "historial":
        if a.example_exog:
            print("AVISO: datos de EJEMPLO sintéticos; la salida NO es un resultado y se guarda con prefijo EJEMPLO_NO_REAL")
            climate = exogenous.make_example_climate(weekly)
            icen = exogenous.make_example_icen(weekly)
            tag = exogenous.EXAMPLE_FLAG + "_"
        else:
            climate = pd.read_csv(a.climate_csv) if Path(a.climate_csv).exists() else None
            icen = pd.read_csv(a.icen_csv) if Path(a.icen_csv).exists() else None
        exog = exogenous.build_exog(a.features, weekly, units, cfg, climate, icen)
    frozen = None
    if a.features != "historial" and (out / "hyperparameters.json").exists():
        frozen = backtest.load_frozen_params(out / "hyperparameters.json")
    suffix = "" if a.features == "historial" else f"_{a.features}"
    pred, info = backtest.run_backtest(weekly, tri, cfg, models=models, out_dir=out, features=a.features, exog=exog,
                                       params_by_h=frozen, out_name=f"{tag}predictions{suffix}.csv.gz")
    b = cfg["backtest"]["bootstrap"]
    summ = metrics.summarize(pred, b["block_weeks"], b["n_resamples"], b["ci"], cfg["backtest"]["seed"])
    summ.to_csv(out / f"{tag}metrics_summary{suffix}.csv", index=False)
    pd_ = metrics.paired_difference(pred, b["block_weeks"], b["n_resamples"], b["ci"], cfg["backtest"]["seed"])
    pd_.to_csv(out / f"{tag}paired_difference{suffix}.csv", index=False)
    print(f"{len(pred)} predicciones; {info['fits']} reentrenamientos; fallbacks NB: {len(info.get('negbin_fallbacks', []))}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dengue_pipeline")
    p.add_argument("--config", default=None, help="ruta de config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("register", help="etapa 1: registrar un CSV crudo en el manifiesto")
    s.add_argument("file")
    s.add_argument("--kind", choices=["national", "regional"], required=True)
    s.add_argument("--source-url", required=True)
    s.add_argument("--downloaded-on", default=None, help="AAAA-MM-DD, lo declara quien descarga")
    s.add_argument("--manifest", default="data/manifest.json")
    s.set_defaults(fn=_cmd_register)

    s = sub.add_parser("validate", help="etapa 2: validar los datos crudos")
    s.add_argument("--national")
    s.add_argument("--regional")
    s.add_argument("--out", default="reports")
    s.set_defaults(fn=_cmd_validate)

    s = sub.add_parser("integrate", help="etapa 3: tabla unidad x semana (serie final)")
    s.add_argument("--national", required=True)
    s.add_argument("--out", default="data/processed")
    s.set_defaults(fn=_cmd_integrate)

    s = sub.add_parser("delay", help="etapa 4: distribución del retraso de notificación (Piura)")
    s.add_argument("--regional", required=True)
    s.add_argument("--out", default="data/processed")
    s.set_defaults(fn=_cmd_delay)

    s = sub.add_parser("simulate", help="etapa 4: triángulo simulado de datos disponibles")
    s.add_argument("--national", required=True)
    s.add_argument("--regional", required=True)
    s.add_argument("--out", default="data/processed")
    s.set_defaults(fn=_cmd_simulate)

    s = sub.add_parser("pair", help="par real de versiones: Piura 2023, corte frente a serie final")
    s.add_argument("--national", required=True)
    s.add_argument("--regional", required=True)
    s.set_defaults(fn=_cmd_pair)

    s = sub.add_parser("population", help="lee el cuadro de población del INEI -> population.csv")
    s.add_argument("xlsx")
    s.add_argument("--districts", default=None, help="cuadro distrital INEI 2018-2026 (XLSX de gob.pe) para las unidades de distritos (Lima Norte)")
    s.add_argument("--out", default="data/processed")
    s.set_defaults(fn=_cmd_population)

    s = sub.add_parser("incidence", help="incidencia acumulada de 4 semanas por 100 000 y niveles de alerta")
    s.add_argument("--weekly", default="data/processed/weekly_final.csv")
    s.add_argument("--population", default="data/processed/population.csv")
    s.add_argument("--out", default="data/processed")
    s.set_defaults(fn=_cmd_incidence)

    s = sub.add_parser("alerts", help="sensibilidad, falsas alarmas y anticipación por episodio sobre predictions.csv.gz")
    s.add_argument("--predictions", default="data/processed/backtest/predictions.csv.gz")
    s.add_argument("--population", default="data/processed/population.csv")
    s.add_argument("--out", default="data/processed/backtest")
    s.set_defaults(fn=_cmd_alerts)

    s = sub.add_parser("eda", help="exploración de datos: figuras y tablas en reports/")
    s.add_argument("--weekly", default="data/processed/weekly_final.csv")
    s.add_argument("--incidence", default="data/processed/weekly_incidence.csv")
    s.add_argument("--delay", default="data/processed/delay_pmf.csv")
    s.add_argument("--out", default="reports")
    s.set_defaults(fn=_cmd_eda)

    s = sub.add_parser("climate", help="descarga NASA POWER por unidad y arma el clima semanal (necesita coordenadas y red)")
    s.add_argument("--coordinates", default="config/coordenadas.csv", help="unit,lat,lon,source (ver config/coordenadas_plantilla.csv)")
    s.add_argument("--start", required=True, help="AAAAMMDD")
    s.add_argument("--end", required=True, help="AAAAMMDD")
    s.add_argument("--downloaded-on", required=True, help="AAAA-MM-DD, lo declara quien descarga")
    s.add_argument("--raw-dir", default="data/raw/nasa_power")
    s.add_argument("--out", default="data/processed")
    s.set_defaults(fn=_cmd_climate)

    s = sub.add_parser("icen-reports", help="arma config/icen.csv leyendo los informes técnicos mensuales del IGP (PDF en una carpeta)")
    s.add_argument("folder", help="carpeta con los PDF descargados de repositorio.igp.gob.pe")
    s.add_argument("--out", default="config")
    s.set_defaults(fn=_cmd_icen_reports)

    s = sub.add_parser("icen", help="ICEN conocido en cada semana a partir de la tabla de versiones publicadas")
    s.add_argument("icen", help="CSV month,value,published_on,kind (ver config/icen_plantilla.csv)")
    s.add_argument("--weekly", default="data/processed/weekly_final.csv")
    s.add_argument("--out", default="data/processed")
    s.set_defaults(fn=_cmd_icen)

    s = sub.add_parser("backtest", help="etapas 5-6: modelos y backtesting en dos condiciones")
    s.add_argument("--features", choices=list(exogenous.CONDITIONS), default="historial",
                   help="ablación: historial | clima | clima_icen (las dos últimas usan los hiperparámetros congelados de historial)")
    s.add_argument("--climate-csv", default="data/processed/climate_weekly.csv")
    s.add_argument("--icen-csv", default="data/processed/icen_weekly.csv")
    s.add_argument("--train-first-year", type=int, default=None,
                   help="primer año de entrenamiento (p. ej. 2014 si hay ICEN solo desde 2014); salida en data/processed/backtest_desdeAAAA. "
                        "Corre las tres condiciones con este mismo valor para que la comparación sea justa")
    s.add_argument("--example-exog", action="store_true", help="usa clima/ICEN SINTÉTICOS solo para probar la interfaz")
    s.add_argument("--weekly", default="data/processed/weekly_final.csv")
    s.add_argument("--triangle", default="data/processed/triangle_simulated.csv.gz")
    s.add_argument("--models", default=None, help="lista separada por comas (por defecto los cuatro)")
    s.add_argument("--out", default="data/processed/backtest")
    s.set_defaults(fn=_cmd_backtest)

    a = p.parse_args(argv)
    return a.fn(a, load_config(a.config))
