"""Línea de comandos. Uso: python -m dengue_pipeline <comando> --help"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import (acquire, alerts_eval, compare_runs, delay_scenarios, eda, exogenous, icen_reports, integrate, population,
               reproduce, snapshots, validate)
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


def _write_pmf(r, path):
    pd.DataFrame({"delay_weeks": range(len(r["pmf"])), "pmf": r["pmf"], "cum": r["cum"]}).to_csv(path, index=False)


def _cmd_delay(a, cfg):
    reg = read_strict(a.regional)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = list(delay_scenarios.scenarios(cfg)) if a.all_scenarios else [a.scenario]
    for name in names:
        r = delay_scenarios.estimate(reg, cfg, name)
        _write_pmf(r, out / ("delay_pmf.csv" if name == delay_scenarios.BASE else f"delay_pmf_{name}.csv"))
        print(name, {k: v for k, v in r.items() if k not in ("pmf", "cum")})
    if a.all_scenarios:
        delay_scenarios.summary(reg, cfg).to_csv(out / "delay_scenarios_summary.csv", index=False)
        print("Resumen de escenarios: delay_scenarios_summary.csv")
    return 0


def _cmd_simulate(a, cfg):
    seed = cfg["delay"]["seed"] if a.seed is None else a.seed
    r = delay_scenarios.estimate(read_strict(a.regional), cfg, a.scenario)
    final = integrate.weekly_table(read_strict(a.national), cfg)
    tri = snapshots.simulate_triangle(final, r["pmf"], seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    is_main = a.scenario == delay_scenarios.BASE and seed == cfg["delay"]["seed"]
    name = "triangle_simulated.csv.gz" if is_main else f"triangle_simulated_{delay_scenarios.tag(a.scenario, seed)}.csv.gz"
    tri.to_csv(out / name, index=False)
    print(f"{len(tri)} filas del triángulo simulado ({a.scenario}, semilla {seed}) -> {name}")
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
    print(f"{len(d)} predicciones con tasa ({d['unit'].nunique()} unidades); {len(ep)} episodios")
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


def _exog_for(a, weekly, units, cfg):
    """(exog, prefijo de archivo) de la condición pedida; (None, "") para `historial`."""
    if a.features == "historial":
        return None, ""
    tag = ""
    if a.example_exog:
        print("AVISO: datos de EJEMPLO sintéticos; la salida NO es un resultado y se guarda con prefijo EJEMPLO_NO_REAL")
        climate = exogenous.make_example_climate(weekly)
        icen = exogenous.make_example_icen(weekly)
        tag = exogenous.EXAMPLE_FLAG + "_"
    else:
        climate = pd.read_csv(a.climate_csv) if Path(a.climate_csv).exists() else None
        icen = pd.read_csv(a.icen_csv) if Path(a.icen_csv).exists() else None
    return exogenous.build_exog(a.features, weekly, units, cfg, climate, icen), tag


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
    exog, tag = _exog_for(a, weekly, units, cfg)
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


def _cmd_delay_sensitivity(a, cfg):
    """Sensibilidad al retraso: cada escenario x semilla simula su propio triángulo y corre solo la condición `disponibles`.

    Reutiliza los hiperparámetros congelados de la corrida principal (hay que haber corrido antes `backtest` con el mismo
    --train-first-year). La serie final no depende del retraso, así que no se repite.
    """
    from . import backtest, metrics
    weekly = pd.read_csv(a.weekly)
    out = Path(a.out)
    models = a.models.split(",") if a.models else None
    units = [u["id"] for u in cfg["units"]]
    if a.train_first_year:
        cfg["backtest"]["train_first_year"] = a.train_first_year
        if a.out == "data/processed/backtest":
            out = Path(f"data/processed/backtest_desde{a.train_first_year}")
    hp = out / "hyperparameters.json"
    if not hp.exists():
        raise SystemExit(f"[FALTA] {hp}: corre antes `backtest` (condición historial) con el mismo --train-first-year")
    frozen = backtest.load_frozen_params(hp)
    exog, tag = _exog_for(a, weekly, units, cfg)
    reg = read_strict(a.regional)
    names = a.scenarios.split(",") if a.scenarios else list(delay_scenarios.scenarios(cfg))
    seeds = delay_scenarios.seeds(cfg)[: a.n_seeds] if a.n_seeds else delay_scenarios.seeds(cfg)
    suffix = "" if a.features == "historial" else f"_{a.features}"
    sdir = out / "sensibilidad_retraso"
    b = cfg["backtest"]["bootstrap"]
    rows = []
    for name in names:
        pmf = delay_scenarios.estimate(reg, cfg, name)["pmf"]
        for seed in seeds:
            tri = snapshots.simulate_triangle(weekly, pmf, seed)
            t = delay_scenarios.tag(name, seed)
            pred, info = backtest.run_backtest(weekly, tri, cfg, models=models, conditions=("disponibles",), out_dir=sdir,
                                               features=a.features, exog=exog, params_by_h=frozen,
                                               out_name=f"{tag}predictions{suffix}_{t}.csv.gz")
            summ = metrics.summarize(pred, b["block_weeks"], b["n_resamples"], b["ci"], cfg["backtest"]["seed"])
            summ.insert(0, "semilla", seed)
            summ.insert(0, "escenario", name)
            summ.insert(0, "features", a.features)
            rows.append(summ)
            print(f"[{t}] {len(pred)} predicciones; fallbacks NB: {len(info.get('negbin_fallbacks', []))}")
    pd.concat(rows, ignore_index=True).to_csv(sdir / f"{tag}metrics_summary{suffix}_sensibilidad_retraso.csv", index=False)
    print(f"Resumen: {sdir / (tag + 'metrics_summary' + suffix + '_sensibilidad_retraso.csv')}")
    return 0


def _cmd_reproduce(a, cfg):
    res = reproduce.verify_inputs({"national": a.national, "regional": a.regional}, a.manifest)
    for r in res:
        print(f"[{'OK' if r['ok'] else 'HASH DISTINTO'}] {r['kind']}: {r['file']}")
        print(f"    sha256   {r['sha256']}")
        print(f"    esperado {r['esperado']}")
    if not all(r["ok"] for r in res):
        if not a.allow_different_data:
            print("Los datos no son los del manifiesto: la comparación con otras corridas no tendría sentido. "
                  "Usa la misma copia de los CSV (el portal puede haberlos actualizado). "
                  "Para correr de todos modos: --allow-different-data.")
            return 2
        print("AVISO: se sigue con datos distintos a los del manifiesto (--allow-different-data).")
    steps = reproduce.build_steps(a.national, a.regional, a.population_xlsx, a.districts, a.climate_downloaded_on,
                                  a.skip_climate, not a.no_ablation, not a.no_sensitivity)
    only = a.only.split(",") if a.only else None
    steps = reproduce.select(steps, a.from_step, only)
    return reproduce.run(steps, dry_run=a.dry_run, log_path="data/processed/reproducir_log.json")


def _cmd_compare_runs(a, cfg):
    rows = compare_runs.compare_dirs(a.dir_a, a.dir_b, a.tol)
    print(compare_runs.format_report(rows))
    return 0 if rows and all(r["estado"] == "OK" for r in rows) else 1


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
    s.add_argument("--scenario", default=delay_scenarios.BASE, help="escenario del retraso (por defecto el base, piura2025)")
    s.add_argument("--all-scenarios", action="store_true", help="escribe los tres escenarios y delay_scenarios_summary.csv")
    s.add_argument("--out", default="data/processed")
    s.set_defaults(fn=_cmd_delay)

    s = sub.add_parser("simulate", help="etapa 4: triángulo simulado de datos disponibles")
    s.add_argument("--national", required=True)
    s.add_argument("--regional", required=True)
    s.add_argument("--scenario", default=delay_scenarios.BASE, help="escenario del retraso (por defecto el base, piura2025)")
    s.add_argument("--seed", type=int, default=None, help="semilla (por defecto delay.seed de config.yaml)")
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

    s = sub.add_parser("delay-sensitivity",
                       help="sensibilidad al retraso: escenarios (piura2025, piura2026, piura2023) x semillas, solo condición disponibles")
    s.add_argument("--regional", required=True)
    s.add_argument("--features", choices=list(exogenous.CONDITIONS), default="historial")
    s.add_argument("--climate-csv", default="data/processed/climate_weekly.csv")
    s.add_argument("--icen-csv", default="data/processed/icen_weekly.csv")
    s.add_argument("--train-first-year", type=int, default=None, help="igual que en backtest (la misma que usó la corrida principal)")
    s.add_argument("--example-exog", action="store_true", help="usa clima/ICEN SINTÉTICOS solo para probar la interfaz")
    s.add_argument("--weekly", default="data/processed/weekly_final.csv")
    s.add_argument("--scenarios", default=None, help="lista separada por comas (por defecto todos)")
    s.add_argument("--n-seeds", type=int, default=None, help="solo las primeras N semillas (por defecto todas)")
    s.add_argument("--models", default=None, help="lista separada por comas (por defecto los cuatro)")
    s.add_argument("--out", default="data/processed/backtest")
    s.set_defaults(fn=_cmd_delay_sensitivity)

    s = sub.add_parser("reproduce", help="corre TODO el pipeline con un solo comando, tras comprobar los hashes de los CSV")
    s.add_argument("--national", required=True)
    s.add_argument("--regional", required=True)
    s.add_argument("--population-xlsx", required=True, help="cuadro departamental INEI (proy_03_4.xlsx)")
    s.add_argument("--districts", default=None, help="cuadro distrital INEI 2018-2026 (Lima Norte)")
    s.add_argument("--manifest", default="data/manifest.json")
    s.add_argument("--climate-downloaded-on", default=None, help="AAAA-MM-DD (por defecto, hoy)")
    s.add_argument("--skip-climate", action="store_true", help="no descarga el clima (ya existe data/processed/climate_weekly.csv)")
    s.add_argument("--no-sensitivity", action="store_true", help="omite la sensibilidad al retraso")
    s.add_argument("--no-ablation", action="store_true", help="omite la ablación desde 2014")
    s.add_argument("--from", dest="from_step", default=None, help="continúa desde este paso (tras un fallo o un corte)")
    s.add_argument("--only", default=None, help="solo estos pasos, separados por comas")
    s.add_argument("--dry-run", action="store_true", help="solo muestra los comandos")
    s.add_argument("--allow-different-data", action="store_true", help="sigue aunque el hash de los CSV no coincida con el manifiesto")
    s.set_defaults(fn=_cmd_reproduce)

    s = sub.add_parser("compare-runs", help="compara dos carpetas data/processed (dos corridas) con una tolerancia numérica")
    s.add_argument("dir_a")
    s.add_argument("dir_b")
    s.add_argument("--tol", type=float, default=1e-6, help="máxima diferencia numérica aceptada (por defecto 1e-6)")
    s.set_defaults(fn=_cmd_compare_runs)

    a = p.parse_args(argv)
    return a.fn(a, load_config(a.config))
