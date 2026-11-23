"""Evaluación de alertas sobre predictions.csv.gz: sensibilidad, falsas alarmas y anticipación por episodio.

Qué se compara. El nivel observado sale de la incidencia observada (casos acumulados de 4 semanas que terminan en la
semana objetivo, serie final, por 100 000 habitantes); el nivel pronosticado, de la incidencia pronosticada. Alerta =
nivel >= umbral de nivel: 1 (amarillo o más) o 2 (solo rojo). Esquemas de cortes: "base" (50/150) y "sens" (100/300).
Los cortes por 100 000 vienen de config.yaml; el esquema en tres niveles es [INFERENCIA] del equipo.

Incertidumbre. Bootstrap por bloques de semanas OBJETIVO consecutivas (conteos tp/fp/fn/tn sumados entre unidades en cada
semana). Por unidad se remuestrean solo las semanas de esa unidad. La anticipación por episodio NO lleva intervalo: son
pocos episodios y no independientes; se informan uno a uno.

Anticipación. `lead_runs` = semanas consecutivas de alerta pronosticada que terminan en el inicio del episodio observado
(definición de metrics.episode_lead_weeks). Como el pronóstico de la semana objetivo s se emite en s - h, la anticipación
total en semanas es `lead_total` = lead_runs + h. Un episodio que ya está en curso en la primera semana evaluada
(censurado por la izquierda) se marca y se excluye del resumen.

Unidades sin población (Lima Norte) quedan fuera: no tienen tasa [FALTA].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics

NOTE = "corrida de prueba: retraso PROVISIONAL (Piura simulado); no citar como resultado"


def attach_levels(pred: pd.DataFrame, pop: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Agrega pop, tasas por 100 000 (pronosticada y observada) y niveles en los dos esquemas de cortes."""
    a = cfg["alerts"]
    schemes = {"base": (a["yellow_per_100k"], a["red_per_100k"]),
               "sens": tuple(a["sensitivity_cuts_per_100k"])}
    d = pred.merge(pop[["unit", "year", "population"]], on=["unit", "year"], how="inner")   # year = año de la semana objetivo
    d["rate_pred"] = metrics.incidence_per_100k(d["pred"].to_numpy(), d["population"].to_numpy())
    d["rate_obs"] = metrics.incidence_per_100k(d["obs"].to_numpy(), d["population"].to_numpy())
    for name, (y, r) in schemes.items():
        d[f"lvl_pred_{name}"] = metrics.alert_level(d["rate_pred"].to_numpy(), y, r)
        d[f"lvl_obs_{name}"] = metrics.alert_level(d["rate_obs"].to_numpy(), y, r)
    return d.sort_values(["condition", "model", "horizon", "unit", "target_week"]).reset_index(drop=True)


def _week_counts(g: pd.DataFrame, scheme: str, thr: int) -> pd.DataFrame:
    p, o = g[f"lvl_pred_{scheme}"] >= thr, g[f"lvl_obs_{scheme}"] >= thr
    c = pd.DataFrame({"target_week": g["target_week"], "tp": (p & o).astype(int), "fp": (p & ~o).astype(int),
                      "fn": (~p & o).astype(int), "tn": (~p & ~o).astype(int)})
    return c.groupby("target_week").sum().sort_index()


def _groups(g: pd.DataFrame):
    yield "todo", "todo", g
    for y, s in g.groupby("year"):
        yield "año", str(int(y)), s
    for u, s in g.groupby("unit"):
        yield "unidad", u, s


RATIOS = {"sensibilidad": ("tp", ("tp", "fn")), "tasa_falsa_alarma": ("fp", ("fp", "tn")),
          "falsas_alarmas_sobre_alertas": ("fp", ("fp", "tp"))}


def alert_metrics_table(d: pd.DataFrame, block: int, n: int, ci: float, seed: int) -> pd.DataFrame:
    rows = []
    for (cond, model, h), g in d.groupby(["condition", "model", "horizon"]):
        for scheme in ("base", "sens"):
            for thr, thr_name in ((1, "amarillo_o_mas"), (2, "rojo")):
                for kind, label, sub in _groups(g):
                    wc = _week_counts(sub, scheme, thr)
                    row = {"condition": cond, "model": model, "horizon": h, "cortes": scheme, "umbral": thr_name,
                           "agrupacion": kind, "grupo": label, "semanas": len(wc),
                           **{k: int(wc[k].sum()) for k in ("tp", "fp", "fn", "tn")}, "nota": NOTE}
                    for name, (num, dens) in RATIOS.items():
                        est, lo, hi = metrics.block_bootstrap_ratio(wc[num].to_numpy(), wc[list(dens)].sum(axis=1).to_numpy(),
                                                                    block, n, ci, seed)
                        row[name], row[name + "_lo"], row[name + "_hi"] = est, lo, hi
                    rows.append(row)
    return pd.DataFrame(rows)


def alert_paired_difference(d: pd.DataFrame, block: int, n: int, ci: float, seed: int) -> pd.DataFrame:
    """Diferencia (datos disponibles - serie final) en sensibilidad y tasa de falsa alarma, con IC por bloques pareado.

    Se remuestrean las MISMAS semanas en las dos condiciones. Positivo en sensibilidad = mejor con datos disponibles.
    """
    rows = []
    for (model, h), g in d.groupby(["model", "horizon"]):
        for scheme in ("base", "sens"):
            for thr, thr_name in ((1, "amarillo_o_mas"), (2, "rojo")):
                wa = _week_counts(g[g["condition"] == "disponibles"], scheme, thr)
                wf = _week_counts(g[g["condition"] == "final"], scheme, thr)
                wk = wa.index.intersection(wf.index)
                wa, wf = wa.loc[wk], wf.loc[wk]
                row = {"model": model, "horizon": h, "cortes": scheme, "umbral": thr_name, "semanas": len(wk), "nota": NOTE}
                for name, (num, dens) in RATIOS.items():
                    if name == "falsas_alarmas_sobre_alertas":
                        continue
                    row[name + "_dif"], row[name + "_dif_lo"], row[name + "_dif_hi"] = _paired_ratio_diff(
                        wa, wf, num, list(dens), block, n, ci, seed)
                rows.append(row)
    return pd.DataFrame(rows)


def _paired_ratio_diff(wa, wf, num, dens, block, n, ci, seed):
    m = len(wa)
    if m == 0:
        return (np.nan, np.nan, np.nan)
    a_n, a_d = wa[num].to_numpy(float), wa[dens].sum(axis=1).to_numpy(float)
    f_n, f_d = wf[num].to_numpy(float), wf[dens].sum(axis=1).to_numpy(float)
    if a_d.sum() == 0 or f_d.sum() == 0:
        return (np.nan, np.nan, np.nan)
    block = max(1, min(block, m))
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(m / block))
    starts = rng.integers(0, m - block + 1, size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :m]
    da, df_ = a_d[idx].sum(1), f_d[idx].sum(1)
    with np.errstate(all="ignore"):
        est = np.where((da > 0) & (df_ > 0), a_n[idx].sum(1) / da - f_n[idx].sum(1) / df_, np.nan)
    est = est[np.isfinite(est)]
    point = float(a_n.sum() / a_d.sum() - f_n.sum() / f_d.sum())
    if len(est) == 0:
        return (point, np.nan, np.nan)
    lo, hi = np.quantile(est, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return (point, float(lo), float(hi))


def episodes_table(d: pd.DataFrame, max_back: int = 8) -> pd.DataFrame:
    """Un renglón por episodio observado, por (condición, modelo, horizonte, cortes, umbral, unidad)."""
    rows = []
    for (cond, model, h, unit), g in d.groupby(["condition", "model", "horizon", "unit"]):
        g = g.sort_values("target_week")
        tw = g["target_week"].to_numpy()
        if len(tw) > 1 and (np.diff(tw) != 1).any():
            raise ValueError("semanas objetivo no consecutivas: no se puede medir la anticipación")
        for scheme in ("base", "sens"):
            for thr, thr_name in ((1, "amarillo_o_mas"), (2, "rojo")):
                p, o = (g[f"lvl_pred_{scheme}"] >= thr).to_numpy(), (g[f"lvl_obs_{scheme}"] >= thr).to_numpy()
                starts = np.flatnonzero(o & ~np.r_[False, o[:-1]])
                leads = metrics.episode_lead_weeks(p, o, max_back)
                for s, k in zip(starts, leads):
                    rows.append({"condition": cond, "model": model, "horizon": h, "cortes": scheme, "umbral": thr_name,
                                 "unit": unit, "target_week_inicio": int(tw[s]), "year": int(g["year"].iloc[s]),
                                 "censurado_izq": bool(s == 0), "detectado": bool(np.isfinite(k)),
                                 "lead_runs": k, "lead_total": k + h if np.isfinite(k) else np.nan, "nota": NOTE})
    return pd.DataFrame(rows)


def episode_summary(ep: pd.DataFrame) -> pd.DataFrame:
    e = ep[~ep["censurado_izq"]]
    keys = ["condition", "model", "horizon", "cortes", "umbral"]
    out = e.groupby(keys).agg(episodios=("detectado", "size"), detectados=("detectado", "sum"),
                              lead_total_mediana=("lead_total", "median"), lead_total_min=("lead_total", "min"),
                              lead_total_max=("lead_total", "max")).reset_index()
    out["nota"] = NOTE
    return out
