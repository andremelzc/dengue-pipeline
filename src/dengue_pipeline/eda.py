"""Exploración de datos (EDA): series por unidad, estacionalidad, años de brote, ceros, autocorrelación y retraso.

Genera figuras PNG y tablas CSV en reports/. Todo sale de la serie final (weekly_final.csv), de la incidencia
(weekly_incidence.csv, necesita población: Lima Norte queda sin tasa) y de delay_pmf.csv (retraso de notificación, patrón base Piura 2025).
Nada aquí es un resultado de desempeño de modelos.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]      # orden fijo (palette.md); etiquetas directas
LEVEL_COLORS = {0: "#c9c8c2", 1: "#eda100", 2: "#e34948"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _fig(nrows, ncols, w, h, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), facecolor=SURFACE, **kw)
    for ax in np.atleast_1d(axes).ravel():
        _style(ax)
    return fig, axes


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    x = np.asarray(x, float)
    x = x - x.mean()
    d = (x * x).sum()
    return np.array([1.0] + [float((x[:-k] * x[k:]).sum() / d) for k in range(1, nlags + 1)])


def _units(weekly):
    return list(weekly["unit"].drop_duplicates())


def fig_series(weekly, out):
    us = _units(weekly)
    fig, axes = _fig(len(us), 1, 10, 1.9 * len(us), sharex=True)
    for ax, u, c in zip(axes, us, COLORS):
        g = weekly[weekly.unit == u]
        ax.plot(pd.to_datetime(g["week_start"]), g["cases"], color=c, lw=0.9)
        ax.set_ylabel(u.replace("_", " "), color=INK, fontsize=8, rotation=0, ha="right", va="center")
        ax.set_ylim(bottom=0)
    axes[0].set_title("Casos semanales por unidad (serie final, inicio de síntomas; cada panel con su propia escala)", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "eda_01_series_por_unidad.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_seasonality(weekly, out):
    us = _units(weekly)
    fig, axes = _fig(2, 3, 11, 5.2, sharex=True)
    rows = []
    for ax, u, c in zip(axes.ravel(), us, COLORS):
        g = weekly[(weekly.unit == u) & (weekly.week <= 52)]
        tot = g.groupby("year")["cases"].transform("sum")
        g = g.assign(share=np.where(tot > 0, g["cases"] / tot.where(tot > 0, 1), np.nan))
        piv = g.pivot(index="week", columns="year", values="share").dropna(axis=1, how="all")   # años sin casos no tienen estacionalidad
        med, q1, q3 = piv.median(axis=1), piv.quantile(0.25, axis=1), piv.quantile(0.75, axis=1)
        ax.fill_between(med.index, q1, q3, color=c, alpha=0.18, lw=0)
        ax.plot(med.index, med, color=c, lw=1.4)
        ax.set_title(u.replace("_", " "), fontsize=9, color=INK, loc="left")
        rows.append({"unit": u, "semana_pico_mediana": int(med.idxmax()), "semana_pico_q1": int(piv.idxmax().quantile(0.25)),
                     "semana_pico_q3": int(piv.idxmax().quantile(0.75))})
    for ax in axes[-1]:
        ax.set_xlabel("semana epidemiológica", fontsize=8, color=INK2)
    fig.suptitle("Estacionalidad: proporción del total anual por semana (mediana de años y rango intercuartil)", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "eda_02_estacionalidad.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return pd.DataFrame(rows)


def fig_outbreak_years(inc, out):
    d = inc[inc["population"].notna()]
    mx = d.groupby(["unit", "year"])["inc4w_per100k"].max().unstack("year")
    fig, ax = _fig(1, 1, 10, 3.4)
    ax.grid(False)
    # color: escala secuencial de un solo tono sobre log10(1 + tasa máxima); los cortes 50 y 150 se anotan en el número
    im = ax.imshow(np.log10(1 + mx.to_numpy()), aspect="auto", cmap="Blues", vmin=0, vmax=np.log10(1 + 1500))
    ax.set_xticks(range(mx.shape[1]), [str(c) for c in mx.columns], fontsize=8, color=INK2)
    ax.set_yticks(range(mx.shape[0]), [u.replace("_", " ") for u in mx.index], fontsize=8, color=INK)
    for i in range(mx.shape[0]):
        for j in range(mx.shape[1]):
            v = mx.iat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if np.log10(1 + v) > 1.9 else INK,
                    fontweight="bold" if v >= 150 else "normal")
    ax.set_title("Tasa máxima de la incidencia acumulada de 4 semanas por año (por 100 000). En negrita: >= 150 (rojo). Lima Norte sin población", loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(out / "eda_03_anios_de_brote.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return mx


def fig_zeros(weekly, out):
    z = weekly.assign(zero=weekly["cases"] == 0)
    byu = z.groupby("unit")["zero"].mean().reindex(_units(weekly))
    fig, ax = _fig(1, 1, 7, 3.2)
    ax.bar([u.replace("_", " ") for u in byu.index], byu.to_numpy() * 100, color="#2a78d6", width=0.55)
    for i, v in enumerate(byu.to_numpy() * 100):
        ax.text(i, v + 1, f"{v:.0f} %", ha="center", fontsize=8, color=INK)
    ax.set_ylabel("% de semanas con 0 casos", fontsize=8, color=INK2)
    ax.set_title("Semanas sin casos por unidad (2010-2024)", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "eda_04_ceros.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return byu


def fig_acf(weekly, out, nlags=26):
    fig, ax = _fig(1, 1, 8, 4)
    rows = []
    for u, c in zip(_units(weekly), COLORS):
        a = acf(np.log1p(weekly[weekly.unit == u]["cases"].to_numpy()), nlags)
        ax.plot(range(nlags + 1), a, color=c, lw=1.2)
        ax.text(nlags + 0.4, a[-1], u.replace("_", " "), color=INK, fontsize=7, va="center")
        rows.append({"unit": u, **{f"acf_lag{k}": a[k] for k in (1, 2, 4, 8, 13, 26)}})
    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_xlim(0, nlags + 6)
    ax.set_xlabel("rezago (semanas)", fontsize=8, color=INK2)
    ax.set_title("Autocorrelación de log(1 + casos semanales)", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "eda_05_autocorrelacion.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return pd.DataFrame(rows)


def fig_delay(pmfs, out, subtitle="últimas 4 semanas, retrasos negativos y mayores de 26 semanas excluidos"):
    """Un panel por patrón de retraso (misma escala): barras = proporción en cada retraso, línea = acumulada.

    `pmfs` es {nombre: DataFrame(delay_weeks, pmf, cum)} o un solo DataFrame (se dibuja como un patrón).
    """
    if isinstance(pmfs, pd.DataFrame):
        pmfs = {"piura2025": pmfs}
    n = len(pmfs)
    fig, axes = _fig(1, n, 3.6 * n + 0.6, 3.8, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, pmf), c in zip(axes, pmfs.items(), COLORS):
        ax.bar(pmf["delay_weeks"], pmf["pmf"], color=c, width=0.7, label="proporción")
        ax.plot(pmf["delay_weeks"], pmf["cum"], color=INK2, lw=1.2, label="acumulada")
        ax.set_ylim(0, 1.02)
        ax.set_title(name, loc="left", fontsize=9, color=INK)
        ax.set_xlabel("retraso de notificación (semanas)", fontsize=8, color=INK2)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="center right")
    fig.suptitle(f"Retraso de notificación por patrón ({subtitle})", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "eda_06_retraso.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def run_eda(weekly: pd.DataFrame, inc: pd.DataFrame, pmf: pd.DataFrame, out_dir: str | Path,
            scenario_pmfs: dict | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_series(weekly, out)
    season = fig_seasonality(weekly, out)
    mx = fig_outbreak_years(inc, out)
    zeros = fig_zeros(weekly, out)
    ac = fig_acf(weekly, out)
    fig_delay(scenario_pmfs or pmf, out)

    annual = weekly.groupby(["unit", "year"])["cases"].sum().unstack("year")
    d = inc[inc["population"].notna()]
    lvl = d.groupby("unit")["level"].apply(lambda s: s[s >= 0].value_counts(normalize=True)).unstack(fill_value=0)
    lvl.columns = ["share_verde", "share_amarillo", "share_rojo"]
    lvl_sens = d.groupby("unit")["level_sens"].apply(lambda s: s[s >= 0].value_counts(normalize=True)).unstack(fill_value=0)
    lvl_sens.columns = ["share_verde_sens", "share_amarillo_sens", "share_rojo_sens"]
    scale = (annual.max(axis=1) / annual.median(axis=1)).rename("cociente_max_mediana_anual")
    stats = (pd.DataFrame({"casos_totales": annual.sum(axis=1), "semana_max_casos": weekly.groupby("unit")["cases"].max(),
                           "pct_semanas_cero": zeros * 100}).join(season.set_index("unit")).join(ac.set_index("unit"))
             .join(lvl).join(lvl_sens).join(scale))
    stats.to_csv(out / "eda_estadisticas_por_unidad.csv")
    annual.to_csv(out / "eda_casos_anuales.csv")
    mx.round(1).to_csv(out / "eda_tasa_maxima_4s_por_anio.csv")
    pmf = pmf.copy()
    summary = {"delay_share_0": float(pmf["cum"].iloc[0]), "delay_share_le1": float(pmf["cum"].iloc[1]),
               "delay_share_le2": float(pmf["cum"].iloc[2]), "delay_share_le4": float(pmf["cum"].iloc[4]),
               "delay_mean_weeks": float((pmf["delay_weeks"] * pmf["pmf"]).sum() / pmf["pmf"].sum()),
               "delay_max_weeks_modelled": int(pmf["delay_weeks"].max())}
    pd.Series(summary).to_csv(out / "eda_retraso_resumen.csv", header=["valor"])
    return {"stats": stats, "annual": annual, "maxrate": mx, "delay": summary}
