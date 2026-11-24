"""Pruebas de la evaluación de alertas (datos SINTÉTICOS hechos a mano)."""
import numpy as np
import pandas as pd
import pytest

from dengue_pipeline import alerts_eval, metrics

CFG = {"alerts": {"yellow_per_100k": 50, "red_per_100k": 150, "sensitivity_cuts_per_100k": [100, 300]}}
DIF = "_dif_disponibles_menos_final"
POP = pd.DataFrame({"unit": ["A", "B"], "year": [2023, 2023], "population": [100_000, 100_000]})   # tasa = casos acumulados


def _pred(obs, pred, unit="A", cond="final", model="m", h=2, year=2023):
    n = len(obs)
    tw = np.arange(100, 100 + n)
    return pd.DataFrame({"condition": cond, "model": model, "horizon": h, "unit": unit, "origin_week": tw - h,
                         "target_week": tw, "year": year, "pred": pred, "obs": obs})


def test_niveles_con_poblacion_y_unidad_sin_poblacion_sale():
    p = pd.concat([_pred([10, 60, 200], [49, 50, 150]), _pred([1, 1, 1], [1, 1, 1], unit="LIMA_NORTE")])
    d = alerts_eval.attach_levels(p, POP, CFG)
    assert set(d["unit"]) == {"A"}                                   # Lima Norte no tiene población: fuera
    assert list(d["lvl_obs_base"]) == [0, 1, 2] and list(d["lvl_pred_base"]) == [0, 1, 2]
    assert list(d["lvl_obs_sens"]) == [0, 0, 1]                      # cortes 100 / 300


def test_conteos_coinciden_con_alert_metrics():
    rng = np.random.default_rng(1)
    obs, pred = rng.integers(0, 300, 80), rng.integers(0, 300, 80)
    d = alerts_eval.attach_levels(_pred(obs, pred), POP, CFG)
    wc = alerts_eval._week_counts(d, "base", 1)
    ref = metrics.alert_metrics(d["lvl_pred_base"].to_numpy(), d["lvl_obs_base"].to_numpy(), 1)
    assert (int(wc.tp.sum()), int(wc.fp.sum()), int(wc.fn.sum()), int(wc.tn.sum())) == (ref["tp"], ref["fp"], ref["fn"], ref["tn"])
    tab = alerts_eval.alert_metrics_table(d, 4, 200, 0.95, 1)
    r = tab[(tab.cortes == "base") & (tab.umbral == "amarillo_o_mas") & (tab.agrupacion == "todo")].iloc[0]
    assert r["sensibilidad"] == pytest.approx(ref["sensibilidad"]) and r["tasa_falsa_alarma"] == pytest.approx(ref["tasa_falsa_alarma"])
    assert r["falsas_alarmas_sobre_alertas"] == pytest.approx(ref["falsas_alarmas_sobre_alertas"])
    assert r["sensibilidad_lo"] <= r["sensibilidad"] <= r["sensibilidad_hi"]


def test_bootstrap_de_razon():
    est, lo, hi = metrics.block_bootstrap_ratio(np.ones(20), np.ones(20), 4, 200, 0.95, 3)
    assert (est, lo, hi) == (1.0, 1.0, 1.0)
    assert all(np.isnan(metrics.block_bootstrap_ratio(np.zeros(5), np.zeros(5), 2, 50, 0.95, 3)))   # denominador 0
    a = metrics.block_bootstrap_ratio(np.arange(30) % 3 == 0, np.ones(30), 5, 300, 0.9, 7)
    assert a == metrics.block_bootstrap_ratio(np.arange(30) % 3 == 0, np.ones(30), 5, 300, 0.9, 7)   # misma semilla, mismo resultado
    assert a[1] <= a[0] <= a[2]


def test_anticipacion_por_episodio_a_mano():
    # observado en alerta (>= 50) en las semanas objetivo índice 5,6,7 y 12; pronóstico en alerta en 3,4,5,6
    obs = np.zeros(15); obs[[5, 6, 7, 12]] = 100
    pred = np.zeros(15); pred[[3, 4, 5, 6]] = 100
    d = alerts_eval.attach_levels(_pred(obs, pred, h=2), POP, CFG)
    ep = alerts_eval.episodes_table(d)
    e = ep[(ep.cortes == "base") & (ep.umbral == "amarillo_o_mas")].sort_values("target_week_inicio")
    assert list(e["target_week_inicio"]) == [105, 112]
    assert e["lead_runs"].iloc[0] == 2 and e["lead_total"].iloc[0] == 4          # alertas en 103 y 104 antes del inicio; + h = 2
    assert not e["detectado"].iloc[1] and np.isnan(e["lead_total"].iloc[1])      # no se alertó en 112
    assert not e["censurado_izq"].any()


def test_episodio_en_curso_en_la_primera_semana_se_excluye_del_resumen():
    obs = np.r_[100, 100, np.zeros(10)]
    d = alerts_eval.attach_levels(_pred(obs, obs), POP, CFG)
    ep = alerts_eval.episodes_table(d)
    assert ep["censurado_izq"].all()
    assert alerts_eval.episode_summary(ep).empty


def test_semanas_no_consecutivas_falla():
    p = _pred([10] * 6, [10] * 6).drop(index=3)
    with pytest.raises(ValueError, match="consecutivas"):
        alerts_eval.episodes_table(alerts_eval.attach_levels(p, POP, CFG))


def test_diferencia_pareada_cero_si_las_condiciones_son_iguales():
    rng = np.random.default_rng(5)
    obs, pred = rng.integers(0, 300, 60), rng.integers(0, 300, 60)
    p = pd.concat([_pred(obs, pred, cond="final"), _pred(obs, pred, cond="disponibles")])
    t = alerts_eval.alert_paired_difference(alerts_eval.attach_levels(p, POP, CFG), 4, 200, 0.95, 1)
    t = t.dropna(subset=["sensibilidad" + DIF])        # sin episodios en rojo con cortes 100/300 (obs < 300): no hay razón que calcular
    assert len(t) == 3
    assert (t["sensibilidad" + DIF].abs() < 1e-12).all() and (t["tasa_falsa_alarma" + DIF + "_lo"].abs() < 1e-12).all()


def test_diferencia_pareada_detecta_peor_desempeno():
    obs = np.array([100.0, 100.0, 0, 0] * 15)
    good, bad = obs.copy(), np.zeros_like(obs)                         # "final" acierta; "disponibles" nunca alerta
    p = pd.concat([_pred(obs, good, cond="final"), _pred(obs, bad, cond="disponibles")])
    t = alerts_eval.alert_paired_difference(alerts_eval.attach_levels(p, POP, CFG), 4, 200, 0.95, 1)
    r = t[(t.cortes == "base") & (t.umbral == "amarillo_o_mas")].iloc[0]
    assert r["sensibilidad" + DIF] == -1.0 and r["sensibilidad" + DIF + "_hi"] == -1.0   # disponibles menos final: nunca alerta, así que -1
    assert r["tasa_falsa_alarma" + DIF] == 0.0


def test_columnas_de_diferencia_pareada_llevan_el_signo_en_el_nombre():
    obs = np.array([100.0, 0] * 20)
    p = pd.concat([_pred(obs, obs, cond="final"), _pred(obs, obs, cond="disponibles")])
    t = alerts_eval.alert_paired_difference(alerts_eval.attach_levels(p, POP, CFG), 4, 50, 0.95, 1)
    dif = [c for c in t.columns if "_dif" in c]
    assert sorted(dif) == sorted(f"{m}{DIF}{s}" for m in ("sensibilidad", "tasa_falsa_alarma") for s in ("", "_lo", "_hi"))
    assert not any(c.endswith(("_dif", "_dif_lo", "_dif_hi")) for c in t.columns)    # los nombres ambiguos ya no existen
