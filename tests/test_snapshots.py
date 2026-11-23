"""La prueba clave del artículo: ninguna instantánea contiene información posterior a su semana."""
import numpy as np
import pandas as pd
import pytest

from dengue_pipeline import snapshots as sn
from dengue_pipeline.epiweek import abs_week_of_date
from helpers import fmt, regional_rows


def tri_random(seed=0, n_weeks=60, max_delay=8):
    rng = np.random.default_rng(seed)
    rows = []
    for unit in ("A", "B"):
        for w in range(1000, 1000 + n_weeks):
            for d in range(max_delay + 1):
                rows.append((unit, w, w + d, int(rng.integers(0, 20))))
    return pd.DataFrame(rows, columns=sn.TRI_COLS)


def test_regla_de_disponibilidad_por_fecha_de_notificacion():
    # inicio el domingo 5-ene-2025; notificación el miércoles 15-ene (semana siguiente a la siguiente)
    reg = regional_rows([dict(FECHA_INICIO_SINTOMAS="20250105", FECHA_NOTIFICACION="20250115")])
    tri = sn.regional_triangle(reg, "20260702", "PIURA")
    onset_w, notif_w = abs_week_of_date(pd.Timestamp("2025-01-05")), abs_week_of_date(pd.Timestamp("2025-01-15"))
    assert tri.iloc[0].onset_week == onset_w and tri.iloc[0].avail_week == notif_w
    assert sn.snapshot(tri, notif_w - 1)["n"].sum() == 0     # la semana anterior a la notificación no lo ve
    assert sn.snapshot(tri, notif_w)["n"].sum() == 1          # en la semana de notificación ya está


def test_caso_notificado_antes_del_inicio_no_se_cuenta_antes_de_sus_sintomas():
    reg = regional_rows([dict(FECHA_INICIO_SINTOMAS="20250120", FECHA_NOTIFICACION="20250105")])
    tri = sn.regional_triangle(reg, "20260702", "PIURA")
    onset_w = abs_week_of_date(pd.Timestamp("2025-01-20"))
    assert tri.iloc[0].avail_week == onset_w
    assert sn.snapshot(tri, onset_w - 1)["n"].sum() == 0


@pytest.mark.parametrize("seed", range(5))
def test_instantanea_no_cambia_si_se_altera_el_futuro(seed):
    tri = tri_random(seed)
    for t in (1005, 1030, 1059):
        before = sn.snapshot(tri, t)
        future = tri["avail_week"] > t
        tri2 = tri.copy()
        tri2.loc[future, "n"] = tri2.loc[future, "n"] + 999          # se altera todo lo que llega después de t
        extra = pd.DataFrame([("A", 1000, t + 1, 5000), ("C", t, t + 3, 7)], columns=sn.TRI_COLS)
        tri2 = pd.concat([tri2, extra], ignore_index=True)           # y se agregan casos que llegan después de t
        after = sn.snapshot(tri2, t)
        pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))
        sn.assert_no_future(tri2, after, t)


def test_detecta_una_instantanea_con_fuga_de_futuro():
    tri = tri_random()
    t = 1020
    snap = sn.snapshot(tri, t)
    leaky = pd.concat([snap, pd.DataFrame([("A", t + 2, 1)], columns=["unit", "onset_week", "n"])])
    with pytest.raises(AssertionError):
        sn.assert_no_future(tri, leaky, t)
    wrong_total = snap.copy()
    wrong_total.loc[wrong_total.index[0], "n"] += 1
    with pytest.raises(AssertionError):
        sn.assert_no_future(tri, wrong_total, t)


def test_instantanea_es_monotona_en_el_tiempo():
    tri = tri_random()
    totals = [sn.snapshot(tri, t)["n"].sum() for t in range(1000, 1070)]
    assert all(b >= a for a, b in zip(totals, totals[1:]))


def test_simulacion_conserva_totales_y_es_reproducible():
    final = pd.DataFrame({"unit": ["A"] * 4 + ["B"] * 4, "abs_week": list(range(100, 104)) * 2,
                          "cases": [0, 10, 200, 3, 7, 0, 55, 1000]})
    pmf = np.array([0.5, 0.3, 0.15, 0.05])
    t1, t2, t3 = (sn.simulate_triangle(final, pmf, s) for s in (7, 7, 8))
    pd.testing.assert_frame_equal(t1, t2)
    assert not t1.equals(t3)
    by_cell = t1.groupby(["unit", "onset_week"])["n"].sum()
    for _, r in final[final.cases > 0].iterrows():
        assert by_cell[(r.unit, r.abs_week)] == r.cases
    assert (t1.avail_week - t1.onset_week).between(0, 3).all()


def test_simulacion_con_retraso_fijo():
    final = pd.DataFrame({"unit": ["A"], "abs_week": [100], "cases": [50]})
    tri = sn.simulate_triangle(final, np.array([0, 0, 1.0]), 1)
    assert list(tri.avail_week) == [102] and list(tri.n) == [50]
    assert sn.snapshot(tri, 101)["n"].sum() == 0 and sn.snapshot(tri, 102)["n"].sum() == 50


def _regional_with_known_delays(pmf, n_per_week=2000, first_onset="2025-01-05", weeks=20, seed=3):
    rng = np.random.default_rng(seed)
    rows, base = [], pd.Timestamp(first_onset)  # domingo
    for w in range(weeks):
        for k in rng.choice(len(pmf), size=n_per_week, p=pmf):
            onset = base + pd.Timedelta(weeks=w, days=1)
            notif = onset + pd.Timedelta(weeks=int(k), days=1)
            rows.append((fmt(onset), fmt(notif)))
    return regional_rows([dict(FECHA_INICIO_SINTOMAS=o, FECHA_NOTIFICACION=n) for o, n in rows])


def test_estimacion_del_retraso_recupera_la_distribucion_conocida():
    true = np.array([0.55, 0.25, 0.12, 0.05, 0.03])
    reg = _regional_with_known_delays(true)
    r = sn.delay_pmf(reg, "20260702", 2025, exclude_last_weeks=4, max_delay_weeks=8)
    assert np.allclose(r["pmf"][:5], true, atol=0.02) and r["pmf"][5:].sum() == 0
    assert r["pmf"].sum() == pytest.approx(1.0) and r["dropped_negative"] == 0


def test_retrasos_negativos_y_demasiado_largos_se_informan():
    reg = regional_rows([
        dict(FECHA_INICIO_SINTOMAS="20250210", FECHA_NOTIFICACION="20250205"),   # negativo
        dict(FECHA_INICIO_SINTOMAS="20250210", FECHA_NOTIFICACION="20250212"),   # 0 semanas
        dict(FECHA_INICIO_SINTOMAS="20250210", FECHA_NOTIFICACION="20250601"),   # demasiado largo
    ])
    r = sn.delay_pmf(reg, "20260702", 2025, exclude_last_weeks=4, max_delay_weeks=10)
    assert (r["dropped_negative"], r["dropped_over_max"], r["n_used"]) == (1, 1, 1)


def test_par_real_corte_vs_serie_final():
    w = abs_week_of_date(pd.Timestamp("2023-05-07"))
    reg = regional_rows([dict(FECHA_CORTE="20230517", FECHA_INICIO_SINTOMAS="20230508", FECHA_NOTIFICACION="20230510")] * 3)
    final = pd.DataFrame({"unit": ["PIURA"], "abs_week": [w], "cases": [12]})   # el lunes 8-may está en la semana de w
    out = sn.cut_vs_final(reg, "20230517", final, "PIURA", [w])
    assert (out.corte[0], out.final[0], out.razon[0]) == (3, 12, 0.25)
