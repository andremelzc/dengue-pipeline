import numpy as np
import pandas as pd
import pytest

from dengue_pipeline import integrate
from dengue_pipeline.epiweek import weeks_in_year
from helpers import national_rows, tiny_cfg


def test_tabla_semanal_completa_y_con_ceros():
    nat = national_rows([
        {"ano": 2023, "semana": 3, "tipo_dx": "C"}, {"ano": 2023, "semana": 3, "tipo_dx": "P"},
        {"ano": 2023, "semana": 3, "tipo_dx": "C"}, {"ano": 2024, "semana": 10, "departamento": "TUMBES", "ubigeo": "240101"},
        {"ano": 2023, "semana": 4, "departamento": "LIMA", "ubigeo": "150106"},   # Carabayllo: Lima Norte
        {"ano": 2023, "semana": 4, "departamento": "LIMA", "ubigeo": "150101"},   # Lima Cercado: fuera
    ])
    t = integrate.weekly_table(nat, tiny_cfg())
    n_weeks = weeks_in_year(2023) + weeks_in_year(2024)
    assert len(t) == 3 * n_weeks                                   # 3 unidades x todas las semanas
    pi = t[(t.unit == "PIURA") & (t.year == 2023) & (t.week == 3)].iloc[0]
    assert (pi.cases_C, pi.cases_P, pi.cases) == (2, 1, 3)
    assert t[(t.unit == "PIURA") & (t.year == 2023) & (t.week == 5)].iloc[0].cases == 0   # semana sin registros = 0
    assert t[(t.unit == "LIMA_NORTE")].cases.sum() == 1                                  # solo el ubigeo de la lista
    assert t[(t.unit == "TUMBES") & (t.year == 2024) & (t.week == 10)].iloc[0].cases == 1
    assert t.cases.sum() == 5


def test_semana_inexistente_se_descarta_y_se_informa():
    nat = national_rows([{"ano": 2024, "semana": 53}, {"ano": 2024, "semana": 52}])
    t = integrate.weekly_table(nat, tiny_cfg())
    assert t.cases.sum() == 1 and t.attrs["dropped_cases_invalid_week"] == 1


def test_definicion_de_caso_solo_confirmados():
    nat = national_rows([{"tipo_dx": "C"}, {"tipo_dx": "P"}])
    t = integrate.weekly_table(nat, tiny_cfg(), case_types=["C"])
    assert t.cases.sum() == 1


def _serie():
    t = integrate.weekly_table(national_rows([{"ano": 2023, "semana": w} for w in range(1, 9) for _ in range(w)]), tiny_cfg())
    return t[t.unit == "PIURA"].reset_index(drop=True)


def test_rezagos_toman_el_pasado():
    t = _serie()
    out = integrate.add_lags(t, ["cases"], [1, 2])
    row = out[(out.year == 2023) & (out.week == 5)].iloc[0]
    assert row.cases_lag1 == 4 and row.cases_lag2 == 3


def test_rezagos_no_usan_el_futuro():
    t = _serie()
    a = integrate.add_lags(t, ["cases"], [1, 2, 3])
    T = int(t.loc[(t.year == 2023) & (t.week == 5), "abs_week"].iloc[0])
    t2 = t.copy()
    t2.loc[t2.abs_week > T, "cases"] = 10_000      # se altera todo lo posterior a la semana T
    b = integrate.add_lags(t2, ["cases"], [1, 2, 3])
    cols = [c for c in a.columns if "_lag" in c]
    # Hasta T+1 los rezagos solo miran semanas <= T, que no cambiaron: deben ser idénticos.
    pd.testing.assert_frame_equal(a[a.abs_week <= T + 1][cols], b[b.abs_week <= T + 1][cols])
    # Control de potencia: en T+2 el rezago 1 mira la semana T+1 (alterada), así que debe cambiar.
    assert a[a.abs_week == T + 2]["cases_lag1"].iloc[0] != b[b.abs_week == T + 2]["cases_lag1"].iloc[0]


def test_rezago_cero_o_negativo_esta_prohibido():
    with pytest.raises(ValueError):
        integrate.add_lags(_serie(), ["cases"], [0])
    with pytest.raises(ValueError):
        integrate.add_lags(_serie(), ["cases"], [-1])


def test_rezagos_exigen_semanas_consecutivas():
    t = _serie().drop(index=[3])
    with pytest.raises(ValueError):
        integrate.add_lags(t, ["cases"], [1])


def test_clima_semanal_y_semana_incompleta():
    days = pd.date_range("2023-01-01", "2023-01-17", freq="D")  # dom 1-ene a mar 17-ene
    daily = pd.DataFrame({"date": days, "T2M": 20.0, "RH2M": 80.0, "PRECTOTCORR": 1.0})
    w = integrate.daily_to_weekly(daily).sort_values("abs_week").reset_index(drop=True)
    assert list(w["complete"]) == [True, True, False]
    assert w.loc[0, "rain_sum"] == 7.0 and w.loc[0, "temp_mean"] == 20.0
    assert w.loc[2, ["temp_mean", "hum_mean", "rain_sum"]].isna().all()   # no se rellena en silencio
