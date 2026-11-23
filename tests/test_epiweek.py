from datetime import date, timedelta

import pandas as pd
import pytest

from dengue_pipeline.epiweek import (abs_week, abs_week_of_date, abs_week_of_dates, abs_week_start, epiweek,
                                     epiweek_of_dates, week_end, week_start, week_table, weeks_in_year)


def test_fechas_conocidas():
    assert epiweek(date(2026, 1, 4)) == (2026, 1)      # el 1-ene-2026 es jueves: la semana 1 empieza el 4
    assert epiweek(date(2026, 1, 1)) == (2025, 53)     # y esa semana pertenece al 2025
    assert epiweek(date(2026, 6, 30)) == (2026, 26)
    assert epiweek(date(2020, 12, 31)) == (2020, 53)
    assert epiweek(date(2021, 1, 2)) == (2020, 53)
    assert epiweek(date(2021, 1, 3)) == (2021, 1)
    assert epiweek(date(2023, 5, 17)) == (2023, 20)


def test_años_con_53_semanas_segun_los_datos_reales():
    # Máximo de `semana` observado en el CSV nacional: 2014, 2020 y 2025 llegan a 53; 2024 a 52.
    assert [weeks_in_year(y) for y in (2014, 2020, 2025)] == [53, 53, 53]
    assert weeks_in_year(2024) == 52


def test_regla_alternativa_la_semana_1_contiene_el_4_de_enero():
    # Formulación independiente: la semana 1 (domingo a sábado) es la que contiene el 4 de enero.
    for y in range(1995, 2045):
        jan4 = date(y, 1, 4)
        assert week_start(y, 1) == jan4 - timedelta(days=(jan4.weekday() + 1) % 7)


def test_ida_y_vuelta_y_continuidad_2000_2030():
    d, prev_abs = date(2000, 1, 1), None
    while d <= date(2030, 12, 31):
        y, w = epiweek(d)
        assert 1 <= w <= weeks_in_year(y)
        assert week_start(y, w) <= d <= week_end(y, w)
        assert week_start(y, w).weekday() == 6  # domingo
        a = abs_week_of_date(d)
        assert a == abs_week(y, w) and abs_week_start(a) == week_start(y, w)
        if prev_abs is not None:
            assert a - prev_abs in (0, 1)  # el índice absoluto nunca salta
        prev_abs, d = a, d + timedelta(days=1)


def test_version_vectorizada_igual_a_la_escalar():
    s = pd.Series(pd.date_range("2013-12-20", "2021-01-10", freq="3D"))
    ew = epiweek_of_dates(s)
    assert [epiweek(x) for x in s] == list(zip(ew["epi_year"].astype(int), ew["epi_week"].astype(int)))
    assert [abs_week_of_date(x) for x in s] == list(abs_week_of_dates(s))


def test_week_table_consecutiva():
    t = week_table(2019, 2021)
    assert t["abs_week"].diff().dropna().eq(1).all()
    assert len(t) == sum(weeks_in_year(y) for y in (2019, 2020, 2021))
