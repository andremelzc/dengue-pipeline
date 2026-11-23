"""Pruebas de población, incidencia de 4 semanas y niveles de alerta (datos SINTÉTICOS, salvo la prueba con el archivo real)."""
import os

import numpy as np
import openpyxl
import pandas as pd
import pytest

from dengue_pipeline import population
from dengue_pipeline.metrics import alert_level
from helpers import tiny_cfg

CFG = {**tiny_cfg(), "alerts": {"yellow_per_100k": 50, "red_per_100k": 150, "sensitivity_cuts_per_100k": [100, 300]}}


def _xlsx(path, bad_sum=False):
    """Cuadro sintético con el mismo formato que el del INEI (dos hojas, nombres con espacios y tildes)."""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for title, years in (("2022-2023", (2022, 2023)), ("2024-2026", (2024, 2025, 2026))):
        ws = wb.create_sheet(title)
        ws.append([f"POBLACIÓN ESTIMADA ... {title}"])
        ws.append(["Ubigeo", "Departamento"] + [v for y in years for v in (y, None, None)])
        ws.append([None, None] + ["Total ", "Hombre", "Mujer"] * len(years))
        ws.append(["000000", "Perú"] + [3000, 1500, 1500] * len(years))
        for ub, name, base in (("200000", "Piura ", 1000), ("240000", "Tumbes", 200), ("150000", "Lima", 5000)):
            row = [ub, name]
            for i, y in enumerate(years):
                tot = base + 10 * (y - 2022)
                row += [tot, tot // 2, tot - tot // 2 + (1 if bad_sum and name == "Tumbes" else 0)]
            ws.append(row)
        ws.append(["Fuente: INEI (sintético)"])
    wb.save(path)


def _cfg_years():
    c = dict(CFG)
    c["study_window"] = {"first_year": 2022, "last_year": 2024, "core_first_year": 2022}
    return c


def test_lectura_del_cuadro_y_union_por_departamento(tmp_path):
    f = tmp_path / "pop.xlsx"
    _xlsx(f)
    raw = population.read_inei_departmental(f)
    assert {"PIURA", "TUMBES", "LIMA", "PERU"} == set(raw["departamento"])        # sin espacios ni tildes
    pop = population.population_by_unit(raw, _cfg_years())
    assert list(pop["unit"].unique()) == ["PIURA", "TUMBES"]                       # Lima Norte no entra
    assert pop[(pop.unit == "PIURA") & (pop.year == 2023)]["population"].iloc[0] == 1010
    assert pop["year"].between(2022, 2024).all()


def test_hombre_mas_mujer_distinto_de_total_falla(tmp_path):
    f = tmp_path / "pop.xlsx"
    _xlsx(f, bad_sum=True)
    with pytest.raises(ValueError, match="Hombre \\+ Mujer"):
        population.read_inei_departmental(f)


def test_falta_un_anio_o_un_departamento_falla(tmp_path):
    f = tmp_path / "pop.xlsx"
    _xlsx(f)
    raw = population.read_inei_departmental(f)
    c = _cfg_years()
    c["study_window"] = {"first_year": 2022, "last_year": 2030, "core_first_year": 2022}
    with pytest.raises(ValueError, match="faltan años"):
        population.population_by_unit(raw, c)
    c = _cfg_years()
    c["units"] = [{"id": "X", "type": "departamento", "departamento": "NOEXISTE"}]
    with pytest.raises(ValueError, match="no está en el cuadro"):
        population.population_by_unit(raw, c)


def test_cortes_en_el_borde():
    inc = np.array([0, 49.999, 50, 99.99, 100, 149.999, 150, 299.9, 300, 1e4])
    assert list(alert_level(inc, 50, 150)) == [0, 0, 1, 1, 1, 1, 2, 2, 2, 2]     # verde < 50, amarillo [50, 150), rojo >= 150
    assert list(alert_level(inc, 100, 300)) == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]     # sensibilidad


def _weekly(cases, unit="PIURA", year=2023):
    n = len(cases)
    return pd.DataFrame({"unit": unit, "year": year, "abs_week": np.arange(1, n + 1), "cases": cases})


def _pop(unit="PIURA", year=2023, p=100_000):
    return pd.DataFrame({"unit": [unit], "year": [year], "population": [p]})


def test_incidencia_de_4_semanas_a_mano():
    # población 100 000: casos 4w = suma de t y las 3 anteriores; tasa = casos4w por 100 000 / 100 000 * 1e5 = casos4w
    out = population.add_incidence(_weekly([10, 20, 5, 15, 40, 0, 0, 0, 100, 1]), _pop(), CFG)
    assert out["cases4w"].isna().sum() == 3 and out["inc4w_per100k"].iloc[:3].isna().all()   # primeras 3 sin tasa
    assert out["cases4w"].iloc[3] == 50 and out["inc4w_per100k"].iloc[3] == 50.0              # 10+20+5+15, justo en el corte
    assert out["level"].iloc[3] == 1 and out["level"].iloc[2] == -1
    assert out["cases4w"].iloc[4] == 80 and out["level"].iloc[4] == 1
    assert out["cases4w"].iloc[8] == 100 and out["level"].iloc[8] == 1 and out["level_sens"].iloc[8] == 1
    out2 = population.add_incidence(_weekly([0, 0, 0, 200, 0, 0, 0, 0, 0, 0]), _pop(), CFG)
    assert out2["level"].iloc[3] == 2 and out2["level_sens"].iloc[3] == 1                       # 200: rojo (150) pero amarillo en sensibilidad (100/300)


def test_la_tasa_usa_la_poblacion_del_anio_de_la_semana():
    w = pd.concat([_weekly([10] * 4, year=2023), _weekly([10] * 4, year=2024).assign(abs_week=np.arange(5, 9))])
    pop = pd.DataFrame({"unit": ["PIURA"] * 2, "year": [2023, 2024], "population": [100_000, 200_000]})
    out = population.add_incidence(w, pop, CFG)
    assert out["inc4w_per100k"].iloc[3] == 40.0           # 40 casos / 100 000
    assert out["inc4w_per100k"].iloc[7] == 20.0           # 40 casos / 200 000 (año 2024)


def test_unidad_sin_poblacion_queda_sin_tasa_no_en_cero():
    w = pd.concat([_weekly([100] * 6), _weekly([100] * 6, unit="LIMA_NORTE")])
    out = population.add_incidence(w, _pop(), CFG)
    ln = out[out.unit == "LIMA_NORTE"]
    assert ln["inc4w_per100k"].isna().all() and (ln["level"] == -1).all()


def test_no_usa_el_futuro():
    base = [3, 8, 1, 9, 4, 7, 2, 6, 5, 10]
    a = population.add_incidence(_weekly(base), _pop(), CFG)
    alt = base[:6] + [999] * 4                             # se altera todo lo posterior a la semana 6
    b = population.add_incidence(_weekly(alt), _pop(), CFG)
    cols = ["cases4w", "inc4w_per100k", "level", "level_sens"]
    pd.testing.assert_frame_equal(a.iloc[:6][cols].reset_index(drop=True), b.iloc[:6][cols].reset_index(drop=True))


def test_semanas_no_consecutivas_falla():
    w = _weekly([1, 2, 3, 4, 5]).drop(index=2)
    with pytest.raises(ValueError, match="consecutivas"):
        population.add_incidence(w, _pop(), CFG)


def test_csv_de_poblacion_exige_columnas_y_positivos(tmp_path):
    f = tmp_path / "p.csv"
    pd.DataFrame({"unit": ["A"], "year": [2020], "population": [0]}).to_csv(f, index=False)
    with pytest.raises(ValueError, match="no positiva"):
        population.read_population_csv(f)
    pd.DataFrame({"a": [1]}).to_csv(f, index=False)
    with pytest.raises(ValueError, match="columnas"):
        population.read_population_csv(f)


REAL = os.environ.get("DENGUE_POPULATION_XLSX")


@pytest.mark.skipif(not REAL, reason="define DENGUE_POPULATION_XLSX con el archivo real del INEI (proy_03_4.xlsx)")
def test_cifras_reales_del_cuadro_inei():
    from dengue_pipeline.config import load_config
    pop = population.population_by_unit(population.read_inei_departmental(REAL), load_config())
    g = lambda u, y: int(pop[(pop.unit == u) & (pop.year == y)]["population"].iloc[0])
    assert g("PIURA", 2010) == 1764979 and g("PIURA", 2020) == 2047954
    assert g("TUMBES", 2010) == 211526 and g("TUMBES", 2020) == 251521
    assert g("LAMBAYEQUE", 2020) == 1310785 and g("SAN_MARTIN", 2020) == 899648 and g("LA_LIBERTAD", 2020) == 2016771
    assert len(pop) == 5 * 15 and "LIMA_NORTE" not in set(pop["unit"])
