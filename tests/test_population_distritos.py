import openpyxl
import pytest

from dengue_pipeline import population

CFG = {"study_window": {"first_year": 2010, "last_year": 2024},
       "units": [{"id": "LN", "type": "ubigeo_group", "ubigeos": ["150102", "150106"]}]}


def _xlsx(tmp_path, rows):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["CUADRO Nº 01"])
    ws.append(["UBIGEO", "Departamento, provincia y distrito", "Población proyectada"])
    ws.append([None, None, 2018, 2019, 2020])
    ws.append([None] * 5)
    for r in rows:
        ws.append(r)
    p = tmp_path / "d.xlsx"
    wb.save(p)
    return p


def test_suma_de_distritos_por_año(tmp_path):
    p = _xlsx(tmp_path, [["000000", "PERÚ", 9, 9, 9], ["150102", "ANCON  ", 10, 20, 30], ["150106", "CARABAYLLO", 1, 2, 3]])
    out = population.population_ubigeo_groups(population.read_inei_districts(p), CFG)
    assert out["year"].tolist() == [2018, 2019, 2020] and out["population"].tolist() == [11, 22, 33]


def test_falla_si_falta_un_distrito(tmp_path):
    p = _xlsx(tmp_path, [["150102", "ANCON", 10, 20, 30]])
    with pytest.raises(ValueError, match="150106"):
        population.population_ubigeo_groups(population.read_inei_districts(p), CFG)
