"""Escenarios del retraso (base + sensibilidad) y semillas múltiples."""
import copy

import numpy as np
import pandas as pd
import pytest

from dengue_pipeline import delay_scenarios as ds
from dengue_pipeline.config import load_config
from dengue_pipeline.snapshots import simulate_triangle
from helpers import regional_rows


def test_la_configuracion_trae_el_base_y_dos_escenarios_de_sensibilidad():
    cfg = load_config()
    assert list(ds.scenarios(cfg)) == ["piura2025", "piura2026", "piura2023"]   # el base va primero
    assert ds.scenarios(cfg)[ds.BASE] == cfg["delay"]["estimation"]


def test_cinco_semillas_distintas_y_la_primera_es_la_de_la_corrida_principal():
    cfg = load_config()
    s = ds.seeds(cfg)
    assert len(s) == 5 and len(set(s)) == 5 and s[0] == cfg["delay"]["seed"]


def test_semillas_repetidas_se_rechazan():
    cfg = copy.deepcopy(load_config())
    cfg["delay"]["seeds"] = [1, 2, 2]
    with pytest.raises(ValueError):
        ds.seeds(cfg)


def test_el_escenario_base_no_se_puede_repetir_en_la_lista_de_sensibilidad():
    cfg = copy.deepcopy(load_config())
    cfg["delay"]["scenarios"]["piura2025"] = {"cut": "20260702", "onset_year": 2025}
    with pytest.raises(ValueError):
        ds.scenarios(cfg)


def test_escenario_desconocido_da_error_claro():
    with pytest.raises(ValueError, match="desconocido"):
        ds.estimate(regional_rows([dict()]), load_config(), "no_existe")


def _reg():
    """Dos cortes con retrasos distintos: 2025 (corte 2026) casi inmediato; 2023 (corte 2023) con retraso de 3 semanas."""
    rows = []
    for _ in range(40):
        rows.append(dict(FECHA_CORTE="20260702", FECHA_INICIO_SINTOMAS="20250210", FECHA_NOTIFICACION="20250211"))
        rows.append(dict(FECHA_CORTE="20230517", FECHA_INICIO_SINTOMAS="20230109", FECHA_NOTIFICACION="20230130"))
    return regional_rows(rows)


def test_cada_escenario_usa_su_propio_corte_y_anio():
    cfg, reg = load_config(), _reg()
    base, s2023 = ds.estimate(reg, cfg, "piura2025"), ds.estimate(reg, cfg, "piura2023")
    assert base["cut"] == "20260702" and base["onset_year"] == 2025 and base["pmf"][0] == pytest.approx(1.0)
    assert s2023["cut"] == "20230517" and s2023["onset_year"] == 2023 and s2023["pmf"][3] == pytest.approx(1.0)


def test_resumen_una_fila_por_escenario_con_percentiles_en_semanas():
    cfg = load_config()
    reg = _reg()
    cfg = copy.deepcopy(cfg)
    cfg["delay"]["scenarios"] = {"piura2023": cfg["delay"]["scenarios"]["piura2023"]}   # 2026 no tiene casos en este ejemplo
    s = ds.summary(reg, cfg)
    assert list(s["escenario"]) == ["piura2025", "piura2023"]
    assert list(s["p99_semanas"]) == [0, 3] and list(s["casos_usados"]) == [40, 40]


def test_misma_semilla_mismo_triangulo_y_semillas_distintas_triangulos_distintos():
    w = pd.DataFrame({"unit": ["A"] * 30, "abs_week": range(1, 31), "cases": [40] * 30})
    pmf = np.array([0.5, 0.3, 0.2])
    a, b = simulate_triangle(w, pmf, 20261120), simulate_triangle(w, pmf, 20261120)
    c = simulate_triangle(w, pmf, 20261121)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c) and a["n"].sum() == c["n"].sum() == 30 * 40   # los totales por celda se conservan
