"""Reproduce cifras ya verificadas en datos.md con los CSV reales.

Se saltan si no se indican los archivos:
    DENGUE_NATIONAL_CSV=/ruta/nacional.csv DENGUE_REGIONAL_CSV=/ruta/regional.csv pytest
"""
import os

import pandas as pd
import pytest

from dengue_pipeline import integrate, snapshots, validate
from dengue_pipeline.config import load_config
from dengue_pipeline.epiweek import abs_week
from dengue_pipeline.io import physical_lines, read_strict

NAT, REG = os.environ.get("DENGUE_NATIONAL_CSV"), os.environ.get("DENGUE_REGIONAL_CSV")
pytestmark = pytest.mark.skipif(not (NAT and REG), reason="faltan los CSV reales")


@pytest.fixture(scope="module")
def nat():
    return read_strict(NAT)


@pytest.fixture(scope="module")
def reg():
    return read_strict(REG)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_nacional_filas_y_tipo_dx(nat):                      # datos.md 3.1
    assert len(nat) == 1_065_274 and physical_lines(NAT) == 1_065_275
    assert nat["tipo_dx"].value_counts().to_dict() == {"C": 955_112, "P": 110_162}


def test_nacional_totales_por_año(nat):                      # datos.md 3.2
    by_year = nat.groupby("ano").size()
    assert by_year["2023"] == 256_641 and by_year["2024"] == 271_531 and by_year["2017"] == 68_290


def test_nacional_validacion_sin_fallas(nat, cfg):
    checks = validate.validate_national(nat, cfg, NAT)
    assert not [c for c in checks if c.status == validate.FAIL]
    # Hallazgo nuevo: 81 casos de 2009 con semana 53, año que tiene 52 semanas MMWR (fuera de la ventana 2010-2024).
    w = next(c for c in checks if c.name == "semana_valida_para_el_año")
    assert w.status == validate.WARN and w.data["invalid"] == {"2009-S53": 81} and w.data["in_window"] == 0
    c = next(c for c in checks if c.name == "ultimo_año_posiblemente_incompleto")
    assert c.status == validate.WARN and {"PIURA", "LAMBAYEQUE", "LA_LIBERTAD"} <= set(c.data["flagged"])  # datos.md 3.9


def test_regional_cortes_y_deriva(reg, cfg):                 # datos.md 2.2, 2.3 y 3.5
    assert len(reg) == 47_082
    assert reg["FECHA_CORTE"].value_counts().to_dict() == {"20230517": 24_566, "20260702": 22_516}
    checks = validate.validate_regional(reg, cfg)
    drift = next(c for c in checks if c.name == "deriva_de_significado_entre_cortes")
    assert drift.status == validate.WARN
    assert drift.data["20230517"]["significado_inferido_de_SEMANA_NOTIFICACION"] == "inicio_de_sintomas"
    assert drift.data["20260702"]["significado_inferido_de_SEMANA_NOTIFICACION"] == "notificacion"
    neg = next(c for c in checks if c.name == "notificacion_anterior_al_inicio_de_sintomas")
    assert neg.data["20230517"] == 66
    post = next(c for c in checks if c.name == "notificacion_posterior_al_corte")   # hallazgo nuevo (ver README)
    assert post.status == validate.WARN and post.data["after_cut"] == {"20230517": 310, "20260702": 0}
    assert post.data["other_week"] == {"20230517": 0, "20260702": 0}


def test_tabla_semanal_conserva_los_casos(nat, cfg):
    t = integrate.weekly_table(nat, cfg)
    assert t.attrs["dropped_cases_invalid_week"] == 0
    assert t[(t.unit == "PIURA") & (t.year == 2023)].cases.sum() == 79_304   # datos.md 3.9


def test_par_real_piura_2023(nat, reg, cfg):                 # datos.md 3.7
    final = integrate.weekly_table(nat, cfg)
    weeks = [abs_week(2023, w) for w in (17, 18, 19, 20)]
    out = snapshots.cut_vs_final(reg, "20230517", final, "PIURA", weeks)
    assert list(out["corte"]) == [2784, 3016, 3473, 1848]
    assert list(out["final"]) == [2203, 3667, 5053, 8151]
