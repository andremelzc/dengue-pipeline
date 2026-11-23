"""Inyección de los defectos de datos observados: cada uno debe ser detectado."""
import numpy as np
import pandas as pd

from dengue_pipeline import validate
from dengue_pipeline.epiweek import epiweek
from dengue_pipeline.io import physical_lines, read_strict
from helpers import fmt, national_rows, regional_rows, tiny_cfg


def status(checks, name):
    return next(c.status for c in checks if c.name == name)


def clean_national():
    rows = [{"ano": a, "semana": s, "ubigeo": "200101"} for a in (2023, 2024) for s in range(1, 30) for _ in range(30)]
    return national_rows(rows)


def test_datos_limpios_sin_fallas():
    checks = validate.validate_national(clean_national(), tiny_cfg())
    assert not [c for c in checks if c.status == validate.FAIL]


def test_ubigeo_corto_se_detecta():
    df = clean_national()
    df.loc[0, "ubigeo"] = "10101"  # perdió el cero inicial: el ubigeo debe leerse como texto
    assert status(validate.validate_national(df, tiny_cfg()), "ubigeo_texto_6_digitos") == validate.FAIL


def test_tipo_dx_fuera_de_dominio():
    df = clean_national()
    df.loc[0, "tipo_dx"] = "X"
    assert status(validate.validate_national(df, tiny_cfg()), "tipo_dx_dominio") == validate.FAIL


def test_semana_53_en_año_de_52_semanas():
    df = clean_national()
    df.loc[0, ["ano", "semana"]] = ["2024", "53"]  # 2024 tiene 52 semanas
    assert status(validate.validate_national(df, tiny_cfg()), "semana_valida_para_el_año") == validate.FAIL


def test_semana_53_en_año_de_53_semanas_es_valida():
    df = clean_national()
    df.loc[0, ["ano", "semana"]] = ["2020", "53"]
    assert status(validate.validate_national(df, tiny_cfg()), "semana_valida_para_el_año") == validate.PASS


def test_caracter_de_reemplazo_en_nombres():
    df = clean_national()
    df.loc[0, "provincia"] = "CA�ETE"
    assert status(validate.validate_national(df, tiny_cfg()), "texto_con_caracteres_dañados") == validate.WARN


def test_ultimo_año_incompleto_en_unidad_de_estudio():
    rows = [{"ano": 2023, "semana": 5} for _ in range(1000)] + [{"ano": 2024, "semana": 5} for _ in range(10)]
    checks = validate.validate_national(national_rows(rows), tiny_cfg())
    c = next(c for c in checks if c.name == "ultimo_año_posiblemente_incompleto")
    assert c.status == validate.WARN and "PIURA" in c.data["flagged"]


def test_filas_perdidas_o_campos_con_salto_de_linea(tmp_path):
    p = tmp_path / "n.csv"
    df = clean_national().head(5).copy()
    df.loc[0, "localidad"] = "UNA\nLOCALIDAD"  # campo con salto de línea entre comillas
    df.to_csv(p, index=False)
    back = read_strict(p)
    assert len(back) == 5 and physical_lines(p) - 1 == 6  # 5 filas, 6 líneas físicas
    assert status(validate.validate_national(back, tiny_cfg(), p), "filas_vs_lineas_fisicas") == validate.WARN


def _regional_with_drift():
    rng = np.random.default_rng(1)
    rows = []
    for cut, meaning in (("20230517", "onset"), ("20260702", "notif")):
        base = pd.Timestamp("2023-02-05" if cut == "20230517" else "2025-02-02")
        for _ in range(400):
            onset = base + pd.Timedelta(days=int(rng.integers(0, 70)))
            notif = onset + pd.Timedelta(days=int(rng.integers(0, 15)))
            ingreso = notif - pd.Timedelta(days=5) if meaning == "onset" else notif + pd.Timedelta(days=1)
            sem = epiweek(onset)[1] if meaning == "onset" else epiweek(notif)[1]
            rows.append(dict(FECHA_CORTE=cut, FECHA_INICIO_SINTOMAS=fmt(onset), FECHA_NOTIFICACION=fmt(notif),
                             FECHA_INGRESO=fmt(ingreso), SEMANA_NOTIFICACION=sem))
    return regional_rows(rows)


def test_deriva_de_significado_entre_cortes():
    checks = validate.validate_regional(_regional_with_drift(), tiny_cfg())
    c = next(c for c in checks if c.name == "deriva_de_significado_entre_cortes")
    assert c.status == validate.WARN
    assert c.data["20230517"]["significado_inferido_de_SEMANA_NOTIFICACION"] == "inicio_de_sintomas"
    assert c.data["20260702"]["significado_inferido_de_SEMANA_NOTIFICACION"] == "notificacion"


def test_notificacion_anterior_al_inicio_de_sintomas():
    df = regional_rows([dict(FECHA_INICIO_SINTOMAS="20250110", FECHA_NOTIFICACION="20250105")] * 3)
    assert status(validate.validate_regional(df, tiny_cfg()), "notificacion_anterior_al_inicio_de_sintomas") == validate.WARN


def test_fecha_invalida_y_notificacion_posterior_al_corte():
    df = regional_rows([dict(FECHA_NOTIFICACION="20250230")])  # 30 de febrero
    assert status(validate.validate_regional(df, tiny_cfg()), "fechas_validas") == validate.FAIL
    df = regional_rows([dict(FECHA_CORTE="20250201", FECHA_NOTIFICACION="20250301")])   # semanas después del corte
    assert status(validate.validate_regional(df, tiny_cfg()), "notificacion_posterior_al_corte") == validate.FAIL


def test_notificacion_un_dia_despues_del_corte_en_la_misma_semana_es_advertencia():
    # corte miércoles 17-may-2023; notificación jueves 18 (misma semana epidemiológica): caso real de datos.md 3.5
    df = regional_rows([dict(FECHA_CORTE="20230517", FECHA_INICIO_SINTOMAS="20230517", FECHA_NOTIFICACION="20230518")])
    assert status(validate.validate_regional(df, tiny_cfg()), "notificacion_posterior_al_corte") == validate.WARN


def test_ultimas_semanas_del_corte_incompletas():
    rows = []
    base = pd.Timestamp("2026-03-01")  # domingo
    for w in range(10):
        n = 100 if w < 8 else (50 if w == 8 else 5)
        for _ in range(n):
            d = base + pd.Timedelta(weeks=w, days=2)
            rows.append(dict(FECHA_INICIO_SINTOMAS=fmt(d), FECHA_NOTIFICACION=fmt(d + pd.Timedelta(days=1)),
                             FECHA_INGRESO=fmt(d + pd.Timedelta(days=2))))
    c = next(c for c in validate.validate_regional(regional_rows(rows), tiny_cfg())
             if c.name == "ultimas_semanas_del_corte_incompletas")
    assert c.status == validate.WARN
