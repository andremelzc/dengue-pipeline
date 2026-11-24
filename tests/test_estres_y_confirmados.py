"""Prueba de estrés del retraso (--delay-scale) y variante de solo confirmados (--case-definition confirmed).

Las corridas largas se sustituyen por funciones falsas: aquí se comprueba qué se calcula y dónde se guarda, no el desempeño.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dengue_pipeline import backtest, cli, delay_scenarios as ds, metrics
from dengue_pipeline import reproduce as rp
from dengue_pipeline import snapshots as sn
from helpers import national_rows, regional_rows


# ---------- escala del retraso ----------

def _final():
    return pd.DataFrame({"unit": ["A"] * 6 + ["B"] * 6, "abs_week": list(range(100, 106)) * 2,
                         "cases": [30, 0, 50, 8, 90, 12, 5, 70, 0, 40, 33, 1]})


PMF = np.array([0.4, 0.3, 0.2, 0.1])


@pytest.mark.parametrize("scale", [2, 4])
def test_la_escala_multiplica_el_retraso_y_no_cambia_los_sorteos(scale):
    base = sn.simulate_triangle(_final(), PMF, 20261120).sort_values(["unit", "onset_week", "avail_week"]).reset_index(drop=True)
    big = sn.simulate_triangle(_final(), PMF, 20261120, delay_scale=scale).sort_values(["unit", "onset_week", "avail_week"]).reset_index(drop=True)
    assert (big["n"] == base["n"]).all() and (big["onset_week"] == base["onset_week"]).all()
    assert ((big["avail_week"] - big["onset_week"]) == scale * (base["avail_week"] - base["onset_week"])).all()
    assert big["n"].sum() == _final()["cases"].sum()                       # los totales por celda se conservan


def test_escala_uno_es_la_simulacion_de_siempre():
    pd.testing.assert_frame_equal(sn.simulate_triangle(_final(), PMF, 7), sn.simulate_triangle(_final(), PMF, 7, delay_scale=1))


@pytest.mark.parametrize("bad", [0, -2, 1.5])
def test_escala_invalida_se_rechaza(bad):
    with pytest.raises(ValueError, match="delay_scale"):
        sn.simulate_triangle(_final(), PMF, 1, delay_scale=bad)


@pytest.mark.parametrize("scale", [2, 4])
def test_instantanea_con_retraso_escalado_no_contiene_futuro(scale):
    """La comprobación central, repetida con el triángulo escalado: ninguna instantánea trae algo posterior a su semana."""
    tri = sn.simulate_triangle(_final(), PMF, 20261121, delay_scale=scale)
    for t in range(100, 125):
        snap = sn.snapshot(tri, t)
        sn.assert_no_future(tri, snap, t)
        assert (snap["onset_week"] <= t).all()
        tri2 = tri.copy()
        tri2.loc[tri2["avail_week"] > t, "n"] += 999                       # alterar lo que llega después no cambia la instantánea
        pd.testing.assert_frame_equal(snap.reset_index(drop=True), sn.snapshot(tri2, t).reset_index(drop=True))


def test_proporcion_de_casos_mas_alla_del_maximo():
    tri = pd.DataFrame({"unit": "A", "onset_week": [1, 1, 1], "avail_week": [1, 5, 9], "n": [10, 10, 20]})
    assert ds.share_beyond_max(tri, 3) == pytest.approx(30 / 40)          # retrasos 0, 4 y 8: superan 3 los dos últimos
    assert ds.share_beyond_max(tri, 4) == pytest.approx(20 / 40)
    assert ds.share_beyond_max(tri, 8) == 0.0
    assert ds.share_beyond_max(tri.iloc[:0], 4) == 0.0


def test_etiqueta_de_archivo_con_y_sin_escala():
    assert ds.tag("piura2025", 20261120) == "piura2025_s20261120"
    assert ds.tag("piura2025", 20261120, 4) == "piura2025_x4_s20261120"


# ---------- delay-sensitivity --delay-scale ----------

class _Fake:
    """Reemplaza run_backtest y summarize: anota cómo se llamó y devuelve tablas mínimas."""

    def __init__(self):
        self.calls = []

    def run_backtest(self, weekly, tri, cfg, models=None, conditions=("final", "disponibles"), out_dir=None, **kw):
        out_dir.mkdir(parents=True, exist_ok=True)                           # como el real
        self.calls.append({"out_dir": out_dir, "conditions": conditions, "tri": tri, "params": kw["params_by_h"], "name": kw["out_name"]})
        return pd.DataFrame({"x": [1]}), {"fits": 0}

    @staticmethod
    def summarize(pred, *a):
        return pd.DataFrame({"model": ["m"], "mae": [1.0]})


def _sens_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "data" / "processed" / "backtest"
    out.mkdir(parents=True)
    (out / "hyperparameters.json").write_text(
        '{"h2": {"rf": {"max_depth": 4}, "gb": {"max_iter": 100}}, "h4": {"rf": {"max_depth": 8}, "gb": {"max_iter": 300}}}', encoding="utf-8")
    weekly = _final().assign(year=2023, week=1)
    weekly["unit"] = "PIURA"
    weekly = weekly.drop_duplicates(["abs_week"])
    wpath = tmp_path / "weekly.csv"
    weekly.to_csv(wpath, index=False)
    reg = regional_rows([dict(FECHA_CORTE="20260702", FECHA_INICIO_SINTOMAS="20250210", FECHA_NOTIFICACION="20250217")] * 20)   # retraso de 1 semana
    monkeypatch.setattr(cli, "read_strict", lambda p: reg)
    fake = _Fake()
    monkeypatch.setattr(backtest, "run_backtest", fake.run_backtest)
    monkeypatch.setattr(metrics, "summarize", fake.summarize)
    return fake, wpath, out


def test_delay_scale_corre_solo_disponibles_con_hiperparametros_congelados_y_cinco_semillas(tmp_path, monkeypatch):
    fake, wpath, out = _sens_env(tmp_path, monkeypatch)
    assert cli.main(["delay-sensitivity", "--regional", "reg.csv", "--weekly", str(wpath), "--delay-scale", "2", "4"]) == 0
    assert len(fake.calls) == 2 * 5                                          # 2 escalas x 5 semillas, solo el escenario base
    assert all(c["conditions"] == ("disponibles",) for c in fake.calls)
    assert all(c["params"][2]["rf"] == {"max_depth": 4} for c in fake.calls)  # los congelados de la corrida principal
    dirs = {c["out_dir"].name for c in fake.calls}
    assert dirs == {"estres_retraso_x2", "estres_retraso_x4"} and "sensibilidad_retraso" not in dirs
    names = [c["name"] for c in fake.calls if c["out_dir"].name == "estres_retraso_x2"]
    assert [n.split("_s")[-1].removesuffix(".csv.gz") for n in names] == ["20261120", "20261121", "20261122", "20261123", "20261124"]
    assert all("piura2025_x2_s" in n for n in names)
    res = pd.read_csv(out / "estres_retraso_x4" / "metrics_summary_estres_retraso_x4.csv")
    assert set(res["escala_retraso"]) == {4} and set(res["escenario"]) == {"piura2025"} and len(res) == 5
    assert "prop_casos_retraso_mayor_que_D" in res.columns
    assert not (out / "sensibilidad_retraso").exists()                       # la carpeta de la sensibilidad no se toca


def test_el_triangulo_del_estres_tiene_el_retraso_multiplicado(tmp_path, monkeypatch):
    fake, wpath, _ = _sens_env(tmp_path, monkeypatch)
    cli.main(["delay-sensitivity", "--regional", "reg.csv", "--weekly", str(wpath), "--delay-scale", "4", "--n-seeds", "1"])
    (call,) = fake.calls
    assert ((call["tri"]["avail_week"] - call["tri"]["onset_week"]) == 4).all()      # retraso observado 1 semana x 4


def test_sin_delay_scale_la_sensibilidad_sigue_igual(tmp_path, monkeypatch):
    fake, wpath, out = _sens_env(tmp_path, monkeypatch)
    cli.main(["delay-sensitivity", "--regional", "reg.csv", "--weekly", str(wpath), "--scenarios", "piura2025", "--n-seeds", "2"])
    assert {c["out_dir"].name for c in fake.calls} == {"sensibilidad_retraso"} and len(fake.calls) == 2
    res = pd.read_csv(out / "sensibilidad_retraso" / "metrics_summary_sensibilidad_retraso.csv")
    assert list(res.columns[:3]) == ["features", "escenario", "semilla"] and "escala_retraso" not in res.columns


def test_delay_scale_solo_acepta_2_y_4():
    with pytest.raises(SystemExit):
        cli.main(["delay-sensitivity", "--regional", "r.csv", "--delay-scale", "3"])


# ---------- --case-definition confirmed ----------

def _nat():
    return national_rows([{"ano": 2023, "semana": 3, "tipo_dx": "C"}] * 3 + [{"ano": 2023, "semana": 3, "tipo_dx": "P"}] * 2)


def test_integrate_confirmed_cuenta_solo_C_y_escribe_en_su_carpeta(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "read_strict", lambda p: _nat())
    assert cli.main(["integrate", "--national", "nat.csv", "--case-definition", "confirmed"]) == 0
    conf = pd.read_csv(tmp_path / "data/processed/solo_confirmados/weekly_final.csv")
    assert conf["cases"].sum() == 3 and conf["cases_P"].sum() == 2            # `cases` = solo C; cases_P se conserva como columna informativa
    assert not (tmp_path / "data/processed/weekly_final.csv").exists()        # no pisa la corrida principal
    cli.main(["integrate", "--national", "nat.csv"])
    assert pd.read_csv(tmp_path / "data/processed/weekly_final.csv")["cases"].sum() == 5   # el valor por defecto sigue siendo C + P


def test_simulate_confirmed_usa_la_serie_de_confirmados(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reg = regional_rows([dict(FECHA_CORTE="20260702", FECHA_INICIO_SINTOMAS="20250210", FECHA_NOTIFICACION="20250211")] * 30)
    monkeypatch.setattr(cli, "read_strict", lambda p: _nat() if "nat" in str(p) else reg)
    cli.main(["simulate", "--national", "nat.csv", "--regional", "reg.csv", "--case-definition", "confirmed"])
    tri = pd.read_csv(tmp_path / "data/processed/solo_confirmados/triangle_simulated.csv.gz")
    assert tri["n"].sum() == 3 and not (tmp_path / "data/processed/triangle_simulated.csv.gz").exists()


def test_backtest_confirmed_lee_y_guarda_en_su_carpeta(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "data/processed/solo_confirmados"
    d.mkdir(parents=True)
    _final().to_csv(d / "weekly_final.csv", index=False)
    pd.DataFrame({"unit": ["A"], "onset_week": [1], "avail_week": [1], "n": [1]}).to_csv(d / "triangle_simulated.csv.gz", index=False)
    fake = _Fake()
    monkeypatch.setattr(backtest, "run_backtest", fake.run_backtest)
    monkeypatch.setattr(metrics, "summarize", fake.summarize)
    monkeypatch.setattr(metrics, "paired_difference", lambda *a: pd.DataFrame({"a": [1]}))
    (d / "backtest").mkdir()
    assert cli.main(["backtest", "--case-definition", "confirmed"]) == 0
    assert fake.calls[0]["out_dir"] == Path("data/processed/solo_confirmados/backtest") and (d / "backtest" / "metrics_summary.csv").exists()
    cli.main(["backtest", "--case-definition", "confirmed", "--train-first-year", "2014"])
    assert fake.calls[1]["out_dir"] == Path("data/processed/solo_confirmados/backtest_desde2014")
    assert not (tmp_path / "data/processed/backtest").exists()                # la corrida principal no se toca


def test_backtest_confirmed_sin_insumos_falla_con_mensaje_claro(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="FALTA"):
        cli.main(["backtest", "--case-definition", "confirmed"])


def test_case_definition_solo_acepta_confirmed():
    with pytest.raises(SystemExit):
        cli.main(["integrate", "--national", "n.csv", "--case-definition", "probable"])


# ---------- pasos opcionales de reproduce ----------

def _names(**kw):
    return [s.name for s in rp.build_steps("n.csv", "r.csv", "pob.xlsx", climate_downloaded_on="2026-11-23", **kw)]


def test_los_pasos_nuevos_son_opcionales_y_no_alteran_los_14():
    assert len(_names()) == 14
    full = _names(stress=True, confirmed=True)
    assert full[:10] == _names()[:10] and len(full) == 19 and full.index("principal") < full.index("estres_retraso")
    assert full.index("integrar_confirmados") < full.index("simular_confirmados") < full.index("principal_confirmados") < full.index("alertas_confirmados")
    s = {x.name: x for x in rp.build_steps("n.csv", "r.csv", "pob.xlsx", stress=True, skip_climate=True)}
    assert s["estres_retraso"].argv == ["delay-sensitivity", "--regional", "r.csv", "--delay-scale", "2", "4"]
