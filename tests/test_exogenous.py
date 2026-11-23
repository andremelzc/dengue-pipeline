"""Pruebas de clima, ICEN y ablación. Los datos son SINTÉTICOS (no son clima ni ICEN reales); la descarga real NO se probó [NO VERIFICADO]."""
import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from dengue_pipeline import backtest, exogenous
from dengue_pipeline.config import load_config
from dengue_pipeline.features import cols_for
from dengue_pipeline.vintage import Vintage


def _coords(tmp_path, rows):
    f = tmp_path / "c.csv"
    pd.DataFrame(rows, columns=["unit", "lat", "lon", "source"]).to_csv(f, index=False)
    return f


def test_coordenadas_vacias_fallan_con_falta(tmp_path):
    f = _coords(tmp_path, [["A", None, None, None], ["B", -5.0, -80.0, "x"]])
    with pytest.raises(ValueError, match=r"\[FALTA\].*'A'"):
        exogenous.read_coordinates(f, ["A", "B"])
    with pytest.raises(ValueError, match="FALTA"):
        exogenous.read_coordinates(f, ["A", "B", "C"])                       # unidad ausente


def test_plantilla_real_del_repositorio_no_se_puede_usar_sin_completar():
    units = [u["id"] for u in load_config()["units"]]
    with pytest.raises(ValueError, match="FALTA"):
        exogenous.read_coordinates("config/coordenadas_plantilla.csv", units)


def test_coordenadas_validas_y_latlon_invertidas(tmp_path):
    ok = exogenous.read_coordinates(_coords(tmp_path, [["A", -5.0, -80.0, "x"]]), ["A"])
    assert ok == {"A": (-5.0, -80.0)}
    with pytest.raises(ValueError, match="invertidas"):
        exogenous.read_coordinates(_coords(tmp_path, [["A", -80.0, -5.0, "x"]]), ["A"])


def test_parse_nasa_power_y_faltantes():
    raw = json.dumps({"properties": {"parameter": {"T2M": {"20240101": 25.0, "20240102": -999}, "RH2M": {"20240101": 70, "20240102": 71},
                                                    "PRECTOTCORR": {"20240101": 0.0, "20240102": 1.5}}}}).encode()
    d = exogenous.parse_nasa_power(raw)
    assert list(d.columns) == ["date", "T2M", "RH2M", "PRECTOTCORR"] and len(d) == 2
    assert np.isnan(d["T2M"].iloc[1]) and d["T2M"].iloc[0] == 25.0


def test_fetch_all_reintenta_guarda_crudo_y_hash(tmp_path):
    calls = {"n": 0}

    def fake(lat, lon, start, end, params):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("sin red")
        return b'{"x": 1}'
    cfg = {"nasa_power": {"parameters": ["T2M"]}}
    e = exogenous.fetch_all(cfg, {"A": (-5.0, -80.0)}, "20240101", "20240102", tmp_path, "2026-11-22", fetcher=fake, sleep=lambda s: None)
    assert calls["n"] == 3 and e[0]["sha256"] == hashlib.sha256(b'{"x": 1}').hexdigest() and e[0]["downloaded_on"] == "2026-11-22"
    assert (tmp_path / "nasa_power_A_20240101_20240102.json").read_bytes() == b'{"x": 1}'

    def always(*a, **k):
        raise RuntimeError("sin red")
    with pytest.raises(RuntimeError):
        exogenous.fetch_all(cfg, {"A": (-5.0, -80.0)}, "20240101", "20240102", tmp_path, "2026-11-22", fetcher=always, sleep=lambda s: None)


def _icen():
    return pd.DataFrame({"month": pd.PeriodIndex(["2020-03", "2020-03", "2020-04"], freq="M"), "value": [0.5, 0.9, 1.4],
                         "published_on": pd.to_datetime(["2020-02-20", "2020-04-10", "2020-03-25"]),
                         "kind": ["temporal", "definitivo", "temporal"]})


def test_icen_usa_la_version_que_existia_en_cada_fecha():
    weeks = pd.DataFrame({"abs_week": [1, 2, 3, 4, 5], "week_end": ["2020-02-10", "2020-03-10", "2020-03-28", "2020-04-05", "2020-04-15"]})
    r = exogenous.icen_weekly(weeks, _icen()).set_index("abs_week")["icen"]
    assert np.isnan(r[1])                          # nada publicado antes del 10-feb
    assert r[2] == 0.5                             # marzo: solo el temporal (publicado 20-feb)
    assert r[3] == 0.5 and True                    # 28-mar: marzo aún solo temporal; el temporal de abril (25-mar) es de otro mes
    assert r[4] == 1.4                             # abril: temporal publicado el 25-mar
    assert r[5] == 1.4                             # el definitivo de marzo (0.9, publicado 10-abr) NO reemplaza a abril


def test_icen_no_mira_versiones_posteriores():
    weeks = pd.DataFrame({"abs_week": [1, 2], "week_end": ["2020-03-10", "2020-03-20"]})
    a = exogenous.icen_weekly(weeks, _icen())
    alt = _icen(); alt.loc[1, "value"] = 99.0       # el definitivo (publicado 10-abr) cambia: las semanas de marzo no deben notarlo
    b = exogenous.icen_weekly(weeks, alt)
    pd.testing.assert_frame_equal(a, b)


def test_read_icen_csv_valida_columnas(tmp_path):
    f = tmp_path / "i.csv"
    pd.DataFrame({"month": ["2020-03"], "value": [0.1]}).to_csv(f, index=False)
    with pytest.raises(ValueError, match="FALTA"):
        exogenous.read_icen_csv(f)
    pd.DataFrame({"month": ["2020-03"], "value": [0.1], "published_on": ["2020-04-01"], "kind": ["otro"]}).to_csv(f, index=False)
    with pytest.raises(ValueError, match="kind"):
        exogenous.read_icen_csv(f)


def _weekly_like(U=2, W=260, w0=1000):
    rows = [{"unit": u, "abs_week": w0 + i, "week": (i % 52) + 1, "year": 2015 + i // 52} for u in "AB"[:U] for i in range(W)]
    return pd.DataFrame(rows)


def _exog(seed=0, condition="clima_icen"):
    w = _weekly_like()
    clim = exogenous.make_example_climate(w, seed)
    icen = exogenous.make_example_icen(w, seed)
    return exogenous.build_exog(condition, w.assign(cases=0), ["A", "B"], load_config(), clim, icen), w


def test_ablacion_condiciones_y_faltantes():
    w = _weekly_like().assign(cases=0)
    cfg = load_config()
    assert exogenous.build_exog("historial", w, ["A", "B"], cfg, None, None) is None
    with pytest.raises(ValueError, match="FALTA"):
        exogenous.build_exog("clima", w, ["A", "B"], cfg, None, None)
    with pytest.raises(ValueError, match="FALTA"):
        exogenous.build_exog("clima_icen", w, ["A", "B"], cfg, exogenous.make_example_climate(w), None)
    with pytest.raises(ValueError, match="desconocida"):
        exogenous.build_exog("otra", w, ["A", "B"], cfg, None, None)
    e_c, _ = _exog(condition="clima")
    e_i, _ = _exog(condition="clima_icen")
    assert not any(c.startswith("icen") for c in e_c.tree_cols) and any(c.startswith("icen") for c in e_i.tree_cols)


def test_clima_con_rezago_cero_es_rechazado():
    with pytest.raises(ValueError, match="rezago >= 1"):
        exogenous.Exog(["A"], 0, 9, ["temp_mean"], {"temp_mean": [0, 1]}, {"temp_mean": np.zeros((1, 10))})


def test_variables_en_t_usan_t_menos_k():
    e, w = _exog()
    t = np.array([1100, 1101]); u = np.array([0, 1])
    F = e.features(u, t)
    base = w[w.unit == "A"].set_index("abs_week")
    raw = exogenous.make_example_climate(w, 0)
    ta = raw[raw.unit == "A"].set_index("abs_week")["temp_mean"]
    assert F["temp_mean_l4"].iloc[0] == pytest.approx(ta[1100 - 4])
    assert F["temp_mean_l1"].iloc[0] == pytest.approx(ta[1099])
    assert e.features(np.array([0]), np.array([1000]))["temp_mean_l1"].isna().all()     # antes del inicio: NaN, no relleno


def test_cols_for_agrega_variables_exogenas():
    extra = {"tree": ["x_l1"], "negbin": ["x_l1"]}
    assert cols_for("rf", 6, extra)[-1] == "x_l1" and cols_for("negbin", 6, extra)[-1] == "x_l1" and "x_l1" not in cols_for("rf", 6)


def _vint(seed, W=260, U=2, D=6):
    rng = np.random.default_rng(seed)
    base = rng.poisson(20, size=(U, W)).astype(float)
    frac = np.array([0.4, 0.7, 0.85, 0.95, 0.99, 1, 1])[: D + 1]
    return Vintage(["A", "B"], 1000, np.round(base[:, :, None] * frac[None, None, :]))


def test_sin_fuga_con_clima_e_icen():
    """Alterar el clima y el ICEN posteriores a t_fit no cambia el pronóstico (rf, gb y binomial negativa)."""
    cfg = load_config()
    bt = copy.deepcopy(cfg["backtest"]); bt["lags_weeks"] = 6
    params = {"rf": {"n_estimators": 20, "max_depth": 4, "min_samples_leaf": 5},
              "gb": {"learning_rate": 0.1, "max_depth": 2, "max_iter": 30, "min_samples_leaf": 10}}
    v = _vint(1)
    t_fit, h = 1000 + 180, 2
    e1, _ = _exog(0)
    e2, _ = _exog(0)
    e2.cube[:, (t_fit + 1 - e2.w0):, :] = e2.cube[:, (t_fit + 1 - e2.w0):, :] * 9 + 7          # todo lo posterior a t_fit
    outs = []
    for e in (e1, e2):
        extra = {"tree": e.tree_cols, "negbin": e.negbin_cols}
        F = backtest.build_feats(v, np.arange(v.w0 + 8, t_fit + 1), 6, e)
        p = backtest.forecast_at(v, F, bt, params, [0, 1], t_fit, np.array([t_fit - 1, t_fit]), h, ["negbin", "rf", "gb"], v.w0 + 8, None, extra)
        outs.append(p.sort_values(["model", "u", "t"]).reset_index(drop=True))
    pd.testing.assert_frame_equal(outs[0], outs[1])


def test_binomial_negativa_no_imputa_faltantes_en_prueba():
    cfg = load_config()
    bt = copy.deepcopy(cfg["backtest"]); bt["lags_weeks"] = 6
    v, e = _vint(2), None
    e, _ = _exog(0)
    t_fit, h = 1000 + 180, 2
    e.cube[:, t_fit - 1 - e.w0, :] = np.nan                                                      # falta una semana justo antes del origen
    extra = {"tree": e.tree_cols, "negbin": e.negbin_cols}
    F = backtest.build_feats(v, np.arange(v.w0 + 8, t_fit + 1), 6, e)
    with pytest.raises(ValueError, match="exógenas"):
        backtest.forecast_at(v, F, bt, {"rf": {}, "gb": {}}, [0, 1], t_fit, np.array([t_fit]), h, ["negbin"], v.w0 + 8, None, extra)


def test_datos_de_ejemplo_estan_marcados():
    w = _weekly_like()
    assert exogenous.make_example_climate(w)["fuente"].str.startswith(exogenous.EXAMPLE_FLAG).all()
    assert exogenous.make_example_icen(w)["fuente"].str.startswith(exogenous.EXAMPLE_FLAG).all()
