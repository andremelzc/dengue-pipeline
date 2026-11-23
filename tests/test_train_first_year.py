"""`train_first_year` limita de verdad el entrenamiento y los pliegues de la elección de hiperparámetros no dependen de él."""
import copy

import numpy as np
import pytest

from dengue_pipeline import backtest, models
from dengue_pipeline.config import load_config
from dengue_pipeline.epiweek import abs_week
from dengue_pipeline.vintage import Vintage


def _bt(first_year):
    bt = copy.deepcopy(load_config(None)["backtest"])
    bt["train_first_year"] = first_year
    bt["lags_weeks"] = 6
    return bt


def test_bordes_de_pliegues_iguales_a_los_de_siempre_y_sin_depender_de_train_first_year():
    esperado = [abs_week(y, 1) for y in (2014, 2016, 2018, 2020)]
    assert backtest.cv_edges(_bt(2010)) == esperado
    assert backtest.cv_edges(_bt(2014)) == esperado


def test_train_origin_con_el_valor_por_defecto_no_cambia_nada():
    primero = abs_week(2010, 1) + 7                    # inicio de la serie más los rezagos (después de la semana 1 de 2010)
    assert backtest.train_origin(_bt(2010), primero) == primero


def test_train_origin_2014_descarta_los_origenes_anteriores():
    assert backtest.train_origin(_bt(2014), abs_week(2010, 1) + 7) == abs_week(2014, 1)
    # si la serie empezara después de train_first_year, manda el inicio de la serie
    assert backtest.train_origin(_bt(2014), abs_week(2016, 1)) == abs_week(2016, 1)


def _vint(W=640, U=2, D=6):
    rng = np.random.default_rng(0)
    base = rng.poisson(20, size=(U, W)).astype(float)
    frac = np.array([0.4, 0.7, 0.85, 0.95, 0.99, 1, 1])[: D + 1]
    return Vintage(["A", "B"], abs_week(2010, 1), np.round(base[:, :, None] * frac[None, None, :]))


@pytest.mark.parametrize("first_year, folds_esperados", [(2010, 3), (2014, 2)])
def test_numero_de_pliegues_de_la_elección_de_hiperparámetros(monkeypatch, first_year, folds_esperados):
    monkeypatch.setattr(models, "GRIDS", {"rf": {"max_depth": [4], "min_samples_leaf": [5]},
                                          "gb": {"learning_rate": [0.1], "max_depth": [2], "max_iter": [20]}})
    monkeypatch.setitem(models.DEFAULT_PARAMS["rf"], "n_estimators", 10)
    bt, v = _bt(first_year), _vint()
    first_origin = v.w0 + bt["lags_weeks"] - 1
    F = backtest.build_feats(v, np.arange(first_origin, v.w_last + 1), bt["lags_weeks"])
    best = backtest.tune_hyperparameters(v, F, bt, [0, 1], 2, backtest.train_origin(bt, first_origin))
    assert best["n_folds"] == folds_esperados


def test_con_2014_la_eleccion_usa_menos_filas_que_con_2010(monkeypatch):
    monkeypatch.setattr(models, "GRIDS", {"rf": {"max_depth": [4], "min_samples_leaf": [5]},
                                          "gb": {"learning_rate": [0.1], "max_depth": [2], "max_iter": [20]}})
    monkeypatch.setitem(models.DEFAULT_PARAMS["rf"], "n_estimators", 10)
    v, rows = _vint(), {}
    for fy in (2010, 2014):
        bt = _bt(fy)
        first_origin = v.w0 + bt["lags_weeks"] - 1
        F = backtest.build_feats(v, np.arange(first_origin, v.w_last + 1), bt["lags_weeks"])
        rows[fy] = backtest.tune_hyperparameters(v, F, bt, [0, 1], 2, backtest.train_origin(bt, first_origin))["rows"]
    assert rows[2014] < rows[2010]
