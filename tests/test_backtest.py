"""El pronóstico en t_fit no puede depender de nada posterior a t_fit."""
import copy

import numpy as np
import pandas as pd

from dengue_pipeline import backtest
from dengue_pipeline.config import load_config
from dengue_pipeline.vintage import Vintage


def _vint(seed, W=260, U=2, D=6):
    rng = np.random.default_rng(seed)
    base = rng.poisson(20, size=(U, W)).astype(float)
    frac = np.array([0.4, 0.7, 0.85, 0.95, 0.99, 1, 1])[: D + 1]
    return Vintage(["A", "B"], 1000, np.round(base[:, :, None] * frac[None, None, :]))


def test_sin_fuga_hacia_el_futuro():
    cfg = load_config(None)
    bt = copy.deepcopy(cfg["backtest"])
    bt["lags_weeks"] = 6
    params = {"rf": {"n_estimators": 20, "max_depth": 4, "min_samples_leaf": 5},
              "gb": {"learning_rate": 0.1, "max_depth": 2, "max_iter": 30, "min_samples_leaf": 10}}
    v1 = _vint(1)
    t_fit, h = 1000 + 180, 2
    v2 = Vintage(v1.units, v1.w0, v1.cum.copy())
    s_future = t_fit - h  # lo que se sabe de semanas > t_fit o de t_fit+1 en adelante se altera
    v2.cum[:, (t_fit + 1 - v2.w0):, :] *= 7  # semanas de inicio posteriores a t_fit
    # y lo que llega después de t_fit sobre semanas anteriores:
    k_late = t_fit - v2.w0
    for s in range(v2.W):
        for k in range(v2.D1):
            if s + k > k_late:  # información que aún no estaba disponible en t_fit
                v2.cum[:, s, k] = v2.cum[:, s, k] * 5 + 3
    outs = []
    for v in (v1, v2):
        F = backtest.build_feats(v, np.arange(v.w0 + 5, t_fit + 1), 6)
        p = backtest.forecast_at(v, F, bt, params, [0, 1], t_fit, np.array([t_fit - 1, t_fit]), h,
                                 ["endemic", "negbin", "rf", "gb"], v.w0 + 5)
        outs.append(p.sort_values(["model", "u", "t"]).reset_index(drop=True))
    pd.testing.assert_frame_equal(outs[0], outs[1])
