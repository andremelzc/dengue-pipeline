"""Los cuatro modelos acordados y una referencia.

  endemic      línea base de canal endémico (mediana histórica de la misma semana); no se entrena
  negbin       binomial negativa con rezagos (GLM/MLE de statsmodels), con offset log1p(c4)
  rf           random forest (scikit-learn)
  gb           gradient boosting (HistGradientBoostingRegressor de scikit-learn; "XGBoost o similar")
  persistence  REFERENCIA (no es uno de los cuatro): el acumulado de 4 semanas actual se repite

Todos devuelven casos acumulados de 4 semanas pronosticados (>= 0). Hiperparámetros: los fija
`tune()` con datos hasta cfg['backtest']['tuning_until_year'] y quedan congelados; los valores
marcados [PENDIENTE DE IMPLEMENTAR] en el Método son los que aparecen en hyperparameters.json.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

MODEL_NAMES = ["endemic", "negbin", "rf", "gb"]
REFERENCE = "persistence"

DEFAULT_PARAMS = {
    "rf": {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 10},
    "gb": {"learning_rate": 0.05, "max_depth": 3, "max_iter": 200, "min_samples_leaf": 20},
}
GRIDS = {
    "rf": {"max_depth": [4, 8, None], "min_samples_leaf": [5, 20]},
    "gb": {"learning_rate": [0.05, 0.1], "max_depth": [2, 3], "max_iter": [100, 300]},
}


class TreeModel:
    """RF o GB sobre el cambio relativo z = log1p(y4) - log1p(c4)."""

    def __init__(self, kind: str, params: dict, seed: int):
        self.kind, self.params, self.seed = kind, dict(params), seed
        self.m = None

    def fit(self, X: pd.DataFrame, z: np.ndarray):
        if self.kind == "rf":
            self.m = RandomForestRegressor(n_jobs=2, random_state=self.seed, **self.params)
        else:
            self.m = HistGradientBoostingRegressor(random_state=self.seed, **self.params)
        self.m.fit(X.to_numpy(), z)
        return self

    def predict(self, X: pd.DataFrame, c4_log: np.ndarray) -> np.ndarray:
        return np.maximum(np.expm1(c4_log + self.m.predict(X.to_numpy())), 0.0)


class NegBin:
    """Binomial negativa con desplazamiento log1p(c4): y4 ~ NB(mu), log mu = log1p(c4) + X b.

    Estima alpha por máxima verosimilitud (statsmodels). Si no converge se cae a una regresión de Poisson
    y se anota en `self.fallback` (así el resultado nunca falla en silencio).
    """

    def __init__(self):
        self.res, self.cols, self.fallback = None, None, None

    def fit(self, X: pd.DataFrame, y: np.ndarray, c4_log: np.ndarray):
        import statsmodels.api as sm
        self.cols = list(X.columns)
        Xc = sm.add_constant(X.to_numpy(), has_constant="add")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r = sm.NegativeBinomial(y, Xc, offset=c4_log).fit(disp=0, maxiter=300, method="bfgs")
                if not np.all(np.isfinite(r.params)) or not r.mle_retvals.get("converged", False):
                    raise RuntimeError("sin convergencia")
                self.res = r
            except Exception as e:  # noqa: BLE001
                self.fallback = f"Poisson ({type(e).__name__}: {e})"
                self.res = sm.GLM(y, Xc, family=sm.families.Poisson(), offset=c4_log).fit()
        return self

    def predict(self, X: pd.DataFrame, c4_log: np.ndarray) -> np.ndarray:
        import statsmodels.api as sm
        Xc = sm.add_constant(X.to_numpy(), has_constant="add")
        eta = np.clip(Xc @ self.res.params[: Xc.shape[1]] + c4_log, -20, 20)
        return np.maximum(np.exp(eta), 0.0)


def tune(X: pd.DataFrame, z: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]], seed: int) -> dict:
    """Elige hiperparámetros de rf y gb por error absoluto en escala log, con pliegues ORDENADOS en el tiempo.

    `folds` es una lista de (índices de entrenamiento, índices de validación) donde la validación es siempre
    posterior al entrenamiento. Solo debe recibir datos anteriores al periodo de prueba.
    """
    best = {}
    for kind, grid in GRIDS.items():
        scores = []
        keys = list(grid)
        for vals in itertools.product(*[grid[k] for k in keys]):
            p = {**DEFAULT_PARAMS[kind], **dict(zip(keys, vals))}
            err = []
            for tr, va in folds:
                m = TreeModel(kind, p, seed).fit(X.iloc[tr], z[tr])
                err.append(np.mean(np.abs(m.m.predict(X.iloc[va].to_numpy()) - z[va])))
            scores.append((float(np.mean(err)), p))
        best[kind] = min(scores, key=lambda s: s[0])[1]
        best[kind + "_cv_error"] = min(scores, key=lambda s: s[0])[0]
    return best
