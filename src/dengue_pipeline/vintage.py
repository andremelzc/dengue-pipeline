"""Vista de 'lo conocido en la semana t' sobre un tensor de conteos acumulados por retraso.

cum[u, s - w0, k] = casos de la unidad u con inicio de síntomas en la semana s que ya están disponibles
cuando han pasado k semanas (k = 0 es la misma semana). known(t, s) devuelve lo que se sabe en t sobre la
semana s: cum[u, s - w0, min(t - s, D)]. Para t < s devuelve 0 (la semana aún no ocurre).

Dos tensores se construyen aquí:
- condición "serie final": todos los casos disponibles desde el retraso 0 (se ve el conteo final siempre);
- condición "datos disponibles": a partir del triángulo (real o simulado) de snapshots.py.
Ambas comparten el mismo código de variables y de modelos: solo cambia el tensor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class Vintage:
    def __init__(self, units: list[str], w0: int, cum: np.ndarray):
        self.units, self.w0, self.cum = list(units), int(w0), cum  # cum: (U, W, D+1), acumulado en k
        self.U, self.W, self.D1 = cum.shape
        self.D = self.D1 - 1

    @property
    def w_last(self) -> int:
        return self.w0 + self.W - 1

    def uidx(self, unit: str) -> int:
        return self.units.index(unit)

    def known(self, u: int | np.ndarray, t: int | np.ndarray, s: int | np.ndarray) -> np.ndarray:
        """Casos de la semana s conocidos en la semana t (arreglos broadcast). t < s -> 0."""
        t, s, u = np.broadcast_arrays(np.asarray(t), np.asarray(s), np.asarray(u))
        k = np.clip(t - s, -1, self.D)
        si = np.clip(s - self.w0, 0, self.W - 1)
        out = self.cum[u, si, np.maximum(k, 0)].astype(float)
        out[(k < 0) | (s < self.w0) | (s > self.w_last)] = 0.0
        return out

    def final(self, u, s):
        return self.known(u, np.full_like(np.asarray(s), self.w_last + self.D + 1), s)


def from_final(weekly: pd.DataFrame, units: list[str], D: int) -> Vintage:
    """Condición serie final: el conteo final está disponible desde el retraso 0."""
    w0, w1 = int(weekly["abs_week"].min()), int(weekly["abs_week"].max())
    cum = np.zeros((len(units), w1 - w0 + 1, D + 1))
    for i, u in enumerate(units):
        sub = weekly[weekly["unit"] == u].set_index("abs_week")["cases"]
        cum[i, :, :] = sub.reindex(range(w0, w1 + 1), fill_value=0).to_numpy()[:, None]
    return Vintage(units, w0, cum)


def from_triangle(tri: pd.DataFrame, weekly: pd.DataFrame, units: list[str], D: int) -> Vintage:
    """Condición datos disponibles: cum[u, s, k] = suma de n con avail_week - onset_week <= k.

    Los casos con retraso mayor que D se acumulan en D (con la pmf de la simulación, truncada en D, no hay).
    Las semanas sin casos en el triángulo quedan en 0.
    """
    w0, w1 = int(weekly["abs_week"].min()), int(weekly["abs_week"].max())
    cum = np.zeros((len(units), w1 - w0 + 1, D + 1))
    t = tri[tri["unit"].isin(units)].copy()
    t = t[(t["onset_week"] >= w0) & (t["onset_week"] <= w1)]
    t["k"] = np.clip(t["avail_week"] - t["onset_week"], 0, D)
    ui = t["unit"].map({u: i for i, u in enumerate(units)}).to_numpy()
    np.add.at(cum, (ui, (t["onset_week"] - w0).to_numpy(), t["k"].to_numpy()), t["n"].to_numpy())
    return Vintage(units, w0, np.cumsum(cum, axis=2))
