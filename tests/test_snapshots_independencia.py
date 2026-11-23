import numpy as np
import pandas as pd

from dengue_pipeline.snapshots import simulate_triangle


def _w(units_cases):
    rows = [(u, w, c) for u, cs in units_cases.items() for w, c in enumerate(cs, start=1)]
    return pd.DataFrame(rows, columns=["unit", "abs_week", "cases"])


def test_los_sorteos_de_una_unidad_no_dependen_de_las_demas():
    pmf = np.array([0.5, 0.3, 0.2])
    a = simulate_triangle(_w({"A": [5, 9, 3, 7], "B": [4, 4, 4, 4]}), pmf, seed=7)
    b = simulate_triangle(_w({"A": [50, 1, 30, 2], "B": [4, 4, 4, 4]}), pmf, seed=7)   # cambian solo los casos de A
    pd.testing.assert_frame_equal(a[a.unit == "B"].reset_index(drop=True), b[b.unit == "B"].reset_index(drop=True))
