"""Comparación de dos corridas: tolerancia numérica, orden de filas indiferente y archivos que faltan."""
import json

import numpy as np
import pandas as pd

from dengue_pipeline import cli
from dengue_pipeline import compare_runs as cr


def _preds(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"unit": ["A", "B"] * (n // 2), "origin_week": np.arange(n), "horizon": 2, "model": "gb",
                         "pred": rng.random(n) * 100, "obs": rng.random(n) * 100})


def _write(root, preds=None, hp=None):
    (root / "backtest").mkdir(parents=True, exist_ok=True)
    (preds if preds is not None else _preds()).to_csv(root / "backtest" / "predictions.csv.gz", index=False)
    (root / "backtest" / "hyperparameters.json").write_text(json.dumps(hp or {"h2": {"rf": {"max_depth": None}, "cv": 0.3}}), encoding="utf-8")


def _states(a, b, tol=1e-6):
    return {r["archivo"]: r["estado"] for r in cr.compare_dirs(a, b, tol)}


def test_corridas_identicas_coinciden(tmp_path):
    _write(tmp_path / "a"); _write(tmp_path / "b")
    assert set(_states(tmp_path / "a", tmp_path / "b").values()) == {"OK"}


def test_el_orden_de_las_filas_no_importa(tmp_path):
    _write(tmp_path / "a"); _write(tmp_path / "b", _preds().sample(frac=1, random_state=1))
    assert _states(tmp_path / "a", tmp_path / "b")["backtest/predictions.csv.gz"] == "OK"


def test_diferencia_pequena_pasa_y_grande_no(tmp_path):
    p = _preds()
    q = p.copy(); q["pred"] += 1e-9
    r = p.copy(); r["pred"] += 0.5
    _write(tmp_path / "a", p); _write(tmp_path / "b", q); _write(tmp_path / "c", r)
    assert _states(tmp_path / "a", tmp_path / "b")["backtest/predictions.csv.gz"] == "OK"
    bad = [x for x in cr.compare_dirs(tmp_path / "a", tmp_path / "c") if x["archivo"].endswith("predictions.csv.gz")][0]
    assert bad["estado"] == "DIFERENTE" and "0.5" in bad["detalle"]


def test_archivo_que_falta_y_filas_distintas(tmp_path):
    _write(tmp_path / "a"); _write(tmp_path / "b", _preds(n=48))
    (tmp_path / "b" / "backtest" / "hyperparameters.json").unlink()
    st = _states(tmp_path / "a", tmp_path / "b")
    assert st["backtest/hyperparameters.json"] == "FALTA" and st["backtest/predictions.csv.gz"] == "DIFERENTE"


def test_json_ignora_fechas_de_descarga_pero_no_los_hashes(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d, when, h in ((a, "2026-11-20", "x"), (b, "2026-11-23", "x")):
        d.mkdir(); (d / "climate_manifest.json").write_text(json.dumps([{"unit": "P", "sha256": h, "downloaded_on": when}]), encoding="utf-8")
    assert _states(a, b)["climate_manifest.json"] == "OK"
    (b / "climate_manifest.json").write_text(json.dumps([{"unit": "P", "sha256": "otro", "downloaded_on": "2026-11-23"}]), encoding="utf-8")
    assert _states(a, b)["climate_manifest.json"] == "DIFERENTE"


def test_el_comando_devuelve_0_si_coinciden_y_1_si_no(tmp_path, capsys):
    _write(tmp_path / "a"); _write(tmp_path / "b"); _write(tmp_path / "c", _preds(seed=9))
    assert cli.main(["compare-runs", str(tmp_path / "a"), str(tmp_path / "b")]) == 0
    assert cli.main(["compare-runs", str(tmp_path / "a"), str(tmp_path / "c")]) == 1
    assert "HAY DIFERENCIAS" in capsys.readouterr().out
