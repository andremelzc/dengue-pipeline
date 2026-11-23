"""Reproducción con un solo comando: orden de los pasos, comprobación de hashes y parada ante un fallo."""
import json

import pytest

from dengue_pipeline import reproduce as rp


def _names(**kw):
    return [s.name for s in rp.build_steps("n.csv", "r.csv", "pob.xlsx", **kw)]


def test_orden_de_los_pasos_respeta_las_dependencias():
    n = _names(climate_downloaded_on="2026-11-23")
    assert n == ["validar", "integrar", "retraso", "simular", "poblacion", "incidencia", "icen", "clima", "principal", "alertas",
                 "sensibilidad", "hist2014", "clima2014", "clima_icen2014"]
    assert n.index("principal") < n.index("sensibilidad")          # la sensibilidad usa los hiperparámetros de la corrida principal
    assert n.index("hist2014") < n.index("clima2014") < n.index("clima_icen2014")
    assert n.index("clima") < n.index("clima2014") and n.index("icen") < n.index("clima_icen2014")


def test_opciones_para_omitir_pasos():
    assert "clima" not in _names(skip_climate=True)
    assert not {"sensibilidad"} & set(_names(sensitivity=False))
    assert not {"hist2014", "clima2014", "clima_icen2014"} & set(_names(ablation=False))


def test_los_comandos_incluyen_los_archivos_pedidos():
    s = {x.name: x for x in rp.build_steps("n.csv", "r.csv", "pob.xlsx", districts="dist.xlsx", climate_downloaded_on="2026-11-23")}
    assert s["poblacion"].argv == ["population", "pob.xlsx", "--districts", "dist.xlsx"]
    assert s["clima"].argv[-2:] == ["--downloaded-on", "2026-11-23"] and s["clima"].needs_network
    assert s["clima_icen2014"].argv == ["backtest", "--train-first-year", "2014", "--features", "clima_icen"]


def test_seleccion_desde_un_paso_y_solo_algunos():
    steps = rp.build_steps("n.csv", "r.csv", "pob.xlsx", climate_downloaded_on="2026-11-23")
    assert [s.name for s in rp.select(steps, start="principal")][0] == "principal"
    assert [s.name for s in rp.select(steps, only=["alertas", "icen"])] == ["icen", "alertas"]
    with pytest.raises(ValueError):
        rp.select(steps, start="no_existe")


def test_verificacion_de_hashes(tmp_path):
    nat, reg = tmp_path / "n.csv", tmp_path / "r.csv"
    nat.write_text("a,b\n1,2\n", encoding="utf-8")
    reg.write_text("c\n3\n", encoding="utf-8")
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps([{"kind": "national", "sha256": rp.sha256_file(nat)},
                               {"kind": "regional", "sha256": "0" * 64}]), encoding="utf-8")
    res = {r["kind"]: r["ok"] for r in rp.verify_inputs({"national": nat, "regional": reg}, man)}
    assert res == {"national": True, "regional": False}


def test_dry_run_no_ejecuta_nada(tmp_path):
    log = tmp_path / "log.json"
    assert rp.run([rp.Step("x", ["no-existe"])], dry_run=True, log=lambda *_: None, log_path=log) == 0
    assert not log.exists()


def test_se_detiene_en_el_primer_fallo_y_deja_registro(tmp_path):
    log = tmp_path / "log.json"
    steps = [rp.Step("bien", ["--help"]), rp.Step("mal", ["comando-que-no-existe"]), rp.Step("nunca", ["--help"])]
    code = rp.run(steps, log=lambda *_: None, log_path=log)
    rec = json.loads(log.read_text(encoding="utf-8"))
    assert code != 0 and [p["paso"] for p in rec["pasos"]] == ["bien", "mal"]
    assert rec["entorno"]["python"] and "pandas" in rec["entorno"]["paquetes"]
