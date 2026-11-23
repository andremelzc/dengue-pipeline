from dengue_pipeline.icen_reports import parse_report_text

TXT = """Informe Técnico PpR/El Niño-IGP/2024-05
INFORME TÉCNICO Nº PpR/El Niño-IGP/2024-05
10/06/2024
Año Mes ICEN Condiciones costeras del Mes
2024 Marzo 0.50 Neutro
2024 Abril 0.80 Cálida débil
Año Mes ICENtmp Condiciones costeras
2024 Mayo 1.10 Cálida débil
Año Mes ONI
2024 Abril 0.20
"""


def test_lee_definitivo_y_temporal_y_ignora_otras_tablas():
    good, review, meta = parse_report_text(TXT, "x.pdf")
    assert meta["published_on"] == "2024-06-10" and meta["report"] == "2024-05"
    assert [(r["month"], r["kind"], r["value"]) for r in good] == [
        ("2024-03", "definitivo", 0.5), ("2024-04", "definitivo", 0.8), ("2024-05", "temporal", 1.1)]
    assert review == []


def test_temporal_de_mes_definitivo_va_a_revision_no_a_la_tabla():
    t = TXT.replace("2024 Mayo 1.10", "2023 Mayo 1.10")
    good, review, _ = parse_report_text(t)
    assert len(good) == 2 and len(review) == 1 and "ya_definitivo" in review[0]["motivo"]


def test_sin_fecha_no_entra():
    good, review, _ = parse_report_text(TXT.replace("10/06/2024", ""))
    assert good == [] and len(review) == 3


def test_columna_icenv5_cuando_viene_despues_de_icenoi():
    t = """INFORME TÉCNICO Nº PpR/El Niño-IGP/2020-04
15 /05/2020
Año Mes ICENOI Categoría ICENv5 Categoría
2020 Marzo 0.47 Cálida Débil 0.10 Neutra
"""
    good, review, meta = parse_report_text(t)
    assert meta["published_on"] == "2020-05-15"
    assert [(r["month"], r["value"], r["serie"]) for r in good] == [("2020-03", 0.10, "ICENv5")]
