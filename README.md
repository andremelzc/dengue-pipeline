# dengue-pipeline

Pipeline reproducible, en Python y por línea de comandos, para evaluar alertas de dengue con los datos
disponibles en cada semana (incluido el retraso de notificación). Es un backtesting, no un sistema en operación.

**Estado (22-nov-2026): etapas 1 a 6 implementadas y probadas, más población/incidencia/niveles de alerta, métricas de alerta, EDA y la ablación con clima (NASA POWER) e ICEN. El CI (Python 3.11 a 3.13), la imagen de Docker y la descarga de clima están verificados. Faltan los autores y las conformidades del equipo: ver "Lo que NO está hecho".**

> **Las salidas de las corridas actuales NO son resultados.** Todas usan un retraso de notificación PROVISIONAL (el de Piura 2025 aplicado a todas las unidades). No se citan ni se publican cifras de desempeño hasta fijar el retraso definitivo.

## Instalación y pruebas

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,notebook,icen]"
pytest                      # 83 pasan y 7 se saltan sin los archivos reales (probado 22-nov-2026)
# pruebas que reproducen cifras de datos.md con los CSV reales (con los tres archivos pasan las 90):
DENGUE_NATIONAL_CSV=/ruta/nacional.csv DENGUE_REGIONAL_CSV=/ruta/regional.csv DENGUE_POPULATION_XLSX=/ruta/proy_03_4.xlsx pytest
```

Probado con Python 3.13 en dos entornos: pandas 3.0.5, numpy 2.5.3, pyyaml 6.0.3, pytest 9.1.1; y una instalación limpia con pandas 3.1.1, numpy 2.5.5, pyyaml 6.0.4, pytest 9.2.0.

## Uso (los CSV se descargan a mano; ver datos.md 1.1)

```bash
python -m dengue_pipeline register data/raw/nacional.csv --kind national --source-url <URL> --downloaded-on 2026-11-20
python -m dengue_pipeline register data/raw/piura.csv    --kind regional --source-url <URL> --downloaded-on 2026-11-20
python -m dengue_pipeline validate  --national data/raw/nacional.csv --regional data/raw/piura.csv   # reports/validation_report.md
python -m dengue_pipeline integrate --national data/raw/nacional.csv                                   # data/processed/weekly_final.csv
python -m dengue_pipeline delay     --regional data/raw/piura.csv                                      # data/processed/delay_pmf.csv
python -m dengue_pipeline simulate  --national data/raw/nacional.csv --regional data/raw/piura.csv     # data/processed/triangle_simulated.csv.gz
python -m dengue_pipeline pair      --national data/raw/nacional.csv --regional data/raw/piura.csv     # par real Piura 2023
```

`validate` termina con código 1 si alguna regla da FALLA (las ADVERTENCIAS no bloquean). Todo parámetro está en `config/config.yaml`.

## Qué hace cada etapa

| Etapa | Módulo | Qué hace |
|---|---|---|
| 1 Adquisición | `acquire.py` | Manifiesto con hash SHA-256, filas, líneas físicas y fecha de descarga (la declara quien descarga) |
| 2 Validación | `validate.py` | Reglas ligadas a los defectos observados; informe en JSON y Markdown |
| 3 Integración | `integrate.py`, `epiweek.py` | Tabla unidad x semana epidemiológica (MMWR), semanas vacías = 0, clima semanal, rezagos sin futuro |
| 4 Instantáneas | `snapshots.py` | Disponibilidad por fecha de notificación, distribución del retraso, simulación para las demás unidades |

**Definición central.** Un caso está disponible en la semana t si su fecha de notificación es anterior o igual al fin de la semana t
(para casos notificados antes de su propio inicio de síntomas, desde el inicio). La instantánea de t nunca contiene nada posterior a t:
lo comprueban `tests/test_snapshots.py` (altera todo lo que llega después de t y exige que la instantánea no cambie) y `assert_no_future`.

## Hallazgos de la primera ejecución con los CSV reales (no estaban en datos.md)

1. **81 casos de 2009 con semana 53** (`semana`), año que tiene 52 semanas MMWR. Fuera de la ventana 2010-2024; no se corrigen. Causa desconocida.
2. **310 casos del corte regional de 2023 con notificación el 18-may-2023**, un día después de `FECHA_CORTE` (17-may). Misma semana epidemiológica, así que no afecta a los conteos semanales.
3. **43 296 filas con el carácter de reemplazo (U+FFFD)** en nombres (p. ej. "CA�ETE"): unir por `ubigeo`, nunca por nombre.
4. **148 787 filas idénticas en todas las columnas** en el CSV nacional. No hay identificador de caso, así que pueden ser casos distintos: no se eliminan.
5. `FECHA_INGRESO` en el corte de 2026: 60,1 % posterior a la notificación y 39,4 % el mismo día (99,5 % "posterior o igual"). `datos.md` 2.3 dice "posterior en el 99,6 %": coincide solo si se lee como "posterior o igual".

Reproducen lo ya anotado en datos.md: filas y `tipo_dx` del nacional (3.1), totales anuales (3.2), la deriva de significado entre cortes (2.3),
los 66 casos con notificación anterior al inicio (3.5), las semanas finales incompletas (3.6), el par real de Piura 2023 (3.7) y el último año incompleto (3.9).

## Lo que NO está hecho

**Datos que hay que traer ([FALTA])**
- Coordenadas: `config/coordenadas.csv` trae una PROPUESTA (punto de la capital departamental; Comas para Lima Norte) con su fuente web, que no es oficial: verificar contra INEI/IGN y confirmar el criterio de un punto por unidad con el profesor.
- **ICEN por versión publicada**: HECHO. Cargado en `config/icen.csv` (150 informes del IGP, 2014-01 a 2026-10; se armó con `python -m dengue_pipeline icen-reports <carpeta con los PDF del IGP>`). Queda por confirmar con el equipo qué hacer con las 22 filas de `reports/icen_revision_manual.csv` (año mal escrito o valores repetidos en los informes): hoy quedan fuera. Limitaciones que se declaran: no cubre 2010-2013 (por eso la ablación entrena desde 2014), la columna publicada cambia con los años (ICEN, ICENv3, ICENv5) y 17 de 158 meses cambian de valor entre informes.
- **Población de Lima Norte**: HECHO. Suma de sus 8 distritos (incluye San Martín de Porres) en el cuadro distrital del INEI 2018-2026 (gob.pe, 31-dic-2025; misma base censo 2017 que el cuadro departamental). Solo existe desde 2018: Lima Norte tiene tasa y niveles de alerta en 2020-2024 (años de prueba); antes de 2018, no. Se corre con `python -m dengue_pipeline population <proy_03_4.xlsx> --districts <cuadro distrital.xlsx>`.
- Fuente del canal endémico (mediana de 5 años) [SIN FUENTE]; fecha y URL de descarga de cada archivo para el manifiesto.

**Verificado el 22-nov-2026**
- La descarga de NASA POWER (`climate`): 6 unidades, 2009-11-01 a 2024-12-31, 5 540 días por unidad, sin valores -999; 6 semanas incompletas quedan marcadas y sin rellenar. Hash SHA-256 en `data/processed/climate_manifest.json`.
- El flujo de GitHub Actions: pasa en Python 3.11, 3.12 y 3.13.
- El `Dockerfile`: la imagen se construye y las pruebas pasan dentro (83 pasan, 7 se saltan).

**Decisiones del equipo [DECISIÓN PENDIENTE]**
- Protocolo `backtest:` APROBADO por el usuario el 22-nov-2026 (falta la conformidad del profesor/equipo). Siguen abiertos los [PROVISIONAL]: retraso de notificación, semillas, bloque del bootstrap.
- Valores [PROVISIONAL] de `config.yaml` (semanas excluidas al estimar el retraso, retraso máximo, semilla, umbrales de validación, rezagos de clima e ICEN).
- Tasa semanal: hoy es la incidencia acumulada de 4 semanas con la población del año epidemiológico (propuesta).
- Autores: `LICENSE` (MIT, elegida el 22-nov-2026; los datos del CDC/INEI tienen sus propios términos) lleva `[FALTA: autores del artículo]`; completar también `docs/CITATION.cff.plantilla`.

## Límites que se declaran

- El retraso se mide solo sobre casos que ya llegaron al archivo, así que aparece subestimado (datos.md 3.6).
- Las instantáneas reales suponen que el archivo de un corte no omite reclasificaciones ni bajas posteriores.
- El retraso simulado en las demás unidades es un supuesto: se observa solo en Piura.
- La serie final del CSV nacional es por semana de **inicio de síntomas**; 2025 está incompleto en la costa (no usar para evaluar falsas alarmas).


## Etapas 5 y 6 (backtesting) y ablación
```bash
python -m dengue_pipeline backtest                                   # condición `historial` (~7 min); hiperparámetros congelados en hyperparameters.json
python -m dengue_pipeline backtest --features clima                  # + clima local (necesita data/processed/climate_weekly.csv)
python -m dengue_pipeline backtest --features clima_icen             # + ICEN (necesita data/processed/icen_weekly.csv)
python -m dengue_pipeline backtest --features clima_icen --example-exog --models gb,negbin   # SOLO prueba de la interfaz, datos SINTÉTICOS
```
Salidas en `data/processed/backtest/`: `predictions*.csv.gz`, `metrics_summary*.csv`, `paired_difference*.csv`, `hyperparameters.json`. Las corridas con datos de
ejemplo llevan el prefijo `EJEMPLO_NO_REAL_`. Todo el bloque `backtest:` de config.yaml es PROPUESTA (ver `reports/protocolo_evaluacion.md`).

```bash
python -m dengue_pipeline climate --coordinates config/coordenadas.csv --start 20091101 --end 20241231 --downloaded-on AAAA-MM-DD   # necesita red
python -m dengue_pipeline icen config/icen.csv                       # ICEN conocido en cada semana (versión publicada hasta esa fecha); hacerlo ANTES de --features clima_icen
```
Reglas: el clima de la semana t-k entra con k >= 1; el ICEN es el valor publicado hasta el fin de cada semana; la ablación reutiliza los hiperparámetros de `historial`.

## Métricas de alerta
```bash
python -m dengue_pipeline alerts      # -> alert_metrics.csv, alert_paired_difference.csv, alert_episodes.csv, alert_episode_summary.csv
```
Sensibilidad, tasa de falsa alarma, falsas alarmas sobre alertas y anticipación por episodio, con cortes 50/150 y 100/300 y umbrales "amarillo o más" y "rojo".
IC por bootstrap de bloques de semanas, por año y por unidad; diferencia pareada entre condiciones. La anticipación por episodio no lleva IC (pocos episodios).
Las salidas sobre el retraso PROVISIONAL llevan la nota "no citar como resultado".

## EDA
`python -m dengue_pipeline eda` -> figuras `reports/eda_0*.png` y tablas; lectura en `reports/eda_resumen.md`.

## Notebook
`notebooks/ejecutar_pipeline.ipynb` ejecuta todo en orden (instalar con `pip install -e ".[dev,notebook]"`). Revisar primero su celda de parámetros.

## Integración continua y contenedor
`.github/workflows/ci.yml` (Python 3.11 a 3.13, `pytest`) y `Dockerfile`: verificados el 22-nov-2026 (CI en GitHub Actions; imagen construida y pruebas dentro del contenedor). `make test` corre las pruebas.

## Población, incidencia y niveles de alerta (nuevo)
```bash
python -m dengue_pipeline population proy_03_4.xlsx     # -> data/processed/population.csv (unit, year, population)
python -m dengue_pipeline incidence                      # -> data/processed/weekly_incidence.csv
```
Fuente: cuadro del INEI "Población estimada al 30 de junio, por años calendario y sexo, según departamento, 2000-2026"
(el propio archivo cita el Boletín especial N.° 25, 1995-2030). Cada semana usa la población de su año epidemiológico; la tasa es la
incidencia acumulada de 4 semanas (t y las 3 anteriores) por 100 000 [DECISIÓN PENDIENTE: propuesta]. Niveles: 0 verde (< 50), 1 amarillo (50 a < 150),
2 rojo (>= 150); `level_sens` usa 100 y 300; -1 = sin tasa. La prueba con el archivo real se activa con `DENGUE_POPULATION_XLSX=/ruta/proy_03_4.xlsx`.


## Cambios del 22-nov-2026 (tarde)
- Lima Norte pasa a 8 distritos (se agregó San Martín de Porres). Sus casos 2010-2024 suben de 40 854 a 46 690; las demás unidades no cambian (serie final: diferencia máxima de predicciones 5e-11).
- La simulación del retraso ahora usa una semilla por unidad (derivada de la semilla y el nombre de la unidad): antes una sola secuencia aleatoria hacía que un cambio en una unidad moviera las simulaciones de otras. Hay una prueba. Las predicciones de la condición `disponibles` cambian respecto a corridas anteriores (siguen siendo de prueba, retraso PROVISIONAL).
- `backtest --train-first-year 2014` corre con entrenamiento desde 2014 (ICEN solo existe desde 2014) y guarda en `data/processed/backtest_desde2014/`. Para la ablación, correr las tres condiciones con este mismo valor.
- `data/manifest.json`: fecha de descarga 2026-11-20 (indicada por el usuario).
