# Reproducir el pipeline: las tres personas corren lo mismo

Objetivo: que cada persona corra el pipeline completo en su computador y se compruebe que los resultados coinciden.
Si coinciden, el pipeline es reproducible. Todo se hace con dos comandos: `reproduce` y `compare-runs`.

> Las salidas de estas corridas **no son resultados citables** (falta la conformidad del profesor). Esto solo comprueba que el
> pipeline da lo mismo en cada computador. Los datos y los resultados no se suben al repositorio.

## 1. Preparar (una sola vez)

1. **Acceso.** El repositorio es privado: pide que te agreguen como colaborador y clónalo.
2. **Python 3.13.3 exacto** (no otra versión) o Docker con el `Dockerfile` del repositorio.
3. **Entorno.**
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate      Linux/Mac: source .venv/bin/activate
   pip install -r requirements.txt
   pip install --no-deps -e .
   python -m pytest -q          # deben pasar 110 y omitirse 7 (sin los archivos reales)
   ```
4. **Archivos de entrada: la misma copia, no los bajes de nuevo.** El portal de datos abiertos puede haber actualizado los CSV y
   entonces el hash cambia. Recíbelos de quien los descargó el 20-nov-2026 (por un canal privado, no por el repositorio: cada
   fuente tiene sus propios términos). El comando comprueba los hashes de los dos CSV contra `data/manifest.json`. Los dos XLSX del
   INEI no están en el manifiesto; compara su SHA-256 a mano:

   | Archivo | SHA-256 |
   |---|---|
   | CSV nacional (CDC Perú) | `7865e8b2f5a7aef1e8dce159444891f2c2e59b311d574f28c55aaca5a9abb8ba` |
   | CSV regional de Piura | `7034519f94396df2077657c453897b1972893f1e7be724c8c8bcb0ce7cc8d093` |
   | `proy_03_4.xlsx` (INEI, departamental) | `4138ba83babb0ecb96e5f2c2ccc684fa1ce217678f4460fc6e2abcb358be4ae6` |
   | `poblacion_distrital_2018_2026.xlsx` (INEI, distrital) | `9436df29b883fd4a9db3705040a6668ff4efe7047c2643249b6b6bedd90d5c8b` |

   Para calcular un hash: `Get-FileHash archivo -Algorithm SHA256` (PowerShell) o `sha256sum archivo` (Linux/Mac).
5. **Red.** El clima se descarga de NASA POWER (el comando lo hace solo). El ICEN ya está en `config/icen.csv`.

## 2. Correr todo (unas 2 horas, sin atención)

Mira primero el plan sin ejecutar nada (también comprueba los hashes):

```bash
python -m dengue_pipeline reproduce --national RUTA/nacional.csv --regional RUTA/piura.csv \
    --population-xlsx RUTA/proy_03_4.xlsx --districts RUTA/poblacion_distrital_2018_2026.xlsx --dry-run
```

Quita `--dry-run` para correrlo. Son 14 pasos, en orden: validar, integrar, retraso, simular, población, incidencia, ICEN, clima,
corrida principal, alertas, sensibilidad al retraso (3 escenarios x 5 semillas) y ablación desde 2014 (historial, clima, clima + ICEN).

- **No apagues ni reinicies el computador** mientras corre. Si se corta: `... reproduce ... --from NOMBRE_DEL_PASO`
  (el paso que dice "ABORTA en ..." o el último que no terminó).
- Si el hash de un CSV no coincide, el comando se detiene: esos datos no son los del manifiesto. (`--allow-different-data`
  fuerza la corrida, pero entonces la comparación no vale.)
- Si el clima ya está descargado: `--skip-climate`. Para correr solo una parte: `--only principal,alertas`.
- Al terminar queda `data/processed/reproducir_log.json` con las versiones de Python y de cada paquete, el sistema y el tiempo de cada paso.

## 3. Comparar con la corrida de referencia

Cada persona le envía su carpeta `data/processed` (sin `data/raw`) a quien coordina, o al revés. Para compararla con otra:

```bash
python -m dengue_pipeline compare-runs data/processed RUTA/otra/processed
```

Para cada archivo dice `OK`, `DIFERENTE` (con la máxima diferencia numérica) o `FALTA`. Los números se comparan con tolerancia
1e-6 (`--tol` la cambia) y el orden de las filas no importa. Las fechas de descarga, rutas y tiempos se ignoran. Termina con código 0 si
todo coincide y con 1 si hay diferencias.

- **Todo OK:** reproducido. Se anota en el Método: "ejecutado de forma independiente por tres personas, con diferencia máxima de X".
- **Hay `DIFERENTE`:** envía el informe completo y el `reproducir_log.json`. Antes de culpar al código, revisa que sean las mismas versiones
  y los mismos datos de entrada. Si solo difieren los archivos de clima, probablemente NASA POWER actualizó valores: usa la misma
  copia de `climate_weekly.csv` y `--skip-climate`.
