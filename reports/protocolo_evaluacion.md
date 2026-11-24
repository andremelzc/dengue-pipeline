# Protocolo de evaluación (PROPUESTA; ninguna fila es decisión del equipo hasta que se apruebe)

| Punto | Propuesta | Estado |
|---|---|---|
| Datos | 2010–2024; núcleo de prueba 2020–2024 | prueba 2020–2024 = propuesta |
| División | Solo temporal. Nunca aleatoria. | propuesta |
| Ventana | Expansiva; reentrenar cada 4 semanas (la deslizante queda como sensibilidad) | propuesta |
| Hiperparámetros | Elegidos una vez con datos ≤ 2019 (4 pliegues ordenados con purga) y congelados | propuesta |
| Información | En la semana de origen t solo lo conocido en t; las etiquetas de entrenamiento también se toman con lo conocido en el momento de entrenar; guarda de 4 semanas | propuesta (probado: tests/test_backtest.py) |
| Objetivo | Casos acumulados de 4 semanas que terminan en t+h (h = 2 y 4). Árboles: cambio relativo frente al nivel actual (log). Binomial negativa: conteo con desplazamiento | propuesta |
| Modelos | canal endémico (referencia histórica), binomial negativa con rezagos, random forest, gradient boosting (+ persistencia como referencia extra) | modelos DECIDIDOS; detalles propuesta |
| Condiciones | serie final vs datos disponibles (triángulo simulado con el retraso de Piura, aprobado el 23-nov-2026) | DECIDIDO |
| Verdad | Serie final (sin retraso) | propuesta |
| Unidades | Un modelo con Piura, Lambayeque, La Libertad, Tumbes, San Martín; Lima Norte solo transferencia (exploratorio) | propuesta |
| Incertidumbre | Bootstrap por bloques de 8 semanas (1000 remuestreos, IC 95 %); resultados por año; los brotes no son independientes | propuesta |
| Alertas | Niveles por tasa por 100 000 (50 / 150; cortes 100 / 300); tasa = acumulado de 4 semanas con población del año de la semana (INEI, cuadro departamental 2000-2026) | implementado (`alerts`); tasa semanal = propuesta. **Lima Norte sin población [FALTA]: sin tasa ni niveles** |
| Clima / ICEN | Ablación de tres condiciones: historial; + clima local; + clima local + ICEN (hiperparámetros congelados de historial; clima con rezago >= 1 semana; ICEN con la versión publicada en cada fecha) | interfaz implementada y probada con datos de EJEMPLO. ICEN real por versión de publicación cargado (`config/icen.csv`, 150 informes del IGP, 2014-2026; ver `reports/icen_revision_manual.csv`). **[FALTA]: verificar coordenadas y descarga real de NASA POWER [NO VERIFICADO]. El ICEN no cubre 2010-2013 y la columna usada cambia con los años (ICEN, ICENv3, ICENv5): decisión del equipo** |

Límites: el retraso simulado usa la distribución de Piura para todas las unidades; el canal endémico es la mediana de 5 años previos [SIN FUENTE]; el árbol/NB no conoce el nivel de población.
