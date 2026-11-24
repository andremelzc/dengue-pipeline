# Exploración de datos (EDA) y qué implica para el modelado

Fuente: serie final por semana de inicio de síntomas, 2010-2024, seis unidades (`weekly_final.csv`); tasas con población INEI (Lima Norte sin población [FALTA]);
retraso de `delay_pmf.csv` (patrón base, Piura 2025). Figuras `eda_01` a `eda_06` y tablas `eda_*.csv` en esta carpeta. Figuras y tablas: `python -m dengue_pipeline eda` (este resumen se escribe a mano).
Nada de lo que sigue es un resultado de desempeño de modelos.

## Lo que muestran los datos

| Hallazgo | Cifras (de `eda_estadisticas_por_unidad.csv` y `eda_casos_anuales.csv`) | Figura |
|---|---|---|
| **Escala muy distinta entre unidades y entre años** | Casos 2010-2024: Piura 214 831; La Libertad 91 178; Lambayeque 50 644; San Martín 41 446; Lima Norte 40 854; Tumbes 40 475. Piura pasa de 70 casos en 2019 a 79 304 en 2023 | eda_01 |
| **Años casi vacíos y años de brote concentrados** | 2018: 4 casos en Lambayeque, 3 en La Libertad, 10 en Lima Norte. El año de mayor número de casos es 2023 en Lambayeque, Piura y Tumbes, y 2024 en La Libertad, Lima Norte y San Martín | eda_01, eda_03 |
| **Años con tasa máxima de 4 semanas >= 150 por 100 000** (rojo) | Piura 6 años (2010, 2015, 2017, 2022, 2023, 2024); Tumbes 5 (2015, 2017, 2020, 2023, 2024); Lambayeque 2 (2023, 2024); La Libertad 2 (2023, 2024); San Martín 1 (2024) | eda_03 |
| **Muchas semanas sin casos** | Lima Norte 67,5 %; La Libertad 41,6 %; Lambayeque 33,0 %; Tumbes 11,0 %; Piura 10,2 %; San Martín 4,6 % | eda_04 |
| **Estacionalidad en la costa** | Semana epidemiológica del pico (mediana de años): Lambayeque 12; Piura 15; Tumbes 16; La Libertad 17; Lima Norte 18 | eda_02 |
| **San Martín no sigue el año calendario** | Pico mediano en la semana 49 con rango intercuartil de picos 6 a 48: la temporada cruza el cambio de año | eda_02 |
| **Autocorrelación fuerte y lenta** (log de casos semanales) | Rezago 1: 0,94 a 0,97. Rezago 8: 0,68 a 0,81. Rezago 26: 0,21 a 0,46 | eda_05 |
| **Retraso de notificación corto (patrón base)** | 54,0 % en la misma semana; 94,4 % en <= 1 semana; 97,1 % en <= 2; 98,2 % en <= 4; media 0,69 semanas | eda_06 |
| **Niveles de alerta desbalanceados** (2010-2024, cortes 50/150) | Semanas en rojo: Tumbes 12,7 %; Piura 9,0 %; La Libertad 4,1 %; Lambayeque 3,3 %; San Martín 0,4 % | eda_03 |

## Qué implica para el modelado

1. **Cambios de escala de órdenes de magnitud.** Se justifica que los árboles aprendan el cambio relativo respecto del nivel actual (ya implementado) y no el conteo: un bosque no extrapola por encima de lo visto. Un único modelo para varias unidades comparte la dinámica, pero no conoce la población (límite que ya declara el protocolo).
2. **Ceros frecuentes** (Lima Norte, La Libertad, Lambayeque). El desplazamiento `log1p(c4)` de la binomial negativa y el objetivo en escala log los manejan, pero una línea base de **mediana histórica** da 0 en semanas sin casos y, por construcción, rara vez alertará: conviene tenerlo presente al leer la sensibilidad del canal endémico (la fuente de esa referencia sigue [SIN FUENTE]).
3. **Autocorrelación alta.** La persistencia (repetir el acumulado actual) es una referencia exigente, sobre todo con horizonte 2, donde las ventanas de 4 semanas del dato y del objetivo se solapan. Cualquier modelo debe compararse contra ella. Además, con autocorrelación 0,68 a 0,81 a 8 semanas, el bootstrap de **bloques de 8 semanas** puede ser corto: conviene una sensibilidad con bloques más largos [DECISIÓN PENDIENTE].
4. **Estacionalidad.** Seno y coseno de periodo anual bastan para la costa; en San Martín el pico cruza el año, así que la semana del año no debe leerse como "inicio de temporada". El canal endémico (misma semana de años previos) sí es compatible.
5. **Pocos años de brote.** La información sobre alertas rojas viene de pocos años por unidad; esto condiciona el tamaño efectivo de la prueba (ver `revision_protocolo.md`).
6. **Retraso corto.** Con el retraso del patrón base, casi todo lo no disponible está en las 1 o 2 semanas más recientes. En el triángulo simulado, el acumulado de 4 semanas conocido en la semana t es, en la mediana, el 85,7 % a 86,9 % del final según la unidad (percentil 5: 73 % a 80 %; semanas con al menos 20 casos finales, de 2020 a 2024). Es una propiedad del retraso simulado a partir de Piura, no un resultado de desempeño de modelos; cambia con el patrón de retraso.
7. **2025** queda fuera: incompleto en la costa (no usar para evaluar falsas alarmas).
