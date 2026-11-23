# Revisión crítica del protocolo de evaluación (antes de aprobarlo)

Estado: el protocolo sigue siendo PROPUESTA. Esta revisión sale de leer `backtest.py`, `features.py`, `models.py`, `vintage.py`, `snapshots.py`, `metrics.py`
y `alerts_eval.py`, y de la EDA (`eda_resumen.md`). Las cifras son propiedades de los datos o del triángulo simulado, no resultados de modelos.
Prioridad: **A** = cambiar o decidir antes de correr la versión definitiva; **M** = declarar y, si es barato, probar; **B** = mejora menor.

## 1. Fuga de información

**Lo que está bien (comprobado en el código y con pruebas).**
- Las variables en la semana de origen t usan solo `known(t, s)` con s <= t. Las etiquetas de entrenamiento también son lo conocido en el momento de entrenar, con guarda de 4 semanas y una aserción explícita en `forecast_at`.
- Hay prueba que altera todo lo posterior a `t_fit` y exige predicciones idénticas (casos, y con clima e ICEN). Los hiperparámetros se eligen con datos hasta 2019, con purga, y se congelan.

**Lo que cambiaría o declararía.**

| # | Punto | Qué pasa | Qué haría | Prior. |
|---|---|---|---|---|
| 1.1 | **El retraso se estima con datos posteriores al periodo de prueba** | `delay_pmf` sale de casos de 2025 con el corte de 2026-07-02 y se aplica a 2010-2024 (prueba 2020-2024). No filtra valores de casos, pero sí el mecanismo de reporte: en 2020 nadie conocía el retraso de 2025 | Declararlo como supuesto de estacionariedad del retraso. No es corregible con los datos actuales (solo existe ese corte regional) | M |
| 1.2 | **Población con proyección posterior** | Las tasas de 2020-2024 usan la proyección INEI vigente en 2026, no la que existía en cada momento. El efecto es pequeño frente a los cambios de escala (decenas de veces), pero es retrospectivo | Declararlo; no hay población "al momento" en los datos | B |
| 1.3 | **Una sola realización del retraso** | `simulate_triangle` sortea con una semilla. Todas las conclusiones de "datos disponibles" dependen de ese sorteo | Repetir con varias semillas (número [PENDIENTE DE DECIDIR]) y reportar la dispersión de la diferencia entre condiciones | A |
| 1.4 | **La condición "serie final" no es realizable** | Usa conteos finales de las semanas recientes, que en tiempo real no existen. Es justo el contraste del artículo, pero debe llamarse "referencia retrospectiva", no "desempeño alcanzable" | Redactarlo así en el Método | M |
| 1.5 | **Hiperparámetros afinados en la condición serie final** | Se afinan con etiquetas finales y se reutilizan en "datos disponibles" y en la ablación | Aceptable (aísla el efecto de los datos), pero declararlo; si se quiere cerrar la duda, afinar también con datos disponibles como sensibilidad | B |
| 1.6 | **Verdad por inicio de síntomas, disponibilidad por notificación** | El nivel observado sale de la serie final por inicio de síntomas. Coherente con el triángulo, pero un sistema real emitiría el nivel con casos notificados | Declarar la definición de nivel observado | B |

## 2. Tamaño efectivo de la prueba

Los números (de `weekly_incidence.csv` y de los episodios observados con cortes 50/150):
- 261 semanas por unidad en 2020-2024, cinco unidades con población: **1 305** filas por modelo y horizonte, pero muy dependientes entre sí.
- **Semanas en rojo: 158**, de las cuales 146 caen en 2023 (84) y 2024 (62); 2022 tiene 2, 2020 tiene 10 y 2021 ninguna. Semanas en amarillo o más: 56 (2020), 34 (2021), 31 (2022), 169 (2023), 136 (2024).
- **Episodios observados en rojo (no censurados): 11 en total** (Lambayeque 2, La Libertad 2, Piura 3, San Martín 1, Tumbes 3); 9 de los 11 en 2023-2024. En amarillo o más: 21 (horizonte 2).
- Con cortes 100/300 hay aún menos (6 episodios en rojo y 17 en amarillo o más).

Consecuencias y qué haría:
1. **El IC por semanas subestima la incertidumbre de la sensibilidad en rojo**, porque 9 de 11 episodios son de dos años y pocas unidades. Con 11 episodios, la anticipación por episodio es descriptiva: sin IC (ya está así en `alerts_eval.py`) y sin generalizar. **A**: reportar siempre conteos (episodios detectados / episodios) junto a la razón.
2. **Sensibilidad "dejando un año fuera"** (2023 y 2024 por separado) para ver cuánto depende todo de esos años. **A** (barato: son filtros sobre las mismas predicciones).
3. **Ampliar la prueba**: el núcleo 2020-2024 deja fuera 2015 y 2017, años con tasas máximas >= 150 en Piura y Tumbes. Una prueba desde 2017 (o con origen móvil desde 2015) sumaría episodios, a costa de reducir el entrenamiento y rehacer la afinación (hoy hasta 2019, con pliegues que llegan a 2020). Es una **decisión del equipo**: [DECISIÓN PENDIENTE]; no la tomo yo.
4. **Elegir de antemano un resultado primario** (por ejemplo, sensibilidad del nivel amarillo o más a 4 semanas, o la diferencia pareada), porque la rejilla de cuadros (5 modelos x 2 horizontes x 2 condiciones x 2 cortes x 2 umbrales x grupos) invita a elegir lo favorable. **A**.
5. **Bloques del bootstrap**: la autocorrelación a 8 semanas es de 0,68 a 0,81 (EDA); 8 semanas puede ser corto. Hacer una sensibilidad con bloques más largos. **M** [DECISIÓN PENDIENTE sobre el valor].

## 3. Lima Norte por transferencia

- **Sin población, no hay tasa ni nivel**: el cuadro del INEI es departamental y Lima Norte son 7 distritos [FALTA]. Hoy Lima Norte solo puede evaluarse por error en casos, no por alertas. Sustituir por la población de todo el departamento de Lima sería incorrecto: sobreestimaría la población y subestimaría las tasas.
- **Es un cambio de dominio**, no solo falta de historia: otra escala, 67,5 % de semanas con 0 casos (la mayor de todas), sistema de salud distinto al de Piura (donde se estimó el retraso) y 7 distritos agrupados con criterio [A CONFIRMAR] con el profesor.
- **Qué haría:** (a) dejar Lima Norte fuera de las métricas de alerta hasta tener la población distrital 2010-2024, y presentarlo como exploratorio con error en casos; (b) si se quiere sostener "transferencia", añadir una **validación de transferencia dejando una unidad fuera** también entre las cinco (entrenar con cuatro y probar en la quinta); con una sola unidad de transferencia, de otra naturaleza, no se puede separar "transferencia" de "unidad rara". **A** para (a), **M** para (b).

## 4. Retraso simulado igual al de Piura

- Es un **supuesto fuerte**: `simulate_triangle` sortea con la misma distribución (media 0,69 semanas; 94,4 % en <= 1 semana) para las seis unidades y todos los años. No varía con el tamaño del brote, la unidad, ni el año.
- Efecto: la brecha entre condiciones es proporcional al retraso supuesto. En el triángulo actual el acumulado de 4 semanas conocido es, en la mediana, el 86 % del final. Si otras regiones o años de brote tuvieran retrasos mayores, la brecha real sería mayor; si menores, menor. Además, el retraso se mide solo sobre casos que ya llegaron al archivo, así que está subestimado (datos.md 3.6).
- **Qué haría:** **A**: tratar el retraso como **parámetro de sensibilidad**, no como un valor único: reportar la diferencia entre condiciones bajo varias distribuciones (por ejemplo, la observada y versiones con retraso mayor), con los escenarios [PENDIENTE DE DECIDIR] por el equipo. Así el artículo dice "si el retraso es como el de Piura, la diferencia es X; cómo cambia con retraso mayor", en vez de afirmar un valor. Pedir al CDC los cortes de otras regiones sigue siendo la única forma de observarlo (datos.md §5).
- El retraso PROVISIONAL (semanas excluidas, retraso máximo, semilla) debe decidirse antes de la corrida definitiva.

## 5. Otros puntos

| # | Punto | Qué haría | Prior. |
|---|---|---|---|
| 5.1 | **Solape de ventanas.** El objetivo es el acumulado de 4 semanas que terminan en t+h: con h = 2 comparte 2 semanas con el dato actual, y con h = 4 queda contiguo. La persistencia es muy fuerte por construcción (autocorrelación 0,94 a 0,97 a 1 semana) | Mantener la persistencia como referencia (ya está) y reportar la mejora **frente a ella**; no solo frente al canal endémico | A |
| 5.2 | **"Anticipación" se mide sobre una serie suavizada.** El episodio empieza cuando el acumulado de 4 semanas cruza el corte, no cuando comienza el brote real; ya se suma el horizonte (`lead_total`) | Definirla así en el Método; no llamarla "anticipación del brote" | M |
| 5.3 | **Canal endémico [SIN FUENTE]** y, con ceros frecuentes, mediana 0 | Citar la referencia (Hussain-Alkhateeb 2018 es la opción del presupuesto) o declararlo como línea base propia | M |
| 5.4 | **Un solo modelo para unidades de escala muy distinta, sin población como variable** | Declararlo como límite (ya está); probar la población como variable solo si el equipo lo aprueba | B |
| 5.5 | **Clima e ICEN: pocos años de El Niño** (2010-2024) | Declarar la ablación con ICEN como exploratoria (ya acordado) | M |
| 5.6 | **Reajustar cada 4 semanas con ventana expansiva** mezcla años de brote muy distintos | Ventana deslizante como sensibilidad (ya está en el protocolo) | B |

## Qué falta aprobar

1. Prueba 2020-2024 frente a ampliarla (3.3 arriba).
2. Resultado primario (2.4).
3. Escenarios de retraso y número de semillas (4 y 1.3).
4. Tratamiento de Lima Norte: sin alertas hasta tener población distrital (3).
5. Los valores [PROVISIONAL] de `config.yaml`, la tasa semanal (acumulado de 4 semanas) y los rezagos de clima e ICEN.
