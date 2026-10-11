# Resultados del experimento

Extracción de la *ratio decidendi*: cada párrafo de la resolución se clasifica como «pertenece a la ratio» o no. Datos reales de la base `arbol` del Tribunal Agroambiental (2011–2024). Archivo generado por `python -m app.report`.

## Particiones

| Partición | Resoluciones | Párrafos | % positivos | Años |
|---|---:|---:|---:|---|
| Entrenamiento | 837 | 89.813 | 4,9% | 2011–2022 |
| Validación | 148 | 16.328 | 5,4% | 2011–2022 |
| Prueba | 499 | 54.442 | 7,2% | 2023–2024 |

La prueba contiene solo resoluciones de 2023–2024, posteriores a todo el entrenamiento. Se excluyeron 114 resoluciones cuya ratio se recupera en menos del 50 % (alineamiento débil).

## Prueba: misma colección para todos los métodos

| Método | F1 por resolución (IC 95 %) | Mediana | Resoluciones con F1 ≥ 0,8 (IC 95 %) | F1 por párrafo |
|---|---|---:|---|---:|
| Regla de sección | **0,356** (0,332–0,382) | 0,336 | 9,0% (6,8%–11,9%) | 0,232 |
| TF-IDF + regresión logística | **0,615** (0,593–0,639) | 0,654 | 27,9% (24,1%–31,9%) | 0,534 |
| BETO con ajuste fino | **0,655** (0,635–0,677) | 0,689 | 32,9% (28,9%–37,1%) | 0,565 |

F1 por resolución: F1 ponderado por palabras entre los párrafos extraídos y la ratio de la ficha, promediado entre resoluciones. Intervalo por bootstrap de resoluciones (1.000 remuestreos); proporción con intervalo de Wilson.

## Comparación pareada (McNemar exacto)

Resultado binario por resolución: F1 ≥ 0,8.

| Comparación | Solo el primero | Solo el segundo | p |
|---|---:|---:|---:|
| Regla de sección vs TF-IDF + regresión logística | 13 | 107 | 0.0000 |
| TF-IDF + regresión logística vs BETO con ajuste fino | 43 | 68 | 0.0223 |

## Desglose del mejor método disponible (BETO con ajuste fino)

| Tipo de resolución | Resoluciones | F1 |
|---|---:|---:|
| Auto Agroambiental Plurinacional | 269 | 0,619 |
| Auto Interlocutorio Definitivo | 119 | 0,734 |
| Sentencia Agroambiental Plurinacional | 111 | 0,656 |

| Analista (idusuario) | Resoluciones | F1 |
|---|---:|---:|
| 1037 | 302 | 0,720 |
| 1042 | 135 | 0,613 |
| 1045 | 36 | 0,449 |
| 1041 | 23 | 0,358 |
| 1044 | 2 | 0,889 |
| 1047 | 1 | 0,457 |

## Selección de TF-IDF en validación

| Contexto vecino | C | Umbral | F1 validación |
|---|---:|---:|---:|
| no | 1.0 | 0.55 | 0,569 |
| no | 4.0 | 0.4 | 0,564 |
| sí | 1.0 | 0.55 | 0,587 |
| sí | 4.0 | 0.35 | 0,580 |

Elegido: contexto sí, C = 1.0, umbral 0.55.

## BETO

Modelo base `dccuchile/bert-base-spanish-wwm-cased`, ajuste fino de 2 épocas, tasa 2e-05, longitud máxima 384 tokens, 28.979 párrafos de entrenamiento submuestreados, umbral 0.35 elegido en validación, GPU Tesla T4.

## Límites

- La etiqueta proviene de la ficha de un analista. Los analistas seleccionan la ratio con criterios distintos; parte del error medido es diferencia de criterio, no error del modelo.
- Las resoluciones de 2011–2021 provienen de una exportación de precedentes vinculados, no de una muestra aleatoria de esos años.
- 2024 aporta pocas resoluciones con ficha; su desempeño se reporta dentro de la prueba pero no por separado.
- Los puntajes no están calibrados como probabilidades de acierto.

## Reproducibilidad

SHA-256 del dataset: `54b1dd9b7c859ed7f1849c688d9e092e4c304507abd2bfe621099e1aea7c13d8`. Versión del modelo: `tfidf-v1-54b1dd9b`. Versiones: `{'python': '3.12.3', 'scikit-learn': '1.9.1', 'pandas': '3.0.6', 'numpy': '2.3.5', 'fastapi': '0.119.1'}`.
