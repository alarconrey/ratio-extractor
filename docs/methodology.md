# Modelos, validación y métricas

## Unidad de observación

El **párrafo** de una resolución, clasificado como «pertenece a la ratio» o no. Los resultados se agregan por resolución, que es la unidad que revisa la persona analista.

## Métodos comparados

| Método | Qué se entrena | Entradas |
|---|---|---|
| Regla de sección | Nada (fijada antes de evaluar) | Sección detectada: todos los párrafos de «Fundamentos jurídicos» |
| TF-IDF + regresión logística | Vocabulario TF-IDF y pesos de la regresión | Texto del párrafo (1–2 gramas), texto de los párrafos vecinos, sección, posición y longitud |
| BETO con ajuste fino | Capa de clasificación + ajuste fino de todos los pesos | Sección, posición, párrafo y 300 caracteres de cada vecino, hasta 384 tokens |

**Lo reutilizado y lo entrenado:** BETO (`dccuchile/bert-base-spanish-wwm-cased`) es un modelo preentrenado reutilizado. En este proyecto se entrena la capa de clasificación y se ajustan sus pesos con los párrafos etiquetados (`notebooks/train_beto_colab.ipynb`). TF-IDF se entrena completo aquí (`app/train.py`).

## Validación

- **Partición temporal:** entrenamiento y validación con 2011–2022; prueba con 2023–2024.
- **Agrupada por resolución:** todos los párrafos de una resolución quedan en la misma partición; las fichas duplicadas no cruzan particiones.
- **Selección en validación:** para TF-IDF se comparan 4 configuraciones (con y sin contexto vecino; C = 1 y 4). Para cada una se elige el umbral entre 0,10 y 0,95 que maximiza el F1 por resolución en validación. BETO elige su umbral igual, en el notebook.
- **Prueba una sola vez:** el modelo y el umbral se congelan antes de puntuar la prueba. No se ajustó nada después de ver la prueba.
- **Desbalance:** ~5 % de párrafos positivos. TF-IDF usa `class_weight="balanced"`; BETO submuestrea negativos al entrenar (todos los positivos, 50 % de los negativos de fundamentos, 15 % del resto). Validación y prueba nunca se submuestrean.

## Métricas

| Métrica | Interpretación |
|---|---|
| **F1 por resolución** (principal) | F1 ponderado por palabras entre los párrafos extraídos y la ratio de la ficha, promediado entre resoluciones. Mide cuánto texto de la propuesta coincide con lo que eligió el analista. |
| Resoluciones bien extraídas | Proporción con F1 ≥ 0,8: la propuesta casi no necesita corrección. Con intervalo de Wilson. |
| F1 por párrafo | Precisión y recall sobre todos los párrafos de la prueba. |
| IC 95 % | Bootstrap de resoluciones (1.000 remuestreos). |
| McNemar exacto | Compara dos métodos sobre las mismas resoluciones (bien extraída sí/no). |

No se usa la exactitud (*accuracy*): con 95 % de negativos, no marcar nada daría 95 %.

## Análisis de errores (TF-IDF, prueba)

| Grupo | Párrafos | Palabras (mediana) | En «fundamentos» | Posición (mediana) | Cita un precedente |
|---|---:|---:|---:|---:|---:|
| Verdaderos positivos | 2.305 | 200 | 82 % | 0,86 | 14 % |
| Falsos positivos | 2.398 | 176 | 82 % | 0,84 | 18 % |
| Falsos negativos | 1.621 | 66 | 44 % | 0,62 | 9 % |

- **Los falsos positivos son indistinguibles de los aciertos** por posición, sección y longitud. El modelo encuentra la zona correcta, pero no distingue *cuál* de los párrafos de fundamentos eligió el analista. Las citas de precedentes anteriores pesan poco (18 % frente a 14 %). Es la ambigüedad que motiva un modelo contextual como BETO, y también un techo: parte de esa elección es criterio del analista.
- **Los falsos negativos son párrafos cortos** y más de la mitad cae fuera de la sección detectada como fundamentos: o la detección de secciones falla en ese documento, o la ratio incluye párrafos breves de transición que el modelo no puntúa alto.
- **Por analista**, el F1 varía de 0,35 a 0,67; las diferencias de criterio entre analistas limitan cualquier modelo entrenado con sus fichas.
- 19 de 499 resoluciones tienen F1 = 0 (ningún párrafo propuesto coincide).

## Política de revisión

En la interfaz, los párrafos con puntaje a ±0,15 del umbral se marcan para revisar. Es una ayuda de lectura: los puntajes no son probabilidades calibradas de acierto y no se informa una exactitud garantizada para ese subconjunto.

## Límites

- La ratio de una ficha no es una verdad única: otro analista podría seleccionar otros párrafos.
- Un texto que no es una resolución también recibe puntajes; las advertencias (sin sección de fundamentos, texto plano) ayudan, pero no lo detectan siempre.
- El texto plano se divide por líneas; el modelo se entrenó con el HTML de la base, por lo que la división puede diferir.
- BETO procesa hasta 384 tokens por párrafo; de los párrafos más largos solo lee el comienzo, y la interfaz lo advierte.
