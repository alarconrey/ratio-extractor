# Datos y etiquetado

## Origen

Base de datos `arbol` del Tribunal Agroambiental (SQL Server), exportada con SSMS («Generar scripts», esquema y datos). Se usan dos tablas:

- `resolucion`: número, gestión, tipo y texto completo (`contenido`, HTML generado desde Word).
- `ficha`: ficha jurisprudencial de la resolución, elaborada por una persona analista. De ella se usa `ratiodecidenti`.

Se procesaron tres exportaciones: 2022 completo, 2023 completo (que trae además resoluciones de 2011–2021 vinculadas como precedentes) y 2024. La tabla `users` no se carga.

| Concepto | Valor |
|---|---:|
| Resoluciones con ficha | 1.598 |
| Párrafos | 171.906 |
| Párrafos «ratio» | ~5 % |
| Excluidas por alineamiento débil | 114 |

## Por qué extracción

El análisis previo de los campos de la ficha mostró que `ratiodecidenti` se copia literalmente de la resolución en el 97–100 % de las fichas, en todos los años y con todos los analistas. Es el único campo estable entre analistas; el tipo de jurisprudencia, el tipo de resolución y el estilo de las síntesis dependen de quién hizo la ficha (V de Cramér 0,44 entre analista y tipo de jurisprudencia). Por eso el problema se plantea como **extracción de la ratio**, no como clasificación de etiquetas de la ficha.

## Etiquetado por alineamiento

1. El HTML de `contenido` se divide en párrafos (`<p>`, `<li>`), quedándose con los bloques más internos (`app/text.py`).
2. Se unen las ratios de todas las fichas distintas de la resolución (una resolución puede tener varias fichas).
3. Un párrafo es `es_ratio = 1` si al menos el **50 %** de sus secuencias de 8 palabras aparece en esa ratio, tras normalizar minúsculas, comillas y puntos.
4. Se mide qué parte de la ratio recuperan los párrafos positivos (`cobertura_ratio`). Mediana: 94–96 % según el año. Las resoluciones con cobertura < 50 % se excluyen (`revisar = 1`).

La etiqueta proviene del trabajo de personas analistas, no del modelo, y se obtiene sin anotación adicional.

## Variables (`data/dataset_ratio_parrafos.csv.gz`)

| Columna | Descripción | ¿Entrada del modelo? |
|---|---|---|
| id_resolucion, resolucion | Identificador | No (agrupa la partición) |
| gestion | Año | No (partición temporal) |
| tipo, tipo_nombre | Tipo de resolución | No |
| analista | `idusuario` de la ficha | **No** (solo para medir sesgo) |
| n_fichas | Fichas distintas de la resolución | No |
| particion | entrenamiento (≤ 2022) / prueba (≥ 2023) | No |
| idx, n_parrafos, pos_rel | Posición del párrafo | Sí |
| seccion, encabezado | Sección detectada (fundamentos, antecedentes, por_tanto…) | Sí (`seccion`) |
| n_palabras | Longitud | Sí |
| solape | Proporción del párrafo encontrada en la ratio | **No** (es la fuente de la etiqueta) |
| es_ratio | Etiqueta | Objetivo |
| texto | Texto del párrafo | Sí |

## Partición (`data/splits.csv`)

| Partición | Resoluciones | Años |
|---|---:|---|
| train | 837 | 2011–2022 |
| validation | 148 | 2011–2022 (15 %, agrupado por resolución) |
| test | 499 | 2023–2024 |
| excluded | 114 | alineamiento débil |

## Limitaciones de los datos

- Las resoluciones de 2011–2021 provienen de una exportación de precedentes vinculados, no de una muestra aleatoria.
- 2024 tiene pocas fichas (33 resoluciones) y casi todas de un tipo.
- Cada analista selecciona la ratio con su propio criterio (varios párrafos en 2022, un bloque en 2023).
- Las resoluciones contienen nombres de las partes. Son públicas, pero el repositorio debe mantenerse privado.
