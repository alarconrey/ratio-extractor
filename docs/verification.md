# Verificación de la entrega

Comprobaciones realizadas el 9 y 10 de octubre de 2026.

## Datos

- `python -m app.build_dataset` sobre las tres exportaciones (2022, 2023 con precedentes 2011–2021, y 2024) reproduce exactamente el dataset analizado: 1.598 resoluciones, 171.906 párrafos, 114 excluidas. Se compararon etiqueta, texto, sección, posición y analista fila por fila: cero diferencias.
- La partición tiene identificadores únicos, sin resoluciones repetidas entre particiones; la prueba solo contiene 2023–2024.

## Entrenamiento y evaluación

- TF-IDF: 4 configuraciones evaluadas en validación; elegida la de contexto vecino con C = 1 (F1 de validación 0,587, umbral 0,55).
- Prueba (499 resoluciones): regla 0,356; TF-IDF 0,615. McNemar exacto: 107 resoluciones bien extraídas solo por TF-IDF frente a 13 solo por la regla (p < 0,001).
- BETO (Colab, Tesla T4, 2 épocas, 67 min de entrenamiento): umbral 0,35 elegido en validación (F1 0,602). Prueba: 0,655 (IC 0,635–0,677), 32,9 % bien extraídas. `python -m app.report` reproduce exactamente las cifras del notebook a partir de `beto_predictions_test.csv`. McNemar frente a TF-IDF: 68 frente a 43, p = 0,022.
- El pipeline guardado reproduce las predicciones de prueba almacenadas, y la API reproduce las predicciones de los ejemplos con una diferencia máxima de 10⁻¹⁶.
- El notebook de BETO se ejecutó completo con un BERT diminuto aleatorio (CPU) para validar el código: datos, entrenamiento, umbral, evaluación y guardado. Esta ejecución corrigió un error de truncado con párrafos largos. **No es una evaluación de BETO**; sus resultados reales provienen de la ejecución en Colab con GPU.
- **Error detectado en la primera ejecución real de BETO:** la evaluación dio un F1 de 0,123 en prueba con el umbral mínimo (0,1), peor que la regla sin aprendizaje. La causa: `group_by_length=True`, usado para acelerar el entrenamiento, también reordena los lotes de evaluación en `transformers` 4.56, así que las probabilidades quedaron asignadas a párrafos equivocados. El entrenamiento no se vio afectado (sus métricas alinean etiquetas y predicciones del mismo lote). Se corrigió prediciendo con un `Trainer` sin agrupación y se agregó una comprobación que detiene el notebook si el orden de las predicciones no coincide con el de los datos. El resultado erróneo no se usa en ningún informe.

## Pruebas automáticas

`python -m pytest -q`: **37 pruebas correctas**, también en una copia limpia instalada desde `requirements-lock.txt` con Python 3.12. Cubren:

- División en párrafos (HTML y texto plano), detección de encabezados y secciones.
- F1 por resolución calculado a mano, elección de umbral, McNemar y Wilson con valores de referencia.
- Partición temporal y disjunta; coherencia entre modelo, umbral e informe; reproducción de predicciones y del F1 reportado.
- API: caso válido (idéntico al modelo evaluado), caso difícil (párrafos a revisar dentro del margen), texto ajeno (advertencias), nueve entradas inválidas o con campos extra (422), texto sin párrafos (422), BETO ausente (503), modelo ausente (503), regla siempre disponible, y la ruta BETO con un doble de prueba declarado (sin torch ni pesos). El doble verifica el flujo, no la calidad de BETO.

Starlette emite un aviso de deprecación de AnyIO en el cliente de pruebas; no afecta la ejecución y no se oculta.

## Interfaz

Revisada en Chromium (Playwright) a 1366 × 900 y 390 × 844: carga, ejemplos, resultado resaltado, revisión de párrafos, coincidencia con la ficha, mensaje de error 422, pestaña Resultados y vista móvil. `node --check frontend/app.js` sin errores. Las fuentes de Google Fonts no cargaron en el entorno de verificación (sin acceso); la interfaz usa Georgia y la fuente del sistema como respaldo.

No se realizó una auditoría completa de accesibilidad ni pruebas en Windows o en teléfonos físicos.
