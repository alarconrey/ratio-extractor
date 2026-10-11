# Extractor de ratio decidendi

Asistente de IA para la **extracción asistida de la *ratio decidendi*** de resoluciones del Tribunal Agroambiental de Bolivia, con supervisión humana. Proyecto final del Capstone

El sistema recibe una resolución (Auto o Sentencia), propone qué párrafos forman su *ratio decidendi* y deja que la persona analista acepte o quite párrafos antes de copiarla a la ficha jurisprudencial.

El código, los nombres de funciones y los comentarios están en inglés. La interfaz y la documentación están en español.

## La pregunta del proyecto

**Dada una resolución completa, ¿qué párrafos conviene proponer como su *ratio decidendi* para que una persona analista los revise?**

La aplicación propone. No redacta las síntesis de la ficha, no clasifica la materia y no guarda nada.

## Resultado principal

Prueba sobre **499 resoluciones de 2023–2024**, posteriores a todo el entrenamiento:

| Método | F1 por resolución (IC 95 %) | Resoluciones bien extraídas (F1 ≥ 0,8) |
|---|---|---|
| Regla de sección (línea base) | 0,356 (0,332–0,382) | 9,0 % |
| TF-IDF + regresión logística | 0,615 (0,593–0,639) | 27,9 % |
| BETO con ajuste fino | **0,655** (0,635–0,677) | 32,9 % |

TF-IDF supera a la regla (McNemar exacto, p < 0,001) y BETO supera a TF-IDF (68 frente a 43 resoluciones bien extraídas solo por uno de los dos, p = 0,022). Las cifras completas, los desgloses por tipo de resolución y por analista, y los límites están en [artifacts/evaluation.md](artifacts/evaluation.md), generado por el código.

## Qué incluye

- **Dataset real:** 1.598 resoluciones con ficha (2011–2024), 171.906 párrafos etiquetados por alineamiento con la ratio de la ficha ([docs/data.md](docs/data.md)).
- **Partición temporal sin fugas:** entrenamiento y validación 2011–2022, prueba 2023–2024, agrupada por resolución.
- **Tres métodos sobre la misma prueba:** regla de sección, TF-IDF + regresión logística (entrenado aquí) y BETO con ajuste fino (entrenado en Colab con `notebooks/train_beto_colab.ipynb`).
- **API FastAPI** con validación de entrada, manejo de errores y respaldo: si BETO no está instalado, usa TF-IDF.
- **Interfaz web** (HTML/CSS/JavaScript, sin compilación): documento marcado como con resaltador, revisión párrafo por párrafo, ratio lista para copiar, resultados y método.
- **37 pruebas automáticas** (datos, métricas, modelo y API).

## Empezar desde cero

### 1. Preparar Python

Se utilizó **Python 3.12**. Instálalo desde [python.org](https://www.python.org/downloads/). Abre una terminal dentro de la carpeta `ratio-extractor` (la que contiene este archivo y `run.py`).

### 2. Instalar las dependencias

macOS o Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

Windows Git Bash:

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements-lock.txt
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

`requirements-lock.txt` fija las versiones probadas; `requirements.txt` enumera las dependencias directas con rangos.

### 3. Iniciar

```bash
python run.py
```

En Windows: `.\.venv\Scripts\python.exe run.py`.

Abre **http://127.0.0.1:8995**. Mantén la terminal abierta; `Ctrl+C` detiene el servidor. El repositorio incluye el modelo TF-IDF entrenado y la evaluación; si faltan, `run.py` los genera (unos 3 minutos).

### 4. Probar la interfaz

1. Pulsa el ejemplo **Resolución bien extraída**: su texto se carga en el formulario. Luego pulsa **Proponer ratio**.
2. Mira los párrafos resaltados en amarillo (propuesta), los de borde punteado (cerca del umbral) y la barra azul del margen (ratio según la ficha del analista).
3. Usa **Siguiente por revisar** (o la tecla `N`) y decide cada párrafo dudoso con su casilla o con «Mantener». La ratio y la coincidencia con la ficha se actualizan; **Deshacer** (`Ctrl+Z`) revierte el último cambio.
4. Prueba **Caso difícil**, **Texto ajeno** (debe mostrar avisos) y **Entrada inválida** (el formulario indica cuántos caracteres faltan y no permite enviar).
5. Abre **Resultados** y **Datos y método**.

## Activar BETO

BETO con ajuste fino pesa ~440 MB y no va dentro del repositorio: está publicado en Hugging Face, en el repositorio privado [`alarconx/beto-ratio-decidendi`](https://huggingface.co/alarconx/beto-ratio-decidendi). Sus predicciones de prueba y su umbral (`artifacts/beto_predictions_test.csv`, `artifacts/beto_summary.json`) sí están incluidos, así que la evaluación y la pestaña Resultados ya muestran BETO sin descargar nada. Descargarlo solo hace falta para usarlo en la interfaz.

### Dónde se indica el modelo

| Archivo | Qué hace |
|---|---|
| `models-lock.json` | Repositorio de Hugging Face (`repo`) y revisión exacta (`revision`) del modelo publicado |
| `app/download_models.py` | Lee `models-lock.json` y descarga esa revisión en `models/beto/` |
| `app/settings.py` | `BETO_DIR = models/beto`, la carpeta de la que se carga el modelo |
| `app/beto_model.py` | Carga el modelo **solo desde esa carpeta local** (sin internet) y aplica el umbral de `beto_summary.json` |

### Pasos

Con el servidor detenido (`Ctrl+C`) y el entorno activado (`source .venv/bin/activate` en macOS o Linux, `source .venv/Scripts/activate` en Git Bash de Windows):

```bash
python -m pip install -r requirements-beto.txt   # torch y transformers (unos minutos)
huggingface-cli login                            # pega un token de tipo Read; el repositorio es privado
python -m app.download_models                    # descarga el modelo en models/beto/
python run.py
```

En PowerShell, usa `.\.venv\Scripts\python.exe` en lugar de `python` y `.\.venv\Scripts\huggingface-cli.exe login`.

Si todo está bien, la cabecera de la interfaz muestra **«Servidor listo · BETO con ajuste fino»** y BETO aparece como método recomendado. Tras analizar, el selector *Método* de la barra de la resolución permite comparar BETO, TF-IDF y la regla sobre la misma resolución. En CPU, BETO tarda unos segundos por resolución.

**Sin Hugging Face:** copia el **contenido** de la carpeta `final` que guardó el notebook de Colab en `models/beto/`, de modo que quede `models/beto/config.json`.

### Volver a entrenar BETO

1. Ejecuta `notebooks/train_beto_colab.ipynb` en Colab con GPU (unos 70 minutos en una T4). Los datos van en `datos_colab.zip` (el CSV del dataset y `data/splits.csv`). Si la sesión se corta después de entrenar, `notebooks/recover_beto_predictions.ipynb` predice desde el último punto de control sin reentrenar.
2. Copia `beto_summary.json` y `beto_predictions_test.csv` a `artifacts/` y ejecuta `python -m app.report` para regenerar la evaluación.
3. Para publicar el modelo nuevo, ejecuta la última celda del notebook y actualiza `revision` en `models-lock.json`.

## Repetir el experimento

```bash
python -m app.train      # entrena TF-IDF, elige en validación, evalúa la prueba una vez y escribe el informe
python -m app.examples   # reconstruye los escenarios de la interfaz
python -m pytest -q      # 37 pruebas
python run.py
```

O en un paso: `python run.py --rebuild`.

Para reconstruir el dataset desde la base `arbol` (scripts SQL Server generados con SSMS, «Esquema y datos», Unicode):

```bash
python -m app.build_dataset script_2022.zip script_2023.zip script_2024.sql
```

## Organización

```text
ratio-extractor/
├── README.md
├── run.py                       # Inicia el sistema local
├── requirements.txt / requirements-lock.txt / requirements-beto.txt
├── models-lock.json             # Repositorio y revisión de BETO en Hugging Face
├── app/
│   ├── settings.py              # Rutas y decisiones fijadas antes de evaluar
│   ├── text.py                  # División en párrafos y secciones (compartido por datos y API)
│   ├── build_dataset.py         # Scripts SQL → párrafos etiquetados y partición
│   ├── baseline.py              # Regla de sección
│   ├── tfidf_model.py           # TF-IDF + regresión logística
│   ├── beto_model.py            # Inferencia BETO (opcional)
│   ├── evaluation.py            # F1 por resolución, intervalos, McNemar
│   ├── train.py                 # Entrenamiento y selección en validación
│   ├── report.py                # metrics.json y evaluation.md
│   ├── examples.py              # Escenarios de la interfaz
│   ├── inference.py             # Texto → propuesta de ratio
│   ├── schemas.py               # Validación de la entrada
│   ├── download_models.py       # Descarga de BETO desde Hugging Face
│   └── main.py                  # API y archivos del frontend
├── frontend/                    # index.html, styles.css, app.js
├── notebooks/                   # Ajuste fino de BETO en Colab
├── data/                        # Dataset (.csv.gz), partición, resumen y ejemplos
├── artifacts/                   # Modelo, predicciones y resultados generados
├── tests/
└── docs/
```

## Guías

- [Datos y etiquetado](docs/data.md)
- [Modelos, validación, métricas y análisis de errores](docs/methodology.md)
- [API](docs/api.md)
- [Verificación de la entrega](docs/verification.md)
- [Resultados calculados](artifacts/evaluation.md)

## Problemas habituales

| Situación | Qué hacer |
|---|---|
| No encuentra Python | Instala Python 3.12; en macOS prueba `python3`, en Windows `py -3.12`. |
| Falta una librería | Usa el Python de `.venv` e instala `requirements-lock.txt`. |
| Puerto ocupado | `python run.py --port 8996` y abre ese puerto. |
| «Falta el modelo entrenado» | Ejecuta `python -m app.train` y reinicia. |
| BETO aparece como «no instalado» | Sigue «Activar BETO»; http://127.0.0.1:8995/api/health indica el motivo exacto en `beto_status`. |
| `401` o `Repository not found` al descargar BETO | Falta `huggingface-cli login`, o tu cuenta no tiene acceso al repositorio privado. |
| BETO tarda en responder | Es normal en CPU (segundos por resolución); para la demo, TF-IDF responde al instante. |
| HTTP 422 | El texto es corto, no tiene párrafos o incluye campos extra. Pega la resolución completa. |
| Texto plano con resultados raros | Pega el HTML de la base o deja un párrafo por línea. |

## Alcance y datos

El servidor escucha solo en `127.0.0.1`, no tiene autenticación y no guarda los textos. Las resoluciones del Tribunal Agroambiental son públicas; aun así contienen nombres de las partes, por lo que el repositorio debe mantenerse **privado**, con acceso para el docente. La tabla `users` de la base nunca se carga.
