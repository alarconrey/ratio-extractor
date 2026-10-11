"""Local API and teaching interface for ratio decidendi extraction."""

import json
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.beto_model import BetoModel
from app.inference import InputProblem, extract_ratio
from app.schemas import ExtractionInput
from app.settings import ARTIFACT_DIR, EXAMPLES_FILE, FRONTEND_DIR, MAX_CHARACTERS, MAX_PARAGRAPHS, MIN_CHARACTERS, MIN_PARAGRAPHS


@asynccontextmanager
async def lifespan(app):
    app.state.tfidf = None
    app.state.report = None
    app.state.beto, app.state.beto_reason = None, "No cargado."
    model_path, report_path = ARTIFACT_DIR / "model.joblib", ARTIFACT_DIR / "metrics.json"
    if model_path.exists() and report_path.exists():
        bundle = joblib.load(model_path)  # Only load this project's own artifact.
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if bundle["model_version"] != report["model_version"]:
            raise RuntimeError("model.joblib and metrics.json come from different trainings. Run python -m app.train.")
        app.state.tfidf, app.state.report = bundle, report
    if getattr(app.state, "load_beto", True):
        app.state.beto, app.state.beto_reason = BetoModel.try_load()
    yield


app = FastAPI(title="Extractor de ratio decidendi", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.exception_handler(RequestValidationError)
async def invalid_input(request: Request, error: RequestValidationError):
    details = [{"loc": item["loc"], "msg": item["msg"], "type": item["type"]} for item in error.errors()]
    return JSONResponse(status_code=422, content={"detail": details})


def available_methods():
    methods = ["rule"]
    if app.state.tfidf is not None:
        methods.append("tfidf")
    if app.state.beto is not None:
        methods.append("beto")
    return methods


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health():
    methods = available_methods()
    return {
        "status": "ok" if app.state.tfidf is not None else "model_missing",
        "methods": methods, "default_method": "beto" if "beto" in methods else "tfidf" if "tfidf" in methods else "rule",
        "beto_status": "ok" if app.state.beto is not None else app.state.beto_reason,
        "limits": {"min_characters": MIN_CHARACTERS, "max_characters": MAX_CHARACTERS,
                   "min_paragraphs": MIN_PARAGRAPHS, "max_paragraphs": MAX_PARAGRAPHS},
    }


@app.get("/api/metrics")
def metrics():
    if app.state.report is None:
        raise HTTPException(503, "Faltan los artefactos. Ejecuta python -m app.train y reinicia.")
    return app.state.report


@app.get("/api/examples")
def examples():
    if not EXAMPLES_FILE.exists():
        return []
    return json.loads(EXAMPLES_FILE.read_text(encoding="utf-8"))


@app.post("/api/extract-ratio")
def extract(payload: ExtractionInput):
    method = payload.method
    if method == "auto":
        method = health()["default_method"]
    if method == "tfidf" and app.state.tfidf is None:
        raise HTTPException(503, "El modelo TF-IDF no está disponible. Ejecuta python -m app.train y reinicia.")
    if method == "beto" and app.state.beto is None:
        raise HTTPException(503, f"BETO no está disponible: {app.state.beto_reason}")
    try:
        return extract_ratio(payload.text, method, app.state.tfidf, app.state.beto)
    except InputProblem as problem:
        raise HTTPException(422, str(problem)) from None
