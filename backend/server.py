"""FastAPI collector server.

Run with:
    uvicorn backend.server:app --reload

Routes:
    GET  /                          → single-page collector (frontend/index.html)
    GET  /static/<path>             → served from frontend/
    GET  /api/questions             → questions.json
    POST /api/cases                 → mint a new case_id
    POST /api/cases/{id}/upload     → multipart: file (wav), q (1..12)
    POST /api/cases/{id}/assess     → runs core.asd_pipeline against the case dir
    GET  /api/health                → ok
"""

from __future__ import annotations

import json
import logging
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend import storage

logger = logging.getLogger("collector")

app = FastAPI(title="Omli Speech Eval — Collector")
app.mount("/static", StaticFiles(directory=storage.FRONTEND_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(storage.FRONTEND_DIR, "index.html"))


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "num_questions": storage.NUM_QUESTIONS}


@app.get("/api/questions")
def questions() -> JSONResponse:
    with open(storage.QUESTIONS_PATH) as f:
        return JSONResponse(json.load(f))


@app.post("/api/cases")
def create_case() -> dict:
    case_id = storage.new_case_id()
    storage.case_dir(case_id, create=True)
    logger.info("Created case %s", case_id)
    return {"case_id": case_id}


@app.post("/api/cases/{case_id}/upload")
async def upload(case_id: str, q: int = Form(...), file: UploadFile = File(...)) -> dict:
    try:
        path = storage.wav_path(case_id, q)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")

    with open(path, "wb") as f:
        f.write(data)
    logger.info("Saved %s (%d bytes)", path, len(data))
    return {"ok": True, "path": os.path.relpath(path, storage.PROJECT_ROOT), "bytes": len(data)}


@app.post("/api/cases/{case_id}/assess")
def assess(case_id: str, child_age_months: int = Form(66)) -> dict:
    # Import here so server boots fast even if librosa/Praat are slow to import.
    from core.asd_pipeline import assess_asd_risk

    wavs = storage.list_wavs(case_id)
    if len(wavs) < 4:
        raise HTTPException(
            status_code=400,
            detail=f"Case {case_id} has only {len(wavs)} recordings; need at least 4.",
        )

    prompted = wavs[:4]
    all_audio = wavs[:storage.NUM_QUESTIONS]

    result = assess_asd_risk(
        prompted_question_audio_paths=prompted,
        all_audio_paths=all_audio,
        child_age_months=child_age_months,
    )
    out_path = os.path.join(storage.case_dir(case_id), "asd_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Wrote %s", out_path)
    return result
