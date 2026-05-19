"""FastAPI collector server.

Run with:
    uvicorn backend.server:app --reload

Routes:
    GET  /                          → single-page collector (frontend/index.html)
    GET  /static/<path>             → served from frontend/
    GET  /api/questions             → questions.json
    GET  /api/cases                 → list all cases with metadata
    POST /api/cases                 → mint a new case_id
    POST /api/cases/{id}/upload     → multipart: file (wav), q (1..12)
    POST /api/cases/{id}/assess     → runs core.asd_pipeline; returns {raw, summary}
    GET  /api/cases/{id}/summary    → re-summarise saved asd_result.json
    GET  /api/cases/{id}/raw        → download saved asd_result.json
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


@app.get("/api/cases")
def list_cases() -> list[dict]:
    """List all cases on disk (newest first) with light summary metadata."""
    from core.asd_consumer_view import summarize_for_consumer

    out = []
    for case_id in storage.list_case_ids():
        rpath = storage.result_path(case_id)
        n_recs = len(storage.list_wavs(case_id))
        row = {
            "case_id": case_id,
            "created_at": storage.parse_created_at(case_id),
            "num_recordings": n_recs,
            "has_result": os.path.exists(rpath),
            "tier": None,
            "verdict": None,
            "color": None,
            "child_age_months": None,
        }
        if row["has_result"]:
            try:
                with open(rpath) as f:
                    raw = json.load(f)
                summary = summarize_for_consumer(raw)
                row["tier"] = raw.get("tier")
                row["verdict"] = summary["headline"]["verdict"]
                row["color"] = summary["headline"]["color"]
                row["child_age_months"] = raw.get("child_age_months")
            except Exception as e:
                logger.warning("Could not read %s: %s", rpath, e)
        out.append(row)
    return out


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
    from core.asd_consumer_view import summarize_for_consumer

    wavs = storage.list_wavs(case_id)
    if len(wavs) < 4:
        raise HTTPException(
            status_code=400,
            detail=f"Case {case_id} has only {len(wavs)} recordings; need at least 4.",
        )

    prompted = wavs[:4]
    all_audio = wavs[:storage.NUM_QUESTIONS]

    raw = assess_asd_risk(
        prompted_question_audio_paths=prompted,
        all_audio_paths=all_audio,
        child_age_months=child_age_months,
    )
    out_path = storage.result_path(case_id)
    with open(out_path, "w") as f:
        json.dump(raw, f, indent=2, default=str)
    logger.info("Wrote %s", out_path)

    return {"raw": raw, "summary": summarize_for_consumer(raw)}


@app.get("/api/cases/{case_id}/summary")
def summary(case_id: str) -> dict:
    """Re-summarise a saved result without re-running the pipeline."""
    from core.asd_consumer_view import summarize_for_consumer

    path = storage.result_path(case_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No result saved for case {case_id}")
    with open(path) as f:
        raw = json.load(f)
    return summarize_for_consumer(raw)


@app.get("/api/cases/{case_id}/raw")
def raw_result(case_id: str) -> FileResponse:
    """Serve the saved asd_result.json as a downloadable file."""
    path = storage.result_path(case_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No result saved for case {case_id}")
    return FileResponse(
        path,
        media_type="application/json",
        filename=f"asd_result_{case_id}.json",
    )
