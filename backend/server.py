"""FastAPI collector server.

Run with:
    uvicorn backend.server:app --reload

Routes:
    GET  /                              → single-page collector (frontend/index.html)
    GET  /static/<path>                 → served from frontend/
    GET  /api/questions                 → questions.json
    GET  /api/cases                     → list all cases with metadata (both pipelines)
    POST /api/cases                     → mint a new case_id
    POST /api/cases/{id}/upload         → multipart: file (wav), q (1..12)
    POST /api/cases/{id}/assess         → runs both pipelines; returns {asd, speech_delay}
    GET  /api/cases/{id}/summary        → re-summarise saved results (both pipelines)
    GET  /api/cases/{id}/raw?which=...  → download asd_result.json or speech_delay_result.json
    GET  /api/health                    → ok
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Load .env before any module that reads env vars at import time (e.g. backend.asr's
# SarvamProvider singleton). Falls through silently if .env doesn't exist.
load_dotenv()

from backend import storage  # noqa: E402 — env loaded above

logger = logging.getLogger("collector")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm Sarvam's connection pool at startup; close it at shutdown."""
    from backend.asr import sarvam
    if sarvam.configured:
        await sarvam.warmup()
    else:
        logger.warning(
            "SARVAM_API_KEY not set — speech_delay ASR-dependent metrics will "
            "report computed=false until you populate .env"
        )
    yield
    if sarvam.configured:
        await sarvam.close()


app = FastAPI(title="Omli Speech Eval — Collector", lifespan=lifespan)
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
        asd_path = storage.result_path(case_id)
        sd_path = storage.speech_delay_result_path(case_id)
        n_recs = len(storage.list_wavs(case_id))
        row = {
            "case_id": case_id,
            "created_at": storage.parse_created_at(case_id),
            "num_recordings": n_recs,
            "has_asd_result": os.path.exists(asd_path),
            "has_speech_delay_result": os.path.exists(sd_path),
            "child_age_months": None,
            # ASD columns
            "asd_tier": None,
            "asd_verdict": None,
            "asd_color": None,
            # Speech-delay columns
            "speech_delay_status": None,
            "speech_delay_band": None,
            "speech_delay_color": None,
            "speech_delay_delay_months": None,
        }

        if row["has_asd_result"]:
            try:
                with open(asd_path) as f:
                    raw = json.load(f)
                summary = summarize_for_consumer(raw)
                row["asd_tier"] = raw.get("tier")
                row["asd_verdict"] = summary["headline"]["verdict"]
                row["asd_color"] = summary["headline"]["color"]
                row["child_age_months"] = raw.get("child_age_months")
            except Exception as e:
                logger.warning("Could not read %s: %s", asd_path, e)

        if row["has_speech_delay_result"]:
            try:
                from core.speech_delay_consumer_view import STATUS_COLOR as _sd_status_color
                with open(sd_path) as f:
                    sd = json.load(f)
                row["speech_delay_status"] = sd.get("delay_status")
                row["speech_delay_band"] = sd.get("developmental_band")
                row["speech_delay_color"] = _sd_status_color.get(sd.get("delay_status"), "gray")
                row["speech_delay_delay_months"] = sd.get("delay_months")
                if row["child_age_months"] is None:
                    row["child_age_months"] = sd.get("child_age_months")
            except Exception as e:
                logger.warning("Could not read %s: %s", sd_path, e)

        # Convenience: a single `has_result` flag for any saved result.
        row["has_result"] = row["has_asd_result"] or row["has_speech_delay_result"]
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


def _load_questions() -> list:
    """Return the question list from data/questions.json."""
    with open(storage.QUESTIONS_PATH) as f:
        return json.load(f)["questions"]


def _build_speech_delay_recordings(case_id: str, wavs: list) -> list:
    """
    Pair each recording with task_type + expected_text from questions.json.
    Recording q01.wav → questions[0], q02.wav → questions[1], etc.
    """
    questions = _load_questions()
    out = []
    for i, path in enumerate(wavs):
        if i >= len(questions):
            break
        q = questions[i]
        out.append({
            "audio_path": path,
            "task_type": storage.QUESTION_TYPE_MAP.get(q["type"], "prompted_question"),
            "expected_text": q.get("expected_text"),
            "asr_transcript_clean": None,  # populated by backend ASR in Phase L
            "asr_transcript_raw": None,
            "pronunciation_scores": None,
        })
    return out


@app.post("/api/cases/{case_id}/assess")
async def assess(case_id: str, child_age_months: int = Form(66)) -> dict:
    """
    Run both pipelines against a case in a single pass.

    Order:
      A) Single quality pass shared by both pipelines.
      B) Sarvam ASR fan-out (parallel) — populates asr_transcript_clean
         per recording.
      C) ASD pipeline (no ASR dependency).
      D) Speech-delay pipeline (consumes ASR transcripts + quality dict).

    Pipeline calls are sync + CPU-bound (librosa, Praat). They run on the
    asyncio threadpool via ``asyncio.to_thread`` so the event loop stays
    free for the ASR HTTP fan-out.
    """
    from core.asd_consumer_view import summarize_for_consumer
    from core.asd_pipeline import assess_asd_risk
    from core.audio_analysis import assess_recording_quality
    from core.speech_delay_consumer_view import summarize_for_consumer as summarize_speech_delay
    from core.speech_delay_pipeline import assess_speech_delay
    from backend.asr import sarvam, transcribe_batch

    wavs = storage.list_wavs(case_id)
    if len(wavs) < 4:
        raise HTTPException(
            status_code=400,
            detail=f"Case {case_id} has only {len(wavs)} recordings; need at least 4.",
        )

    # ---- Phase A: Single quality pass shared across both pipelines ----
    quality_dict = await asyncio.to_thread(
        lambda: {p: assess_recording_quality(p) for p in wavs}
    )

    # ---- Phase B: ASR via Sarvam (parallel) ----
    asr_results: dict = {}
    if sarvam.configured:
        usable_paths = [p for p in wavs if quality_dict[p].usable]
        if usable_paths:
            asr_results = await transcribe_batch(usable_paths)

    # ---- Phase C: ASD pipeline ----
    prompted = wavs[:4]
    all_audio = wavs[:storage.NUM_QUESTIONS]
    asd_raw = await asyncio.to_thread(
        assess_asd_risk,
        prompted_question_audio_paths=prompted,
        all_audio_paths=all_audio,
        child_age_months=child_age_months,
        recording_quality=quality_dict,
    )
    with open(storage.result_path(case_id), "w") as f:
        json.dump(asd_raw, f, indent=2, default=str)

    # ---- Phase D: Speech-delay pipeline ----
    sd_recordings = _build_speech_delay_recordings(case_id, all_audio)
    # Splice ASR transcripts in. asr_transcript_raw stays None per the
    # MVP decision (see backend/asr.py module docstring).
    for rec in sd_recordings:
        result = asr_results.get(rec["audio_path"])
        if result:
            rec["asr_transcript_clean"] = result["clean"]
            rec["asr_transcript_raw"] = result["raw"]  # None in MVP

    sd_raw = await asyncio.to_thread(
        assess_speech_delay,
        sd_recordings,
        child_age_months=child_age_months,
        recording_quality=quality_dict,
    )
    with open(storage.speech_delay_result_path(case_id), "w") as f:
        json.dump(sd_raw, f, indent=2, default=str)

    transcripts_ok = sum(1 for r in asr_results.values() if r.get("clean"))
    logger.info(
        "Case %s assessed — asd.tier=%s sd.status=%s band=%s sarvam=%d/%d",
        case_id, asd_raw.get("tier"), sd_raw.get("delay_status"),
        sd_raw.get("developmental_band"), transcripts_ok, len(wavs),
    )

    return {
        "asd": {"raw": asd_raw, "summary": summarize_for_consumer(asd_raw)},
        "speech_delay": {"raw": sd_raw, "summary": summarize_speech_delay(sd_raw)},
    }


@app.get("/api/cases/{case_id}/summary")
def summary(case_id: str) -> dict:
    """
    Re-summarise saved results without re-running pipelines.
    Returns {asd: <summary or None>, speech_delay: <summary or None>}.
    """
    from core.asd_consumer_view import summarize_for_consumer
    from core.speech_delay_consumer_view import summarize_for_consumer as summarize_speech_delay

    out: dict = {"asd": None, "speech_delay": None}

    asd_path = storage.result_path(case_id)
    if os.path.exists(asd_path):
        with open(asd_path) as f:
            out["asd"] = summarize_for_consumer(json.load(f))

    sd_path = storage.speech_delay_result_path(case_id)
    if os.path.exists(sd_path):
        with open(sd_path) as f:
            out["speech_delay"] = summarize_speech_delay(json.load(f))

    if out["asd"] is None and out["speech_delay"] is None:
        raise HTTPException(status_code=404, detail=f"No saved results for case {case_id}")

    return out


@app.get("/api/cases/{case_id}/raw")
def raw_result(case_id: str, which: str = "asd") -> FileResponse:
    """
    Serve a saved result file as a download.

    `which=asd` (default) → asd_result.json
    `which=speech_delay`  → speech_delay_result.json
    """
    if which == "asd":
        path = storage.result_path(case_id)
        download_name = f"asd_result_{case_id}.json"
    elif which == "speech_delay":
        path = storage.speech_delay_result_path(case_id)
        download_name = f"speech_delay_result_{case_id}.json"
    else:
        raise HTTPException(status_code=400, detail=f"which must be 'asd' or 'speech_delay', got {which!r}")

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No {which} result for case {case_id}")
    return FileResponse(path, media_type="application/json", filename=download_name)
