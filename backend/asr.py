"""
ASR (Automatic Speech Recognition) integration for the collector backend.
=========================================================================

Wraps Sarvam AI Speech-to-Text — the production ASR for the Omli stack.
Mirrors the production ``services/speech_provider_service.py`` wrapper:
singleton, persistent aiohttp session, identical form fields, identical
response-parser fallback chain.

Why no Whisper here (MVP decision, 2026-05-19)
-----------------------------------------------
Production has both Sarvam (primary) and openai-whisper (optional secondary),
but the Whisper path only runs when Sarvam fails — and Sarvam currently
strips disfluencies server-side, so the original disfluency-aware-speaking-rate
rationale collapses if Sarvam is reliable enough.

For the MVP we ship **Sarvam-only**. Consequences:
- ``asr_transcript_clean`` populated from Sarvam.
- ``asr_transcript_raw`` always None — speech_delay's ``speaking_rate`` falls
  back to the cleaned transcript and surfaces a ``mode_note`` flagging that
  disfluent children may be under-counted.
- If Sarvam errors on a recording, that recording's transcripts stay None and
  the affected metrics report ``computed=false`` with reason.

**Future Whisper-fallback insertion point** (when needed):
    1. Add a ``WhisperProvider`` class in this module — local model via
       ``openai-whisper`` package or cloud via OpenAI SDK.
    2. In ``transcribe_file`` below, after Sarvam call: if Sarvam errored,
       try Whisper and assign its output to both ``clean`` and ``raw``.
       Alternatively, always run Whisper alongside Sarvam to populate ``raw``
       with disfluency-preserving transcripts for honest speaking_rate.
    3. Add ``ENABLE_WHISPER_FALLBACK`` env var to gate the behaviour.

Until then, ``raw`` stays None and the architecture acknowledges this.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Optional

import aiohttp

logger = logging.getLogger("asr")


# Sentinel for "caller didn't specify language_code" — distinguishes from
# explicit None (which means "auto-detect / send no language_code field").
_UNSET = object()


class SarvamProvider:
    """
    Singleton wrapper around Sarvam AI Speech-to-Text.

    Reuses one persistent aiohttp.ClientSession across the process lifetime —
    eliminates TLS-handshake latency on subsequent calls. Warmed up at FastAPI
    startup via ``warmup()``.
    """

    _instance: Optional["SarvamProvider"] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.api_key: str = os.getenv("SARVAM_API_KEY", "").strip()
        self.url: str = (
            os.getenv("SARVAM_STT_URL", "https://api.sarvam.ai/speech-to-text")
            .strip()
            .rstrip("/")
        )
        self.language_code: str = os.getenv("SARVAM_STT_LANGUAGE_CODE", "").strip()
        self.model: str = os.getenv("SARVAM_STT_MODEL", "").strip()
        self.mode: str = os.getenv("SARVAM_STT_MODE", "").strip()
        self._session: Optional[aiohttp.ClientSession] = None
        SarvamProvider._initialized = True

    # -- session lifecycle ----------------------------------------------------

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                ssl=True,
                limit=20,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
            )
            timeout = aiohttp.ClientTimeout(total=60)
            self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
            logger.info("Sarvam aiohttp session created")
        return self._session

    async def warmup(self) -> None:
        """Pre-establish the TCP/SSL pool so the first real request is fast."""
        try:
            await self._get_session()
            logger.info("Sarvam connection pool warmed up")
        except Exception as exc:
            logger.warning("Sarvam warmup failed (non-fatal): %s", exc)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("Sarvam aiohttp session closed")

    # -- transcription --------------------------------------------------------

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str,
        content_type: Optional[str] = None,
        language_code: object = _UNSET,
        model: Optional[str] = None,
    ) -> str:
        """
        Send one audio file to Sarvam STT. Returns the transcript string.

        language_code semantics match production:
          _UNSET → fall back to ``SARVAM_STT_LANGUAGE_CODE`` env (default ``hi-IN``)
          None   → omit the field; Sarvam auto-detects
          str    → send as-is

        Raises ``RuntimeError`` on HTTP 4xx/5xx — caller catches.
        """
        if not audio_bytes:
            raise ValueError("Audio bytes empty")
        if not self.api_key:
            raise RuntimeError("SARVAM_API_KEY is not configured")

        safe_filename = (filename or "speech.wav").strip() or "speech.wav"
        ct = (content_type or "audio/wav").strip() or "audio/wav"

        if language_code is _UNSET:
            lc = self.language_code
        elif language_code is None:
            lc = ""
        else:
            lc = str(language_code).strip()

        md = (model or "").strip() or self.model
        mode = self.mode.strip()

        form = aiohttp.FormData()
        form.add_field("file", audio_bytes, filename=safe_filename, content_type=ct)
        if lc:
            form.add_field("language_code", lc)
        if md:
            form.add_field("model", md)
        # codemix mode is only valid when language_code is also sent; sending
        # mode without language_code causes Sarvam to return 400 (matches the
        # production wrapper's safety guard at speech_provider_service.py:231).
        if mode and lc:
            form.add_field("mode", mode)

        headers = {"api-subscription-key": self.api_key, "Accept": "application/json"}
        session = await self._get_session()

        async with session.post(self.url, headers=headers, data=form) as resp:
            payload_text = await resp.text()
            if resp.status >= 400:
                logger.error("Sarvam STT failed (%s): %s", resp.status, payload_text[:500])
                raise RuntimeError(f"Sarvam transcription failed: HTTP {resp.status}")
            try:
                payload = json.loads(payload_text) if payload_text else {}
            except json.JSONDecodeError:
                logger.warning("Sarvam returned non-JSON: %s", payload_text[:300])
                payload = {}
            return _extract_transcript(payload)


def _extract_transcript(payload: dict) -> str:
    """
    Walk the Sarvam response shape and pull out the transcript.

    Sarvam's API has returned the transcript in several places over time —
    this fallback chain matches the production wrapper's parser
    (speech_provider_service.py:240-269).
    """
    if not payload:
        return ""

    direct = payload.get("transcript") or payload.get("text")
    if isinstance(direct, str):
        return direct.strip()

    data = payload.get("data")
    if isinstance(data, dict):
        dt = data.get("transcript") or data.get("text")
        if isinstance(dt, str):
            return dt.strip()
        results = data.get("results")
        if isinstance(results, list):
            combined = _join_segments(results)
            if combined:
                return combined

    segs = payload.get("segments") or payload.get("results")
    if isinstance(segs, list):
        return _join_segments(segs)

    return ""


def _join_segments(segments: Any) -> str:
    if not isinstance(segments, list):
        return ""
    return " ".join(
        item.get("text", item.get("transcript", "")).strip()
        for item in segments
        if isinstance(item, dict) and (item.get("text") or item.get("transcript"))
    ).strip()


# Module-level singleton — created at import time so env vars are read once.
sarvam = SarvamProvider()


# =============================================================================
# Top-level orchestration helpers used by the backend
# =============================================================================

async def transcribe_file(audio_path: str) -> dict:
    """
    Transcribe one audio file via Sarvam.

    Returns ``{"clean": str | None, "raw": None, "error": str | None}``.
    ``raw`` is always None in the MVP — see module docstring for the
    Whisper-fallback insertion point.
    """
    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
    except Exception as exc:
        logger.warning("Could not read %s: %s", audio_path, exc)
        return {"clean": None, "raw": None, "error": f"file read failed: {exc}"}

    filename = os.path.basename(audio_path)
    try:
        transcript = await sarvam.transcribe(audio_bytes, filename, content_type="audio/wav")
        return {"clean": transcript or None, "raw": None, "error": None}
    except Exception as exc:
        logger.warning("Sarvam transcribe failed for %s: %s", audio_path, exc)
        return {"clean": None, "raw": None, "error": str(exc)}


async def transcribe_batch(audio_paths: list) -> dict:
    """
    Run Sarvam transcription on each path concurrently.

    Sarvam's aiohttp session pool (limit=20) handles fan-out; on a typical
    12-recording case this completes in roughly the latency of a single call.

    Returns ``{path: {clean, raw, error}}`` for every input path.
    """
    results = await asyncio.gather(
        *[transcribe_file(p) for p in audio_paths],
        return_exceptions=False,
    )
    return dict(zip(audio_paths, results))
