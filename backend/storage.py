"""Filesystem layout helpers for the collector.

Cases live at:
    data/cases/<case_id>/q01.wav … q12.wav

`case_id` is restricted to [A-Za-z0-9_-] to keep it safe inside path joins.
"""

from __future__ import annotations

import os
import re
import secrets
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES_ROOT = os.path.join(PROJECT_ROOT, "data", "cases")
QUESTIONS_PATH = os.path.join(PROJECT_ROOT, "data", "questions.json")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
NUM_QUESTIONS = 12

_CASE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def new_case_id() -> str:
    """Generate a fresh case id: 20260518-191523-a1b2."""
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


def validate_case_id(case_id: str) -> str:
    if not _CASE_ID_RE.match(case_id):
        raise ValueError(f"Invalid case_id: {case_id!r}. Allowed: A-Z, a-z, 0-9, _, -, up to 64 chars.")
    return case_id


def case_dir(case_id: str, create: bool = False) -> str:
    validate_case_id(case_id)
    path = os.path.join(CASES_ROOT, case_id)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def wav_path(case_id: str, q_index: int) -> str:
    """Path for question q_index (1-based)."""
    if not 1 <= q_index <= NUM_QUESTIONS:
        raise ValueError(f"q_index must be 1..{NUM_QUESTIONS}, got {q_index}")
    return os.path.join(case_dir(case_id, create=True), f"q{q_index:02d}.wav")


def list_wavs(case_id: str) -> list[str]:
    """Sorted .wav files for a case. Empty list if case has none."""
    cdir = case_dir(case_id)
    if not os.path.isdir(cdir):
        return []
    return sorted(
        os.path.join(cdir, f)
        for f in os.listdir(cdir)
        if f.endswith(".wav")
    )


def list_case_ids() -> list[str]:
    """All case directories under data/cases/, sorted by case_id (newest first)."""
    if not os.path.isdir(CASES_ROOT):
        return []
    ids = [
        name for name in os.listdir(CASES_ROOT)
        if os.path.isdir(os.path.join(CASES_ROOT, name))
        and _CASE_ID_RE.match(name)
    ]
    return sorted(ids, reverse=True)


def parse_created_at(case_id: str) -> str | None:
    """Extract YYYY-MM-DDTHH:MM:SS from a case_id like 20260518-191523-xxxx."""
    parts = case_id.split("-")
    if len(parts) < 2 or len(parts[0]) != 8 or len(parts[1]) != 6:
        return None
    d, t = parts[0], parts[1]
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}T{t[:2]}:{t[2:4]}:{t[4:6]}"


def result_path(case_id: str) -> str:
    return os.path.join(case_dir(case_id), "asd_result.json")
