"""
Consumer-facing summary of a speech-delay pipeline result.
===========================================================
Pure translation layer — converts ``assess_speech_delay()`` output into a
dict shaped for parent / clinician at-a-glance UIs. Mirrors the structure
of ``core.asd_consumer_view`` so the frontend can render both screenings
side-by-side with one rendering codepath per primitive (headline card,
domain card, metric chart, alert banner, next-steps list).

Single source of truth for: delay-status → verdict text, domain →
plain-language label, per-metric display metadata (label, tech-label,
unit, description). Update copy here, not in the frontend.

Each metric is structured for a **percentile bar chart**: a 0-100 axis
with quartile bands (red < p10, yellow p10-p25, green ≥ p25) and the
child's percentile as a dot. Different visual from ASD's range bars,
but the chart data is shaped to be straightforward to render.
"""

from __future__ import annotations

from typing import Any, Optional


DISCLAIMER = (
    "This is a speech-delay screening tool, not a diagnosis. All norms are "
    "provisional and have not yet been calibrated against SLP ground truth "
    "for Indian Hindi/English/Hinglish children. Always have a qualified "
    "specialist review the results before acting on them."
)


DELAY_DISPLAY = {
    "on_track": {
        "verdict": "On track for age",
        "color": "green",
        "subtext": "Speech metrics align with age expectations.",
    },
    "behind": {
        "verdict": "Worth monitoring",
        "color": "yellow",
        "subtext": "Some metrics fall below age expectations. Re-screen soon.",
    },
    "significantly_behind": {
        "verdict": "Talk to a specialist",
        "color": "red",
        "subtext": "Multiple metrics significantly below age expectations.",
    },
    "insufficient_data": {
        "verdict": "Need clearer recordings",
        "color": "gray",
        "subtext": "Not enough usable audio or transcripts for a reliable screening.",
    },
    "not_computed": {
        "verdict": "Not assessed",
        "color": "gray",
        "subtext": "Speech-delay metrics have not been computed for this case.",
    },
}


DOMAIN_DISPLAY = {
    "articulation": {
        "name": "Articulation",
        "description": "How clearly speech sounds are produced.",
    },
    "language": {
        "name": "Language",
        "description": "Words used, sentences repeated, things named.",
    },
    "fluency": {
        "name": "Fluency & voice",
        "description": "Speaking pace, pauses, voice quality.",
    },
}


METRIC_DISPLAY: dict[str, dict[str, str]] = {
    "single_word_pcc": {
        "domain": "articulation",
        "label": "Single-word accuracy",
        "tech_label": "Single-word PCC (overall)",
        "unit": "%",
        "description": "How accurately individual words are pronounced.",
    },
    "connected_pcc": {
        "domain": "articulation",
        "label": "Sentence accuracy",
        "tech_label": "Connected-speech PCC",
        "unit": "%",
        "description": "How accurately words are pronounced in connected sentences.",
    },
    "word_coverage": {
        "domain": "language",
        "label": "Word repetition",
        "tech_label": "Word coverage on sentence repetition",
        "unit": "%",
        "description": "Share of expected words the child repeated back.",
    },
    "naming_accuracy": {
        "domain": "language",
        "label": "Naming accuracy",
        "tech_label": "Picture-naming accuracy",
        "unit": "%",
        "description": "How often the child names objects correctly.",
    },
    "speaking_rate": {
        "domain": "fluency",
        "label": "Speaking pace",
        "tech_label": "Speaking rate (words per minute)",
        "unit": "WPM",
        "description": "How fast the child speaks across all recordings.",
    },
    "pause_ratio": {
        "domain": "fluency",
        "label": "Pause amount",
        "tech_label": "Pause ratio (silent / total frames)",
        "unit": "",
        "description": "Share of time spent silent vs speaking.",
    },
    "pitch_mean": {
        "domain": "fluency",
        "label": "Voice pitch",
        "tech_label": "Mean fundamental frequency (F0)",
        "unit": "Hz",
        "description": "Average pitch of the child's voice.",
    },
    "jitter": {
        "domain": "fluency",
        "label": "Pitch stability",
        "tech_label": "Jitter (local %)",
        "unit": "%",
        "description": "Small pitch instabilities between voice cycles.",
    },
    "shimmer": {
        "domain": "fluency",
        "label": "Loudness stability",
        "tech_label": "Shimmer (local %)",
        "unit": "%",
        "description": "Small loudness instabilities between voice cycles.",
    },
    "hnr": {
        "domain": "fluency",
        "label": "Voice clarity",
        "tech_label": "Harmonics-to-noise ratio",
        "unit": "dB",
        "description": "Ratio of clean voice signal to noise. Higher is clearer.",
    },
}


# Status pill color (single source of truth — also used by the list endpoint
# to colour the case-row pill before any UI rendering happens).
STATUS_COLOR = {
    "on_track": "green",
    "behind": "yellow",
    "significantly_behind": "red",
    "insufficient_data": "gray",
    "not_computed": "gray",
}


SPEAKING_RATE_MODE_NOTE = {
    "asr_words_raw": None,
    "asr_words_clean_fallback": "Computed from cleaned transcript — disfluencies may be under-counted, slightly under-estimating speaking rate.",
    "acoustic_estimate_unreliable": "Acoustic estimate only — language-specific syllable-to-word conversion not yet calibrated. Interpret with caution.",
    "unavailable": None,
}


# =============================================================================
# HELPERS
# =============================================================================

def _confidence_level(score: float) -> str:
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "moderate"
    return "low"


def _confidence_note(conf: dict) -> str:
    warnings = conf.get("warnings", [])
    if warnings:
        return warnings[0]
    computed = conf.get("computed_metrics", 0)
    total = conf.get("total_metrics", 0)
    return f"{computed} of {total} metrics computed."


def _delay_label(raw: dict, age_groups_config: Optional[dict] = None) -> str:
    """Human-readable summary of developmental_band + delay_months."""
    band = raw.get("developmental_band")
    months = raw.get("delay_months")
    chrono_group = raw.get("age_group")

    if band == "insufficient_data":
        return "Developmental level couldn't be determined from these recordings — articulation and language metrics were not computable."
    if band == chrono_group and months == 0:
        return f"Performing at age level ({band} years)."
    if band and band.startswith("below_"):
        below_of = band.split("_", 1)[1]
        if months:
            return f"Below the {below_of} year expected range — approximately {months} months behind chronological age."
        return f"Below the {below_of} year expected range."
    if band and months:
        return f"Performing at the {band} year level — approximately {months} months behind chronological age."
    if band:
        return f"Performing at the {band} year level."
    return "Developmental level not assessed."


def _next_steps(delay_status: str, likely_child) -> list[str]:
    if delay_status == "on_track":
        steps = [
            "No action needed from a screening standpoint.",
            "Re-screen periodically to track development.",
        ]
    elif delay_status == "behind":
        steps = [
            "Re-record the same set of questions in about 2 weeks.",
            "If the same patterns appear again, share results with a pediatric SLP.",
        ]
    elif delay_status == "significantly_behind":
        steps = [
            "Share these results with a pediatric speech-language pathologist.",
            "A screening result is not a diagnosis — a specialist can interpret in context.",
            "Re-recording in 2 weeks helps confirm whether patterns are stable.",
        ]
    elif delay_status == "insufficient_data":
        steps = [
            "Record again in a quieter room and ensure the child speaks close to the microphone.",
            "Aim for at least 5 seconds of speech per open-ended question.",
            "Make sure ASR is configured — sentence repetition, naming, and speaking rate depend on transcripts.",
        ]
    else:
        steps = []

    # Adult voice caveat — same wording as ASD's consumer view.
    if likely_child is False:
        steps = [
            "Adult voice identified — the screening targets children 3–8 and assumes a child's voice. "
            "Confirm a child was the one recording, then re-record before relying on the verdict.",
        ] + steps

    return steps


def _percentile_to_status(percentile: Optional[int], thresholds: dict) -> str:
    if percentile is None:
        return "not_computed"
    if percentile >= thresholds.get("on_track_min_percentile", 25):
        return "on_track"
    if percentile >= thresholds.get("behind_min_percentile", 10):
        return "behind"
    return "significantly_behind"


def _rejection_notes(quality_report: dict) -> list[str]:
    out = []
    for r in quality_report.get("all_recordings", []):
        if r.get("usable"):
            continue
        label = (r.get("path") or "").rsplit("/", 1)[-1].replace(".wav", "") or "recording"
        reason = r.get("rejection_reason") or "rejected"
        out.append(f"{label}: {reason}")
    return out


# =============================================================================
# MAIN ENTRY
# =============================================================================

def summarize_for_consumer(raw: dict) -> dict:
    """Translate raw speech_delay pipeline output into consumer-facing structure."""
    delay_status = raw.get("delay_status", "not_computed")
    display = DELAY_DISPLAY.get(delay_status, DELAY_DISPLAY["not_computed"])

    voice_check = raw.get("voice_check") or {}
    likely_child = voice_check.get("likely_child")

    headline = {
        "delay_status": delay_status,
        "verdict": display["verdict"],
        "color": display["color"],
        "subtext": display["subtext"],
        "developmental_band": raw.get("developmental_band"),
        "delay_months": raw.get("delay_months"),
        "delay_label": _delay_label(raw),
        "composite_percentile": raw.get("composite_percentile"),
        "developmental_composite": raw.get("developmental_composite"),
    }

    alerts: list[dict[str, Any]] = []
    if likely_child is False:
        alerts.append({
            "level": "warning",
            "title": "Adult voice identified",
            "body": voice_check.get("reason")
                or "The recorded voice appears to be an adult's, not a child's. "
                   "The screening targets children 3–8 — results may not be meaningful.",
        })
    if raw.get("calibration_status") == "provisional":
        alerts.append({
            "level": "info",
            "title": "Provisional norms",
            "body": (
                "Percentile tables are educated estimates from published research on Western "
                "English-speaking children. SLP calibration on Indian children is still pending — "
                "exact percentiles and delay-month numbers may shift after calibration."
            ),
        })

    # Domain cards
    domain_detail = raw.get("domain_detail", {})
    domains_out = []
    for key in ("articulation", "language", "fluency"):
        info = domain_detail.get(key, {})
        display_info = DOMAIN_DISPLAY[key]
        domains_out.append({
            "key": key,
            "name": display_info["name"],
            "description": display_info["description"],
            "percentile": info.get("percentile"),
            "status": info.get("status", "not_computed"),
            "computed_count": info.get("computed_count", 0),
            "total_count": info.get("total_count", 0),
        })

    # Per-metric — chart-ready data for percentile-bar rendering.
    metrics_raw = raw.get("metrics", {})
    thresholds = {"on_track_min_percentile": 25, "behind_min_percentile": 10}
    metrics_out = []
    for key, display_info in METRIC_DISPLAY.items():
        m = metrics_raw.get(key, {})
        computed = bool(m.get("computed"))
        percentile = m.get("percentile")
        status = _percentile_to_status(percentile, thresholds) if computed else "not_computed"
        entry = {
            "key": key,
            "domain": display_info["domain"],
            "label": display_info["label"],
            "tech_label": display_info["tech_label"],
            "unit": display_info["unit"],
            "description": display_info["description"],
            "value": m.get("value"),
            "percentile": percentile,
            "computed": computed,
            "status": status,
        }
        if not computed:
            entry["reason"] = m.get("reason")
        # speaking_rate carries a mode flag — surface the unreliable-estimate caveat.
        if key == "speaking_rate":
            mode = m.get("mode")
            entry["mode"] = mode
            note = SPEAKING_RATE_MODE_NOTE.get(mode)
            if note:
                entry["mode_note"] = note
        metrics_out.append(entry)

    # Confidence
    conf = raw.get("confidence", {})
    score = conf.get("confidence_score", 0.0)
    confidence = {
        "level": _confidence_level(score),
        "score": score,
        "note": _confidence_note(conf),
    }

    # Session
    qr = raw.get("quality_report", {})
    recs = qr.get("all_recordings", [])
    usable = sum(1 for r in recs if r.get("usable"))
    session = {
        "usable": usable,
        "total": len(recs),
        "rejection_notes": _rejection_notes(qr),
    }

    return {
        "headline": headline,
        "alerts": alerts,
        "domains": domains_out,
        "metrics": metrics_out,
        "confidence": confidence,
        "session": session,
        "next_steps": _next_steps(delay_status, likely_child),
        "disclaimer": DISCLAIMER,
        "calibration_status": raw.get("calibration_status"),
    }
