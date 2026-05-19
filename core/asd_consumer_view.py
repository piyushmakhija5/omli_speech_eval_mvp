"""Consumer-facing summary of a pipeline result.

Pure translation layer — takes the raw output of `assess_asd_risk()` and
returns a dict shaped for non-technical readers (parents, clinicians at a
glance, dashboards). No pipeline logic here; only copy and categorisation.

Single source of truth for tier-to-language mapping and color tokens.
Update copy here, not in the frontend.
"""

from __future__ import annotations

import os
from typing import Any

DISCLAIMER = (
    "This is an ASD screening tool, not a diagnosis. "
    "Results should always be reviewed by a qualified specialist."
)


# Binary-ish speech-pattern stamp shown under the headline.
# Independent of tier color (which encodes severity); this is just yes/no/?.
SPEECH_STATUS = {
    "no_indicators":        {"label": "Typical",      "color": "green"},
    "monitor":              {"label": "Atypical",     "color": "yellow"},
    "recommend_evaluation": {"label": "Atypical",     "color": "red"},
    "insufficient_data":    {"label": "Inconclusive", "color": "gray"},
}

TIER_DISPLAY = {
    "no_indicators": {
        "verdict": "No concerns",
        "color": "green",
        "subtext": "No atypical speech patterns detected in this screening.",
    },
    "monitor": {
        "verdict": "Worth monitoring",
        "color": "yellow",
        "subtext": "Some atypical patterns detected. Consider re-screening in 2 weeks.",
    },
    "recommend_evaluation": {
        "verdict": "Talk to a specialist",
        "color": "red",
        "subtext": "Multiple speech patterns differ from age expectations.",
    },
    "insufficient_data": {
        "verdict": "Need clearer recordings",
        "color": "gray",
        "subtext": "Not enough usable audio for a reliable screening.",
    },
}

GROUP_DISPLAY = {
    "prosody": {
        "label": "Speech melody",
        "typical": "Pitch variation looks age-appropriate.",
        "atypical": "Pitch variation differs from age expectations.",
        "not_computed": "Not enough data to assess.",
    },
    "spectral": {
        "label": "Voice quality",
        "typical": "Voice texture looks age-appropriate.",
        "atypical": "Voice texture differs from age expectations.",
        "not_computed": "Not enough data to assess.",
    },
    "interaction": {
        "label": "Conversation rhythm",
        "typical": "Response timing looks age-appropriate.",
        "atypical": "Response timing varies more than expected.",
        "not_computed": "Not enough data to assess.",
    },
    "voice_stability": {
        "label": "Voice stability",
        "typical": "Voice steadiness and pauses look age-appropriate.",
        "atypical": "Voice steadiness or pause patterns differ from age expectations.",
        "not_computed": "Not enough data to assess.",
    },
}

GROUP_ORDER = ["prosody", "spectral", "interaction", "voice_stability"]


# Per-marker display metadata. Plain label first, technical label as subtitle.
MARKER_DEFS = [
    # (group, key, label, tech_label, unit, direction, description, norm_source)
    # norm_source values:
    #   "<key>_norms"           — picks CONFIG[<key>_norms][age_group] → {mean, std}
    #   ("voice_stability", "X_mean", "X_std")
    #   "threshold:<config_key>" — uses CONFIG[<config_key>] as the threshold
    ("prosody", "pitch_variability",
     "Pitch variability", "F0 standard deviation", "Hz", "high",
     "How much the voice's pitch rises and falls. Very flat speech is associated with atypical prosody.",
     "pitch_variability_norms"),
    ("prosody", "pitch_range",
     "Pitch range", "F0 max − min", "Hz", "high",
     "Span between the highest and lowest pitch in speech.",
     "pitch_range_norms"),
    ("spectral", "spectral_entropy",
     "Spectral variety", "Spectral entropy", "", "both",
     "Diversity of frequencies in the voice. Both very low and very high values can be atypical.",
     "spectral_entropy_norms"),
    ("spectral", "ltas_slope",
     "Spectral tilt", "LTAS slope (long-term avg spectrum)", "", "high",
     "Balance between low and high frequencies across the recording.",
     "ltas_slope_norms"),
    ("interaction", "latency_mean",
     "Response time", "Mean response latency", "ms", "high",
     "Average time between the prompt ending and the child starting to speak.",
     "latency_norms"),
    ("interaction", "turn_taking_cv",
     "Response variability", "Latency CV (std / mean)", "", "threshold",
     "How much the child's response time varies across the four open-ended questions.",
     "threshold:turn_taking_cv_threshold"),
    ("voice_stability", "jitter",
     "Pitch instability", "Jitter (local %)", "%", "high",
     "Small pitch instabilities between consecutive voice cycles.",
     ("voice_stability_norms", "jitter_mean", "jitter_std")),
    ("voice_stability", "shimmer",
     "Loudness instability", "Shimmer (local %)", "%", "high",
     "Small loudness instabilities between consecutive voice cycles.",
     ("voice_stability_norms", "shimmer_mean", "shimmer_std")),
    ("voice_stability", "hnr",
     "Voice clarity", "Harmonics-to-noise ratio", "dB", "low",
     "Ratio of clean voice signal to noise. Higher is clearer.",
     ("voice_stability_norms", "hnr_mean", "hnr_std")),
    ("voice_stability", "pause_variance",
     "Pause variability", "Variance of silent-segment durations", "ms²", "threshold",
     "How uneven the pauses between speech segments are.",
     "threshold:pause_variance_threshold"),
]


def _is_atypical(value: float, low: float, high: float, direction: str) -> bool:
    if direction == "high":
        return value > high
    if direction == "low":
        return value < low
    return value < low or value > high  # both


def _marker_value(key: str, biomarkers: dict) -> float | None:
    """Most markers are direct lookups; turn_taking_cv is derived."""
    if key == "turn_taking_cv":
        lat_mean = biomarkers.get("latency_mean")
        lat_std = biomarkers.get("latency_std")
        if lat_mean and lat_std and lat_mean > 0:
            return lat_std / lat_mean
        return None
    return biomarkers.get(key)


def _norm_for(norm_source, config: dict, age_group: str):
    """Resolve a MARKER_DEFS norm_source into (mean, std) or ('threshold', t)."""
    if isinstance(norm_source, str):
        if norm_source.startswith("threshold:"):
            t_key = norm_source.split(":", 1)[1]
            return ("threshold", config.get(t_key))
        norms = config.get(norm_source, {})
        n = norms.get(age_group)
        if not n:
            return None
        return ("norm", n["mean"], n["std"])
    # Tuple form: voice_stability_norms with prefixed keys
    norms_key, mean_k, std_k = norm_source
    n = config.get(norms_key, {}).get(age_group, {})
    if mean_k not in n or std_k not in n:
        return None
    return ("norm", n[mean_k], n[std_k])


def _build_markers(raw: dict) -> dict:
    """Return {group_key: [marker_dict, ...]} with all per-marker chart data."""
    from core.asd_pipeline import CONFIG  # avoid circular at import time

    age_group = raw.get("age_group")
    biomarkers = raw.get("biomarkers", {})
    threshold_sd = CONFIG.get("atypical_threshold_sd", 2.0)

    out: dict[str, list] = {g: [] for g in GROUP_ORDER}
    if not age_group:
        return out

    for group, key, label, tech_label, unit, direction, description, norm_source in MARKER_DEFS:
        resolved = _norm_for(norm_source, CONFIG, age_group)
        value = _marker_value(key, biomarkers)
        marker = {
            "key": key,
            "label": label,
            "tech_label": tech_label,
            "unit": unit,
            "direction": direction,
            "description": description,
            "value": float(value) if value is not None else None,
            "computed": value is not None,
        }

        if resolved is None or resolved[0] == "threshold" and resolved[1] is None:
            marker["atypical"] = None
        elif resolved[0] == "norm":
            _, mean, std = resolved
            low = mean - threshold_sd * std
            high = mean + threshold_sd * std
            marker.update({
                "norm_mean": float(mean), "norm_std": float(std),
                "norm_low": float(low), "norm_high": float(high),
            })
            marker["atypical"] = _is_atypical(value, low, high, direction) if value is not None else None
        else:  # threshold
            t = float(resolved[1])
            marker["threshold"] = t
            marker["atypical"] = (value > t) if value is not None else None

        out[group].append(marker)

    return out


def _confidence_level(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "moderate"
    return "low"


def _confidence_note(conf: dict) -> str:
    warnings = conf.get("warnings", [])
    if warnings:
        return warnings[0]
    return "All biomarkers computed from clean recordings."


def _next_steps(tier: str, likely_child) -> list[str]:
    if tier == "no_indicators":
        steps = [
            "No action needed from a screening standpoint.",
            "You may re-screen periodically to track development.",
        ]
    elif tier == "monitor":
        steps = [
            "Re-record the same set of questions in about 2 weeks.",
            "If patterns persist, share the results with a pediatric SLP.",
        ]
    elif tier == "recommend_evaluation":
        steps = [
            "Share these results with a pediatric speech-language pathologist or developmental pediatrician.",
            "A screening result is not a diagnosis — a specialist can interpret it in context.",
            "Re-recording in 2 weeks can help confirm whether patterns are stable.",
        ]
    elif tier == "insufficient_data":
        steps = [
            "Record again in a quieter room.",
            "Aim for at least 5 seconds per open-ended question.",
            "Make sure the child is the one speaking, close to the microphone.",
        ]
    else:
        steps = []

    # When voice_check flags an adult voice, prepend a caveat as the first
    # action — without suppressing the tier-based guidance that follows.
    if likely_child is False:
        steps = [
            "Adult voice identified — the screening targets children 3–8 and assumes a child's voice. "
            "Confirm a child was the one recording, then re-record before relying on the verdict.",
        ] + steps

    return steps


def _rejection_notes(quality_report: dict) -> list[str]:
    notes = []
    for r in quality_report.get("all_recordings", []):
        if r.get("usable"):
            continue
        reason = r.get("rejection_reason") or "rejected"
        label = os.path.basename(r.get("path", "")).replace(".wav", "") or "recording"
        notes.append(f"{label}: {reason}")
    return notes


def summarize_for_consumer(raw: dict) -> dict:
    """Translate a raw pipeline result into a consumer-facing summary."""
    tier = raw.get("tier", "insufficient_data")
    tier_display = TIER_DISPLAY.get(tier, TIER_DISPLAY["insufficient_data"])

    voice_check = raw.get("voice_check") or {}
    likely_child = voice_check.get("likely_child")

    # Trust the pipeline's tier output: headline color and speech stamp both
    # reflect what the screening actually computed. The voice_check concern is
    # surfaced via the alert banner and as the first item in next_steps.
    headline = {
        "tier": tier,
        "verdict": tier_display["verdict"],
        "color": tier_display["color"],
        "subtext": tier_display["subtext"],
        "speech_status": SPEECH_STATUS.get(tier, {"label": "Inconclusive", "color": "gray"}),
    }

    alerts = []
    if likely_child is False:
        alerts.append({
            "level": "warning",
            "title": "Adult voice identified",
            "body": voice_check.get("reason")
                or "The recorded voice appears to be an adult's, not a child's. "
                   "The screening targets children 3–8 — results may not be meaningful for this speaker.",
        })

    group_details = raw.get("group_details", {})
    markers_by_group = _build_markers(raw)
    groups: list[dict[str, Any]] = []
    for key in GROUP_ORDER:
        info = GROUP_DISPLAY[key]
        gd = group_details.get(key, {})
        if not gd.get("computable", False):
            status = "not_computed"
        elif gd.get("atypical", False):
            status = "atypical"
        else:
            status = "typical"
        groups.append({
            "key": key,
            "name": info["label"],
            "status": status,
            "plain": info[status],
            "markers": markers_by_group.get(key, []),
        })

    conf = raw.get("confidence", {})
    score = conf.get("confidence_score", 0.0)
    confidence = {
        "level": _confidence_level(score),
        "score": score,
        "note": _confidence_note(conf),
    }

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
        "groups": groups,
        "confidence": confidence,
        "session": session,
        "next_steps": _next_steps(tier, likely_child),
        "disclaimer": DISCLAIMER,
    }
