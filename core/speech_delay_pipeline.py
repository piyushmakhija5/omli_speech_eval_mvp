"""
Speech Delay Detection Pipeline
================================
Takes the same 12 recordings as the ASD pipeline, computes 10 metrics across
articulation / language / fluency domains, scores against percentile norms,
and reports a developmental band + delay status.

Input contract (see assess_speech_delay()):
    recordings: list of per-recording dicts with audio_path, task_type,
                expected_text (None for prompted_question), and
                asr_transcript_clean / asr_transcript_raw / pronunciation_scores
                — all optional. Missing ASR/pronunciation inputs cause the
                relevant metrics to report computed=False with a reason.

The pipeline does NOT call ASR or pronunciation services. It is a
consumer of those outputs. The backend (or test code) is responsible for
populating the transcript and pronunciation fields where available.

Composition: lowest_domain by default — the worst-performing domain
drives delay_status. This prevents a child with severe articulation
issues from being masked by good fluency.

Developmental band: computed from articulation + language only
(developmental_composite). Fluency is reported as a separate signal but
doesn't factor into the band/delay calculation, because fluency varies
for reasons unrelated to language acquisition (microphone, mood, prompt
phrasing).

All norms are PROVISIONAL — see core/speech_delay_config.json header.
"""

from __future__ import annotations

import json
import logging
import os
from enum import Enum
from typing import Optional

import numpy as np
from rapidfuzz import fuzz

from core.audio_analysis import (
    MetricResult,
    RecordingQuality,
    _load_audio_safe,
    assess_recording_quality as _assess_recording_quality,
    compute_voice_check as _compute_voice_check_shared,
    extract_pause_distribution as _extract_pause_distribution,
    extract_pitch_metrics as _extract_pitch_metrics,
    extract_voice_stability as _extract_voice_stability,
)

logger = logging.getLogger("speech_delay_pipeline")


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class DelayStatus(str, Enum):
    ON_TRACK = "on_track"
    BEHIND = "behind"
    SIGNIFICANTLY_BEHIND = "significantly_behind"
    INSUFFICIENT_DATA = "insufficient_data"
    NOT_COMPUTED = "not_computed"


VALID_TASK_TYPES = {"sentence_repetition", "picture_naming", "prompted_question"}


DOMAINS: dict[str, list[str]] = {
    "articulation": ["single_word_pcc", "connected_pcc"],
    "language": ["word_coverage", "naming_accuracy"],
    "fluency": ["speaking_rate", "pause_ratio", "pitch_mean", "jitter", "shimmer", "hnr"],
}


# Maps metric name → norm-table key in the config.
_NORM_KEY_BY_METRIC: dict[str, str] = {
    "single_word_pcc": "single_word_pcc",
    "connected_pcc": "single_word_pcc",  # uses same table; offset applied to value
    "word_coverage": "word_coverage",
    "naming_accuracy": "naming_accuracy",
    "speaking_rate": "speaking_rate_wpm",
    "pause_ratio": "pause_ratio",
    "pitch_mean": "pitch_hz",
    "jitter": "jitter_pct",
    "shimmer": "shimmer_pct",
    "hnr": "hnr_db",
}


# =============================================================================
# CONFIG LOADER
# =============================================================================

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speech_delay_config.json")


def _load_config(config_path: Optional[str] = None) -> dict:
    path = config_path or _CONFIG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Speech-delay config not found at {path}")
    with open(path, "r") as f:
        raw = json.load(f)
    logger.info(f"Speech-delay config loaded from {path} — {len(raw.get('age_groups', {}))} age groups")
    return raw


def reload_config(config_path: Optional[str] = None) -> dict:
    """Reload config at runtime."""
    global CONFIG
    CONFIG = _load_config(config_path)
    return CONFIG


CONFIG = _load_config()


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def _validate_input(recordings: list, child_age_months: int) -> None:
    """Raise ValueError with a specific message on malformed input."""
    if not isinstance(recordings, list):
        raise ValueError("recordings must be a list")
    if len(recordings) == 0:
        raise ValueError("recordings must be non-empty")

    if not isinstance(child_age_months, int):
        raise ValueError(f"child_age_months must be int, got {type(child_age_months).__name__}")
    if child_age_months < 24 or child_age_months > 144:
        raise ValueError(f"child_age_months must be 24-144 (2-12y), got {child_age_months}")

    seen_paths = set()
    for i, rec in enumerate(recordings):
        if not isinstance(rec, dict):
            raise ValueError(f"recordings[{i}] must be a dict, got {type(rec).__name__}")
        if "audio_path" not in rec or not rec["audio_path"]:
            raise ValueError(f"recordings[{i}] missing or empty audio_path")
        if rec["audio_path"] in seen_paths:
            raise ValueError(f"recordings[{i}] duplicate audio_path: {rec['audio_path']}")
        seen_paths.add(rec["audio_path"])
        if "task_type" not in rec:
            raise ValueError(f"recordings[{i}] missing task_type")
        if rec["task_type"] not in VALID_TASK_TYPES:
            raise ValueError(
                f"recordings[{i}].task_type must be one of {sorted(VALID_TASK_TYPES)}, "
                f"got {rec['task_type']!r}"
            )


# =============================================================================
# AGE GROUP LOOKUP
# =============================================================================

def get_age_group(age_months: int, config: Optional[dict] = None) -> str:
    """Map age in months to age group name."""
    cfg = config or CONFIG
    for name, group in cfg["age_groups"].items():
        low, high = group["age_range_months"]
        if low <= age_months <= high:
            return name
    # Out of band — pick the closest group. Voice_check will also flag.
    if age_months < cfg["age_groups"][next(iter(cfg["age_groups"]))]["age_range_months"][0]:
        return next(iter(cfg["age_groups"]))
    return list(cfg["age_groups"].keys())[-1]


# =============================================================================
# PERCENTILE LOOKUP — direction-agnostic, breakpoints sorted by VALUE
# =============================================================================

def _percentile_lookup(value: float, norm: dict) -> int:
    """
    norm = {"p10", "p25", "p50", "p75", "p90"} where p10 is the *worst* 10% cutoff
    (the value below which 10% of typically-developing children score, for
    higher-is-better metrics; or the value above which 10% have worse readings,
    for lower-is-better metrics).

    Returns integer percentile in [10, 90]. Below the worst breakpoint or
    above the best, clamps to 10 or 90 respectively.
    """
    breakpoints = sorted(
        [(10, norm["p10"]), (25, norm["p25"]), (50, norm["p50"]),
         (75, norm["p75"]), (90, norm["p90"])],
        key=lambda kv: kv[1],
    )

    if value <= breakpoints[0][1]:
        return breakpoints[0][0]
    if value >= breakpoints[-1][1]:
        return breakpoints[-1][0]

    for i in range(len(breakpoints) - 1):
        p1, v1 = breakpoints[i]
        p2, v2 = breakpoints[i + 1]
        if v1 <= value <= v2:
            t = (value - v1) / (v2 - v1) if v2 > v1 else 0
            return int(round(p1 + t * (p2 - p1)))
    return 50


# =============================================================================
# PER-METRIC COMPUTATION (returns MetricResult)
# =============================================================================

def _normalise_word(w: str) -> str:
    return w.lower().strip(" .,!?;:\"'()[]")


def _compute_word_coverage(rec: dict, fuzzy_threshold: float) -> MetricResult:
    """Sentence repetition: % of expected words present in transcript (fuzzy)."""
    transcript = rec.get("asr_transcript_clean")
    expected = rec.get("expected_text")
    if not transcript:
        return MetricResult(reason="ASR transcript (clean) not provided")
    if not expected:
        return MetricResult(reason="Expected text not provided")

    expected_words = [w for w in (_normalise_word(t) for t in expected.split()) if w]
    if not expected_words:
        return MetricResult(reason="Expected text yields no tokens")

    transcript_words = [w for w in (_normalise_word(t) for t in transcript.split()) if w]

    matched = 0
    for ew in expected_words:
        for tw in transcript_words:
            if fuzz.ratio(ew, tw) >= fuzzy_threshold:
                matched += 1
                break

    return MetricResult(value=matched / len(expected_words) * 100, computed=True)


def _compute_naming_accuracy(recs: list, fuzzy_threshold: float) -> MetricResult:
    """% of picture-naming items where the expected word appears in the transcript."""
    if not recs:
        return MetricResult(reason="No naming recordings")
    correct = 0
    total = 0
    for rec in recs:
        transcript = rec.get("asr_transcript_clean")
        expected = rec.get("expected_text")
        if not transcript or not expected:
            continue
        total += 1
        ew = _normalise_word(expected)
        transcript_words = [_normalise_word(t) for t in transcript.split()]
        for tw in transcript_words:
            if fuzz.ratio(ew, tw) >= fuzzy_threshold:
                correct += 1
                break
    if total == 0:
        return MetricResult(reason="No naming recordings have both transcript and expected text")
    return MetricResult(value=correct / total * 100, computed=True)


def _compute_pcc_for(recs: list) -> MetricResult:
    """PCC from pronunciation_scores.overall_score, averaged across given recordings."""
    scores = []
    for rec in recs:
        ps = rec.get("pronunciation_scores")
        if not ps:
            continue
        overall = ps.get("overall_score")
        if overall is not None:
            scores.append(float(overall))
    if not scores:
        return MetricResult(reason="Pronunciation model output not provided")
    # Pronunciation overall_score is on [0,1]; convert to percentage.
    return MetricResult(value=float(np.mean(scores)) * 100, computed=True)


def _compute_speaking_rate(
    recordings_with_quality: list,
    syllables_per_word_default: float,
) -> dict:
    """
    Words-per-minute aggregated across usable recordings.

    Prefers ``asr_transcript_raw`` (Whisper — preserves disfluencies → honest
    word count). Falls back to ``asr_transcript_clean`` with a mode flag. If
    only acoustic syllables are available, divides by ``syllables_per_word_default``
    and flags mode as ``acoustic_estimate_unreliable`` (language-dependent
    conversion ratio not yet calibrated).
    """
    word_counts: list[int] = []
    durations: list[float] = []
    mode = "asr_words_raw"

    for rec, quality in recordings_with_quality:
        if not quality.usable:
            continue
        raw = rec.get("asr_transcript_raw")
        if raw is None or not str(raw).strip():
            raw = rec.get("asr_transcript_clean")
            if raw is not None and mode == "asr_words_raw":
                mode = "asr_words_clean_fallback"
        if raw is None or not str(raw).strip():
            continue
        words = [w for w in str(raw).split() if w]
        if words and quality.duration_s > 0:
            word_counts.append(len(words))
            durations.append(quality.duration_s)

    if not word_counts:
        return {
            "value": MetricResult(reason="No usable transcript for speaking_rate"),
            "mode": "unavailable",
        }

    total_words = sum(word_counts)
    total_duration_min = sum(durations) / 60.0
    if total_duration_min <= 0:
        return {"value": MetricResult(reason="Zero total duration"), "mode": "unavailable"}

    wpm = total_words / total_duration_min
    return {"value": MetricResult(value=float(wpm), computed=True), "mode": mode}


def _aggregate_audio_metrics(recordings_with_quality: list) -> dict:
    """Pitch_mean / pause_ratio / jitter / shimmer / hnr averaged over usable recs."""
    pitch_means: list[float] = []
    pause_ratios: list[float] = []
    jitters: list[float] = []
    shimmers: list[float] = []
    hnrs: list[float] = []

    for rec, quality in recordings_with_quality:
        if not quality.usable:
            continue
        path = rec["audio_path"]

        y, sr = _load_audio_safe(path)
        if y is not None:
            pitch = _extract_pitch_metrics(y, sr)
            if pitch["pitch_mean"].computed:
                pitch_means.append(pitch["pitch_mean"].value)

        pause = _extract_pause_distribution(path)
        if pause["pause_ratio"].computed:
            pause_ratios.append(pause["pause_ratio"].value)

        vs = _extract_voice_stability(path)
        if vs["jitter"].computed:
            jitters.append(vs["jitter"].value)
        if vs["shimmer"].computed:
            shimmers.append(vs["shimmer"].value)
        if vs["hnr"].computed:
            hnrs.append(vs["hnr"].value)

    def _avg_or_skip(vals: list, name: str) -> MetricResult:
        if not vals:
            return MetricResult(reason=f"No usable recordings yielded a {name} value")
        return MetricResult(value=float(np.mean(vals)), computed=True)

    return {
        "pitch_mean": _avg_or_skip(pitch_means, "pitch"),
        "pause_ratio": _avg_or_skip(pause_ratios, "pause_ratio"),
        "jitter": _avg_or_skip(jitters, "jitter"),
        "shimmer": _avg_or_skip(shimmers, "shimmer"),
        "hnr": _avg_or_skip(hnrs, "HNR"),
    }


# =============================================================================
# COMPOSITION — domain & overall
# =============================================================================

def _percentile_to_status(percentile: Optional[int], thresholds: dict) -> str:
    if percentile is None:
        return "not_computed"
    if percentile >= thresholds["on_track_min_percentile"]:
        return "on_track"
    if percentile >= thresholds["behind_min_percentile"]:
        return "behind"
    return "significantly_behind"


def _compose_domain(domain_name: str, percentiles: dict, thresholds: dict) -> dict:
    """Average computed percentiles within a domain."""
    metrics = DOMAINS[domain_name]
    computed = [percentiles[m] for m in metrics if percentiles.get(m) is not None]
    if not computed:
        return {
            "percentile": None,
            "status": "not_computed",
            "computed_count": 0,
            "total_count": len(metrics),
        }
    avg = int(round(sum(computed) / len(computed)))
    return {
        "percentile": avg,
        "status": _percentile_to_status(avg, thresholds),
        "computed_count": len(computed),
        "total_count": len(metrics),
    }


def _compose_overall(domain_results: dict, method: str = "lowest_domain") -> Optional[int]:
    """Composite percentile across the supplied domain_results."""
    computed = [d["percentile"] for d in domain_results.values() if d["percentile"] is not None]
    if not computed:
        return None
    if method == "lowest_domain":
        return min(computed)
    if method == "simple_average":
        return int(round(sum(computed) / len(computed)))
    logger.warning(f"Unknown composition_method {method!r}, falling back to simple_average")
    return int(round(sum(computed) / len(computed)))


# =============================================================================
# DEVELOPMENTAL BAND MAPPING
# =============================================================================

def _round_to_6mo_bucket(months: float) -> int:
    """
    Round delay to nearest 6-month bucket, floored at 0. Honest precision only.

    Uses half-up rounding (15 → 18, not 12 via Python's banker's rounding) — for
    delay reporting, slight over-flagging on the boundary is clinically safer
    than under-flagging.
    """
    if months <= 0:
        return 0
    return int((months + 3) // 6) * 6


def _band_midpoint_months(age_range: list) -> float:
    return (age_range[0] + age_range[1]) / 2.0


def _compute_developmental_composite_for_age_group(
    metric_values: dict,
    age_group_norms: dict,
) -> Optional[int]:
    """developmental_composite (articulation + language) under a given age group's norms."""
    domain_percentiles = []
    for domain_name in ("articulation", "language"):
        ps_for_domain = []
        for metric in DOMAINS[domain_name]:
            v = metric_values.get(metric)
            if v is None:
                continue
            norm_key = _NORM_KEY_BY_METRIC[metric]
            if norm_key not in age_group_norms:
                continue
            v_eff = v + age_group_norms.get("connected_pcc_offset", 0) if metric == "connected_pcc" else v
            ps_for_domain.append(_percentile_lookup(v_eff, age_group_norms[norm_key]))
        if ps_for_domain:
            domain_percentiles.append(sum(ps_for_domain) / len(ps_for_domain))
    if not domain_percentiles:
        return None
    return int(round(min(domain_percentiles)))


def compute_developmental_band(
    metric_values: dict,
    chronological_age_months: int,
    age_groups_config: dict,
    chronological_age_group: str,
) -> dict:
    """
    For each age group, recompute the developmental composite (articulation +
    language) under THAT group's norms. Walk from youngest to oldest; the
    developmental band is the YOUNGEST age group where composite >= 50.

    Returns: {band, delay_months, composites_per_band, resolution}
    """
    ordered = list(age_groups_config.keys())
    composites_per_band: dict[str, Optional[int]] = {}
    for ag in ordered:
        composites_per_band[ag] = _compute_developmental_composite_for_age_group(
            metric_values, age_groups_config[ag]["norms"]
        )

    # If NO age group could produce a developmental composite (typically because
    # articulation + language metrics are all uncomputable — e.g. ASR missing),
    # band/delay are not meaningful. Return insufficient_data; don't fabricate.
    if all(c is None for c in composites_per_band.values()):
        return {
            "band": "insufficient_data",
            "delay_months": None,
            "composites_per_band": composites_per_band,
            "resolution": "age_group",
        }

    chrono_composite = composites_per_band.get(chronological_age_group)

    # On-track at chronological age
    if chrono_composite is not None and chrono_composite >= 50:
        return {
            "band": chronological_age_group,
            "delay_months": 0,
            "composites_per_band": composites_per_band,
            "resolution": "age_group",
        }

    chrono_idx = ordered.index(chronological_age_group)
    for ag in ordered[:chrono_idx]:
        c = composites_per_band.get(ag)
        if c is not None and c >= 50:
            band_mid = _band_midpoint_months(age_groups_config[ag]["age_range_months"])
            return {
                "band": ag,
                "delay_months": _round_to_6mo_bucket(chronological_age_months - band_mid),
                "composites_per_band": composites_per_band,
                "resolution": "age_group",
            }

    # Below even the youngest age group (some composites computable, none ≥ 50)
    youngest = ordered[0]
    youngest_low = age_groups_config[youngest]["age_range_months"][0]
    return {
        "band": f"below_{youngest}",
        "delay_months": _round_to_6mo_bucket(chronological_age_months - youngest_low),
        "composites_per_band": composites_per_band,
        "resolution": "age_group",
    }


# =============================================================================
# CONFIDENCE + QUALITY REPORT
# =============================================================================

def _compute_confidence(
    recordings_with_quality: list,
    metric_percentiles: dict,
) -> dict:
    total = len(recordings_with_quality)
    usable = sum(1 for _, q in recordings_with_quality if q.usable)
    avg_snr = np.mean([q.snr_db for _, q in recordings_with_quality if q.usable and q.snr_db > 0]) if usable > 0 else 0.0
    computed_metric_count = sum(1 for v in metric_percentiles.values() if v is not None)
    total_metric_count = len(_NORM_KEY_BY_METRIC)

    recording_factor = usable / max(total, 1)
    metric_factor = computed_metric_count / max(total_metric_count, 1)
    snr_factor = min(1.0, max(0.0, (avg_snr - 5) / 20))

    confidence = 0.40 * recording_factor + 0.40 * metric_factor + 0.20 * snr_factor

    warnings = []
    if usable < total:
        warnings.append(f"Only {usable}/{total} recordings usable")
    if computed_metric_count < total_metric_count:
        warnings.append(f"Only {computed_metric_count}/{total_metric_count} metrics computed")
    if avg_snr < 10:
        warnings.append(f"Low average audio quality (SNR: {avg_snr:.1f} dB)")

    return {
        "confidence_score": round(float(confidence), 2),
        "usable_recordings": usable,
        "total_recordings": total,
        "computed_metrics": computed_metric_count,
        "total_metrics": total_metric_count,
        "avg_snr_db": round(float(avg_snr), 1),
        "warnings": warnings,
    }


def _build_quality_report(recordings_with_quality: list) -> dict:
    return {
        "all_recordings": [
            {
                "path": q.path,
                "task_type": rec.get("task_type"),
                "usable": q.usable,
                "flags": [f.value for f in q.flags],
                "snr_db": round(q.snr_db, 1),
                "duration_s": round(q.duration_s, 1),
                "speech_ratio": round(q.speech_ratio, 2),
                "rejection_reason": q.rejection_reason,
            }
            for rec, q in recordings_with_quality
        ],
        "rejected_count": sum(1 for _, q in recordings_with_quality if not q.usable),
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def assess_speech_delay(
    recordings: list,
    child_age_months: int,
    *,
    recording_quality: Optional[dict] = None,
    config_path: Optional[str] = None,
) -> dict:
    """
    Run the speech-delay screening on a session's recordings.

    Args:
        recordings: list of per-recording dicts. See module docstring.
        child_age_months: child's chronological age in months.
        recording_quality: optional map of audio_path -> RecordingQuality. When
            called from the backend, both ASD and speech_delay pipelines share
            a single quality pass — the dict is mandatory there. For standalone
            CLI / test use, omit and the pipeline computes quality internally.
        config_path: optional override for speech_delay_config.json.

    Returns: dict with delay_status, developmental_band, delay_months,
             composite_percentile, developmental_composite, per-metric values
             and percentiles, domain_detail, voice_check, confidence, etc.
    """
    _validate_input(recordings, child_age_months)
    config = _load_config(config_path) if config_path else CONFIG

    age_group = get_age_group(child_age_months, config)

    # ---- Phase 1: Quality (use provided dict or compute) ----
    if recording_quality is None:
        recording_quality = {
            rec["audio_path"]: _assess_recording_quality(rec["audio_path"])
            for rec in recordings
        }
    else:
        # Validate that every recording has a corresponding quality entry.
        missing = [r["audio_path"] for r in recordings if r["audio_path"] not in recording_quality]
        if missing:
            raise ValueError(f"recording_quality missing entries for: {missing}")

    recordings_with_quality = [(rec, recording_quality[rec["audio_path"]]) for rec in recordings]

    # ---- Phase 2: Per-metric computation ----
    naming_recs = [r for r, q in recordings_with_quality if r["task_type"] == "picture_naming" and q.usable]
    repetition_recs = [r for r, q in recordings_with_quality if r["task_type"] == "sentence_repetition" and q.usable]

    fuzzy_t = config.get("fuzzy_match_threshold", 75)
    sylpw = config.get("syllables_per_word_default", 1.85)

    single_word_pcc = _compute_pcc_for(naming_recs)
    connected_pcc = _compute_pcc_for(repetition_recs)
    naming_accuracy = _compute_naming_accuracy(naming_recs, fuzzy_t)

    # word_coverage averaged across sentence-repetition recordings
    coverage_vals = []
    for rec in repetition_recs:
        res = _compute_word_coverage(rec, fuzzy_t)
        if res.computed:
            coverage_vals.append(res.value)
    word_coverage = (
        MetricResult(value=float(np.mean(coverage_vals)), computed=True)
        if coverage_vals
        else MetricResult(reason="No sentence_repetition recordings yielded word_coverage")
    )

    sr_result = _compute_speaking_rate(recordings_with_quality, sylpw)
    speaking_rate = sr_result["value"]
    speaking_rate_mode = sr_result["mode"]

    audio = _aggregate_audio_metrics(recordings_with_quality)

    metrics_raw: dict[str, MetricResult] = {
        "single_word_pcc": single_word_pcc,
        "connected_pcc": connected_pcc,
        "word_coverage": word_coverage,
        "naming_accuracy": naming_accuracy,
        "speaking_rate": speaking_rate,
        "pause_ratio": audio["pause_ratio"],
        "pitch_mean": audio["pitch_mean"],
        "jitter": audio["jitter"],
        "shimmer": audio["shimmer"],
        "hnr": audio["hnr"],
    }

    # ---- Phase 3: Percentile lookup against chronological age group ----
    age_norms = config["age_groups"][age_group]["norms"]
    metric_percentiles: dict[str, Optional[int]] = {}
    for m, mr in metrics_raw.items():
        if not mr.computed:
            metric_percentiles[m] = None
            continue
        norm_key = _NORM_KEY_BY_METRIC[m]
        if norm_key not in age_norms:
            metric_percentiles[m] = None
            continue
        v_eff = mr.value + age_norms.get("connected_pcc_offset", 0) if m == "connected_pcc" else mr.value
        metric_percentiles[m] = _percentile_lookup(v_eff, age_norms[norm_key])

    # ---- Phase 4: Domain detail ----
    thresholds = config["scoring"]["delay_thresholds"]
    domain_detail = {
        d: _compose_domain(d, metric_percentiles, thresholds) for d in DOMAINS
    }

    # ---- Phase 5: Composite + developmental band ----
    method = config["scoring"].get("composition_method", "lowest_domain")
    composite_percentile = _compose_overall(domain_detail, method)
    developmental_composite = _compose_overall(
        {k: domain_detail[k] for k in ("articulation", "language")},
        method,
    )

    metric_values_for_band = {m: mr.value for m, mr in metrics_raw.items() if mr.computed}
    band = compute_developmental_band(
        metric_values_for_band,
        child_age_months,
        config["age_groups"],
        age_group,
    )

    # ---- Phase 6: Delay status from composite ----
    if composite_percentile is None:
        delay_status = DelayStatus.INSUFFICIENT_DATA.value
    else:
        delay_status = _percentile_to_status(composite_percentile, thresholds)

    # ---- Phase 7: Voice check (shared with ASD) ----
    pitch_mean_val = metrics_raw["pitch_mean"].value if metrics_raw["pitch_mean"].computed else None
    voice_check = _compute_voice_check_shared(pitch_mean_val, age_group)

    # ---- Phase 8: Confidence + quality report ----
    confidence = _compute_confidence(recordings_with_quality, metric_percentiles)
    quality_report = _build_quality_report(recordings_with_quality)

    # ---- Assemble output ----
    metrics_out: dict[str, dict] = {}
    for m, mr in metrics_raw.items():
        entry = {
            "value": float(mr.value) if mr.computed and mr.value is not None else None,
            "percentile": metric_percentiles.get(m),
            "computed": mr.computed,
        }
        if not mr.computed:
            entry["reason"] = mr.reason
        if m == "speaking_rate":
            entry["mode"] = speaking_rate_mode
        metrics_out[m] = entry

    logger.info(
        f"Speech-delay assessment complete: delay_status={delay_status}, "
        f"developmental_band={band['band']}, delay_months={band['delay_months']}, "
        f"composite={composite_percentile}, computed_metrics={confidence['computed_metrics']}/{confidence['total_metrics']}"
    )

    return {
        "delay_status": delay_status,
        "developmental_band": band["band"],
        "delay_months": band["delay_months"],
        "composite_percentile": composite_percentile,
        "developmental_composite": developmental_composite,
        "calibration_status": "provisional",
        "metrics": metrics_out,
        "domain_detail": domain_detail,
        "composites_per_band": band["composites_per_band"],
        "band_resolution": band["resolution"],
        "voice_check": voice_check,
        "confidence": confidence,
        "quality_report": quality_report,
        "age_group": age_group,
        "child_age_months": child_age_months,
    }


# =============================================================================
# CLI — print config summary
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s — %(levelname)s — %(message)s")
    print("Speech Delay Pipeline")
    print(f"  Composition method: {CONFIG['scoring'].get('composition_method')}")
    print(f"  Delay thresholds:   {CONFIG['scoring']['delay_thresholds']}")
    print(f"  Fuzzy match:        ≥{CONFIG.get('fuzzy_match_threshold')}%")
    print(f"  Syllables/word:     {CONFIG.get('syllables_per_word_default')} (default; language-aware later)")
    print(f"  Age groups:         {list(CONFIG['age_groups'].keys())}")
    print(f"  Metrics ({len(_NORM_KEY_BY_METRIC)}): {list(_NORM_KEY_BY_METRIC.keys())}")
    print(f"  Domains:")
    for d, ms in DOMAINS.items():
        print(f"    {d:13s} ← {ms}")
