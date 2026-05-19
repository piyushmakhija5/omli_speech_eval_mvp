"""
ASD Risk Detection Pipeline (Production)
=========================================
Extracts acoustic biomarkers from child speech recordings and produces
a tiered risk classification (no_indicators / monitor / recommend_evaluation).

All metrics are computed from raw audio. Zero ASR dependency.
Works identically for English, Hindi, and Hinglish.

Usage:
    from asd_pipeline import assess_asd_risk
    result = assess_asd_risk(
        prompted_question_audio_paths=["q1.wav", "q2.wav", "q3.wav", "q4.wav"],
        all_audio_paths=["q1.wav", ..., "q12.wav"],
        child_age_months=66,
    )
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from scipy.signal import welch
from scipy.stats import entropy
import webrtcvad
import struct

logger = logging.getLogger("asd_pipeline")


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class RiskTier(str, Enum):
    NO_INDICATORS = "no_indicators"
    MONITOR = "monitor"
    RECOMMEND_EVALUATION = "recommend_evaluation"
    INSUFFICIENT_DATA = "insufficient_data"


class QualityFlag(str, Enum):
    GOOD = "good"
    LOW_SNR = "low_snr"
    CLIPPED = "clipped"
    TOO_SHORT = "too_short"
    SILENCE_ONLY = "silence_only"
    CORRUPT = "corrupt"
    SAMPLE_RATE_MISMATCH = "sample_rate_mismatch"


@dataclass
class RecordingQuality:
    """Quality assessment for a single audio recording."""
    path: str
    duration_s: float = 0.0
    snr_db: float = 0.0
    clipping_ratio: float = 0.0
    speech_ratio: float = 0.0
    flags: list = field(default_factory=list)
    usable: bool = True
    rejection_reason: Optional[str] = None


@dataclass
class MetricResult:
    """Wrapper for a computed metric — distinguishes 'computed' from 'failed' from 'insufficient data'."""
    value: Optional[float] = None
    computed: bool = False
    reason: Optional[str] = None  # why it wasn't computed


# =============================================================================
# CONFIG — loaded from external JSON, not hardcoded
# =============================================================================

import json

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asd_config.json")


def _load_config(config_path: str = None) -> dict:
    """Load config from JSON. Falls back to defaults if file not found."""
    path = config_path or _CONFIG_PATH

    if not os.path.exists(path):
        logger.warning(f"Config file not found at {path}, using built-in defaults")
        return _default_config()

    try:
        with open(path, "r") as f:
            raw = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {path}: {e}. Using defaults.")
        return _default_config()

    # Flatten the JSON structure into the format the code expects
    age_groups = raw.get("age_groups", {})
    quality = raw.get("quality_thresholds", {})
    scoring = raw.get("scoring", {})

    config = {
        # Quality thresholds
        "min_snr_db": quality.get("min_snr_db", 5.0),
        "max_clipping_ratio": quality.get("max_clipping_ratio", 0.01),
        "min_speech_ratio": quality.get("min_speech_ratio", 0.10),
        "min_duration_s": quality.get("min_duration_s", 3.0),
        "target_sr": quality.get("target_sr", 16000),
        "min_total_spontaneous_s": quality.get("min_total_spontaneous_s", 10.0),
        "min_usable_prompted": quality.get("min_usable_prompted", 2),
        "min_usable_total": quality.get("min_usable_total", 6),
        "min_voiced_frames": quality.get("min_voiced_frames", 10),

        # Scoring
        "atypical_threshold_sd": scoring.get("atypical_threshold_sd", 2.0),
        "turn_taking_cv_threshold": scoring.get("turn_taking_cv_threshold", 0.5),
        "pause_variance_threshold": scoring.get("pause_variance_threshold", 50000),

        # Age norms (restructured from age_groups)
        "pitch_mean_norms": {},
        "pitch_variability_norms": {},
        "pitch_range_norms": {},
        "spectral_entropy_norms": {},
        "ltas_slope_norms": {},
        "latency_norms": {},
        "voice_stability_norms": {},

        # Keep raw age groups for age lookup
        "_age_groups": age_groups,
    }

    for group_name, group_data in age_groups.items():
        config["pitch_mean_norms"][group_name] = group_data.get("pitch_mean", {"mean": 275, "std": 28})
        config["pitch_variability_norms"][group_name] = group_data.get("pitch_variability", {"mean": 50, "std": 15})
        config["pitch_range_norms"][group_name] = group_data.get("pitch_range", {"mean": 150, "std": 45})
        config["spectral_entropy_norms"][group_name] = group_data.get("spectral_entropy", {"mean": 0.72, "std": 0.09})
        config["ltas_slope_norms"][group_name] = group_data.get("ltas_slope", {"mean": -0.018, "std": 0.005})
        config["latency_norms"][group_name] = group_data.get("latency", {"mean": 1800, "std": 600})
        config["voice_stability_norms"][group_name] = group_data.get("voice_stability", {
            "jitter_mean": 2.0, "jitter_std": 1.0,
            "shimmer_mean": 7.0, "shimmer_std": 3.0,
            "hnr_mean": 17.0, "hnr_std": 3.5,
        })

    logger.info(f"Config loaded from {path} — {len(age_groups)} age groups: {list(age_groups.keys())}")
    return config


def _default_config() -> dict:
    """Hardcoded fallback if JSON is missing."""
    return {
        "min_snr_db": 5.0, "max_clipping_ratio": 0.01, "min_speech_ratio": 0.10,
        "min_duration_s": 3.0, "target_sr": 16000, "min_total_spontaneous_s": 10.0,
        "min_usable_prompted": 2, "min_usable_total": 6, "min_voiced_frames": 10,
        "atypical_threshold_sd": 2.0, "turn_taking_cv_threshold": 0.5, "pause_variance_threshold": 50000,
        "pitch_mean_norms": {"3-4": {"mean": 295, "std": 28}, "5-6": {"mean": 275, "std": 28}, "7-8": {"mean": 245, "std": 28}},
        "pitch_variability_norms": {"3-4": {"mean": 65, "std": 18}, "5-6": {"mean": 50, "std": 15}, "7-8": {"mean": 42, "std": 12}},
        "pitch_range_norms": {"3-4": {"mean": 180, "std": 50}, "5-6": {"mean": 150, "std": 45}, "7-8": {"mean": 120, "std": 40}},
        "spectral_entropy_norms": {"3-4": {"mean": 0.75, "std": 0.10}, "5-6": {"mean": 0.72, "std": 0.09}, "7-8": {"mean": 0.70, "std": 0.08}},
        "ltas_slope_norms": {"3-4": {"mean": -0.015, "std": 0.005}, "5-6": {"mean": -0.018, "std": 0.005}, "7-8": {"mean": -0.020, "std": 0.004}},
        "latency_norms": {"3-4": {"mean": 2200, "std": 800}, "5-6": {"mean": 1800, "std": 600}, "7-8": {"mean": 1500, "std": 500}},
        "voice_stability_norms": {
            "3-4": {"jitter_mean": 2.5, "jitter_std": 1.2, "shimmer_mean": 8.0, "shimmer_std": 3.5, "hnr_mean": 15.0, "hnr_std": 4.0},
            "5-6": {"jitter_mean": 2.0, "jitter_std": 1.0, "shimmer_mean": 7.0, "shimmer_std": 3.0, "hnr_mean": 17.0, "hnr_std": 3.5},
            "7-8": {"jitter_mean": 1.8, "jitter_std": 0.8, "shimmer_mean": 6.0, "shimmer_std": 2.5, "hnr_mean": 18.0, "hnr_std": 3.0},
        },
        "_age_groups": {},
    }


def reload_config(config_path: str = None):
    """Reload config at runtime (e.g., after editing asd_config.json)."""
    global CONFIG
    CONFIG = _load_config(config_path)
    return CONFIG


CONFIG = _load_config()


# =============================================================================
# AUDIO QUALITY ASSESSMENT
# =============================================================================

def assess_recording_quality(audio_path: str) -> RecordingQuality:
    """
    Assess quality of a single audio recording.
    Returns a RecordingQuality object with flags and usability decision.
    """
    quality = RecordingQuality(path=audio_path)

    # Check file exists
    if not os.path.exists(audio_path):
        quality.usable = False
        quality.rejection_reason = f"File not found: {audio_path}"
        quality.flags.append(QualityFlag.CORRUPT)
        logger.error(quality.rejection_reason)
        return quality

    # Try loading
    try:
        y, sr = librosa.load(audio_path, sr=CONFIG["target_sr"])
    except Exception as e:
        quality.usable = False
        quality.rejection_reason = f"Failed to load audio: {e}"
        quality.flags.append(QualityFlag.CORRUPT)
        logger.error(quality.rejection_reason)
        return quality

    if len(y) == 0:
        quality.usable = False
        quality.rejection_reason = "Empty audio file"
        quality.flags.append(QualityFlag.CORRUPT)
        logger.warning(quality.rejection_reason)
        return quality

    quality.duration_s = len(y) / sr

    # Duration check
    if quality.duration_s < CONFIG["min_duration_s"]:
        quality.flags.append(QualityFlag.TOO_SHORT)
        quality.usable = False
        quality.rejection_reason = f"Too short: {quality.duration_s:.1f}s < {CONFIG['min_duration_s']}s"
        logger.warning(f"{audio_path}: {quality.rejection_reason}")
        return quality

    # SNR estimation (signal vs noise in non-speech regions)
    rms_signal = np.sqrt(np.mean(y ** 2))
    if rms_signal < 1e-10:
        quality.flags.append(QualityFlag.SILENCE_ONLY)
        quality.usable = False
        quality.rejection_reason = "No detectable signal (silence only)"
        logger.warning(f"{audio_path}: {quality.rejection_reason}")
        return quality

    # Estimate noise from the quietest 10% of frames
    frame_length = int(sr * 0.025)
    hop_length = int(sr * 0.010)
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=0))
    noise_floor = np.percentile(frame_rms, 10)
    snr = 20 * np.log10(rms_signal / (noise_floor + 1e-12))
    quality.snr_db = float(snr)

    if snr < CONFIG["min_snr_db"]:
        quality.flags.append(QualityFlag.LOW_SNR)
        logger.warning(f"{audio_path}: Low SNR ({snr:.1f} dB)")
        # Low SNR is a warning, not automatic rejection — metrics may still be usable

    # Clipping detection
    max_amplitude = np.max(np.abs(y))
    clipping_threshold = 0.99 * max_amplitude if max_amplitude > 0.5 else 0.99
    clipping_ratio = np.mean(np.abs(y) > clipping_threshold)
    quality.clipping_ratio = float(clipping_ratio)

    if clipping_ratio > CONFIG["max_clipping_ratio"]:
        quality.flags.append(QualityFlag.CLIPPED)
        quality.usable = False
        quality.rejection_reason = f"Audio clipped ({clipping_ratio * 100:.1f}% samples)"
        logger.warning(f"{audio_path}: {quality.rejection_reason}")
        return quality

    # Speech ratio via VAD
    try:
        speech_ratio = _compute_speech_ratio(y, sr)
        quality.speech_ratio = speech_ratio
        if speech_ratio < CONFIG["min_speech_ratio"]:
            quality.flags.append(QualityFlag.SILENCE_ONLY)
            quality.usable = False
            quality.rejection_reason = f"Almost no speech detected ({speech_ratio * 100:.1f}%)"
            logger.warning(f"{audio_path}: {quality.rejection_reason}")
            return quality
    except Exception as e:
        logger.warning(f"{audio_path}: VAD failed ({e}), proceeding without speech ratio check")

    if not quality.flags:
        quality.flags.append(QualityFlag.GOOD)

    return quality


def _compute_speech_ratio(y: np.ndarray, sr: int) -> float:
    """Compute ratio of speech frames to total frames using VAD."""
    audio_int16 = (y * 32768).astype(np.int16)
    vad = webrtcvad.Vad(2)
    frame_duration_ms = 30
    frame_size = int(sr * frame_duration_ms / 1000)

    speech_frames = 0
    total_frames = 0

    for i in range(len(audio_int16) // frame_size):
        start = i * frame_size
        end = start + frame_size
        if end > len(audio_int16):
            break
        frame_bytes = struct.pack(f"{frame_size}h", *audio_int16[start:end])
        try:
            if vad.is_speech(frame_bytes, sr):
                speech_frames += 1
            total_frames += 1
        except Exception:
            total_frames += 1

    return speech_frames / max(total_frames, 1)


# =============================================================================
# SAFE AUDIO LOADER
# =============================================================================

def _load_audio_safe(path: str) -> tuple:
    """Load audio with validation. Returns (y, sr) or (None, None) on failure."""
    try:
        y, sr = librosa.load(path, sr=CONFIG["target_sr"])
        if len(y) == 0:
            return None, None
        return y, sr
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return None, None


# =============================================================================
# BIOMARKER EXTRACTION (each returns MetricResult)
# =============================================================================

def extract_pitch_metrics(y: np.ndarray, sr: int) -> dict:
    """Extract pitch variability and range from audio."""
    try:
        f0, _, _ = librosa.pyin(y, fmin=75, fmax=600, sr=sr)
        f0_voiced = f0[~np.isnan(f0)]

        if len(f0_voiced) < CONFIG["min_voiced_frames"]:
            return {
                "pitch_variability": MetricResult(reason=f"Only {len(f0_voiced)} voiced frames, need {CONFIG['min_voiced_frames']}"),
                "pitch_range": MetricResult(reason="Insufficient voiced frames"),
                "pitch_mean": MetricResult(reason="Insufficient voiced frames"),
                "f0_voiced": np.array([]),
            }

        return {
            "pitch_variability": MetricResult(value=float(np.std(f0_voiced)), computed=True),
            "pitch_range": MetricResult(value=float(np.max(f0_voiced) - np.min(f0_voiced)), computed=True),
            "pitch_mean": MetricResult(value=float(np.mean(f0_voiced)), computed=True),
            "f0_voiced": f0_voiced,
        }
    except Exception as e:
        logger.error(f"Pitch extraction failed: {e}")
        return {
            "pitch_variability": MetricResult(reason=f"Extraction error: {e}"),
            "pitch_range": MetricResult(reason=f"Extraction error: {e}"),
            "pitch_mean": MetricResult(reason=f"Extraction error: {e}"),
            "f0_voiced": np.array([]),
        }


def extract_spectral_entropy(y: np.ndarray, sr: int) -> MetricResult:
    """Compute spectral entropy."""
    try:
        nperseg = min(1024, len(y))
        if nperseg < 64:
            return MetricResult(reason="Audio too short for spectral analysis")

        freqs, psd = welch(y, fs=sr, nperseg=nperseg)
        psd_norm = psd / (np.sum(psd) + 1e-12)
        max_entropy = np.log2(len(psd_norm))

        if max_entropy == 0:
            return MetricResult(reason="Degenerate spectrum")

        se = float(entropy(psd_norm, base=2) / max_entropy)
        return MetricResult(value=se, computed=True)

    except Exception as e:
        logger.error(f"Spectral entropy failed: {e}")
        return MetricResult(reason=f"Computation error: {e}")


def extract_ltas(y: np.ndarray, sr: int) -> MetricResult:
    """Compute Long-Term Average Spectrum slope."""
    try:
        S = np.abs(librosa.stft(y))
        if S.shape[1] < 2:
            return MetricResult(reason="Audio too short for STFT")

        avg_spectrum = np.mean(S, axis=1)
        if len(avg_spectrum) < 2:
            return MetricResult(reason="Spectrum too short for slope computation")

        log_spectrum = np.log10(avg_spectrum + 1e-12)
        coeffs = np.polyfit(np.arange(len(log_spectrum)), log_spectrum, 1)
        return MetricResult(value=float(coeffs[0]), computed=True)

    except Exception as e:
        logger.error(f"LTAS computation failed: {e}")
        return MetricResult(reason=f"Computation error: {e}")


def extract_response_latency(audio_path: str, prompt_end_ms: float = 0) -> MetricResult:
    """Measure time from prompt end to child's first voiced frame."""
    y, sr = _load_audio_safe(audio_path)
    if y is None:
        return MetricResult(reason="Could not load audio")

    try:
        audio_int16 = (y * 32768).astype(np.int16)
        vad = webrtcvad.Vad(2)
        frame_duration_ms = 30
        frame_size = int(sr * frame_duration_ms / 1000)
        prompt_end_frame = int(prompt_end_ms / frame_duration_ms)

        first_voiced_frame = None
        for i in range(prompt_end_frame, len(audio_int16) // frame_size):
            start = i * frame_size
            end = start + frame_size
            if end > len(audio_int16):
                break
            frame_bytes = struct.pack(f"{frame_size}h", *audio_int16[start:end])
            try:
                if vad.is_speech(frame_bytes, sr):
                    first_voiced_frame = i
                    break
            except Exception:
                continue

        if first_voiced_frame is None:
            return MetricResult(reason="No speech detected in recording")

        latency_ms = float((first_voiced_frame - prompt_end_frame) * frame_duration_ms)
        return MetricResult(value=latency_ms, computed=True)

    except Exception as e:
        logger.error(f"Latency extraction failed for {audio_path}: {e}")
        return MetricResult(reason=f"Extraction error: {e}")


def extract_voice_stability(audio_path: str) -> dict:
    """Extract jitter, shimmer, HNR using Parselmouth/Praat."""
    try:
        snd = parselmouth.Sound(audio_path)
        duration = snd.get_total_duration()

        if duration < 2.5:
            reason = f"Audio too short for Praat analysis ({duration:.1f}s, need 2.5s)"
            return {
                "jitter": MetricResult(reason=reason),
                "shimmer": MetricResult(reason=reason),
                "hnr": MetricResult(reason=reason),
            }

        pitch = call(snd, "To Pitch", 0.0, 75, 600)
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)

        jitter_val = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer_val = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        hnr_val = call(harmonicity, "Get mean", 0, 0)

        return {
            "jitter": MetricResult(value=float(jitter_val * 100), computed=True) if jitter_val else MetricResult(reason="Praat returned None for jitter"),
            "shimmer": MetricResult(value=float(shimmer_val * 100), computed=True) if shimmer_val else MetricResult(reason="Praat returned None for shimmer"),
            "hnr": MetricResult(value=float(hnr_val), computed=True) if hnr_val else MetricResult(reason="Praat returned None for HNR"),
        }

    except Exception as e:
        logger.error(f"Voice stability extraction failed for {audio_path}: {e}")
        reason = f"Praat error: {e}"
        return {
            "jitter": MetricResult(reason=reason),
            "shimmer": MetricResult(reason=reason),
            "hnr": MetricResult(reason=reason),
        }


def extract_pause_distribution(audio_path: str) -> dict:
    """Analyze pause patterns — ratio and distribution variance."""
    y, sr = _load_audio_safe(audio_path)
    if y is None:
        reason = "Could not load audio"
        return {"pause_ratio": MetricResult(reason=reason), "pause_variance": MetricResult(reason=reason)}

    duration = len(y) / sr
    if duration < 1.0:
        reason = f"Audio too short ({duration:.1f}s, need 1.0s)"
        return {"pause_ratio": MetricResult(reason=reason), "pause_variance": MetricResult(reason=reason)}

    try:
        audio_int16 = (y * 32768).astype(np.int16)
        vad = webrtcvad.Vad(2)
        frame_duration_ms = 30
        frame_size = int(sr * frame_duration_ms / 1000)

        speech_frames = 0
        silent_frames = 0
        current_pause_length = 0
        pause_lengths = []

        for i in range(len(audio_int16) // frame_size):
            start = i * frame_size
            end = start + frame_size
            if end > len(audio_int16):
                break
            frame_bytes = struct.pack(f"{frame_size}h", *audio_int16[start:end])
            try:
                is_speech = vad.is_speech(frame_bytes, sr)
            except Exception:
                continue

            if is_speech:
                speech_frames += 1
                if current_pause_length > 0:
                    pause_lengths.append(current_pause_length * frame_duration_ms)
                    current_pause_length = 0
            else:
                silent_frames += 1
                current_pause_length += 1

        total_frames = speech_frames + silent_frames
        if total_frames == 0:
            return {
                "pause_ratio": MetricResult(reason="No frames processed"),
                "pause_variance": MetricResult(reason="No frames processed"),
            }

        return {
            "pause_ratio": MetricResult(value=float(silent_frames / total_frames), computed=True),
            "pause_variance": MetricResult(
                value=float(np.var(pause_lengths)) if len(pause_lengths) > 1 else 0.0,
                computed=True,
            ),
        }

    except Exception as e:
        logger.error(f"Pause distribution failed for {audio_path}: {e}")
        reason = f"VAD error: {e}"
        return {"pause_ratio": MetricResult(reason=reason), "pause_variance": MetricResult(reason=reason)}


# =============================================================================
# HELPERS
# =============================================================================

def get_age_group(age_months: int) -> str:
    """Map age in months to age group using config ranges."""
    age_groups = CONFIG.get("_age_groups", {})

    # Try config-defined ranges first
    for group_name, group_data in age_groups.items():
        age_range = group_data.get("age_range_months", [])
        if len(age_range) == 2 and age_range[0] <= age_months <= age_range[1]:
            return group_name

    # Fallback to hardcoded ranges if config doesn't define them
    if age_months < 60:
        return "3-4"
    elif age_months < 84:
        return "5-6"
    else:
        return "7-8"


def _get_value(metric_result) -> Optional[float]:
    """Safely extract value from MetricResult or raw float."""
    if isinstance(metric_result, MetricResult):
        return metric_result.value
    return metric_result


def is_atypical(value: Optional[float], mean: float, std: float, direction: str = "both") -> bool:
    """Check if value is >threshold SDs from norm. Returns False if value is None."""
    if value is None:
        return False
    threshold = CONFIG["atypical_threshold_sd"]
    deviation = (value - mean) / (std + 1e-12)

    if direction == "high":
        return deviation > threshold
    elif direction == "low":
        return deviation < -threshold
    else:
        return abs(deviation) > threshold


# =============================================================================
# GROUP-LEVEL CONVERGENCE
# =============================================================================

def evaluate_biomarker_groups(biomarkers: dict, age_group: str) -> dict:
    """Evaluate each of 4 biomarker groups. Returns group-level flags + details."""
    cfg = CONFIG
    results = {}

    # Group 1: Prosody
    pv_norms = cfg["pitch_variability_norms"][age_group]
    pr_norms = cfg["pitch_range_norms"][age_group]
    pv = biomarkers.get("pitch_variability")
    pr = biomarkers.get("pitch_range")
    pv_atyp = is_atypical(pv, pv_norms["mean"], pv_norms["std"], "high")
    pr_atyp = is_atypical(pr, pr_norms["mean"], pr_norms["std"], "high")
    pv_computed = pv is not None
    pr_computed = pr is not None

    results["prosody"] = {
        "atypical": pv_atyp or pr_atyp,
        "computable": pv_computed or pr_computed,
        "details": {
            "pitch_variability": pv, "pitch_variability_atypical": pv_atyp, "pitch_variability_computed": pv_computed,
            "pitch_range": pr, "pitch_range_atypical": pr_atyp, "pitch_range_computed": pr_computed,
        },
    }

    # Group 2: Spectral
    se_norms = cfg["spectral_entropy_norms"][age_group]
    lt_norms = cfg["ltas_slope_norms"][age_group]
    se = biomarkers.get("spectral_entropy")
    lt = biomarkers.get("ltas_slope")
    se_atyp = is_atypical(se, se_norms["mean"], se_norms["std"])
    lt_atyp = is_atypical(lt, lt_norms["mean"], lt_norms["std"], "high")
    se_computed = se is not None
    lt_computed = lt is not None

    results["spectral"] = {
        "atypical": se_atyp or lt_atyp,
        "computable": se_computed or lt_computed,
        "details": {
            "spectral_entropy": se, "spectral_entropy_atypical": se_atyp, "spectral_entropy_computed": se_computed,
            "ltas_slope": lt, "ltas_slope_atypical": lt_atyp, "ltas_slope_computed": lt_computed,
        },
    }

    # Group 3: Interaction
    lat_norms = cfg["latency_norms"][age_group]
    lat_mean = biomarkers.get("latency_mean")
    lat_std = biomarkers.get("latency_std")
    lat_atyp = is_atypical(lat_mean, lat_norms["mean"], lat_norms["std"], "high")

    tt_atyp = False
    if lat_std is not None and lat_mean is not None and lat_mean > 0:
        tt_atyp = (lat_std / lat_mean) > cfg["turn_taking_cv_threshold"]

    lat_computed = lat_mean is not None

    results["interaction"] = {
        "atypical": lat_atyp or tt_atyp,
        "computable": lat_computed,
        "details": {
            "latency_mean_ms": lat_mean, "latency_atypical": lat_atyp, "latency_computed": lat_computed,
            "latency_std_ms": lat_std, "turn_taking_atypical": tt_atyp,
        },
    }

    # Group 4: Voice stability
    vs_norms = cfg["voice_stability_norms"][age_group]
    ji = biomarkers.get("jitter")
    sh = biomarkers.get("shimmer")
    hn = biomarkers.get("hnr")
    pvar = biomarkers.get("pause_variance", 0)

    ji_atyp = is_atypical(ji, vs_norms["jitter_mean"], vs_norms["jitter_std"], "high")
    sh_atyp = is_atypical(sh, vs_norms["shimmer_mean"], vs_norms["shimmer_std"], "high")
    hn_atyp = is_atypical(hn, vs_norms["hnr_mean"], vs_norms["hnr_std"], "low")
    p_atyp = (pvar or 0) > cfg["pause_variance_threshold"]
    any_computed = any(x is not None for x in [ji, sh, hn])

    results["voice_stability"] = {
        "atypical": ji_atyp or sh_atyp or hn_atyp or p_atyp,
        "computable": any_computed,
        "details": {
            "jitter_pct": ji, "jitter_atypical": ji_atyp,
            "shimmer_pct": sh, "shimmer_atypical": sh_atyp,
            "hnr_db": hn, "hnr_atypical": hn_atyp,
            "pause_variance": pvar, "pause_atypical": p_atyp,
        },
    }

    return results


# =============================================================================
# TIERED RISK OUTPUT
# =============================================================================

def compute_risk_tier(group_results: dict) -> dict:
    """
    Count atypical groups → tiered classification.
    Only counts groups that were computable.
    """
    computable_groups = {k: v for k, v in group_results.items() if v.get("computable", False)}
    atypical_count = sum(1 for g in computable_groups.values() if g["atypical"])
    atypical_groups = [name for name, g in computable_groups.items() if g["atypical"]]
    total_computable = len(computable_groups)
    non_computable = [name for name, g in group_results.items() if not g.get("computable", False)]

    if total_computable < 3:
        tier = RiskTier.INSUFFICIENT_DATA
        message = (
            f"Only {total_computable} of 4 biomarker groups could be computed. "
            f"Need at least 3 for reliable screening. "
            f"Groups with insufficient data: {', '.join(non_computable)}."
        )
    elif atypical_count >= 3:
        tier = RiskTier.RECOMMEND_EVALUATION
        message = "Consistent atypical patterns detected across multiple domains. Recommend clinical evaluation by a specialist."
    elif atypical_count == 2:
        tier = RiskTier.MONITOR
        message = "Some atypical speech patterns detected. Recommend re-assessment in 2 weeks."
    else:
        tier = RiskTier.NO_INDICATORS
        message = "No atypical patterns detected."

    return {
        "tier": tier.value,
        "message": message,
        "atypical_group_count": atypical_count,
        "atypical_groups": atypical_groups,
        "computable_group_count": total_computable,
        "non_computable_groups": non_computable,
        "group_details": group_results,
    }


# =============================================================================
# VOICE CHECK (soft warning if speaker doesn't look child-like)
# =============================================================================

def compute_voice_check(pitch_mean: Optional[float], age_group: str) -> dict:
    """
    Soft sanity check: is the speaker plausibly a child in the target age band?
    Flags if mean F0 is more than `atypical_threshold_sd` SDs below the age-group
    norm. Adult voices (~120 Hz male, ~210 Hz female) fall far outside the 3-8yo
    range (~250-300 Hz), so this catches the common "wrong speaker" case.
    """
    if pitch_mean is None:
        return {
            "pitch_mean_hz": None,
            "likely_child": None,
            "reason": "pitch_mean not computed",
        }

    norms = CONFIG["pitch_mean_norms"].get(age_group)
    if norms is None:
        return {
            "pitch_mean_hz": round(pitch_mean, 1),
            "likely_child": None,
            "reason": f"No pitch_mean norm for age group {age_group}",
        }

    threshold = CONFIG["atypical_threshold_sd"]
    expected_low = norms["mean"] - threshold * norms["std"]
    expected_high = norms["mean"] + threshold * norms["std"]
    likely_child = pitch_mean >= expected_low

    reason = None
    if not likely_child:
        reason = (
            f"Mean F0 ({pitch_mean:.0f} Hz) is far below expected range for age "
            f"{age_group} ({expected_low:.0f}-{expected_high:.0f} Hz). "
            f"Voice may not be a child's — interpret tier with caution."
        )

    return {
        "pitch_mean_hz": round(pitch_mean, 1),
        "expected_range_for_age_hz": [round(expected_low, 1), round(expected_high, 1)],
        "likely_child": likely_child,
        "reason": reason,
    }


# =============================================================================
# CONFIDENCE SCORE
# =============================================================================

def compute_confidence(
    recording_quality: list,
    prompted_quality: list,
    group_results: dict,
) -> dict:
    """
    Compute a confidence score for the overall ASD assessment.
    Based on: recording quality, number of usable samples, computable groups.
    """
    total_recordings = len(recording_quality)
    usable_recordings = sum(1 for q in recording_quality if q.usable)
    usable_prompted = sum(1 for q in prompted_quality if q.usable)
    computable_groups = sum(1 for g in group_results.values() if g.get("computable", False))
    avg_snr = np.mean([q.snr_db for q in recording_quality if q.usable and q.snr_db > 0]) if usable_recordings > 0 else 0

    # Confidence factors (0-1 each)
    recording_factor = usable_recordings / max(total_recordings, 1)
    prompted_factor = usable_prompted / max(len(prompted_quality), 1)
    group_factor = computable_groups / 4
    snr_factor = min(1.0, max(0.0, (avg_snr - 5) / 20))  # 5dB=0, 25dB=1

    confidence = 0.25 * recording_factor + 0.30 * prompted_factor + 0.25 * group_factor + 0.20 * snr_factor

    warnings_list = []
    if usable_recordings < CONFIG["min_usable_total"]:
        warnings_list.append(f"Only {usable_recordings}/{total_recordings} recordings usable")
    if usable_prompted < CONFIG["min_usable_prompted"]:
        warnings_list.append(f"Only {usable_prompted}/{len(prompted_quality)} prompted question recordings usable")
    if avg_snr < 10:
        warnings_list.append(f"Low average audio quality (SNR: {avg_snr:.1f} dB)")
    if computable_groups < 4:
        warnings_list.append(f"Only {computable_groups}/4 biomarker groups computed")

    return {
        "confidence_score": round(float(confidence), 2),
        "usable_recordings": usable_recordings,
        "total_recordings": total_recordings,
        "usable_prompted": usable_prompted,
        "avg_snr_db": round(avg_snr, 1),
        "computable_groups": computable_groups,
        "warnings": warnings_list,
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def assess_asd_risk(
    prompted_question_audio_paths: list,
    all_audio_paths: list,
    child_age_months: int,
    prompt_end_ms_per_question: list = None,
) -> dict:
    """
    Full ASD risk assessment from a set of audio recordings.

    Args:
        prompted_question_audio_paths: list of audio paths from prompted questions (ideally 4)
        all_audio_paths: list of all audio paths (ideally 12)
        child_age_months: child's age in months
        prompt_end_ms_per_question: prompt end timestamps per prompted question (None = push-to-talk)

    Returns:
        dict with tier, message, biomarkers, group details, confidence, quality report
    """
    age_group = get_age_group(child_age_months)
    if prompt_end_ms_per_question is None:
        prompt_end_ms_per_question = [0] * len(prompted_question_audio_paths)

    # ---- Phase 1: Quality assessment for ALL recordings ----
    logger.info(f"Assessing {len(all_audio_paths)} recordings for age group {age_group}")

    all_quality = [assess_recording_quality(p) for p in all_audio_paths]
    prompted_quality = [assess_recording_quality(p) for p in prompted_question_audio_paths]

    usable_all = [(p, q) for p, q in zip(all_audio_paths, all_quality) if q.usable]
    usable_prompted = [(p, q, ms) for (p, q), ms in
                       zip(zip(prompted_question_audio_paths, prompted_quality), prompt_end_ms_per_question)
                       if q.usable]

    rejected = [q for q in all_quality if not q.usable]
    if rejected:
        logger.warning(f"{len(rejected)} recordings rejected: {[q.rejection_reason for q in rejected]}")

    # ---- Phase 2: Check minimum data ----
    total_spontaneous = sum(q.duration_s for _, q in zip(prompted_question_audio_paths, prompted_quality) if q.usable)

    if total_spontaneous < CONFIG["min_total_spontaneous_s"]:
        return {
            "tier": RiskTier.INSUFFICIENT_DATA.value,
            "message": (
                f"Only {total_spontaneous:.1f}s of usable spontaneous speech. "
                f"Need {CONFIG['min_total_spontaneous_s']}s minimum. "
                f"{len(rejected)} recordings were rejected due to quality issues."
            ),
            "atypical_group_count": 0,
            "atypical_groups": [],
            "computable_group_count": 0,
            "non_computable_groups": ["prosody", "spectral", "interaction", "voice_stability"],
            "biomarkers": {},
            "group_details": {},
            "voice_check": compute_voice_check(None, age_group),
            "confidence": compute_confidence(all_quality, prompted_quality, {}),
            "quality_report": {
                "all_recordings": [{"path": q.path, "usable": q.usable, "flags": [f.value for f in q.flags], "snr_db": round(q.snr_db, 1), "duration_s": round(q.duration_s, 1), "rejection_reason": q.rejection_reason} for q in all_quality],
                "rejected_count": len(rejected),
                "total_spontaneous_speech_s": round(total_spontaneous, 1),
            },
            "age_group": age_group,
            "child_age_months": child_age_months,
        }

    # ---- Phase 3: Extract biomarkers from USABLE prompted questions ----
    all_f0_values = []
    all_spectral_entropies = []
    all_ltas_slopes = []
    latencies = []

    for path, quality, prompt_ms in usable_prompted:
        y, sr = _load_audio_safe(path)
        if y is None:
            latencies.append(None)
            continue

        # Pitch
        pitch_results = extract_pitch_metrics(y, sr)
        f0_arr = pitch_results.get("f0_voiced", np.array([]))
        if len(f0_arr) > 0:
            all_f0_values.extend(f0_arr.tolist())

        # Spectral entropy
        se = extract_spectral_entropy(y, sr)
        if se.computed:
            all_spectral_entropies.append(se.value)

        # LTAS
        ltas = extract_ltas(y, sr)
        if ltas.computed:
            all_ltas_slopes.append(ltas.value)

        # Response latency
        lat = extract_response_latency(path, prompt_end_ms=prompt_ms)
        latencies.append(lat.value)

    # ---- Phase 4: Extract voice stability from ALL usable recordings ----
    all_jitter = []
    all_shimmer = []
    all_hnr = []
    all_pause_variance = []

    for path, quality in usable_all:
        vs = extract_voice_stability(path)
        if vs["jitter"].computed:
            all_jitter.append(vs["jitter"].value)
        if vs["shimmer"].computed:
            all_shimmer.append(vs["shimmer"].value)
        if vs["hnr"].computed:
            all_hnr.append(vs["hnr"].value)

        pd = extract_pause_distribution(path)
        if pd["pause_variance"].computed:
            all_pause_variance.append(pd["pause_variance"].value)

    # ---- Phase 5: Aggregate biomarkers ----
    valid_latencies = [l for l in latencies if l is not None]
    lat_mean = float(np.mean(valid_latencies)) if len(valid_latencies) >= 2 else None
    lat_std = float(np.std(valid_latencies)) if len(valid_latencies) >= 2 else None

    biomarkers = {
        "pitch_variability": float(np.std(all_f0_values)) if len(all_f0_values) >= CONFIG["min_voiced_frames"] else None,
        "pitch_range": float(np.max(all_f0_values) - np.min(all_f0_values)) if len(all_f0_values) >= CONFIG["min_voiced_frames"] else None,
        "pitch_mean": float(np.mean(all_f0_values)) if len(all_f0_values) >= CONFIG["min_voiced_frames"] else None,
        "spectral_entropy": float(np.mean(all_spectral_entropies)) if all_spectral_entropies else None,
        "ltas_slope": float(np.mean(all_ltas_slopes)) if all_ltas_slopes else None,
        "latency_mean": lat_mean,
        "latency_std": lat_std,
        "jitter": float(np.mean(all_jitter)) if all_jitter else None,
        "shimmer": float(np.mean(all_shimmer)) if all_shimmer else None,
        "hnr": float(np.mean(all_hnr)) if all_hnr else None,
        "pause_variance": float(np.mean(all_pause_variance)) if all_pause_variance else None,
    }

    # ---- Phase 6: Evaluate groups + compute tier ----
    group_results = evaluate_biomarker_groups(biomarkers, age_group)
    risk = compute_risk_tier(group_results)

    # ---- Phase 7: Confidence + voice check ----
    confidence = compute_confidence(all_quality, prompted_quality, group_results)
    voice_check = compute_voice_check(biomarkers.get("pitch_mean"), age_group)

    # ---- Assemble output ----
    risk["biomarkers"] = biomarkers
    risk["voice_check"] = voice_check
    risk["confidence"] = confidence
    risk["quality_report"] = {
        "all_recordings": [
            {
                "path": q.path,
                "usable": q.usable,
                "flags": [f.value for f in q.flags],
                "snr_db": round(q.snr_db, 1),
                "duration_s": round(q.duration_s, 1),
                "speech_ratio": round(q.speech_ratio, 2),
                "rejection_reason": q.rejection_reason,
            }
            for q in all_quality
        ],
        "rejected_count": len(rejected),
        "total_spontaneous_speech_s": round(total_spontaneous, 1),
    }
    risk["age_group"] = age_group
    risk["child_age_months"] = child_age_months

    logger.info(
        f"ASD assessment complete: tier={risk['tier']}, confidence={confidence['confidence_score']}, "
        f"likely_child={voice_check.get('likely_child')}"
    )
    return risk


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s — %(levelname)s — %(message)s")
    print("ASD Pipeline (Production)")
    print(f"  Quality thresholds: SNR>{CONFIG['min_snr_db']}dB, clipping<{CONFIG['max_clipping_ratio']*100}%, speech>{CONFIG['min_speech_ratio']*100}%")
    print(f"  Atypical threshold: {CONFIG['atypical_threshold_sd']} SD")
    print(f"  Min spontaneous speech: {CONFIG['min_total_spontaneous_s']}s")
    print(f"  Groups: prosody, spectral, interaction, voice_stability")
    print(f"  Tiers: no_indicators (0-1), monitor (2), recommend_evaluation (3+)")
