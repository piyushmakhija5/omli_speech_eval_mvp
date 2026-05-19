"""
Shared audio analysis primitives.
=================================
Used by both ``core.asd_pipeline`` and ``core.speech_delay_pipeline``.

Contains the per-recording quality gate, low-level audio loading, and the
acoustic extractors whose outputs are consumed by more than one pipeline
(pitch, voice stability, pause distribution). Also provides the
voice-check helper — both pipelines compare measured pitch against the
same child-pitch reference (``CHILD_PITCH_NORMS``) so they cannot drift.

Every function takes its tunable parameters as explicit kwargs with
sensible defaults — no hidden dependency on a global CONFIG. Pipelines
that load configs from JSON pass the relevant values through.
"""

from __future__ import annotations

import logging
import os
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import librosa
import numpy as np
import parselmouth
import webrtcvad
from parselmouth.praat import call

logger = logging.getLogger("audio_analysis")


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

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
    reason: Optional[str] = None


# =============================================================================
# CHILD PITCH NORMS — single source of truth for voice_check across pipelines
# =============================================================================
# Mean F0 (Hz) for typically-developing children. Used by `compute_voice_check`
# to flag adult voices against the expected child range for the selected age.
# Both ASD and speech-delay pipelines share these — keeping them as a module
# constant prevents config drift between the two pipelines.

CHILD_PITCH_NORMS: dict[str, dict[str, float]] = {
    "3-4": {"mean": 295.0, "std": 28.0},
    "5-6": {"mean": 275.0, "std": 28.0},
    "7-8": {"mean": 245.0, "std": 28.0},
}


# =============================================================================
# AUDIO LOADING
# =============================================================================

DEFAULT_TARGET_SR = 16000


def _load_audio_safe(path: str, target_sr: int = DEFAULT_TARGET_SR) -> tuple:
    """Load audio with validation. Returns (y, sr) or (None, None) on failure."""
    try:
        y, sr = librosa.load(path, sr=target_sr)
        if len(y) == 0:
            return None, None
        return y, sr
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return None, None


# =============================================================================
# AUDIO QUALITY ASSESSMENT
# =============================================================================

def assess_recording_quality(
    audio_path: str,
    *,
    target_sr: int = DEFAULT_TARGET_SR,
    min_snr_db: float = 5.0,
    max_clipping_ratio: float = 0.01,
    min_speech_ratio: float = 0.10,
    min_duration_s: float = 3.0,
) -> RecordingQuality:
    """
    Per-recording quality gate. Returns a RecordingQuality with flags and a
    usable/not-usable decision. Hard rejections set ``usable=False``; soft
    flags (e.g. low SNR) are recorded but the recording is still usable.
    """
    quality = RecordingQuality(path=audio_path)

    if not os.path.exists(audio_path):
        quality.usable = False
        quality.rejection_reason = f"File not found: {audio_path}"
        quality.flags.append(QualityFlag.CORRUPT)
        logger.error(quality.rejection_reason)
        return quality

    try:
        y, sr = librosa.load(audio_path, sr=target_sr)
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

    if quality.duration_s < min_duration_s:
        quality.flags.append(QualityFlag.TOO_SHORT)
        quality.usable = False
        quality.rejection_reason = f"Too short: {quality.duration_s:.1f}s < {min_duration_s}s"
        logger.warning(f"{audio_path}: {quality.rejection_reason}")
        return quality

    # SNR — quietest 10% of frames as noise floor.
    rms_signal = np.sqrt(np.mean(y ** 2))
    if rms_signal < 1e-10:
        quality.flags.append(QualityFlag.SILENCE_ONLY)
        quality.usable = False
        quality.rejection_reason = "No detectable signal (silence only)"
        logger.warning(f"{audio_path}: {quality.rejection_reason}")
        return quality

    frame_length = int(sr * 0.025)
    hop_length = int(sr * 0.010)
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=0))
    noise_floor = np.percentile(frame_rms, 10)
    snr = 20 * np.log10(rms_signal / (noise_floor + 1e-12))
    quality.snr_db = float(snr)

    if snr < min_snr_db:
        quality.flags.append(QualityFlag.LOW_SNR)
        logger.warning(f"{audio_path}: Low SNR ({snr:.1f} dB)")
        # Soft flag — metrics may still be usable.

    # Clipping.
    max_amplitude = np.max(np.abs(y))
    clipping_threshold = 0.99 * max_amplitude if max_amplitude > 0.5 else 0.99
    clipping_ratio = np.mean(np.abs(y) > clipping_threshold)
    quality.clipping_ratio = float(clipping_ratio)

    if clipping_ratio > max_clipping_ratio:
        quality.flags.append(QualityFlag.CLIPPED)
        quality.usable = False
        quality.rejection_reason = f"Audio clipped ({clipping_ratio * 100:.1f}% samples)"
        logger.warning(f"{audio_path}: {quality.rejection_reason}")
        return quality

    # Speech ratio via VAD.
    try:
        speech_ratio = _compute_speech_ratio(y, sr)
        quality.speech_ratio = speech_ratio
        if speech_ratio < min_speech_ratio:
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
    """Speech-frame ratio via webrtcvad."""
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
# PITCH (F0) — std/range used by ASD prosody, mean used by speech_delay fluency
# =============================================================================

def extract_pitch_metrics(
    y: np.ndarray,
    sr: int,
    *,
    min_voiced_frames: int = 10,
    fmin: float = 75.0,
    fmax: float = 600.0,
) -> dict:
    """Extract pitch variability, range, and mean from audio (via librosa.pyin)."""
    try:
        f0, _, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr)
        f0_voiced = f0[~np.isnan(f0)]

        if len(f0_voiced) < min_voiced_frames:
            reason = f"Only {len(f0_voiced)} voiced frames, need {min_voiced_frames}"
            return {
                "pitch_variability": MetricResult(reason=reason),
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
        reason = f"Extraction error: {e}"
        return {
            "pitch_variability": MetricResult(reason=reason),
            "pitch_range": MetricResult(reason=reason),
            "pitch_mean": MetricResult(reason=reason),
            "f0_voiced": np.array([]),
        }


# =============================================================================
# VOICE STABILITY — jitter, shimmer, HNR via Praat/Parselmouth
# =============================================================================

def extract_voice_stability(audio_path: str, *, min_duration_s: float = 2.5) -> dict:
    """Jitter (local %), shimmer (local %), HNR (dB) via Praat."""
    try:
        snd = parselmouth.Sound(audio_path)
        duration = snd.get_total_duration()

        if duration < min_duration_s:
            reason = f"Audio too short for Praat analysis ({duration:.1f}s, need {min_duration_s}s)"
            return {
                "jitter": MetricResult(reason=reason),
                "shimmer": MetricResult(reason=reason),
                "hnr": MetricResult(reason=reason),
            }

        call(snd, "To Pitch", 0.0, 75, 600)  # warm-up; consumed via point process
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


# =============================================================================
# PAUSE DISTRIBUTION — ratio (speech_delay) + variance (ASD voice_stability)
# =============================================================================

def extract_pause_distribution(
    audio_path: str,
    *,
    target_sr: int = DEFAULT_TARGET_SR,
    min_duration_s: float = 1.0,
) -> dict:
    """Pause ratio + pause-duration variance via webrtcvad."""
    y, sr = _load_audio_safe(audio_path, target_sr=target_sr)
    if y is None:
        reason = "Could not load audio"
        return {"pause_ratio": MetricResult(reason=reason), "pause_variance": MetricResult(reason=reason)}

    duration = len(y) / sr
    if duration < min_duration_s:
        reason = f"Audio too short ({duration:.1f}s, need {min_duration_s}s)"
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
# VOICE CHECK — soft warning if speaker pitch is far from child norms
# =============================================================================

def compute_voice_check(
    pitch_mean: Optional[float],
    age_group: str,
    *,
    atypical_threshold_sd: float = 2.0,
) -> dict:
    """
    Flags 'adult voice identified' when measured pitch is more than
    ``atypical_threshold_sd`` SDs below the child norm for ``age_group``.
    Uses the shared CHILD_PITCH_NORMS — both pipelines call this directly.
    """
    if pitch_mean is None:
        return {
            "pitch_mean_hz": None,
            "likely_child": None,
            "reason": "pitch_mean not computed",
        }

    norms = CHILD_PITCH_NORMS.get(age_group)
    if norms is None:
        return {
            "pitch_mean_hz": round(pitch_mean, 1),
            "likely_child": None,
            "reason": f"No pitch_mean norm for age group {age_group}",
        }

    expected_low = norms["mean"] - atypical_threshold_sd * norms["std"]
    expected_high = norms["mean"] + atypical_threshold_sd * norms["std"]
    likely_child = pitch_mean >= expected_low

    reason = None
    if not likely_child:
        reason = (
            f"Adult voice identified: mean F0 ({pitch_mean:.0f} Hz) is far below "
            f"expected range for age {age_group} ({expected_low:.0f}-{expected_high:.0f} Hz). "
            f"The screening targets children 3–8 — interpret tier with caution."
        )

    return {
        "pitch_mean_hz": round(pitch_mean, 1),
        "expected_range_for_age_hz": [round(expected_low, 1), round(expected_high, 1)],
        "likely_child": likely_child,
        "reason": reason,
    }
