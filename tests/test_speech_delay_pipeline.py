"""
Scenario tests for the speech-delay pipeline.
==============================================
Pure-input unit tests for composition logic live in ``test_composition``.
This module tests the pipeline end-to-end against a small synthetic
audio set with mocked ASR / pronunciation inputs — verifying graceful
degradation when inputs are missing, age-group routing, validation,
and quality-gate behaviour.

Run with:
    python -m unittest tests.test_speech_delay_pipeline

Synthetic audio is generated once at setUpClass (12 sine-wave WAVs in
data/synthetic/) and quality is assessed once across the suite. Tests
assert on structural correctness (computed=True / False, fields
present, reason text) rather than specific metric values — those depend
on the synthetic waveform and are covered by the composition tests with
controlled inputs.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from core.audio_analysis import (
    assess_recording_quality,
    QualityFlag,
    RecordingQuality,
)
from core.speech_delay_pipeline import assess_speech_delay


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTHETIC_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic")


# A 12-recording session split that matches the production layout:
# q1-q4 prompted, q5-q8 sentence_repetition, q9-q11 picture_naming, q12 counting.
SESSION_LAYOUT = [
    ("prompted_question", None),
    ("prompted_question", None),
    ("prompted_question", None),
    ("prompted_question", None),
    ("sentence_repetition", "the cat is sleeping on the mat"),
    ("sentence_repetition", "my birthday is in the summer"),
    ("sentence_repetition", "we went to the park yesterday"),
    ("sentence_repetition", "i have a red ball and a blue car"),
    ("picture_naming", "apple apple apple"),
    ("picture_naming", "elephant elephant elephant"),
    ("picture_naming", "butterfly butterfly butterfly"),
    ("prompted_question", "one two three four five six seven eight nine ten"),
]


def _ensure_synthetic_audio() -> list[str]:
    """Generate 12 sine-wave WAV files if missing. Returns sorted paths."""
    import soundfile as sf
    os.makedirs(SYNTHETIC_DIR, exist_ok=True)
    paths = []
    sr = 16000
    rng = np.random.default_rng(seed=1234)  # deterministic across runs
    for i in range(1, 13):
        path = os.path.join(SYNTHETIC_DIR, f"q{i}.wav")
        if not os.path.exists(path):
            duration = float(rng.uniform(3.5, 6.0))
            t = np.linspace(0, duration, int(sr * duration), endpoint=False)
            y = (0.3 * np.sin(2 * np.pi * 300 * t)
                 + 0.15 * np.sin(2 * np.pi * 600 * t)
                 + 0.08 * np.sin(2 * np.pi * 900 * t)
                 + 0.05 * rng.standard_normal(len(t)))
            silence = np.zeros(int(sr * float(rng.uniform(0.3, 1.0))))
            y = np.concatenate([silence, y]).astype(np.float32)
            y = y / (np.max(np.abs(y)) + 0.1) * 0.7
            sf.write(path, y, sr)
        paths.append(path)
    return paths


def _build_recordings(
    audio_paths: list[str],
    *,
    with_asr: bool = True,
    with_pronunciation: bool = True,
) -> list[dict]:
    """Build the speech_delay input shape from a list of paths + layout."""
    out = []
    for path, (task_type, expected) in zip(audio_paths, SESSION_LAYOUT):
        rec = {
            "audio_path": path,
            "task_type": task_type,
            "expected_text": expected,
            "asr_transcript_clean": None,
            "asr_transcript_raw": None,
            "pronunciation_scores": None,
        }
        if with_asr:
            # When expected_text is set, transcript exactly matches → 100% coverage.
            if expected is not None:
                rec["asr_transcript_clean"] = expected
                rec["asr_transcript_raw"] = expected + " um"  # raw includes a disfluency
            else:
                rec["asr_transcript_clean"] = "i like playing with my toys at home"
                rec["asr_transcript_raw"] = "i uh like playing with my toys at home"
        if with_pronunciation:
            rec["pronunciation_scores"] = {
                "overall_score": 0.88,
                "phonemes": [{"phone": "p", "score": 0.9}],
            }
        out.append(rec)
    return out


def _force_unusable_quality(paths: list[str]) -> dict[str, RecordingQuality]:
    """Build a quality dict marking everything as too-short and unusable."""
    return {
        p: RecordingQuality(
            path=p,
            duration_s=0.5,
            snr_db=5.0,
            speech_ratio=0.0,
            flags=[QualityFlag.TOO_SHORT],
            usable=False,
            rejection_reason="Too short (synthetic test override)",
        )
        for p in paths
    }


class SpeechDelayPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audio_paths = _ensure_synthetic_audio()
        # Compute quality once — slow part of the suite.
        cls.quality_dict = {p: assess_recording_quality(p) for p in cls.audio_paths}

    # -----------------------------------------------------------------
    # Scenario: full inputs (ASR + pronunciation) → maximum metric coverage
    # -----------------------------------------------------------------

    def test_full_inputs_all_metrics_attempted(self):
        recordings = _build_recordings(self.audio_paths, with_asr=True, with_pronunciation=True)
        result = assess_speech_delay(recordings, 60, recording_quality=self.quality_dict)
        # PCC, word_coverage, naming_accuracy, speaking_rate should all compute.
        for key in ("single_word_pcc", "connected_pcc", "word_coverage", "naming_accuracy", "speaking_rate"):
            with self.subTest(metric=key):
                self.assertTrue(result["metrics"][key]["computed"], result["metrics"][key].get("reason"))
        # Articulation + Language domains have at least one computed metric.
        self.assertGreater(result["domain_detail"]["articulation"]["computed_count"], 0)
        self.assertGreater(result["domain_detail"]["language"]["computed_count"], 0)
        # Developmental composite is computable when articulation + language have data.
        self.assertIsNotNone(result["developmental_composite"])
        # Band should NOT be insufficient_data when articulation + language are present.
        self.assertNotEqual(result["developmental_band"], "insufficient_data")

    def test_full_inputs_speaking_rate_uses_raw_transcript(self):
        recordings = _build_recordings(self.audio_paths, with_asr=True, with_pronunciation=True)
        result = assess_speech_delay(recordings, 60, recording_quality=self.quality_dict)
        sr = result["metrics"]["speaking_rate"]
        self.assertTrue(sr["computed"])
        self.assertEqual(sr.get("mode"), "asr_words_raw")

    # -----------------------------------------------------------------
    # Scenario: ASR missing → ASR-dependent metrics skipped with reason
    # -----------------------------------------------------------------

    def test_asr_missing_skips_word_metrics_with_reason(self):
        recordings = _build_recordings(self.audio_paths, with_asr=False, with_pronunciation=True)
        result = assess_speech_delay(recordings, 60, recording_quality=self.quality_dict)
        # word_coverage, naming_accuracy, speaking_rate need transcripts → not_computed
        for key in ("word_coverage", "naming_accuracy", "speaking_rate"):
            with self.subTest(metric=key):
                self.assertFalse(result["metrics"][key]["computed"])
                self.assertIn(("transcript" in (result["metrics"][key].get("reason") or "")
                               or "naming" in (result["metrics"][key].get("reason") or "")
                               or "sentence_repetition" in (result["metrics"][key].get("reason") or "")),
                              [True])
        # PCC metrics still work via pronunciation_scores.
        self.assertTrue(result["metrics"]["single_word_pcc"]["computed"])
        self.assertTrue(result["metrics"]["connected_pcc"]["computed"])

    # -----------------------------------------------------------------
    # Scenario: pronunciation missing → PCC skipped, rest still works
    # -----------------------------------------------------------------

    def test_pronunciation_missing_skips_pcc_with_reason(self):
        recordings = _build_recordings(self.audio_paths, with_asr=True, with_pronunciation=False)
        result = assess_speech_delay(recordings, 60, recording_quality=self.quality_dict)
        for key in ("single_word_pcc", "connected_pcc"):
            with self.subTest(metric=key):
                self.assertFalse(result["metrics"][key]["computed"])
                self.assertIn("Pronunciation", result["metrics"][key]["reason"])
        # Other metrics still compute.
        self.assertTrue(result["metrics"]["word_coverage"]["computed"])
        self.assertTrue(result["metrics"]["naming_accuracy"]["computed"])

    # -----------------------------------------------------------------
    # Scenario: neither ASR nor pronunciation → only audio metrics
    # -----------------------------------------------------------------

    def test_all_text_inputs_missing_only_audio_metrics_compute(self):
        recordings = _build_recordings(self.audio_paths, with_asr=False, with_pronunciation=False)
        result = assess_speech_delay(recordings, 60, recording_quality=self.quality_dict)

        # ASR/pronunciation-dependent → not computed
        not_computed_expected = {
            "single_word_pcc", "connected_pcc",
            "word_coverage", "naming_accuracy", "speaking_rate",
        }
        for key in not_computed_expected:
            with self.subTest(metric=key):
                self.assertFalse(result["metrics"][key]["computed"])

        # 5 audio metrics should still compute (synthetic sine waves yield values).
        audio_metrics = {"pause_ratio", "pitch_mean", "jitter", "shimmer", "hnr"}
        for key in audio_metrics:
            with self.subTest(metric=key):
                self.assertTrue(result["metrics"][key]["computed"],
                                f"{key}: {result['metrics'][key].get('reason')}")

        # Articulation + Language domains are wholly empty.
        self.assertEqual(result["domain_detail"]["articulation"]["status"], "not_computed")
        self.assertEqual(result["domain_detail"]["language"]["status"], "not_computed")
        # Developmental band correctly reports insufficient_data.
        self.assertEqual(result["developmental_band"], "insufficient_data")
        self.assertIsNone(result["delay_months"])

    # -----------------------------------------------------------------
    # Scenario: bad audio quality → audio metrics skip, ASR paths still work
    # -----------------------------------------------------------------

    def test_bad_audio_quality_skips_audio_metrics(self):
        recordings = _build_recordings(self.audio_paths, with_asr=True, with_pronunciation=True)
        bad_quality = _force_unusable_quality(self.audio_paths)
        result = assess_speech_delay(recordings, 60, recording_quality=bad_quality)

        # With everything unusable, all per-recording paths are filtered out, so
        # nothing can compute — even ASR metrics, because their pools are empty.
        for key in result["metrics"]:
            with self.subTest(metric=key):
                self.assertFalse(result["metrics"][key]["computed"])
        # Quality report reflects all rejections.
        self.assertEqual(result["quality_report"]["rejected_count"], 12)
        # delay_status should fall back to insufficient_data when composite is None.
        self.assertEqual(result["delay_status"], "insufficient_data")

    # -----------------------------------------------------------------
    # Scenario: age group routing → correct norms picked per age
    # -----------------------------------------------------------------

    def test_age_group_routing(self):
        recordings = _build_recordings(self.audio_paths, with_asr=False, with_pronunciation=False)
        for age, expected_group in [(42, "3-4"), (72, "5-6"), (96, "7-8")]:
            with self.subTest(age=age):
                result = assess_speech_delay(recordings, age, recording_quality=self.quality_dict)
                self.assertEqual(result["age_group"], expected_group)

    # -----------------------------------------------------------------
    # Scenario: voice_check reuses ASD's shared check
    # -----------------------------------------------------------------

    def test_voice_check_shared_with_asd(self):
        recordings = _build_recordings(self.audio_paths, with_asr=False, with_pronunciation=False)
        result = assess_speech_delay(recordings, 60, recording_quality=self.quality_dict)
        # Synthetic 300 Hz sine sits inside the 5-6yo child norm — should look like a child.
        self.assertIn(result["voice_check"]["likely_child"], (True, None))

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def test_validation_empty_recordings(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            assess_speech_delay([], 60)

    def test_validation_bad_task_type(self):
        bad = [{"audio_path": "x.wav", "task_type": "free_form"}]
        with self.assertRaisesRegex(ValueError, "task_type"):
            assess_speech_delay(bad, 60)

    def test_validation_missing_audio_path(self):
        bad = [{"task_type": "prompted_question"}]
        with self.assertRaisesRegex(ValueError, "audio_path"):
            assess_speech_delay(bad, 60)

    def test_validation_out_of_range_age(self):
        good = [{"audio_path": "x.wav", "task_type": "prompted_question"}]
        with self.assertRaisesRegex(ValueError, "24-144"):
            assess_speech_delay(good, 200)

    def test_validation_quality_dict_missing_path(self):
        """If recording_quality is passed but missing a path, raise explicit error."""
        recordings = _build_recordings(self.audio_paths, with_asr=False, with_pronunciation=False)
        # Drop one path from the quality dict — pipeline should raise, not silently skip.
        partial_quality = {p: self.quality_dict[p] for p in self.audio_paths[:11]}
        with self.assertRaisesRegex(ValueError, "missing entries"):
            assess_speech_delay(recordings, 60, recording_quality=partial_quality)


if __name__ == "__main__":
    unittest.main()
