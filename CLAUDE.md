# CLAUDE.md — Omli ASD Pipeline

## What this is

ASD (Autism Spectrum Disorder) risk detection pipeline for children aged 3-8. Extracts acoustic biomarkers from speech recordings and produces a tiered risk classification. Part of a larger Omli speech assessment system that also includes speech delay detection (not built yet).

## How it works

Child completes a 12-question voice assessment via the Omli app. 4 questions are "prompted questions" (open-ended, child speaks freely), 8 are structured tasks (sentence repetition, picture naming). The ASD pipeline:

1. Quality-checks each audio recording (SNR, clipping, duration, speech presence)
2. Extracts acoustic biomarkers from raw audio (no ASR/transcription needed)
3. Compares each biomarker to age-adjusted norms
4. Groups biomarkers into 4 groups (prosody, spectral, interaction, voice stability)
5. Counts how many groups are atypical → produces tiered risk output

All metrics are language-independent. Works identically for English, Hindi, and Hinglish.

## Files

```
core/asd_pipeline.py        # Main pipeline. Single entry point: assess_asd_risk()
core/asd_config.json        # All thresholds, age norms, quality gates. Edit this, not the code.
tests/test_asd_pipeline.py  # Test script. Run with: python -m tests.test_asd_pipeline
backend/                    # FastAPI collector server (records 12 voice samples → data/cases/)
frontend/                   # Single-page collector UI served by backend/
data/questions.json         # Question text shown by the collector
data/cases/<case_id>/       # Collected sessions, q01.wav … q12.wav (gitignored)
data/synthetic/             # Synthetic test audio (gitignored)
README.md                   # Setup and usage docs
```

ASD-specific files keep the `asd_` prefix. A future speech-delay pipeline will live
alongside as `core/speech_delay_pipeline.py` + `core/speech_delay_config.json` and
share the same `data/cases/` recordings.

## Architecture decisions

- **Config separated from code.** All norms and thresholds live in `asd_config.json`. Adding a new age group or changing thresholds requires zero code changes. The pipeline has `reload_config()` for runtime swaps.
- **Every metric returns a `MetricResult` object**, not a raw float. This distinguishes "computed and the value is 0" from "couldn't compute because audio was too short." Never return None silently.
- **Per-recording quality gate** runs before any metric extraction. Recordings are flagged (low_snr, clipped, too_short, silence_only, corrupt) and marked usable/not-usable. Pipeline continues with usable recordings only.
- **Graceful degradation.** If 3 of 4 prompted questions fail quality checks, pipeline still runs on the 1 usable recording but reports low confidence. If fewer than 3 biomarker groups are computable, output is `insufficient_data` rather than a potentially wrong tier.
- **Group-level convergence, not individual markers.** 4 groups: prosody, spectral, interaction, voice_stability. Multiple atypical markers within the same group count as 1. Prevents correlated markers (e.g., pitch variability + pitch range) from double-counting. 3+ atypical groups → recommend_evaluation.

## Key types

```python
RiskTier: "no_indicators" | "monitor" | "recommend_evaluation" | "insufficient_data"
QualityFlag: "good" | "low_snr" | "clipped" | "too_short" | "silence_only" | "corrupt"
RecordingQuality: per-recording quality assessment (path, duration, snr, flags, usable, rejection_reason)
MetricResult: wrapper for computed metrics (value, computed: bool, reason: str if failed)
```

## Biomarker groups

| Group | Markers | Primary source |
|-------|---------|---------------|
| Prosody | Pitch variability (F0 std), pitch range (F0 max-min) | Prompted questions |
| Spectral | Spectral entropy, LTAS slope | Prompted questions |
| Interaction | Response latency, turn-taking timing pattern | Prompted questions |
| Voice stability | Jitter, shimmer, HNR, pause variance | All 12 recordings |

## Dependencies

```
librosa          # audio loading, pitch (pyin), STFT, spectral features
praat-parselmouth # jitter, shimmer, HNR
webrtcvad        # voice activity detection, pause analysis
scipy            # spectral entropy (welch + Shannon entropy)
numpy            # everything
soundfile        # test audio generation only
```

## Running tests

```bash
# Synthetic audio (no real recordings needed)
python -m tests.test_asd_pipeline

# Real recordings
python -m tests.test_asd_pipeline --audio-dir ./data/cases/<case_id> --age 60

# age is in months (60 = 5 years)
```

## Collecting samples

```bash
uvicorn backend.server:app --reload   # http://localhost:8000
```

Each session writes `data/cases/<case_id>/q01.wav` … `q12.wav` — feed that folder
straight to `tests.test_asd_pipeline --audio-dir`.

## Target age band

This pipeline is calibrated for **children aged 3-8** (36-107 months). The only age
groups defined are `3-4`, `5-6`, `7-8`. We do not plan to support 0-2, 9+, or adult
voices in this codebase — those are different screening problems with different
markers.

Out-of-band voices (e.g. a parent recording instead of the child) are caught by
the `voice_check` field in the result: if mean F0 is more than
`atypical_threshold_sd` SDs below the age-group `pitch_mean` norm, `likely_child`
is set to `false` with a human-readable reason. The tier is NOT overridden — both
values are returned so the caller can decide. See `compute_voice_check()` in
`core/asd_pipeline.py`.

## Language independence

All metrics are acoustic (pitch, spectral shape, jitter/shimmer/HNR, VAD-based
timing) and do not depend on phonology, so the pipeline is designed to work
identically across English, Hindi, and Hinglish. **No language detection or
language-specific code anywhere — recordings can mix languages within a single
session.** Case `20260518-221449-15da` is a mixed Hindi/Hinglish/English session
that ran through the full pipeline cleanly; informal evidence that the
language-independent claim holds. Formal cross-language validation is still TODO.

## What's provisional / needs calibration

ALL age-adjusted norms in `core/asd_config.json` are educated guesses from
published research on Western, English-speaking children. They have NOT been
validated on:
- Hindi-speaking children
- Indian English-speaking children
- Children in noisy home environments (mobile recordings)

The code will produce valid numbers. Whether those numbers, compared against
these thresholds, produce correct ASD classifications is what SLP validation
will determine. Expect the numbers in the config to change significantly after
calibration.

## What's NOT in this pipeline

- **Speech delay detection** — separate pipeline, not built yet. Will share the same audio recordings.
- **Parent behavioral questions** — 3-5 M-CHAT-R style questions. Deferred for v1.
- **Two-session confirmation** — app-level logic, not in the scoring pipeline.
- **ASR/transcription** — deliberately excluded. All metrics are raw audio. This is by design.

## Coding conventions in this codebase

- Type hints on all function signatures
- Every extraction function handles its own errors and returns a MetricResult with a reason on failure — never raises to caller
- Logging via Python `logging` module, logger name is `asd_pipeline`
- No silent failures. If something can't be computed, the output says why.
- Dataclasses for structured data (RecordingQuality, MetricResult)
- Enums for categorical values (RiskTier, QualityFlag)

## Common tasks

**Change a threshold:** Edit `core/asd_config.json`. No code change needed.

**Add an age group:** Add entry to `age_groups` in `core/asd_config.json` with `age_range_months` and all norm values.

**Add a new biomarker:**
1. Write extraction function in `core/asd_pipeline.py` returning `MetricResult`
2. Add norm values to all age groups in `core/asd_config.json`
3. Add to the appropriate group in `evaluate_biomarker_groups()`
4. The convergence and tiering logic handles it automatically

**Integrate with Omli backend:**
```python
from core.asd_pipeline import assess_asd_risk

# After child completes 12-question session:
result = assess_asd_risk(
    prompted_question_audio_paths=[path1, path2, path3, path4],
    all_audio_paths=[path1, ..., path12],
    child_age_months=child.age_months,
)
# result["tier"] → store in database
# result["confidence"] → show on dashboard
# result["quality_report"] → flag bad recordings for re-collection
```
