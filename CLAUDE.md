# CLAUDE.md — Omli Speech Screening

## What this is

Two speech-screening pipelines for children aged 3-8, sharing one 12-recording
voice session:

1. **ASD pipeline** — acoustic biomarkers across 4 groups (prosody, spectral,
   interaction, voice stability). Tiered output. No ASR dependency.
2. **Speech-delay pipeline** — 10 metrics across 3 domains (articulation,
   language, fluency). Percentile-based output with developmental band + delay
   status. ASR + pronunciation scoring are optional inputs (production wires
   Sarvam + Whisper-base in a deferred Phase L).

Both pipelines reuse shared audio primitives from `core/audio_analysis.py`:
quality gate, audio loading, pitch extraction, voice stability, pause
distribution, voice_check. Quality assessment runs ONCE per recording in the
backend and the result dict is passed to both pipelines.

## How it works

Child completes a 12-question voice assessment via the Omli app. 4 prompted
(open-ended) + 4 sentence repetition + 3 picture naming + 1 counting. Each
recording is paired with a `task_type` and `expected_text` derived from
`data/questions.json`.

Per recording: quality-check (SNR, clipping, duration, speech presence).
Quality results are shared between both pipelines (no double-processing).

**ASD pipeline:** extracts acoustic biomarkers → compares to age-adjusted
norms (mean ± SD) → groups by domain → counts atypical groups → tier output
(`no_indicators` / `monitor` / `recommend_evaluation` / `insufficient_data`).

**Speech-delay pipeline:** computes 10 metrics (5 from acoustic, 4 from
ASR/pronunciation inputs, 1 dual-mode). Each metric → percentile via age-norm
table lookup. Domain averages → composite (lowest_domain by default). Status
mapped via percentile thresholds (`on_track` ≥ 25, `behind` ≥ 10, else
`significantly_behind`). Developmental band mapping uses articulation +
language only.

All metrics designed to be language-independent. Voice_check (shared between
pipelines) flags adult voices via `CHILD_PITCH_NORMS` in `audio_analysis.py`.

## Files

```
core/audio_analysis.py            # Shared primitives — quality gate, audio loaders,
                                  # extract_pitch_metrics, extract_voice_stability,
                                  # extract_pause_distribution, compute_voice_check,
                                  # CHILD_PITCH_NORMS constant (single source of truth)

core/asd_pipeline.py              # ASD-specific extractors (spectral entropy, LTAS,
                                  # response latency), group convergence, tier mapping.
                                  # Entry: assess_asd_risk(prompted, all, age, recording_quality=None)
core/asd_config.json              # ASD norms (mean/std per metric per age group)
core/asd_consumer_view.py         # ASD raw → consumer summary translator

core/speech_delay_pipeline.py     # 10 metrics, percentile lookup, lowest-domain composition,
                                  # compute_developmental_band, voice_check via shared helper.
                                  # Entry: assess_speech_delay(recordings, age, recording_quality=None)
core/speech_delay_config.json     # Percentile tables (p10/p25/p50/p75/p90) + connected_pcc_offset
                                  # per age group, delay_thresholds, fuzzy_match_threshold
core/speech_delay_consumer_view.py # Speech-delay raw → consumer summary translator

backend/server.py                 # FastAPI. /assess runs BOTH pipelines from a single
                                  # quality pass, saves two result files, returns combined
                                  # {asd, speech_delay} response
backend/storage.py                # File paths, case_id validation, QUESTION_TYPE_MAP

frontend/index.html               # Two-column result viewer (≥900px); stacked below
frontend/recorder.js              # Mic capture, both summary renderers, percentile-bar SVG
frontend/styles.css               # Tier-coloured tokens, charts, responsive layout

data/questions.json               # 12 question prompts + expected_text per non-prompted item
data/cases/<case_id>/             # Collected sessions + asd_result.json + speech_delay_result.json
data/synthetic/                   # Sine-wave WAVs for tests (gitignored)

tests/test_asd_pipeline.py        # ASD CLI runner (synthetic or real audio)
tests/test_composition.py         # 36 unit tests — pure functions: percentile lookup, compose,
                                  # band mapping, validation
tests/test_speech_delay_pipeline.py # 13 scenario tests with synthetic audio + mocked ASR
```

## Architecture decisions

- **Config separated from code.** Each pipeline owns a JSON config — `core/asd_config.json` and `core/speech_delay_config.json`. Both reload at runtime via `reload_config()`.
- **Quality assessed ONCE per case.** Backend calls `assess_recording_quality` per WAV and passes the dict to both pipelines via the `recording_quality` kwarg. Pipelines fall back to internal computation when called standalone (tests + CLI).
- **`CHILD_PITCH_NORMS` is a module constant**, not in config. Both pipelines call `compute_voice_check` from `audio_analysis.py` — voice_check cannot drift between them.
- **Pipelines are STT-consumers, not STT-producers.** Speech_delay receives `asr_transcript_clean` (Sarvam) and `asr_transcript_raw` (Whisper-base) as per-recording inputs. Backend wires the ASR calls in deferred Phase L; production code shares Sarvam + Whisper-base infrastructure.
- **Every metric returns a `MetricResult` object.** Distinguishes "computed and value is 0" from "couldn't compute because input was missing". Never returns None silently.
- **Graceful degradation everywhere.** Missing ASR → 4 speech-delay metrics report `computed=false` with explicit reason; pipeline still produces a fluency-domain composite. If articulation + language are both uncomputable, `developmental_band="insufficient_data"` rather than fabricating a delay from fluency alone.
- **Lowest-domain composition** for speech delay. A severe articulation deficit can't be masked by good fluency — composite picks the worst domain.
- **Developmental band uses articulation + language only.** Fluency varies for reasons unrelated to language acquisition (microphone, mood) and would dilute band signal.
- **Delay months rounded to 6-month buckets** (half-up — 15→18). Honest precision given norms are provisional.
- **Side-by-side UI** at ≥900px (clinical-report layout); stacks on phones.

## Key types

```python
# In core/audio_analysis.py (used by both pipelines):
RecordingQuality   # path, duration, snr, flags, usable, rejection_reason
MetricResult       # value, computed: bool, reason: str if not computed
QualityFlag        # good | low_snr | clipped | too_short | silence_only | corrupt
CHILD_PITCH_NORMS  # {"3-4": {mean, std}, ...} — voice_check reference

# In core/asd_pipeline.py:
RiskTier           # no_indicators | monitor | recommend_evaluation | insufficient_data

# In core/speech_delay_pipeline.py:
DelayStatus        # on_track | behind | significantly_behind | insufficient_data
DOMAINS            # {"articulation": [...], "language": [...], "fluency": [...]}
```

## Speech-delay metrics (10 total)

| Domain | Metric | Source / needs |
|--------|--------|----------------|
| Articulation | `single_word_pcc` | `pronunciation_scores.overall_score` on picture-naming recordings |
| Articulation | `connected_pcc` | Same on sentence-repetition. Looked up against `single_word_pcc` norm table + `connected_pcc_offset` per age group |
| Language | `word_coverage` | `asr_transcript_clean` + `expected_text` on repetition; fuzzy match via rapidfuzz ≥75 |
| Language | `naming_accuracy` | Same on naming |
| Fluency | `speaking_rate` | `asr_transcript_raw` (Whisper preserves disfluencies); falls back to `_clean` with `mode="asr_words_clean_fallback"` and an unreliability note |
| Fluency | `pause_ratio` | webrtcvad on each usable recording, averaged |
| Fluency | `pitch_mean` | librosa pyin, averaged |
| Fluency | `jitter` | Praat/Parselmouth, averaged |
| Fluency | `shimmer` | Praat/Parselmouth, averaged |
| Fluency | `hnr` | Praat/Parselmouth, averaged |

## Dependencies

```
librosa             # audio loading, pitch (pyin), STFT, spectral features
praat-parselmouth   # jitter, shimmer, HNR
webrtcvad           # VAD, pause analysis
scipy               # spectral entropy (welch + Shannon)
numpy               # numerics
soundfile           # synthetic audio generation for tests
rapidfuzz           # fuzzy word matching for word_coverage / naming_accuracy
fastapi             # backend
uvicorn[standard]   # ASGI server
python-multipart    # form uploads
setuptools<81       # webrtcvad needs pkg_resources (removed in 81)
```

## Running tests

```bash
# Composition + scenario tests (unittest)
python -m unittest discover tests

# ASD CLI runner (synthetic audio)
python -m tests.test_asd_pipeline

# Run pipeline on a real case folder
python -m tests.test_asd_pipeline --audio-dir ./data/cases/<case_id> --age 60
```

## Collecting samples

```bash
uvicorn backend.server:app --reload   # http://localhost:8000
```

Each session: `data/cases/<case_id>/q01.wav … q12.wav` + (after assess) two
result JSONs. The browser viewer shows both screenings side-by-side at ≥900px.

## Target age band

Calibrated for **3-8 years** (36-107 months). Age groups: `3-4`, `5-6`, `7-8`. No
plans for 0-2, 9+, or adult voices.

Out-of-band voices are flagged via `voice_check` in BOTH pipeline outputs — both
call the same `compute_voice_check()` against `CHILD_PITCH_NORMS`. Tier is NOT
overridden; the caller decides. Consumer views render the "Adult voice
identified" warning as an alert + the first item in `next_steps`.

## Language independence

Acoustic metrics are language-agnostic. Speech-delay's `word_coverage` and
`naming_accuracy` use fuzzy matching that works cross-language as long as ASR
returns reasonable transcripts (Sarvam is Indic-trained, code-mix-capable).
Speaking_rate's syllable-to-word conversion is language-dependent — when ASR is
absent the `syllables_per_word_default` (1.85) is the midpoint of English (1.5)
and Hindi (~2.2); mode flagged `acoustic_estimate_unreliable`.

Case `20260518-221449-15da` is a mixed Hindi/Hinglish/English session — informal
validation that pipeline runs cleanly on mixed-language input. Formal cross-
language validation is still TODO.

## What's provisional / needs calibration

ALL norms in BOTH config files are educated guesses from published research on
Western, English-speaking children. Result JSONs include `"calibration_status":
"provisional"` to remind consumers. Both consumer views render a provisional-
norms info banner.

## What's NOT in this pipeline (yet)

- **ASR + pronunciation integration in backend** — Phase L, deferred. Production
  uses Sarvam (saaras:v3) primary + Whisper-base secondary; backend will wrap
  both in `backend/asr.py` and populate `asr_transcript_clean` / `asr_transcript_raw`
  per recording before invoking speech_delay. Until then, 4 speech-delay metrics
  report `computed=false`.
- **Parent behavioral questions** — M-CHAT-R style. Deferred.
- **Two-session confirmation** — app-level logic, not pipeline.
- **Atomic write across both result files** — backend writes them sequentially;
  if it crashes between, case is in inconsistent state. Re-run button is the
  recovery path. Acceptable for v1.
- **Parallel pipeline execution** — currently sequential (~20s for both
  pipelines). Future: asyncio.gather once Whisper is async-wrapped.

## Coding conventions

- Type hints on all function signatures
- Every extractor returns `MetricResult` with `reason` on failure — never raises
- Logging via `logging` module; loggers `asd_pipeline`, `speech_delay_pipeline`,
  `audio_analysis`, `collector`
- No silent failures. If something can't be computed, the output says why.
- Dataclasses for structured data; enums for categorical values
- Per-pipeline configs in JSON, runtime-reloadable
- Shared helpers go in `core/audio_analysis.py`; pipeline-specific stays in its own module
- Consumer-facing copy lives in `*_consumer_view.py` — single source of truth for tier
  labels, plain-language descriptions, alert text. UI is dumb rendering.

## Common tasks

**Change a threshold:** Edit the relevant `*_config.json`. No code change needed.

**Add an age group (e.g. 9-12):**
1. Add to `age_groups` in BOTH configs.
2. Add corresponding entry to `CHILD_PITCH_NORMS` in `core/audio_analysis.py`.
3. (Optional) update consumer view if labels need to change.

**Add a new ASD biomarker:**
1. Write extractor in `core/asd_pipeline.py` returning `MetricResult`.
2. Add norm values to all age groups in `core/asd_config.json`.
3. Add to the appropriate group in `evaluate_biomarker_groups()`.
4. Convergence and tiering handle it automatically.
5. Add `MARKER_DEFS` entry in `core/asd_consumer_view.py` for the chart.

**Add a new speech-delay metric:**
1. Add per-metric extraction in `assess_speech_delay()`.
2. Add metric → norm-table mapping in `_NORM_KEY_BY_METRIC`.
3. Add the metric to its domain in `DOMAINS`.
4. Add norms to all age groups in `core/speech_delay_config.json`.
5. Add `METRIC_DISPLAY` entry in `core/speech_delay_consumer_view.py`.
6. Composition + percentile lookup handle the rest.

**Integrate with Omli backend (server-side use):**
```python
from core.audio_analysis import assess_recording_quality
from core.asd_pipeline import assess_asd_risk
from core.speech_delay_pipeline import assess_speech_delay

# One quality pass shared between pipelines
quality = {p: assess_recording_quality(p) for p in audio_paths}

asd = assess_asd_risk(prompted, audio_paths, child.age_months, recording_quality=quality)
sd = assess_speech_delay(recordings_with_asr, child.age_months, recording_quality=quality)

# asd["tier"], sd["delay_status"] → store in database
# asd["confidence"]["confidence_score"], sd["confidence"]["confidence_score"] → dashboard
# Both quality_reports → flag bad recordings for re-collection
```
