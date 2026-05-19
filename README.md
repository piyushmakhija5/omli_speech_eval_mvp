# Omli Speech Screening

Two screening pipelines that share the same 12-recording session:

- **ASD pipeline** — acoustic biomarkers across prosody, spectral, interaction, voice stability. Tier output: `no_indicators` / `monitor` / `recommend_evaluation` / `insufficient_data`.
- **Speech-delay pipeline** — 10 metrics across articulation, language, fluency. Percentile-based scoring with developmental band + delay status (`on_track` / `behind` / `significantly_behind`).

All metrics are designed to be language-independent (English / Hindi / Hinglish). Speech-delay's ASR-dependent metrics will use Sarvam + Whisper-base once Phase L wires them in; until then they report `computed=false` with a reason and the pipeline produces a partial result.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m core.asd_pipeline           # quick config check
python -m core.speech_delay_pipeline  # quick config check
```

> `requirements.txt` pins `setuptools<81` because `webrtcvad` imports the legacy
> `pkg_resources` module, which setuptools 81+ removed.

## Quick test (synthetic audio)

```bash
python -m tests.test_asd_pipeline             # ASD CLI runner — generates 12 synthetic WAVs
python -m unittest discover tests             # full unittest suite (composition + scenarios)
```

The full suite covers ~49 tests: pure-function composition (`test_composition.py`) and end-to-end scenario tests for the speech-delay pipeline (`test_speech_delay_pipeline.py`).

## Test with real recordings

```bash
# Put 4-12 .wav files in a folder
python -m tests.test_asd_pipeline --audio-dir ./my_recordings --age 60
```

For the speech-delay pipeline against a real case, easiest is the web collector below — once recordings are uploaded, click "See results" or POST to `/api/cases/{id}/assess`.

## Collecting samples (web app)

```bash
uvicorn backend.server:app --reload
# open http://localhost:8000 — record 12 takes → "See results" runs BOTH pipelines
```

Each session is saved to `data/cases/<case_id>/q01.wav … q12.wav` and produces two result files:
- `asd_result.json`
- `speech_delay_result.json`

The browser viewer shows both screenings side-by-side at ≥900px and stacked on smaller screens.

## Use in your code

```python
from core.audio_analysis import assess_recording_quality
from core.asd_pipeline import assess_asd_risk
from core.speech_delay_pipeline import assess_speech_delay

# Single quality pass shared by both pipelines
paths = ["q01.wav", "q02.wav", ..., "q12.wav"]
quality = {p: assess_recording_quality(p) for p in paths}

asd_result = assess_asd_risk(
    prompted_question_audio_paths=paths[:4],
    all_audio_paths=paths,
    child_age_months=66,
    recording_quality=quality,
)

speech_delay_result = assess_speech_delay(
    recordings=[
        # Per-recording dict with audio_path, task_type, expected_text,
        # asr_transcript_clean (Sarvam), asr_transcript_raw (Whisper),
        # pronunciation_scores (Azure/ZIPA). All optional except audio_path
        # and task_type.
        {"audio_path": p, "task_type": "prompted_question", "expected_text": None,
         "asr_transcript_clean": None, "asr_transcript_raw": None,
         "pronunciation_scores": None}
        for p in paths
    ],
    child_age_months=66,
    recording_quality=quality,
)

print(asd_result["tier"])                       # ASD tier
print(speech_delay_result["delay_status"])      # on_track / behind / etc.
print(speech_delay_result["developmental_band"])  # 3-4 / 5-6 / 7-8 / below_3-4 / insufficient_data
print(speech_delay_result["delay_months"])      # rounded to 6-month buckets
```

## ASD pipeline

| Group | Markers | Extracted from |
|-------|---------|----------------|
| Prosody | Pitch variability, pitch range | Prompted questions |
| Spectral | Spectral entropy, LTAS slope | Prompted questions |
| Interaction | Response latency, turn-taking pattern | Prompted questions |
| Voice stability | Jitter, shimmer, HNR, pause patterns | All 12 recordings |

Output tier: 0-1 atypical groups = `no_indicators`, 2 = `monitor`, 3+ = `recommend_evaluation`. Voice-check separately flags adult voices (mean F0 > 2 SD below child norm) — see `core/audio_analysis.CHILD_PITCH_NORMS`.

## Speech-delay pipeline

| Domain | Metrics | Notes |
|--------|---------|-------|
| Articulation | `single_word_pcc`, `connected_pcc` | Both need pronunciation_scores (Azure/ZIPA). connected_pcc uses single_word_pcc norm table + per-age-group offset. |
| Language | `word_coverage`, `naming_accuracy` | Need `asr_transcript_clean` + `expected_text` per recording. Fuzzy match at 75% via rapidfuzz. |
| Fluency | `speaking_rate`, `pause_ratio`, `pitch_mean`, `jitter`, `shimmer`, `hnr` | `speaking_rate` needs ASR; the other 5 are pure-audio. |

Composition: **lowest_domain** (worst-performing domain drives `delay_status`) — a child with severe articulation issues can't be masked by good fluency.

Developmental band uses articulation + language only (the `developmental_composite`). Fluency varies for reasons unrelated to language acquisition (microphone, mood) and would dilute the band signal. If both articulation and language are uncomputable, band returns `insufficient_data` rather than fabricating a delay.

Delay months are rounded to the nearest 6-month bucket (half-up — `15mo → 18mo`) — honest precision given how provisional the norms are.

## Output tiers

ASD:
| Tier | Condition |
|------|-----------|
| `no_indicators` | 0-1 biomarker groups atypical |
| `monitor` | 2 groups atypical |
| `recommend_evaluation` | 3+ groups atypical |
| `insufficient_data` | <10s spontaneous speech or <3 groups computable |

Speech-delay:
| Status | Condition |
|--------|-----------|
| `on_track` | composite percentile ≥ 25 |
| `behind` | composite 10-24 |
| `significantly_behind` | composite < 10 |
| `insufficient_data` | no composite computable |

## Configuration

All thresholds and norms live in JSON, separate from code:

- `core/asd_config.json` — ASD age-group norms, scoring thresholds
- `core/speech_delay_config.json` — speech-delay percentile tables, delay thresholds, fuzzy-match threshold, syllables-per-word default

Both reload at runtime: `python -c "from core.asd_pipeline import reload_config; reload_config()"`.

**All norms are PROVISIONAL.** Calibrate against SLP ground truth before clinical use. Pipeline reports include `calibration_status: "provisional"` to remind consumers.

## Project layout

```
core/                              # Pipelines + configs (importable)
  audio_analysis.py                # Shared: quality gate, audio loaders, pitch/voice-stability/pause
                                   # extractors, voice_check + CHILD_PITCH_NORMS constant
  asd_pipeline.py                  # ASD-specific extractors (spectral entropy, LTAS, latency),
                                   # group convergence, tier mapping
  asd_config.json
  asd_consumer_view.py             # ASD raw → consumer summary translator
  speech_delay_pipeline.py         # 10 metrics, percentile lookup, lowest-domain composition,
                                   # developmental band mapping
  speech_delay_config.json
  speech_delay_consumer_view.py    # Speech-delay raw → consumer summary translator

backend/                           # FastAPI collector
  server.py                        # /api/cases (list), /assess (runs BOTH pipelines), /summary,
                                   #   /raw?which=asd|speech_delay, ...
  storage.py

frontend/                          # Single-page collector + viewer
  index.html                       # Two-column result pane at ≥900px
  recorder.js                      # Mic capture, ASD + speech-delay rendering, percentile charts
  worklet.js
  styles.css

data/
  questions.json                   # 12 questions with expected_text per item
  cases/<case_id>/                 # Collected sessions + asd_result.json + speech_delay_result.json
  synthetic/                       # Sine-wave WAVs for tests (gitignored)

tests/
  test_asd_pipeline.py             # ASD CLI runner (synthetic or real audio)
  test_composition.py              # 36 unit tests for pure functions
  test_speech_delay_pipeline.py    # 13 scenario tests (full / no-ASR / no-pron / bad quality / ages)

requirements.txt
CLAUDE.md
```

## Phases (build status)

- Phase 1 — `core/audio_analysis.py` shared module — done
- Phase 2 — speech_delay pipeline + config + composition tests — done
- Phase 3 — backend orchestrates both pipelines (single quality pass) — done
- Phase 4 — speech_delay consumer view translator — done
- Phase 5 — two-column UI with side-by-side rendering — done
- Phase 6 — scenario tests + docs — done
- **Phase L (deferred)** — Sarvam ASR + Whisper-base secondary in backend; populates `asr_transcript_clean` + `asr_transcript_raw` per recording. Until shipped, 4 speech-delay metrics report `computed=false`.
