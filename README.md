# ASD Risk Detection Pipeline

Extracts acoustic biomarkers from child speech and produces a tiered risk classification.
Works for English, Hindi, and Hinglish — all metrics are language-independent.

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python -m core.asd_pipeline
```

> `requirements.txt` pins `setuptools<81` because `webrtcvad` imports the legacy
> `pkg_resources` module, which setuptools 81+ removed.

## Quick test (synthetic audio)

```bash
python -m tests.test_asd_pipeline
```

This generates 12 synthetic audio files in `data/synthetic/` and runs the full pipeline.
You should see output like:

```
TIER:       no_indicators
CONFIDENCE: 1.0
ATYPICAL:   1 / 4 groups
RECORDINGS: 12 usable / 12 total (0 rejected)
```

## Test with real recordings

```bash
# Put .wav files in a folder (at least 4, ideally 12)
# First 4 files are treated as "prompted question" responses
python -m tests.test_asd_pipeline --audio-dir ./my_recordings --age 60
```

- `--audio-dir`: folder with .wav files (16kHz mono preferred, will be resampled if needed)
- `--age`: child's age in months (60 = 5 years)

Results are printed and saved as `asd_result.json` in the audio directory.

## Collecting samples (dummy web app)

A small FastAPI collector lets you record 12 voice samples in the browser and drop
them into `data/cases/<case_id>/q01.wav … q12.wav`, ready to be fed straight into
the pipeline.

```bash
uvicorn backend.server:app --reload
# open http://localhost:8000
```

Then run the pipeline against the collected case:

```bash
python -m tests.test_asd_pipeline --audio-dir data/cases/<case_id> --age 60
```

## Use in your code

```python
from core.asd_pipeline import assess_asd_risk

result = assess_asd_risk(
    prompted_question_audio_paths=["q01.wav", "q02.wav", "q03.wav", "q04.wav"],
    all_audio_paths=["q01.wav", "q02.wav", ..., "q12.wav"],
    child_age_months=66,  # 5.5 years
)

print(result["tier"])        # "no_indicators" / "monitor" / "recommend_evaluation" / "insufficient_data"
print(result["message"])     # human-readable explanation
print(result["confidence"])  # confidence score + warnings
print(result["biomarkers"])  # all extracted values
```

## What it checks per recording

Before computing any metrics, each recording is assessed for:
- **File integrity**: exists, loadable, non-empty
- **Duration**: minimum 3 seconds
- **SNR**: signal-to-noise ratio (warns below 5 dB)
- **Clipping**: rejects if >1% samples are clipped
- **Speech presence**: rejects if <10% of frames contain speech

Rejected recordings are logged with a reason. The pipeline continues with usable recordings.

## Output tiers

| Tier | Condition | What it means |
|------|-----------|---------------|
| `no_indicators` | 0-1 biomarker groups atypical | No atypical patterns detected |
| `monitor` | 2 groups atypical | Some atypical patterns. Recommend re-assessment in 2 weeks |
| `recommend_evaluation` | 3+ groups atypical | Consistent atypical patterns. Recommend clinical evaluation |
| `insufficient_data` | <10s spontaneous speech or <3 groups computable | Not enough data for reliable screening |

## 4 biomarker groups

| Group | Markers | Extracted from |
|-------|---------|----------------|
| Prosody | Pitch variability, pitch range | Prompted questions (spontaneous speech) |
| Spectral | Spectral entropy, LTAS slope | Prompted questions |
| Interaction | Response latency, turn-taking pattern | Prompted questions |
| Voice stability | Jitter, shimmer, HNR, pause patterns | All 12 recordings |

Multiple atypical markers within the same group count as 1 atypical group.

## Configuring thresholds

All thresholds live in `core/asd_config.json`, separate from code. Edit the JSON, not the Python.

```bash
# Edit thresholds
vim core/asd_config.json

# Changes take effect on next pipeline run. Or reload at runtime:
python -c "from core.asd_pipeline import reload_config; reload_config()"
```

**Adding a new age group** (e.g., 9-12 years): add a new entry to `age_groups` in the JSON. No code changes needed.

```json
"9-12": {
    "age_range_months": [108, 155],
    "pitch_variability": {"mean": 35.0, "std": 10.0},
    "pitch_range": {"mean": 100.0, "std": 35.0},
    ...
}
```

**Changing sensitivity**: lower `atypical_threshold_sd` (e.g., 1.5) flags more children. Higher (e.g., 2.5) flags fewer.

**Important**: All age-adjusted norms are provisional. They MUST be calibrated with SLP ground truth data before clinical use.

## Project layout

```
core/                          # pipeline + config (importable)
  asd_pipeline.py
  asd_config.json
backend/                       # FastAPI collector server
  server.py
  storage.py
frontend/                      # single-page collector UI
  index.html
  recorder.js
  worklet.js
  styles.css
data/
  questions.json               # 12 questions used by the collector
  cases/<case_id>/q01.wav…     # collected sessions (gitignored)
  synthetic/                   # synthetic test audio (gitignored)
tests/
  test_asd_pipeline.py
requirements.txt
```
