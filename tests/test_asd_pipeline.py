"""
Test script for ASD Pipeline
=============================
Generates synthetic test audio and runs the full pipeline.
Use this to verify the pipeline works on your machine before testing with real recordings.

Usage:
    python test_asd_pipeline.py                    # runs with synthetic audio
    python test_asd_pipeline.py --audio-dir ./my_recordings --age 60
"""

import os
import sys
import argparse
import logging
import json
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger("test")


def generate_synthetic_audio(output_dir: str, n_files: int = 12):
    """Generate synthetic child-like audio for testing."""
    import soundfile as sf

    os.makedirs(output_dir, exist_ok=True)
    sr = 16000
    paths = []

    for i in range(n_files):
        duration = np.random.uniform(3.5, 6.0)
        t = np.linspace(0, duration, int(sr * duration))

        # Simulate child voice: ~300Hz fundamental + harmonics + light noise
        y = (0.3 * np.sin(2 * np.pi * 300 * t)
             + 0.15 * np.sin(2 * np.pi * 600 * t)
             + 0.08 * np.sin(2 * np.pi * 900 * t)
             + 0.05 * np.random.randn(len(t)))

        # Add silence at start (simulates response latency)
        silence = np.zeros(int(sr * np.random.uniform(0.3, 1.5)))
        y = np.concatenate([silence, y]).astype(np.float32)

        # Normalize to safe range
        y = y / (np.max(np.abs(y)) + 0.1) * 0.7

        path = os.path.join(output_dir, f"q{i + 1}.wav")
        sf.write(path, y, sr)
        paths.append(path)

    return paths


def run_test_synthetic():
    """Run pipeline on synthetic audio."""
    from core.asd_pipeline import assess_asd_risk

    logger.info("Generating 12 synthetic audio files...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(project_root, "data", "synthetic")
    paths = generate_synthetic_audio(test_dir)

    prompted = paths[:4]
    all_audio = paths

    logger.info(f"Running ASD pipeline (age: 66 months / 5.5 years)...")
    result = assess_asd_risk(
        prompted_question_audio_paths=prompted,
        all_audio_paths=all_audio,
        child_age_months=66,
    )

    print_result(result)
    return result


def run_test_real(audio_dir: str, age_months: int):
    """Run pipeline on real audio files from a directory."""
    from core.asd_pipeline import assess_asd_risk

    # Find all wav files
    wav_files = sorted([
        os.path.join(audio_dir, f)
        for f in os.listdir(audio_dir)
        if f.endswith(".wav")
    ])

    if len(wav_files) == 0:
        logger.error(f"No .wav files found in {audio_dir}")
        sys.exit(1)

    if len(wav_files) < 4:
        logger.error(f"Need at least 4 .wav files, found {len(wav_files)}")
        sys.exit(1)

    # First 4 = prompted questions, all = full set
    prompted = wav_files[:4]
    all_audio = wav_files[:12]  # cap at 12

    logger.info(f"Found {len(wav_files)} recordings in {audio_dir}")
    logger.info(f"Using first 4 as prompted questions, first {len(all_audio)} as full set")
    logger.info(f"Child age: {age_months} months ({age_months / 12:.1f} years)")

    result = assess_asd_risk(
        prompted_question_audio_paths=prompted,
        all_audio_paths=all_audio,
        child_age_months=age_months,
    )

    print_result(result)

    # Save full result as JSON
    output_path = os.path.join(audio_dir, "asd_result.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Full result saved to {output_path}")

    return result


def print_result(result: dict):
    """Pretty-print the pipeline result."""
    print()
    print("=" * 60)
    print(f"  TIER:       {result['tier']}")
    print(f"  MESSAGE:    {result['message']}")
    print()

    conf = result.get("confidence", {})
    print(f"  CONFIDENCE: {conf.get('confidence_score', 'N/A')}")
    if conf.get("warnings"):
        for w in conf["warnings"]:
            print(f"    ⚠  {w}")
    print()

    print(f"  ATYPICAL:   {result.get('atypical_group_count', 0)} / {result.get('computable_group_count', 0)} groups")
    if result.get("atypical_groups"):
        print(f"              → {', '.join(result['atypical_groups'])}")
    print()

    qr = result.get("quality_report", {})
    rejected = qr.get("rejected_count", 0)
    total = len(qr.get("all_recordings", []))
    print(f"  RECORDINGS: {total - rejected} usable / {total} total ({rejected} rejected)")
    print()

    if result.get("biomarkers"):
        print("  BIOMARKERS:")
        for k, v in result["biomarkers"].items():
            if v is not None:
                print(f"    {k}: {round(v, 3) if isinstance(v, float) else v}")
            else:
                print(f"    {k}: — (insufficient data)")
        print()

    if result.get("group_details"):
        print("  GROUP RESULTS:")
        for group, data in result["group_details"].items():
            status = "ATYPICAL" if data.get("atypical") else "typical"
            computable = "✓" if data.get("computable") else "✗"
            print(f"    {group:20s} [{computable}] {status}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ASD Pipeline")
    parser.add_argument("--audio-dir", type=str, default=None,
                        help="Directory with .wav recordings. If not provided, uses synthetic audio.")
    parser.add_argument("--age", type=int, default=66,
                        help="Child's age in months (default: 66 = 5.5 years)")
    args = parser.parse_args()

    if args.audio_dir:
        run_test_real(args.audio_dir, args.age)
    else:
        run_test_synthetic()
