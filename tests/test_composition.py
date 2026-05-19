"""
Composition logic tests for the speech-delay pipeline.
=======================================================
Fixed-input unit tests for the pure functions that drive scoring:
percentile lookup, domain composition, overall composition, developmental
band mapping, and the 6-month rounding helper. These are the subtle
bits where bugs are easy to miss — exercise edge cases here, not just
happy paths.

Run with:
    python -m unittest tests.test_composition

Uses unittest from stdlib — no new test deps.
"""

import unittest

from core.speech_delay_pipeline import (
    DOMAINS,
    _band_midpoint_months,
    _compose_domain,
    _compose_overall,
    _compute_developmental_composite_for_age_group,
    _percentile_lookup,
    _percentile_to_status,
    _round_to_6mo_bucket,
    _validate_input,
    compute_developmental_band,
)


# Sample norm tables — match the shape of asd_config.json entries.
PCC_NORM = {"p10": 75, "p25": 82, "p50": 90, "p75": 95, "p90": 98}     # higher better
JITTER_NORM = {"p10": 5.0, "p25": 3.5, "p50": 2.0, "p75": 1.0, "p90": 0.5}  # lower better
THRESHOLDS = {"on_track_min_percentile": 25, "behind_min_percentile": 10}


class TestPercentileLookup(unittest.TestCase):
    """Direction-agnostic lookup against breakpoints sorted by value."""

    def test_value_at_p50_returns_50_higher_better(self):
        self.assertEqual(_percentile_lookup(90, PCC_NORM), 50)

    def test_value_at_p10_returns_10_higher_better(self):
        self.assertEqual(_percentile_lookup(75, PCC_NORM), 10)

    def test_value_at_p90_returns_90_higher_better(self):
        self.assertEqual(_percentile_lookup(98, PCC_NORM), 90)

    def test_value_below_p10_clamps_to_10(self):
        self.assertEqual(_percentile_lookup(50, PCC_NORM), 10)
        self.assertEqual(_percentile_lookup(0, PCC_NORM), 10)

    def test_value_above_p90_clamps_to_90(self):
        self.assertEqual(_percentile_lookup(99, PCC_NORM), 90)
        self.assertEqual(_percentile_lookup(100, PCC_NORM), 90)

    def test_interpolation_between_p25_and_p50(self):
        # p25=82, p50=90 → midpoint 86 should give percentile ≈ 37.5 → rounds to 38
        self.assertEqual(_percentile_lookup(86, PCC_NORM), 38)

    def test_interpolation_between_p50_and_p75(self):
        # p50=90, p75=95 → midpoint 92.5 should give percentile ≈ 62.5 → 62 or 63
        p = _percentile_lookup(92.5, PCC_NORM)
        self.assertIn(p, (62, 63))

    def test_lower_is_better_value_at_p50_returns_50(self):
        # jitter=2.0% is exactly p50
        self.assertEqual(_percentile_lookup(2.0, JITTER_NORM), 50)

    def test_lower_is_better_low_value_is_best(self):
        # jitter=0.3% is below p90 (0.5%) → in the best end → clamps to 90
        self.assertEqual(_percentile_lookup(0.3, JITTER_NORM), 90)

    def test_lower_is_better_high_value_is_worst(self):
        # jitter=6.0% is above p10 (5.0%) → in the worst end → clamps to 10
        self.assertEqual(_percentile_lookup(6.0, JITTER_NORM), 10)


class TestDomainCompose(unittest.TestCase):
    def test_all_metrics_computed_averages_correctly(self):
        # Articulation domain: 2 metrics
        percentiles = {"single_word_pcc": 50, "connected_pcc": 30, "word_coverage": 70}
        result = _compose_domain("articulation", percentiles, THRESHOLDS)
        self.assertEqual(result["percentile"], 40)  # (50 + 30) / 2
        self.assertEqual(result["computed_count"], 2)
        self.assertEqual(result["total_count"], 2)
        self.assertEqual(result["status"], "on_track")

    def test_partial_metrics_uses_only_computed(self):
        percentiles = {"single_word_pcc": 50, "connected_pcc": None}
        result = _compose_domain("articulation", percentiles, THRESHOLDS)
        self.assertEqual(result["percentile"], 50)
        self.assertEqual(result["computed_count"], 1)
        self.assertEqual(result["total_count"], 2)

    def test_no_metrics_computed_returns_not_computed(self):
        percentiles = {"single_word_pcc": None, "connected_pcc": None}
        result = _compose_domain("articulation", percentiles, THRESHOLDS)
        self.assertIsNone(result["percentile"])
        self.assertEqual(result["status"], "not_computed")
        self.assertEqual(result["computed_count"], 0)

    def test_status_thresholds(self):
        # >=25 → on_track, >=10 → behind, else significantly_behind
        self.assertEqual(_percentile_to_status(30, THRESHOLDS), "on_track")
        self.assertEqual(_percentile_to_status(25, THRESHOLDS), "on_track")
        self.assertEqual(_percentile_to_status(20, THRESHOLDS), "behind")
        self.assertEqual(_percentile_to_status(10, THRESHOLDS), "behind")
        self.assertEqual(_percentile_to_status(5, THRESHOLDS), "significantly_behind")
        self.assertEqual(_percentile_to_status(None, THRESHOLDS), "not_computed")


class TestOverallCompose(unittest.TestCase):
    def test_lowest_domain_picks_worst(self):
        domain_results = {
            "articulation": {"percentile": 30},
            "language": {"percentile": 70},
            "fluency": {"percentile": 50},
        }
        # Severe articulation deficit must dominate — that's the whole point.
        self.assertEqual(_compose_overall(domain_results, "lowest_domain"), 30)

    def test_simple_average_averages_all(self):
        domain_results = {
            "articulation": {"percentile": 30},
            "language": {"percentile": 70},
            "fluency": {"percentile": 50},
        }
        self.assertEqual(_compose_overall(domain_results, "simple_average"), 50)

    def test_skips_not_computed_domain(self):
        domain_results = {
            "articulation": {"percentile": None},
            "language": {"percentile": 80},
            "fluency": {"percentile": 60},
        }
        # lowest_domain over only the computed values
        self.assertEqual(_compose_overall(domain_results, "lowest_domain"), 60)

    def test_returns_none_when_no_domain_computed(self):
        domain_results = {
            "articulation": {"percentile": None},
            "language": {"percentile": None},
            "fluency": {"percentile": None},
        }
        self.assertIsNone(_compose_overall(domain_results, "lowest_domain"))


class TestRoundTo6MoBucket(unittest.TestCase):
    def test_negative_returns_zero(self):
        self.assertEqual(_round_to_6mo_bucket(-3), 0)
        self.assertEqual(_round_to_6mo_bucket(0), 0)

    def test_rounds_to_nearest_6(self):
        self.assertEqual(_round_to_6mo_bucket(4), 6)    # closer to 6 than 0
        self.assertEqual(_round_to_6mo_bucket(8), 6)    # closer to 6 than 12
        self.assertEqual(_round_to_6mo_bucket(11), 12)
        self.assertEqual(_round_to_6mo_bucket(15), 18)
        self.assertEqual(_round_to_6mo_bucket(17), 18)
        self.assertEqual(_round_to_6mo_bucket(24), 24)

    def test_band_midpoint(self):
        self.assertEqual(_band_midpoint_months([36, 59]), 47.5)
        self.assertEqual(_band_midpoint_months([60, 83]), 71.5)


# Minimal age_groups_config to drive band mapping tests. We use only
# articulation + language norms because that's what developmental_composite
# uses (per design — fluency doesn't drive band mapping).
SAMPLE_AGE_GROUPS = {
    "3-4": {
        "age_range_months": [36, 59],
        "norms": {
            "single_word_pcc":      {"p10": 65, "p25": 75, "p50": 85, "p75": 92, "p90": 97},
            "connected_pcc_offset": 7,
            "word_coverage":        {"p10": 50, "p25": 65, "p50": 78, "p75": 88, "p90": 95},
            "naming_accuracy":      {"p10": 50, "p25": 65, "p50": 80, "p75": 90, "p90": 100},
        },
    },
    "5-6": {
        "age_range_months": [60, 83],
        "norms": {
            "single_word_pcc":      {"p10": 75, "p25": 82, "p50": 90, "p75": 95, "p90": 98},
            "connected_pcc_offset": 6,
            "word_coverage":        {"p10": 60, "p25": 72, "p50": 82, "p75": 90, "p90": 96},
            "naming_accuracy":      {"p10": 60, "p25": 75, "p50": 85, "p75": 92, "p90": 100},
        },
    },
    "7-8": {
        "age_range_months": [84, 107],
        "norms": {
            "single_word_pcc":      {"p10": 82, "p25": 88, "p50": 93, "p75": 97, "p90": 99},
            "connected_pcc_offset": 5,
            "word_coverage":        {"p10": 70, "p25": 80, "p50": 88, "p75": 94, "p90": 98},
            "naming_accuracy":      {"p10": 70, "p25": 82, "p50": 90, "p75": 96, "p90": 100},
        },
    },
}


class TestDevelopmentalBand(unittest.TestCase):
    def test_on_track_5yo_at_p50_for_age(self):
        # 5yo (60 mo) scoring at p50 for 5-6 norms: well above 50 (mean) → on track
        metric_values = {
            "single_word_pcc": 90,
            "connected_pcc": 84,
            "word_coverage": 82,
            "naming_accuracy": 85,
        }
        result = compute_developmental_band(metric_values, 60, SAMPLE_AGE_GROUPS, "5-6")
        self.assertEqual(result["band"], "5-6")
        self.assertEqual(result["delay_months"], 0)

    def test_behind_5yo_scores_like_3_4yo(self):
        # 5yo (60 mo) scoring at typical 3-4 levels. Should land in 3-4 band.
        # 3-4 p50: pcc=85, word_coverage=78, naming=80
        metric_values = {
            "single_word_pcc": 85,
            "connected_pcc": 78,
            "word_coverage": 78,
            "naming_accuracy": 80,
        }
        result = compute_developmental_band(metric_values, 60, SAMPLE_AGE_GROUPS, "5-6")
        self.assertEqual(result["band"], "3-4")
        # delay = 60 - 47.5 = 12.5 → rounds to 12
        self.assertEqual(result["delay_months"], 12)

    def test_significantly_behind_falls_below_youngest(self):
        # 7yo with metrics below even 3-4 p10. Should hit "below_3-4".
        metric_values = {
            "single_word_pcc": 50,
            "connected_pcc": 40,
            "word_coverage": 30,
            "naming_accuracy": 30,
        }
        result = compute_developmental_band(metric_values, 96, SAMPLE_AGE_GROUPS, "7-8")
        self.assertEqual(result["band"], "below_3-4")
        # delay = 96 - 36 = 60 → rounds to 60
        self.assertEqual(result["delay_months"], 60)

    def test_no_metrics_computed_returns_insufficient_data(self):
        # Empty metric_values — articulation + language can't be computed in any
        # age group → band/delay are not meaningful. Should NOT fabricate a
        # "below" band; should explicitly report insufficient_data.
        result = compute_developmental_band({}, 60, SAMPLE_AGE_GROUPS, "5-6")
        self.assertEqual(result["band"], "insufficient_data")
        self.assertIsNone(result["delay_months"])

    def test_single_metric_in_one_domain(self):
        # Only naming_accuracy computed — only the language domain has a number.
        # Developmental composite uses lowest_domain across articulation+language,
        # so when articulation is None it should fall back to language alone.
        metric_values = {"naming_accuracy": 95}  # high
        result = compute_developmental_band(metric_values, 60, SAMPLE_AGE_GROUPS, "5-6")
        # naming_accuracy=95 in 5-6 norms is between p75 (92) and p90 (100) →
        # somewhere in the 80s — well above 50, so on track.
        self.assertEqual(result["band"], "5-6")
        self.assertEqual(result["delay_months"], 0)


class TestComputeDevelopmentalCompositeForAgeGroup(unittest.TestCase):
    def test_uses_lowest_of_two_domains(self):
        # Strong language, weak articulation. Min-of-domain should pick articulation.
        metric_values = {
            "single_word_pcc": 75,   # 5-6 norm: p10 = 75 → percentile 10
            "connected_pcc": 70,     # +6 = 76 → still around p10
            "word_coverage": 90,     # 5-6 norm: p75 = 90 → percentile 75
            "naming_accuracy": 92,   # 5-6 norm: p75 = 92 → percentile 75
        }
        composite = _compute_developmental_composite_for_age_group(
            metric_values, SAMPLE_AGE_GROUPS["5-6"]["norms"]
        )
        # Articulation should be ~10, language ~75. Min = articulation = ~10.
        self.assertLess(composite, 25)

    def test_connected_pcc_offset_applied(self):
        # Verify the +6 offset actually shifts the connected_pcc lookup.
        metric_values_no_offset = {"single_word_pcc": 84}    # = p50-ish (84)
        metric_values_offset = {"connected_pcc": 78}         # +6 = 84 → same lookup
        c1 = _compute_developmental_composite_for_age_group(
            metric_values_no_offset, SAMPLE_AGE_GROUPS["5-6"]["norms"]
        )
        c2 = _compute_developmental_composite_for_age_group(
            metric_values_offset, SAMPLE_AGE_GROUPS["5-6"]["norms"]
        )
        # Both produce a single-metric articulation domain → identical composite
        # because connected_pcc 78 + offset 6 = 84 matches single_word_pcc 84.
        self.assertEqual(c1, c2)


class TestValidateInput(unittest.TestCase):
    def test_rejects_non_list_recordings(self):
        with self.assertRaisesRegex(ValueError, "recordings must be a list"):
            _validate_input("not a list", 60)

    def test_rejects_empty_recordings(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            _validate_input([], 60)

    def test_rejects_non_int_age(self):
        with self.assertRaisesRegex(ValueError, "child_age_months must be int"):
            _validate_input([{"audio_path": "a.wav", "task_type": "prompted_question"}], "60")

    def test_rejects_out_of_band_age(self):
        with self.assertRaisesRegex(ValueError, "24-144"):
            _validate_input([{"audio_path": "a.wav", "task_type": "prompted_question"}], 200)

    def test_rejects_missing_audio_path(self):
        with self.assertRaisesRegex(ValueError, "audio_path"):
            _validate_input([{"task_type": "prompted_question"}], 60)

    def test_rejects_duplicate_audio_path(self):
        with self.assertRaisesRegex(ValueError, "duplicate audio_path"):
            _validate_input([
                {"audio_path": "a.wav", "task_type": "prompted_question"},
                {"audio_path": "a.wav", "task_type": "picture_naming"},
            ], 60)

    def test_rejects_unknown_task_type(self):
        with self.assertRaisesRegex(ValueError, "task_type"):
            _validate_input([{"audio_path": "a.wav", "task_type": "free_form"}], 60)

    def test_accepts_valid_input(self):
        _validate_input(
            [{"audio_path": "q01.wav", "task_type": "prompted_question"}],
            60,
        )  # Should not raise


if __name__ == "__main__":
    unittest.main()
