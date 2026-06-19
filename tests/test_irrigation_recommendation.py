"""
tests/test_irrigation_recommendation.py

FIXES applied:
  - Added sys.path insert so `irrigation_recommendation` (top-level module) is importable
  - Added tests for get_water_amount_mm() helper
  - Added cotton (low sensitivity) edge case test
"""

import sys
from pathlib import Path

# FIXED: irrigation_recommendation.py lives at project root, not in a package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from irrigation_recommendation import (
    recommend_irrigation,
    get_water_amount_mm,
    get_recommendation_rationale,
    IrrigationRecommender,
)


class TestIrrigationRecommendation:

    # ── Basic recommendation logic ────────────────────────────────────────────

    def test_high_stress_no_rain_gives_irrigate_now(self):
        assert recommend_irrigation("wheat", "high stress", 0.0) == "irrigate now"

    def test_high_stress_heavy_rain_gives_delay(self):
        assert recommend_irrigation("wheat", "high stress", 25.0) == "irrigate in 2 days"

    def test_medium_stress_heavy_rain_no_irrigation(self):
        assert recommend_irrigation("rice", "medium stress", 25.0) == "no irrigation needed"

    def test_low_stress_heavy_rain_no_irrigation(self):
        assert recommend_irrigation("cotton", "low stress", 20.0) == "no irrigation needed"

    def test_rice_high_sensitivity_low_stress_low_rain(self):
        # rice sensitivity > 1.0 → low stress + low rain → irrigate in 2 days
        assert recommend_irrigation("rice", "low stress", 2.0) == "irrigate in 2 days"

    def test_cotton_low_sensitivity_low_stress_low_rain(self):
        # cotton sensitivity = 0.8 ≤ 1.0 → no irrigation
        assert recommend_irrigation("cotton", "low stress", 2.0) == "no irrigation needed"

    def test_medium_stress_moderate_rain_delays(self):
        assert recommend_irrigation("maize", "medium stress", 10.0) == "irrigate in 2 days"

    # ── Validation errors ─────────────────────────────────────────────────────

    def test_invalid_crop_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported crop type"):
            recommend_irrigation("banana", "low stress", 5.0)

    def test_invalid_stress_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported stress level"):
            recommend_irrigation("wheat", "extreme stress", 5.0)

    def test_negative_rainfall_raises_value_error(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            recommend_irrigation("wheat", "low stress", -1.0)

    # ── Water amount helper ───────────────────────────────────────────────────

    def test_water_amount_irrigate_now_high_stress(self):
        mm = get_water_amount_mm("irrigate now", "high stress")
        assert mm > 0

    def test_water_amount_no_irrigation(self):
        mm = get_water_amount_mm("no irrigation needed", "low stress")
        assert mm == 0

    # ── Rationale text ────────────────────────────────────────────────────────

    def test_rationale_returns_string(self):
        r = get_recommendation_rationale("wheat", "high stress", 0.0)
        assert isinstance(r, str) and len(r) > 0

    def test_rationale_mentions_crop(self):
        r = get_recommendation_rationale("rice", "medium stress", 12.0)
        assert "rice" in r.lower()
