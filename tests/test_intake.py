"""Tests for audit intake (no Claude, no full-universe dry-count)."""

from __future__ import annotations

from pathlib import Path

from ht_backtest.discovery.intake import validate_intake
from ht_backtest.discovery.pipeline import run_intake_pipeline
from ht_backtest.discovery.risk_scan import scan_risks
from ht_backtest.discovery.translate import translate_with_fixture


def test_validate_intake_example():
    raw = {
        "title": "t",
        "entry": {
            "plain_english": "First raid of the killzone then reclaim inside the edge.",
            "indicator_parameters": {"reclaim_bars": 3},
        },
        "stop": {"plain_english": "Beyond sweep extreme", "parameters": {}},
        "filters": [],
        "timeframe": "15m",
        "instrument_type": "crypto",
    }
    out = validate_intake(raw)
    assert out["instrument_type"] == "crypto"


def test_fixture_maps_first_raid():
    intake = validate_intake(
        {
            "title": "First raid",
            "entry": {
                "plain_english": "Take the first raid of the London killzone and reclaim.",
                "indicator_parameters": {},
            },
            "stop": {"plain_english": "Stop beyond extreme", "parameters": {}},
            "timeframe": "15m",
            "instrument_type": "crypto",
            "theoretical_spine": "Judas swing first raid",
        }
    )
    cand = translate_with_fixture(intake)
    assert cand["dry_count"]["method"] == "kz_first_raid_reclaim"
    assert 0.0 <= cand["confidence"] <= 1.0


def test_fx_rejected_insufficient_data():
    root = Path("data/candidates/_test_fx")
    intake = {
        "title": "FX idea",
        "entry": {
            "plain_english": "Buy EURUSD when RSI crosses above 30 on the London open session.",
            "indicator_parameters": {"rsi": 14},
        },
        "stop": {"plain_english": "1 ATR stop", "parameters": {}},
        "timeframe": "15m",
        "instrument_type": "FX",
    }
    result = run_intake_pipeline(intake, fixture=True, candidates_root=root, skip_dry_count=True)
    assert result["accepted"] is False
    assert any("insufficient_data" in r for r in result["reject_reasons"])


def test_repaint_flagged():
    intake = {
        "title": "ZigZag",
        "entry": {
            "plain_english": "Enter when the ZigZag indicator confirms a swing low repaint pattern.",
            "indicator_parameters": {},
        },
        "stop": {"plain_english": "fixed stop", "parameters": {}},
        "filters": [],
        "timeframe": "15m",
        "instrument_type": "crypto",
        "theoretical_spine": "",
        "source": "",
    }
    from ht_backtest.discovery.intake import validate_intake as v

    intake = v(intake)
    risk = scan_risks(intake)
    assert risk["repainting_risk"] is True
