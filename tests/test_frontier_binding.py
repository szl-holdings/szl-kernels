from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BINDING = ROOT / "frontier" / "hf-wave-2026-09-04.json"


def test_webgpu_frontier_binding_is_fail_closed_and_inventory_pinned() -> None:
    binding = json.loads(BINDING.read_text(encoding="utf-8"))
    assert binding["schema"] == "szl.frontier.binding.v1"
    assert binding["owner"] == "szl-kernels"
    assert binding["default_effect"] == "HOLD"
    assert binding["production_promotion"] is False
    assert binding["canonical_frontier_revision"] == "94a039d086d1343d1fcfe5bca617f601006ca05b"

    candidate = binding["candidates"][0]
    assert candidate["id"] == "hf-webgpu-kernels-2026-09-01"
    assert candidate["status"] == "EVALUATE"
    assert candidate["authority"] == "none"
    assert candidate["inventory_baseline"] == 207
    assert candidate["production_authority"] is False
    assert "reference correctness parity" in candidate["required_evidence"]
    assert "rollback verification" in candidate["required_evidence"]
    assert "discovery events only" in candidate["inventory_policy"]
