# SPDX-License-Identifier: Apache-2.0
"""Estate catalog is complete and actually calls members (or honest UNAVAILABLE)."""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "torch-universal"))

import szl_kernels as sk
from szl_kernels.estate import ESTATE, probe_member
import szl_kernels.estate as estate


REQUIRED = [
    "szl-kernels",
    "szl-receipt-attn",
    "szl-maskmod",
    "szl-block-kv",
    "YARQA-ATTN",
    "szl-governed-norm",
    "szl-lambda-gate",
    "szl-ouroboros",
    "szl-invariants",
    "szl-formulas",
    "szl-blocked",
    "szl-govsign",
    "szl-provctl",
    "szl-nemo",
    "governed-inference-meter",
    "szl-serve",
]


def test_estate_lists_every_published_kernel():
    keys = [e["key"] for e in sk.list_estate()]
    assert set(keys) == set(REQUIRED)
    assert len(keys) == 16


def test_probe_calls_in_suite_numerics_live():
    report = sk.probe_estate()
    by_key = {k["key"]: k for k in report["kernels"]}
    assert by_key["szl-governed-norm"]["status"] == "LIVE"
    assert by_key["szl-governed-norm"]["called"] is True
    assert by_key["szl-lambda-gate"]["status"] == "LIVE"
    assert by_key["szl-lambda-gate"]["advisory"] is True
    energy = by_key["governed-inference-meter"]
    assert energy["status"] == "LIVE"
    assert energy["called"] is True
    if "joules" in energy:
        assert energy["joules"] is None
        assert "UNAVAILABLE" in str(energy.get("label", ""))
    assert report["cuda"]["status"] in ("LIVE", "UNAVAILABLE")
    if report["cuda"]["status"] == "UNAVAILABLE":
        assert "ROADMAP" in report["cuda"]["note"]
    assert report["joblib"] == "QUARANTINED"
    assert report["pickle"] == "QUARANTINED"


def test_missing_optional_kernel_is_unavailable_not_fake_green():
    entry = next(e for e in ESTATE if e["key"] == "szl-receipt-attn")
    rec = probe_member(dict(entry))
    assert rec["status"] == "UNAVAILABLE"
    assert rec["called"] is False
    assert rec["joblib"] == "QUARANTINED"


def test_injected_package_is_actually_called(monkeypatch):
    called = {"n": 0}

    def selfcheck():
        called["n"] += 1
        return {"ok": True, "path": "stub"}

    stub = types.ModuleType("szl_receipt_attn")
    stub.selfcheck = selfcheck
    monkeypatch.setitem(sys.modules, "szl_receipt_attn", stub)
    entry = next(e for e in ESTATE if e["key"] == "szl-receipt-attn")
    rec = probe_member(dict(entry))
    assert called["n"] == 1
    assert rec["status"] == "LIVE"
    assert rec["called"] is True
    assert rec["probe_result"]["ok"] is True


@pytest.mark.parametrize("result,status", [
    (False, "FAILED"),
    ((False, ["broken invariant"]), "FAILED"),
    ({"ok": False}, "FAILED"),
    ({"arithmetic_ok": False}, "FAILED"),
    ({"passed": False}, "FAILED"),
    ({"ok": True, "checks": {"receipt_valid": False}}, "FAILED"),
    ({"ok": "false"}, "UNVERIFIED"),
    ({"arithmetic_ok": "false"}, "UNVERIFIED"),
    ({"passed": "false"}, "UNVERIFIED"),
    ({"version": "1.0"}, "UNVERIFIED"),
    (None, "UNVERIFIED"),
    (True, "LIVE"),
    ((True, []), "LIVE"),
    ({"arithmetic_ok": True}, "LIVE"),
    ({"passed": True}, "LIVE"),
])
def test_probe_preserves_explicit_failed_or_indeterminate_verdict(monkeypatch, result, status):
    stub = types.ModuleType("szl_receipt_attn")
    stub.selfcheck = lambda: result
    monkeypatch.setitem(sys.modules, "szl_receipt_attn", stub)
    entry = next(e for e in ESTATE if e["key"] == "szl-receipt-attn")
    rec = probe_member(dict(entry))
    assert rec["called"] is True
    assert rec["status"] == status


def test_in_suite_fallback_keeps_failed_chain_failed(monkeypatch):
    monkeypatch.setitem(sys.modules, "szl_governed_norm", None)
    chain = types.SimpleNamespace(
        verify=lambda: (False, 1, 0),
        emit_norm=lambda *args: None,
    )
    monkeypatch.setattr(sk, "UnifiedReceiptChain", lambda: chain)
    entry = next(e for e in ESTATE if e["key"] == "szl-governed-norm")
    rec = probe_member(dict(entry))
    assert rec["called"] is True
    assert rec["package"] == "UNAVAILABLE"
    assert rec["chain_ok"] is False
    assert rec["status"] == "FAILED"


@pytest.mark.parametrize("other_status,expected", [
    ("FAILED", "FAILED"),
    ("UNVERIFIED", "INCOMPLETE"),
    ("UNAVAILABLE", "INCOMPLETE"),
    ("LIVE", "VERIFIED"),
])
def test_estate_aggregate_does_not_hide_a_failed_or_missing_member(monkeypatch, other_status, expected):
    entries = ({"key": "first"}, {"key": "second"})
    monkeypatch.setattr(estate, "ESTATE", entries)
    monkeypatch.setattr(estate, "_extend_sys_path", lambda: None)
    monkeypatch.setattr(estate, "cuda_status", lambda: {"status": "UNAVAILABLE"})
    monkeypatch.setattr(estate, "probe_member", lambda entry: {
        "key": entry["key"],
        "status": "LIVE" if entry["key"] == "first" else other_status,
    })
    report = estate.probe_estate()
    assert report["status"] == expected
    assert report["ok"] is (other_status == "LIVE")
    assert report["some_members_available"] is True
    assert report["live"] == (2 if other_status == "LIVE" else 1)
    assert report[other_status.lower() if other_status != "LIVE" else "live"] >= 1


def test_estate_runtime_mirrors_keep_the_same_probe_contract():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    package = root / "torch-ext" / "szl_kernels" / "estate.py"
    mirror = root / "build" / "torch-universal" / "szl_kernels" / "estate.py"
    assert package.read_text(encoding="utf-8") == mirror.read_text(encoding="utf-8")
