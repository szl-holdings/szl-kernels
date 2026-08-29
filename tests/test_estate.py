# SPDX-License-Identifier: Apache-2.0
"""Estate catalog is complete and actually calls members (or honest UNAVAILABLE)."""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "torch-universal"))

import szl_kernels as sk
from szl_kernels.estate import ESTATE, probe_estate, probe_member


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
