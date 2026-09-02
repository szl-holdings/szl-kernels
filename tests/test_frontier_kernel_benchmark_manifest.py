import json
from pathlib import Path


MANIFEST = Path("frontier/kernel_benchmark_manifest.json")


def _load():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_is_explicitly_non_promotional():
    data = _load()
    assert data["status"] == "EXPERIMENTAL"
    assert "No performance claim" in data["claim_policy"]


def test_frontier_targets_cover_published_szl_attention_kernels():
    ids = {item["id"] for item in _load()["targets"]}
    assert {"YARQA-ATTN", "szl-receipt-attn", "szl-block-kv", "szl-maskmod"} <= ids


def test_every_target_has_reference_and_candidate_backends():
    for target in _load()["targets"]:
        assert target["reference_backend"] == "torch"
        assert target["candidate_backends"]
        assert target["hf_kernel"].startswith("SZLHOLDINGS/")


def test_measurement_contract_is_hardware_source_and_correctness_bound():
    fields = set(_load()["required_measurements"])
    required = {
        "source_revision",
        "kernel_revision",
        "hardware_fingerprint",
        "software_fingerprint",
        "p50_us",
        "p95_us",
        "throughput",
        "peak_memory_bytes",
        "max_abs_error",
        "correctness_parity_pass",
        "receipt_sha256",
    }
    assert required <= fields
