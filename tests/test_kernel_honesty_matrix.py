import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "frontier" / "kernel_honesty_matrix.json"
BENCH = ROOT / "frontier" / "kernel_benchmark_manifest.json"

REQUIRED = {
    "SZLHOLDINGS/szl-lambda-gate",
    "SZLHOLDINGS/szl-governed-norm",
    "SZLHOLDINGS/szl-blocked",
    "SZLHOLDINGS/szl-provctl",
}
LEGAL_CI = {"FAILED", "BLOCKED", "UNKNOWN", "GREEN"}
LEGAL_MATURITY = {"BLOCKED", "SOFTWARE_LIMITED", "SOFTWARE", "CANDIDATE"}
LEGAL_UNSAFE_INVENTORY = {
    "NOT_IN_SCOPE",
    "CONFIRMED_UNSAFE",
    "PARTIAL_VERIFICATION",
    "UNVERIFIED",
    "VERIFIED_CLEAN",
}


def test_honesty_matrix_fail_closed() -> None:
    data = json.loads(MATRIX.read_text(encoding="utf-8"))
    assert data["schema"] == "szl.kernel-honesty-matrix/v1"
    assert data["hub_write"] == "DENIED_IN_THIS_CHANGE"
    assert data["acceleration_story"] is False
    assert "empty list is not a clean-provider claim" in data["unsafe_inventory_policy"]

    seen = {row["hub_id"] for row in data["packages"]}
    assert REQUIRED <= seen

    for row in data["packages"]:
        assert row["published_ci"] in LEGAL_CI
        assert row["maturity"] in LEGAL_MATURITY
        assert row["acceleration_claim"] is False
        status = row["unsafe_inventory_status"]
        assert status in LEGAL_UNSAFE_INVENTORY

        if "C4" in row["clusters"]:
            assert status != "NOT_IN_SCOPE"
            if row["unsafe_files"]:
                assert status == "CONFIRMED_UNSAFE"
            if row["published_ci"] == "UNKNOWN":
                assert status != "VERIFIED_CLEAN"
        else:
            assert status == "NOT_IN_SCOPE"
            assert row["unsafe_files"] == []

        if row["hub_id"] in REQUIRED:
            assert row["published_ci"] != "GREEN"
            assert row["maturity"] != "CANDIDATE"


def test_benchmark_targets_respect_all_declared_promotion_exclusions() -> None:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    bench = json.loads(BENCH.read_text(encoding="utf-8"))

    package_ids = {row["hub_id"] for row in matrix["packages"]}
    excluded = set(bench["excluded_from_promotion"])
    targets = {row["hf_kernel"] for row in bench["targets"]}

    assert package_ids == excluded
    assert targets.isdisjoint(excluded)
    assert bench["status"] == "EXPERIMENTAL"
