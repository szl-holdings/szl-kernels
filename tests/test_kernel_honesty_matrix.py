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


def test_honesty_matrix_fail_closed() -> None:
    data = json.loads(MATRIX.read_text(encoding="utf-8"))
    assert data["schema"] == "szl.kernel-honesty-matrix/v1"
    assert data["hub_write"] == "DENIED_IN_THIS_CHANGE"
    assert data["acceleration_story"] is False
    seen = {row["hub_id"] for row in data["packages"]}
    assert REQUIRED <= seen
    for row in data["packages"]:
        assert row["published_ci"] in LEGAL_CI
        assert row["maturity"] in LEGAL_MATURITY
        assert row["acceleration_claim"] is False
        if row["hub_id"] in REQUIRED:
            assert row["published_ci"] != "GREEN"
            assert row["maturity"] != "CANDIDATE"


def test_benchmark_targets_are_not_the_blocked_admission_kernels() -> None:
    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    targets = {row["hf_kernel"] for row in bench["targets"]}
    assert "SZLHOLDINGS/szl-blocked" not in targets
    assert "SZLHOLDINGS/szl-provctl" not in targets
    assert "SZLHOLDINGS/szl-lambda-gate" not in targets
    assert bench["status"] == "EXPERIMENTAL"
