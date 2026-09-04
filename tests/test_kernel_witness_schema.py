import json
from pathlib import Path

SCHEMA = Path(__file__).resolve().parents[1] / "frontier" / "kernel_witness_schema.json"


def test_witness_schema_holds_acceleration() -> None:
    data = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert data["promotion"] == "HOLD"
    fields = set(data["required_fields"])
    assert {"max_abs_error", "correctness_parity_pass", "compile_mode", "receipt_sha256"} <= fields
    assert "fullgraph" in data["compile_modes"]
