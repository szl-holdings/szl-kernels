#!/usr/bin/env python3
"""Wave 4 — kernel correctness pulse.

Investor read:
    Published kernel CI stays failed until a Hub republish. This file
    only checks the source contract: witness schema, honesty matrix,
    no acceleration story. It does not run torch.compile.

Developer run:
    python3 frontier/wave4.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTIER = ROOT / "frontier"


def load(name: str) -> dict:
    return json.loads((FRONTIER / name).read_text(encoding="utf-8"))


def pulse() -> dict:
    schema = load("kernel_witness_schema.json")
    matrix = load("kernel_honesty_matrix.json")
    bench = load("kernel_benchmark_manifest.json")

    if schema.get("promotion") != "HOLD":
        raise SystemExit("REFUSED: witness schema promotion is not HOLD")
    required = set(schema["required_fields"])
    if not {"max_abs_error", "correctness_parity_pass", "compile_mode", "receipt_sha256"} <= required:
        raise SystemExit("REFUSED: witness schema dropped a correctness field")
    if matrix.get("acceleration_story") is not False:
        raise SystemExit("REFUSED: honesty matrix opened an acceleration story")
    if matrix.get("hub_write") != "DENIED_IN_THIS_CHANGE":
        raise SystemExit("REFUSED: honesty matrix implies a Hub write")

    blocked = []
    for row in matrix["packages"]:
        if row.get("acceleration_claim"):
            raise SystemExit(f"REFUSED: {row['hub_id']} claims acceleration")
        if row.get("published_ci") == "GREEN" and row["hub_id"].endswith(("lambda-gate", "governed-norm", "blocked", "provctl")):
            raise SystemExit(f"REFUSED: {row['hub_id']} published_ci flipped GREEN without a Hub republish")
        blocked.append(
            {
                "hub_id": row["hub_id"],
                "published_ci": row["published_ci"],
                "maturity": row["maturity"],
            }
        )

    targets = [row["hf_kernel"] for row in bench.get("targets", [])]
    for banned in ("SZLHOLDINGS/szl-blocked", "SZLHOLDINGS/szl-provctl", "SZLHOLDINGS/szl-lambda-gate"):
        if banned in targets:
            raise SystemExit(f"REFUSED: benchmark targets include admission kernel {banned}")

    return {
        "schema": "szl.wave4-kernel/v1",
        "wave": 4,
        "source": "COMPLETE",
        "metal": "BLOCKED_NO_METAL",
        "promotion": "HOLD",
        "acceleration_story": False,
        "witness_status": schema["status"],
        "packages": blocked,
        "benchmark_targets": targets,
        "winner": None,
        "ready": False,
    }


def main() -> int:
    print(json.dumps(pulse(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
