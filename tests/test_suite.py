# SPDX-License-Identifier: Apache-2.0
"""Suite tests for szl_kernels — cross-kernel provenance, advisory Λ, MEASURED energy."""
import sys, os, json, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "torch-universal"))
import torch
import szl_kernels as sk


def test_selfcheck_all_pass():
    r = sk.selfcheck()
    assert r["ok"] is True, r
    assert all(r["checks"].values()), r["checks"]
    assert r["kernels_touched"] == ["governed_norm", "lambda_gate", "energy_core"]


def test_one_chain_spans_three_kernels():
    chain = sk.UnifiedReceiptChain()
    x = torch.randn(2, 32)
    sk.governed_rms_norm(chain, x, eps=1e-6)
    sk.governed_lambda_gate(chain, torch.tensor([0.9, 0.8, 0.95]), threshold=0.5)
    sk.governed_measure_energy(chain)
    ok, depth, brk = chain.verify()
    assert ok and depth == 3 and brk == -1
    assert chain.kernels_touched() == ["governed_norm", "lambda_gate", "energy_core"]


def test_tensor_digest_matches_legacy_little_endian_int64_bytes():
    tensor = torch.tensor([-2.0, -0.25, 0.0, 0.5, 3.0], dtype=torch.float32)
    scaled = (-2000000, -250000, 0, 500000, 3000000)
    legacy_bytes = b"".join(
        value.to_bytes(8, byteorder="little", signed=True) for value in scaled
    )
    assert sk.tensor_digest(tensor) == hashlib.sha3_256(legacy_bytes).hexdigest()


def test_tensor_digest_preserves_bytes_across_chunk_boundaries():
    tensor = torch.arange(5000, dtype=torch.float32) / 8
    scaled = torch.round(tensor * 1000000).to(torch.int64).tolist()
    legacy_bytes = b"".join(
        value.to_bytes(8, byteorder="little", signed=True) for value in scaled
    )
    assert sk.tensor_digest(tensor) == hashlib.sha3_256(legacy_bytes).hexdigest()


def test_lambda_is_advisory():
    chain = sk.UnifiedReceiptChain()
    g = sk.governed_lambda_gate(chain, torch.tensor([0.9, 0.9, 0.9]))
    assert g["advisory"] is True
    rec = chain.tail(1)[0]
    assert rec["attrs"]["advisory"] is True
    assert "Conjecture 1" in rec["attrs"]["lambda_status"]


def test_lambda_threshold_must_be_finite_and_bounded():
    for threshold in (-0.1, 1.1, float("nan"), float("inf"), float("-inf")):
        chain = sk.UnifiedReceiptChain()
        try:
            sk.governed_lambda_gate(
                chain,
                torch.tensor([0.0, 1.0, 1.0]),
                threshold=threshold,
            )
        except ValueError as exc:
            assert str(exc) == "threshold must be finite and within [0, 1]"
        else:
            raise AssertionError(f"invalid threshold accepted: {threshold!r}")
        assert chain.tail(1) == []
    assert sk.governed_lambda_gate(
        sk.UnifiedReceiptChain(), torch.tensor([0.0, 1.0, 1.0]), threshold=0.0
    )["passed"]
    assert sk.governed_lambda_gate(
        sk.UnifiedReceiptChain(), torch.tensor([1.0, 1.0, 1.0]), threshold=1.0
    )["passed"]


def test_energy_measured_only_never_fabricated():
    chain = sk.UnifiedReceiptChain()
    e = sk.governed_measure_energy(chain)  # no NVML in CI
    assert e["joules"] is None
    assert e["label"] == "UNAVAILABLE_NO_NVML"


def test_tamper_is_detected():
    chain = sk.UnifiedReceiptChain()
    x = torch.randn(2, 16)
    sk.governed_rms_norm(chain, x, eps=1e-6)
    sk.governed_measure_energy(chain)
    recs = json.loads(chain.to_json())
    recs[0]["attrs"]["eps"] = 9.99e-3
    ok, _, brk = sk.UnifiedReceiptChain.verify_json(json.dumps(recs))
    assert ok is False and brk == 0


def test_governed_block_composes_and_verifies():
    blk = sk.GovernedBlock()
    res = blk.forward(torch.randn(2, 32), gov_axes=torch.tensor([0.95, 0.9, 0.92]))
    assert res["chain_ok"] and res["chain_depth"] == 4
    assert res["kernels_touched"][-1] == "governed_block"


def test_norm_matches_reference():
    chain = sk.UnifiedReceiptChain()
    x = torch.randn(4, 64); w = torch.randn(64)
    y = sk.governed_rms_norm(chain, x, weight=w, eps=1e-6)
    ref = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)) * w
    assert torch.allclose(y, ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    p = f = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS {fn.__name__}"); p += 1
        except Exception:
            print(f"  FAIL {fn.__name__}"); traceback.print_exc(); f += 1
    print(f"\n{p} passed, {f} failed")
    sys.exit(1 if f else 0)
