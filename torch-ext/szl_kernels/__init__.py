# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""szl_kernels — SZL Holdings' unified governed-kernel SUITE.

ONE get_kernel-discoverable package that ties the three governed kernels plus
the energy meter into a single suite with a SHARED, op-agnostic
``UnifiedReceiptChain`` — cross-kernel provenance in one auditable forward pass.
This is the gap the Kernel Hub leaders leave open: they optimize FLOPs per op;
SZL governs provenance ACROSS ops.

Suite members (each independently published as its own Hub kernel):
  * governed_norm  -> SZLHOLDINGS/szl-governed-norm   (RMSNorm/LayerNorm + receipts)
  * lambda_gate    -> SZLHOLDINGS/szl-lambda-gate     (advisory Λ gate; Conjecture 1)
  * energy_core    -> szl_energy_core / governed-inference-meter (MEASURED joules)
  * govsign        -> SZLHOLDINGS/szl-govsign         (signed governance attestation; DSSE / in-toto)
  * blocked        -> SZLHOLDINGS/szl-blocked         (honest-BLOCKED first-class state + EU AI Act Annex IV)
  * provctl        -> SZLHOLDINGS/szl-provctl         (provenance-DAG verify + in-toto v1 / SLSA v1 interop + per-kernel MEASURED energy)

The first three are the NUMERIC core exercised by the shared forward pass.
govsign + blocked are GOVERNANCE-LAYER series companions: govsign makes a
verdict third-party-verifiable (sign/verify offline), and blocked makes a
refusal a first-class, provenanced state (the op never runs — no fake-green)
and derives an EU AI Act Annex IV DRAFT skeleton from the same chain.

Companion table (NOT a numeric kernel, NOT a transformer LM, NOT Chaski/Khipu):
  * MiniEmbed      -> SZL-MiniEmbed v1 (PPMI+TruncatedSVD word-embedding table)

Discover + use the suite:

    from kernels import get_kernel
    suite = get_kernel("SZLHOLDINGS/szl-kernels", revision="main", trust_remote_code=True)
    print(suite.list_kernels())            # {'governed_norm':..., 'lambda_gate':..., 'energy_core':...}
    print(suite.list_series())             # {'govsign':..., 'blocked':..., 'provctl':...}
    print(suite.selfcheck())               # one-shot suite health (CPU only)

    # ONE shared chain spanning multiple ops:
    chain = suite.UnifiedReceiptChain()
    import torch
    x = torch.randn(4, 64)
    y = suite.governed_rms_norm(chain, x, eps=1e-6)
    gate = suite.governed_lambda_gate(chain, torch.tensor([0.9, 0.8, 0.95]), threshold=0.5)
    e = suite.governed_measure_energy(chain)   # None joules when no NVML — never faked
    ok, depth, brk = chain.verify()            # the WHOLE pass verifies as one chain
    print(chain.kernels_touched())             # ['governed_norm','lambda_gate','energy_core']

    # Flagship composition — a governed transformer sub-block:
    blk = suite.GovernedBlock()
    res = blk.forward(x, gov_axes=torch.tensor([0.95, 0.9, 0.92]))
    print(res["chain_ok"], res["chain_depth"], res["kernels_touched"])

HONESTY (SZL doctrine v11):
  * Λ is the weighted-geometric-mean ADVISORY aggregator; its uniqueness is
    Conjecture 1 (OPEN). A recorded gate "pass" is advisory, NEVER proven trust.
  * Energy joules are MEASURED-only (real NVML delta). No GPU/NVML => None /
    UNAVAILABLE_NO_NVML, NEVER a fabricated joule.
  * Receipt digests are SHA3-256 integrity fingerprints — tamper-evidence +
    ordering, NOT signatures and NOT proof of authorship.
  * This is a UNIVERSAL (pure-Python) suite: a correctness + provenance
    reference, not a hand-tuned CUDA speed record. No fabricated benchmarks.
"""
from typing import Any, Dict

import torch

from ._chain import GENESIS, UnifiedReceiptChain, tensor_digest
from ._ops import (
    GovernedBlock,
    governed_layer_norm,
    governed_lambda_gate,
    governed_measure_energy,
    governed_rms_norm,
)
from .miniembed import MiniEmbed, PUBLISHED_SHA256

__all__ = [
    "UnifiedReceiptChain",
    "tensor_digest",
    "GENESIS",
    "governed_rms_norm",
    "governed_layer_norm",
    "governed_lambda_gate",
    "governed_measure_energy",
    "GovernedBlock",
    "MiniEmbed",
    "PUBLISHED_SHA256",
    "list_kernels",
    "list_series",
    "get_member",
    "selfcheck",
    "DOCTRINE_FOOTER",
    "PROVENANCE",
    "__version__",
]

__version__ = "0.1.1"

DOCTRINE_FOOTER = (
    "SZL Holdings · unified governed-kernel suite · cross-kernel provenance · "
    "Λ = Conjecture 1 (advisory) · energy MEASURED-only · honesty over checklist"
)

# Suite registry: maps a suite-member key to its Hub coordinates + role. In
# production, get_member() would resolve these via kernels.get_kernel(); the
# reference package ships byte-faithful numerics so it runs standalone.
_REGISTRY: Dict[str, Dict[str, str]] = {
    "governed_norm": {
        "hub_id": "SZLHOLDINGS/szl-governed-norm",
        "role": "RMSNorm/LayerNorm + SHA3-256 hash-chained receipts",
        "honesty": "integrity fingerprint, not a signature",
    },
    "lambda_gate": {
        "hub_id": "SZLHOLDINGS/szl-lambda-gate",
        "role": "advisory Λ weighted-geometric-mean gate",
        "honesty": "Λ uniqueness = Conjecture 1 (open); advisory, NOT proven trust",
    },
    "energy_core": {
        "hub_id": "SZLHOLDINGS/governed-inference-meter",
        "role": "MEASURED-joule energy accounting (real NVML delta)",
        "honesty": "no GPU/NVML => UNAVAILABLE_NO_NVML, joules None — never fabricated",
    },
}

# Governance-layer series companions (not part of the numeric forward pass).
# Each is independently published and operates on the SAME UnifiedReceiptChain
# provenance, extending the suite from "govern provenance ACROSS ops" to
# "sign that provenance" and "refuse-and-record honestly."
_SERIES: Dict[str, Dict[str, str]] = {
    "govsign": {
        "hub_id": "SZLHOLDINGS/szl-govsign",
        "role": "signed governance attestation (DSSE / in-toto, ECDSA P-256)",
        "honesty": "signature proves authorship+integrity, NOT Λ uniqueness; proven_trust locked False; a BLOCKED verdict is signed as BLOCKED",
    },
    "blocked": {
        "hub_id": "SZLHOLDINGS/szl-blocked",
        "role": "honest-BLOCKED first-class state + EU AI Act Annex IV DRAFT derivation",
        "honesty": "a BLOCKED op never executes (no fake-green); Annex IV output is a DRAFT skeleton, NOT legal advice",
    },
    "provctl": {
        "hub_id": "SZLHOLDINGS/szl-provctl",
        "role": "provenance-DAG verify + in-toto v1 / SLSA v1 interop + per-kernel MEASURED energy",
        "honesty": "in-toto/SLSA field names spec-exact; proven_trust locked False; energy MEASURED-only (never fabricated); honest-BLOCKED DAG nodes surfaced, never dropped",
    },
}

PROVENANCE = {
    "suite": "szl_kernels",
    "members": _REGISTRY,
    "series_companions": _SERIES,
    "miniembed": {
        "kind": "distributional_word_embedding_table",
        "not": ["transformer_lm", "chaski", "khipu"],
        "files": {"vectors": "vectors.npz", "vocab": "vocab.json"},
        "method": "PPMI+TruncatedSVD",
        "receipt_chain": "UnifiedReceiptChain",
    },
    "lean_repo": "szl-holdings/lutar-lean",
    "doi_lutar_lean": "10.5281/zenodo.20434308",
    "lambda_status": "Conjecture 1 (open) — uniqueness unproven; advisory only",
    "shared_chain": "UnifiedReceiptChain — op-agnostic SHA3-256 cross-kernel provenance",
}


def list_kernels() -> Dict[str, Dict[str, str]]:
    """Return the suite registry: member key -> {hub_id, role, honesty}."""
    return {k: dict(v) for k, v in _REGISTRY.items()}


def list_series() -> Dict[str, Dict[str, str]]:
    """Return the governance-layer series companions (govsign, blocked, provctl).

    These extend the numeric suite with signed attestation (govsign),
    honest-BLOCKED + EU AI Act Annex IV derivation (blocked), and provenance-DAG
    verification + in-toto/SLSA interop + per-kernel MEASURED energy (provctl).
    They operate on the same UnifiedReceiptChain but are not part of the numeric
    forward pass.
    """
    return {k: dict(v) for k, v in _SERIES.items()}


def get_member(key: str) -> Dict[str, str]:
    """Resolve one suite member's Hub coordinates + honest role string."""
    if key not in _REGISTRY:
        raise KeyError(f"unknown suite member {key!r}; have {list(_REGISTRY)}")
    return dict(_REGISTRY[key])


def selfcheck() -> Dict[str, Any]:
    """One-shot CPU-only suite health check; never raises.

    Proves the suite's CENTRAL claim end-to-end: a single shared chain spans
    governed_norm + lambda_gate + energy_core in ONE forward pass, verifies as
    one tamper-evident sequence, AND a deliberately mutated copy fails
    verification (so the tamper-evidence is real, not decorative).

    Returns {ok, version, checks:{...}, chain_head, kernels_touched,
             tamper_detected, error}.
    """
    checks: Dict[str, bool] = {}
    error = None
    chain_head = GENESIS
    kernels_touched = []
    tamper_detected = False
    try:
        torch.manual_seed(0)
        x = torch.randn(4, 64, dtype=torch.float32)
        w = torch.randn(64, dtype=torch.float32)

        chain = UnifiedReceiptChain()

        # norm matches a fp32 RMSNorm reference
        y = governed_rms_norm(chain, x, weight=w, eps=1e-6)
        ref = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)) * w
        checks["norm_correct"] = bool(torch.allclose(y, ref, rtol=1e-5, atol=1e-5))

        # advisory Λ gate runs and is recorded as advisory
        gate = governed_lambda_gate(
            chain, torch.tensor([0.9, 0.8, 0.95]), threshold=0.5
        )
        checks["lambda_advisory"] = bool(gate["advisory"] is True)

        # energy is MEASURED-only: no NVML here => joules None, honest label
        e = governed_measure_energy(chain)
        checks["energy_honest"] = bool(
            e["joules"] is None and e["label"] == "UNAVAILABLE_NO_NVML"
        )

        # the WHOLE multi-op pass verifies as ONE chain
        ok, depth, brk = chain.verify()
        checks["cross_kernel_verify"] = bool(ok and depth == 3 and brk == -1)
        chain_head = chain.head()
        kernels_touched = chain.kernels_touched()
        checks["spans_three_kernels"] = bool(
            kernels_touched == ["governed_norm", "lambda_gate", "energy_core"]
        )

        # offline re-verification of the exported chain
        ok2, _, _ = UnifiedReceiptChain.verify_json(chain.to_json())
        checks["offline_reverify"] = bool(ok2)

        # tamper-evidence is REAL: mutate one recorded attr and confirm break
        blob = chain.to_json()
        import json as _json
        recs = _json.loads(blob)
        recs[0]["attrs"]["eps"] = 9.99e-3  # flip a recorded value
        ok3, _, brk3 = UnifiedReceiptChain.verify_json(_json.dumps(recs))
        tamper_detected = bool((not ok3) and brk3 == 0)
        checks["tamper_detected"] = tamper_detected

        # GovernedBlock composition runs and self-verifies
        blk = GovernedBlock()
        res = blk.forward(x, weight=w, gov_axes=torch.tensor([0.95, 0.9, 0.92]))
        checks["block_forward"] = bool(
            res["chain_ok"]
            and res["chain_depth"] == 4
            and res["kernels_touched"]
            == ["governed_norm", "lambda_gate", "energy_core", "governed_block"]
        )
    except Exception as exc:  # never raise from a health probe
        error = f"{type(exc).__name__}: {exc}"

    ok = bool(checks) and all(checks.values()) and error is None
    return {
        "ok": ok,
        "version": __version__,
        "checks": checks,
        "chain_head": chain_head,
        "kernels_touched": kernels_touched,
        "tamper_detected": tamper_detected,
        "error": error,
    }
