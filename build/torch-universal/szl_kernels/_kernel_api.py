# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SZL Holdings
"""Pure-Torch Kernel Hub API, separate from the broader Python distribution.

This entrypoint preserves the fifteen public exports of the published v1
CPU kernel and adds governed_cosine_topk. The release builder stages this file
as the variant's __init__.py; it must never import the distribution entrypoint.
MiniEmbed, estate clients and fitted navigator models are separate companions.
Receipts establish hash consistency, not signatures, authorship or quality.
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
from .retrieval import governed_cosine_topk

__all__ = [
    "UnifiedReceiptChain",
    "tensor_digest",
    "GENESIS",
    "governed_rms_norm",
    "governed_layer_norm",
    "governed_lambda_gate",
    "governed_measure_energy",
    "GovernedBlock",
    "list_kernels",
    "list_series",
    "get_member",
    "selfcheck",
    "DOCTRINE_FOOTER",
    "PROVENANCE",
    "__version__",
    "governed_cosine_topk",
]

__version__ = "0.2.0"

DOCTRINE_FOOTER = (
    "SZL Holdings · unified governed-kernel suite · cross-kernel provenance · "
    "Λ = Conjecture 1 (advisory) · energy MEASURED-only · honesty over checklist"
)

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
    "lean_repo": "szl-holdings/lutar-lean",
    "doi_lutar_lean": "10.5281/zenodo.20434308",
    "lambda_status": "Conjecture 1 (open) — uniqueness unproven; advisory only",
    "shared_chain": "UnifiedReceiptChain — op-agnostic SHA3-256 cross-kernel provenance",
}


def list_kernels() -> Dict[str, Dict[str, str]]:
    """Return numeric member coordinates; this function performs no Hub access."""
    return {key: dict(value) for key, value in _REGISTRY.items()}


def list_series() -> Dict[str, Dict[str, str]]:
    """Return companion coordinates without importing or executing companions."""
    return {key: dict(value) for key, value in _SERIES.items()}


def get_member(key: str) -> Dict[str, str]:
    """Resolve a numeric suite member's existing v1 registry entry."""
    if key not in _REGISTRY:
        raise KeyError(f"unknown suite member {key!r}; have {list(_REGISTRY)}")
    return dict(_REGISTRY[key])


def selfcheck() -> Dict[str, Any]:
    """Run the v1 CPU checks and retrieval check; return errors without raising."""
    checks: Dict[str, bool] = {}
    error = None
    chain_head = GENESIS
    kernels_touched = []
    tamper_detected = False
    try:
        generator = torch.Generator(device="cpu").manual_seed(0)
        x = torch.randn(4, 64, generator=generator, dtype=torch.float32, device="cpu")
        weight = torch.randn(64, generator=generator, dtype=torch.float32, device="cpu")
        chain = UnifiedReceiptChain()
        output = governed_rms_norm(chain, x, weight=weight, eps=1e-6)
        reference = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)) * weight
        checks["norm_correct"] = bool(torch.allclose(output, reference, rtol=1e-5, atol=1e-5))
        gate = governed_lambda_gate(chain, torch.tensor([0.9, 0.8, 0.95], device="cpu"), threshold=0.5)
        checks["lambda_advisory"] = gate["advisory"] is True
        energy = governed_measure_energy(chain)
        checks["energy_honest"] = energy["joules"] is None and energy["label"] == "UNAVAILABLE_NO_NVML"
        checks["cross_kernel_verify"] = chain.verify() == (True, 3, -1)
        chain_head = chain.head()
        kernels_touched = chain.kernels_touched()
        checks["spans_three_kernels"] = kernels_touched == ["governed_norm", "lambda_gate", "energy_core"]
        checks["offline_reverify"] = bool(UnifiedReceiptChain.verify_json(chain.to_json())[0])
        import json
        records = json.loads(chain.to_json())
        records[0]["attrs"]["eps"] = 9.99e-3
        tampered_ok, _, first_break = UnifiedReceiptChain.verify_json(json.dumps(records))
        tamper_detected = not tampered_ok and first_break == 0
        checks["tamper_detected"] = tamper_detected
        block = GovernedBlock().forward(x, weight=weight, gov_axes=torch.tensor([0.95, 0.9, 0.92], device="cpu"))
        checks["block_forward"] = bool(
            block["chain_ok"] and block["chain_depth"] == 4
            and block["kernels_touched"] == ["governed_norm", "lambda_gate", "energy_core", "governed_block"]
        )
        retrieval_chain = UnifiedReceiptChain()
        result = governed_cosine_topk(
            retrieval_chain, torch.tensor([1.0, 0.0], device="cpu"),
            torch.tensor([[0.0, 1.0], [1.0, 0.0]], device="cpu"), k=1, block_rows=1,
        )
        checks["retrieval_correct"] = result["indices"].tolist() == [[1]]
        checks["retrieval_receipt"] = retrieval_chain.verify() == (True, 1, -1)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "ok": bool(checks) and all(checks.values()) and error is None,
        "version": __version__,
        "checks": checks,
        "chain_head": chain_head,
        "kernels_touched": kernels_touched,
        "tamper_detected": tamper_detected,
        "error": error,
    }
