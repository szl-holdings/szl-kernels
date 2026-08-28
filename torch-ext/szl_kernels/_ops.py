# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""Governed ops + a composed GovernedBlock, all writing ONE unified chain.

This module is the thin "suite" layer. It does NOT reimplement the kernels'
numerics — the canonical math lives in szl_governed_norm / szl_lambda_gate /
szl_energy_core. Here we provide:

  * self-contained REFERENCE numerics (rms_norm, layer_norm, Λ aggregate) that
    are byte-faithful ports of the published kernels, so this reference package
    RUNS standalone with only torch installed (no Hub fetch needed for the
    self-test). In production the suite delegates to the installed kernels via
    get_kernel(); see NEXT_ARTIFACT_SPEC.md "Delegation".
  * governed wrappers that emit op-agnostic receipts into a shared
    UnifiedReceiptChain (cross-kernel provenance).
  * GovernedBlock: a pre-norm transformer sub-block that composes governed-norm
    + an ADVISORY Λ gate + an energy measurement into ONE auditable forward
    pass with an end-to-end receipt.

HONESTY: Λ stays advisory (Conjecture 1, open). Energy stays MEASURED-only
(None when no NVML — never fabricated). No benchmarks are claimed.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional, Tuple

import torch

from ._chain import UnifiedReceiptChain

# ---------------------------------------------------------------------------
# Reference numerics — faithful ports of the published kernel math.
# (rms_norm / layer_norm mirror szl_governed_norm._norm; lambda_aggregate
#  mirrors szl_lambda_gate._lambda. Kept minimal but numerically identical.)
# ---------------------------------------------------------------------------
def _rms_norm(x: torch.Tensor, weight: Optional[torch.Tensor], eps: float) -> torch.Tensor:
    xf = x.to(torch.float32)
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        out = out * weight.to(torch.float32)
    return out.to(x.dtype)


def _layer_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    return torch.nn.functional.layer_norm(
        x, (x.shape[-1],), weight=weight, bias=bias, eps=eps
    )


def _lambda_aggregate(axes: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    """Weighted geometric mean Λ(x)=∏ xᵢ^{wᵢ} with non-compensatory zero-routing.

    Faithful to szl_lambda_gate: any zero or non-finite axis drives Λ to 0.
    """
    cdt = torch.float32 if axes.dtype in (torch.float16, torch.bfloat16) else axes.dtype
    xf = axes.to(cdt)
    k = xf.shape[-1]
    if weights is None:
        w = torch.full((k,), 1.0 / k, dtype=cdt, device=xf.device)
    else:
        w = weights.to(cdt)
        w = w / w.sum()
    finite = torch.isfinite(xf)
    xc = xf.clamp(0.0, 1.0)
    bad = (~finite) | (xc <= 0.0)
    any_bad = torch.any(bad, dim=-1)
    safe = torch.where(bad, torch.ones_like(xc), xc)
    val = torch.exp((torch.log(safe) * w).sum(dim=-1))
    out = torch.where(any_bad, torch.zeros_like(val), val).clamp(0.0, 1.0)
    return out.to(axes.dtype)


# ---------------------------------------------------------------------------
# Governed wrappers — numerics + a receipt into the SHARED unified chain.
# ---------------------------------------------------------------------------
def governed_rms_norm(
    chain: UnifiedReceiptChain,
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    out = _rms_norm(x, weight, eps)
    chain.emit_norm("rms_norm", x, out, eps)
    return out


def governed_layer_norm(
    chain: UnifiedReceiptChain,
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    out = _layer_norm(x, weight, bias, eps)
    chain.emit_norm("layer_norm", x, out, eps)
    return out


def governed_lambda_gate(
    chain: UnifiedReceiptChain,
    axes: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """ADVISORY Λ gate that records an advisory receipt. NOT proven trust.

    Returns a dict {score, passed, threshold, advisory=True}. ``passed`` is an
    advisory, non-compensatory signal only (Λ uniqueness = Conjecture 1, open).
    """
    threshold_value = float(threshold)
    if not math.isfinite(threshold_value) or not 0.0 <= threshold_value <= 1.0:
        raise ValueError("threshold must be finite and within [0, 1]")
    score = _lambda_aggregate(axes, weights)
    s = float(score.reshape(-1)[0]) if score.dim() else float(score)
    passed = s >= threshold_value
    chain.emit_lambda(s, threshold_value, passed, int(axes.shape[-1]))
    return {"score": s, "passed": passed, "threshold": threshold_value, "advisory": True}


def governed_measure_energy(
    chain: UnifiedReceiptChain,
    measurement: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record an energy reading. MEASURED-only honesty.

    If ``measurement`` is None we DO NOT fabricate joules: we record the honest
    UNAVAILABLE_NO_NVML label with joules=None. (In production this delegates to
    szl_energy_core.measure_energy(), which reads a real NVML cumulative-energy
    delta when a GPU is present.)
    """
    if measurement is None:
        measurement = {
            "joules": None,
            "label": "UNAVAILABLE_NO_NVML",
            "source": "no GPU/NVML in this reference run — joules NOT fabricated",
        }
    chain.emit_energy(measurement)
    return measurement


# ---------------------------------------------------------------------------
# GovernedBlock — the flagship composition (candidate (c)) on the meta-package.
# ---------------------------------------------------------------------------
class GovernedBlock:
    """A pre-norm transformer sub-block with end-to-end cross-kernel provenance.

    Forward pass, all into ONE shared chain, in order:
      1. governed RMSNorm of the hidden state            (governed_norm)
      2. an ADVISORY Λ gate over caller-supplied governance axis scores
         (lambda_gate) — recorded as advisory, never gating the math
      3. an energy measurement of the step               (energy_core)
      4. a final 'block_forward' receipt binding the block's input/output digests

    The result is a single hash-chained sequence a third party can re-walk to
    confirm: norm ran, an advisory gate was evaluated, energy was accounted
    (honestly labeled), and the block output is the one that was produced — with
    nothing inserted or reordered between steps.

    HONESTY: the Λ gate is ADVISORY; ``forward`` does NOT block, mask, or alter
    the numerics based on the gate. The gate result is recorded for audit only.
    Energy is MEASURED-only. No speed claims.
    """

    def __init__(self, chain: Optional[UnifiedReceiptChain] = None) -> None:
        self.chain = chain if chain is not None else UnifiedReceiptChain()

    def forward(
        self,
        hidden: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
        gov_axes: Optional[torch.Tensor] = None,
        gov_weights: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        energy_measurement: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        c = self.chain
        in_digest = __import__("hashlib").sha3_256(
            __import__("json").dumps(list(hidden.shape)).encode()
        ).hexdigest()

        # 1. governed normalization (numerics + receipt)
        normed = governed_rms_norm(c, hidden, weight=weight, eps=eps)

        # 2. ADVISORY Λ gate (recorded, does NOT alter numerics)
        if gov_axes is None:
            # default: neutral advisory axes; still recorded honestly as advisory
            gov_axes = torch.tensor([0.9, 0.9, 0.9], dtype=torch.float32)
        gate = governed_lambda_gate(c, gov_axes, weights=gov_weights, threshold=threshold)

        # 3. energy measurement (MEASURED-only; None when no NVML)
        energy = governed_measure_energy(c, energy_measurement)

        # 4. bind the block I/O into a final receipt
        from ._chain import tensor_digest
        c.emit(
            "governed_block",
            "block_forward",
            {
                "in_shape_digest": in_digest,
                "out_digest": tensor_digest(normed),
                "gate_passed_advisory": bool(gate["passed"]),
                "energy_label": str(energy.get("label", "UNKNOWN")),
            },
        )

        ok, depth, brk = c.verify()
        return {
            "output": normed,
            "gate": gate,                # advisory only
            "energy": energy,            # MEASURED-only label/joules
            "chain_head": c.head(),
            "chain_depth": depth,
            "chain_ok": ok,
            "chain_first_break": brk,
            "kernels_touched": c.kernels_touched(),
        }
