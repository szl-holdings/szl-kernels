# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""Unified, op-agnostic, SHA3-256 hash-chained provenance for the SZL kernel suite.

THE FRONTIER GAP THIS CLOSES
----------------------------
Every governed SZL kernel today owns its OWN receipt log:
  * szl_governed_norm  -> ReceiptChain over normalization calls
  * szl_energy_core    -> CheapestWattLedger over placement decisions
  * szl_lambda_gate    -> (no chain; advisory self-checks only)
So a real forward pass that touches norm AND an advisory Λ gate AND the energy
meter produces THREE disconnected logs. There is no single artifact a third
party can re-walk to prove "these ops happened, in this order, on these
tensors, and nothing was inserted or reordered between them". That cross-op
provenance is the genuine gap the Kernel Hub leaders leave open: they optimize
FLOPs per op, not auditable provenance ACROSS ops.

``UnifiedReceiptChain`` is that missing artifact: one append-only,
SHA3-256 hash-chained log whose entries are op-agnostic. A norm call, an
advisory Λ-gate call, and an energy reading all hash-chain into the SAME chain,
in call order, so the whole forward pass verifies as one tamper-evident
sequence.

HONESTY (SZL doctrine v11)
--------------------------
* The digest is a real SHA3-256 over a canonical JSON body. It is an INTEGRITY
  fingerprint (tamper-evidence + ordering), NOT a cryptographic signature and
  NOT a proof of authorship. DSSE / sigstore signing is a separate, out-of-band
  concern and is explicitly NOT claimed here.
* Tensor fingerprints round to a fixed decimal precision so they reproduce
  across devices/dtypes for the same logical values. This is the same scheme
  szl_governed_norm already ships — we reuse it verbatim so a unified-chain
  tensor digest equals the per-kernel chain's digest for the same tensor.
* Any Λ entry is recorded as ADVISORY metadata only. A recorded "passed=True"
  is a non-compensatory advisory signal, NEVER "proven trust" (Λ uniqueness =
  Conjecture 1, OPEN).
* Any energy entry carries the kernel's honest label verbatim (MEASURED /
  SAMPLE / UNAVAILABLE_NO_NVML / ESTIMATE / UNKNOWN). The chain NEVER upgrades
  an UNAVAILABLE reading to a number — if joules is None it stays None.
* Stdlib + torch only (Kernel Hub universal-kernel requirement). Nothing is
  written to disk or the network from inside the chain.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any, Dict, List

try:  # torch is optional at import time so the chain stays inspectable headless
    import torch  # noqa: F401
    _HAS_TORCH = True
except Exception:  # pragma: no cover - exercised only in torch-less envs
    _HAS_TORCH = False

# Genesis link: 64 hex zeros, identical to szl_governed_norm and szl_energy_core
# so the unified chain's genesis is byte-compatible with the per-kernel chains.
GENESIS = "0" * 64

# Default rounding precision for tensor fingerprints. MUST match
# szl_governed_norm._receipt._tensor_digest so a unified-chain norm receipt's
# out_digest is bit-identical to the standalone kernel's.
_DECIMALS = 6


def tensor_digest(t: "Any", decimals: int = _DECIMALS) -> str:
    """Deterministic SHA3-256 over a tensor's rounded float32 contents.

    Identical scheme to szl_governed_norm._receipt._tensor_digest: round to a
    fixed number of decimals, integerize, hash the raw little-endian bytes. This
    makes the digest stable across devices/dtypes for the same logical values
    (tiny FP noise won't change it). Integrity fingerprint, NOT a signature.

    Falls back to hashing the repr only if torch is unavailable (headless
    inspection); in a real kernel run torch is always present.
    """
    if _HAS_TORCH and hasattr(t, "detach"):
        flat = t.detach().to(torch.float32).reshape(-1)
        scaled = (
            torch.round(flat * (10 ** decimals)).to(torch.int64).cpu().numpy().tobytes()
        )
        return hashlib.sha3_256(scaled).hexdigest()
    return hashlib.sha3_256(repr(t).encode("utf-8")).hexdigest()


class UnifiedReceiptChain:
    """Append-only, SHA3-256 hash-chained log spanning MANY kernel ops.

    Each receipt body is canonical JSON with a fixed, sorted schema:
        {seq, kernel, op, attrs, prev}
    where:
        seq    -- 0-based position in the chain
        kernel -- which suite member emitted it ('governed_norm', 'lambda_gate',
                  'energy_core', or a composed block name)
        op     -- the operation ('rms_norm', 'lambda_gate', 'measure_energy', ...)
        attrs  -- an op-agnostic, JSON-able dict of honest, reproducible
                  attributes (shapes, eps, out_digest, advisory Λ fields,
                  energy label/joules, ...). NEVER fabricated.
        prev   -- digest of the previous receipt (GENESIS for seq 0)

    digest = SHA3-256 over the canonical JSON of {seq,kernel,op,attrs,prev}
    (ts is excluded so the digest is reproducible offline).

    verify() re-walks the chain and returns (ok, depth, first_break_seq):
    it recomputes every digest and checks every prev-link, so ANY insertion,
    deletion, reordering, or mutation of a recorded attribute is detected.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: List[Dict[str, Any]] = []

    # -- canonical hashing (sorted keys, tight separators, strict no-NaN) --
    @staticmethod
    def _digest_body(body: Dict[str, Any]) -> str:
        raw = json.dumps(
            body, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha3_256(raw).hexdigest()

    def emit(self, kernel: str, op: str, attrs: Dict[str, Any]) -> Dict[str, Any]:
        """Append one op-agnostic receipt and return it (with digest + ts).

        ``attrs`` must be JSON-able and finite (no NaN/Inf) — a non-finite value
        would make the receipt un-attestable by a strict third-party verifier,
        so it is rejected at hash time by allow_nan=False. Callers screen
        non-finite numbers to None/UNKNOWN upstream (the energy kernel already
        does this).
        """
        with self._lock:
            prev = self._records[-1]["digest"] if self._records else GENESIS
            seq = len(self._records)
            body = {
                "seq": seq,
                "kernel": str(kernel),
                "op": str(op),
                "attrs": attrs,
                "prev": prev,
            }
            digest = self._digest_body(body)
            rec = dict(body, digest=digest, ts=time.time())
            self._records.append(rec)
            return rec

    # -- convenience emitters that enforce the honesty schema per kernel ----
    def emit_norm(self, op: str, x: "Any", out: "Any", eps: float) -> Dict[str, Any]:
        """Record a governed-norm op. out_digest matches szl_governed_norm."""
        return self.emit(
            "governed_norm",
            op,
            {
                "in_shape": list(getattr(x, "shape", [])),
                "in_dtype": str(getattr(x, "dtype", "")).replace("torch.", ""),
                "eps": float(eps),
                "out_digest": tensor_digest(out),
            },
        )

    def emit_lambda(
        self,
        score: float,
        threshold: float,
        passed: bool,
        k: int,
    ) -> Dict[str, Any]:
        """Record an ADVISORY Λ-gate evaluation.

        ``advisory`` is hard-coded True and ``lambda_status`` is stamped on every
        entry so the chain itself is self-documenting: a recorded pass is NEVER
        proven trust (Λ uniqueness = Conjecture 1, open).
        """
        return self.emit(
            "lambda_gate",
            "lambda_gate",
            {
                "score": float(score),
                "threshold": float(threshold),
                "passed": bool(passed),
                "k": int(k),
                "advisory": True,
                "lambda_status": "Conjecture 1 (open) — advisory only, NOT proven trust",
            },
        )

    def emit_energy(self, measurement: Dict[str, Any]) -> Dict[str, Any]:
        """Record an energy reading VERBATIM (label + joules as-given).

        ``joules`` may be None (UNAVAILABLE/UNKNOWN) — it is recorded as None,
        never upgraded to a fabricated number. ``label`` is the kernel's honest
        MEASURED/SAMPLE/UNAVAILABLE_NO_NVML/ESTIMATE/UNKNOWN string.
        """
        joules = measurement.get("joules", None)
        return self.emit(
            "energy_core",
            "measure_energy",
            {
                "label": str(measurement.get("label", "UNKNOWN")),
                "joules": (None if joules is None else float(joules)),
                "source": str(measurement.get("source", "")),
            },
        )

    # -- read surface --------------------------------------------------------
    def head(self) -> str:
        with self._lock:
            return self._records[-1]["digest"] if self._records else GENESIS

    def count(self) -> int:
        with self._lock:
            return len(self._records)

    def tail(self, n: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._records[-n:])

    def kernels_touched(self) -> List[str]:
        """Distinct kernels that appear in the chain, in first-seen order.

        The cross-kernel provenance summary: proof that a single auditable chain
        actually spanned multiple suite members in one run.
        """
        with self._lock:
            seen: List[str] = []
            for r in self._records:
                if r["kernel"] not in seen:
                    seen.append(r["kernel"])
            return seen

    def verify(self):
        """Re-walk the chain. Returns (ok: bool, depth: int, first_break: int).

        Recomputes every digest from {seq,kernel,op,attrs,prev} and checks each
        prev-link. Detects insertion, deletion, reorder, or any attribute
        mutation. first_break is the seq of the first bad record, or -1 if clean.
        """
        with self._lock:
            prev = GENESIS
            for i, rec in enumerate(self._records):
                body = {k: rec[k] for k in ("seq", "kernel", "op", "attrs", "prev")}
                if rec["prev"] != prev or rec["digest"] != self._digest_body(body):
                    return (False, len(self._records), i)
                prev = rec["digest"]
            return (True, len(self._records), -1)

    def to_json(self) -> str:
        """Export the full chain as canonical JSON for OFFLINE re-verification.

        A third party can load this, recompute each digest, and confirm the
        chain independently — no trust in the emitting process required.
        """
        with self._lock:
            return json.dumps(self._records, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def verify_json(blob: str):
        """Verify an exported chain offline. Returns (ok, depth, first_break)."""
        records = json.loads(blob)
        prev = GENESIS
        for i, rec in enumerate(records):
            body = {k: rec[k] for k in ("seq", "kernel", "op", "attrs", "prev")}
            if rec["prev"] != prev or rec["digest"] != UnifiedReceiptChain._digest_body(body):
                return (False, len(records), i)
            prev = rec["digest"]
        return (True, len(records), -1)
