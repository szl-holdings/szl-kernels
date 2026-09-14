# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""SHA3-256 linked receipts shared by SZL kernel operations.

Unanchored verification checks internal consistency, not complete history,
authorship or execution. An independently retained head/depth checkpoint can
bind the expected extent and recorded contents. The caller must authenticate
that checkpoint separately and bind it to the relevant run/source/policy.

Receipts preserve the legacy hashed fields and digest encoding. Timestamps
remain outside the hashed body; finite timestamps are not authenticated time.
Tensor fingerprints retain the existing rounded-float32 encoding; equal
fingerprints are not evidence that the original tensor bytes were equal.

Public methods return detached snapshots. This prevents ordinary aliasing from
mutating stored records, but is not protection against a hostile Python process.
No network, persistent storage, signing, training or publication is performed.
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import re
import struct
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

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
_DIGEST_CHUNK_ELEMENTS = 4096


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
        scaled = torch.round(flat * (10 ** decimals)).to(torch.int64).cpu()
        digest = hashlib.sha3_256()
        for chunk in scaled.split(_DIGEST_CHUNK_ELEMENTS):
            values = chunk.tolist()
            digest.update(struct.pack(f"<{len(values)}q", *values))
        return digest.hexdigest()
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
    it checks every sequence number, digest and prev-link. Without a separately
    retained checkpoint this establishes only internal consistency: a valid
    prefix or a fully recomputed history can still pass. Use expected_head and
    expected_depth from trusted external custody to detect such replacement.
    Timestamps are finite metadata but remain outside the hashed body. This is
    not a signature, a proof of execution, or protection from a hostile process.
    Public read methods return detached snapshots, not references to storage.
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
                # Normalize to a detached JSON value before hashing. Later caller
                # mutation cannot invalidate stored history or future prev-links.
                "attrs": self._snapshot_attrs(attrs),
                "prev": prev,
            }
            digest = self._digest_body(body)
            rec = dict(body, digest=digest, ts=time.time())
            self._records.append(rec)
            return copy.deepcopy(rec)

    @staticmethod
    def _snapshot_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(attrs, dict):
            raise TypeError("receipt attrs must be a JSON object")
        return json.loads(json.dumps(attrs, sort_keys=True, allow_nan=False))

    def checkpoint(self) -> Dict[str, Any]:
        """Atomically capture depth/head; preserve it OUTSIDE this chain.

        This unsigned dictionary has no authentication by itself. The caller
        must bind it to a separately trusted signed receipt or retained record.
        Two separate calls to count() and head() are not an atomic checkpoint.
        """
        with self._lock:
            if not self._verify_records(self._records)[0]:
                raise ValueError("cannot checkpoint an inconsistent chain")
            return {
                "schema": "szl.receipt-checkpoint/v1",
                "depth": len(self._records),
                "head": self._records[-1]["digest"] if self._records else GENESIS,
            }

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
        if type(n) is not int or n < 0:
            raise ValueError("tail length must be a nonnegative integer")
        with self._lock:
            return copy.deepcopy(self._records[-n:]) if n else []

    def kernels_touched(self) -> List[str]:
        """Distinct kernels that appear in the chain, in first-seen order.

        This summarizes the recorded kernel labels; it does not prove that
        any named implementation actually executed.
        """
        with self._lock:
            seen: List[str] = []
            for r in self._records:
                if r["kernel"] not in seen:
                    seen.append(r["kernel"])
            return seen

    @staticmethod
    def _validate_checkpoint(expected_head: Optional[str], expected_depth: Optional[int]) -> None:
        if expected_head is None and expected_depth is None:
            return
        if (not isinstance(expected_head, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected_head) is None
                or type(expected_depth) is not int or expected_depth < 0):
            raise ValueError("checkpoint requires a full lowercase head and nonnegative depth")
        if (expected_depth == 0) != (expected_head == GENESIS):
            raise ValueError("genesis checkpoint must have exactly zero depth")

    @staticmethod
    def _verify_records(
        records: Any,
        *,
        expected_head: Optional[str] = None,
        expected_depth: Optional[int] = None,
    ) -> Tuple[bool, int, int]:
        # Checkpoint argument errors are programmer errors, not corrupt history.
        UnifiedReceiptChain._validate_checkpoint(expected_head, expected_depth)
        if not isinstance(records, list):
            return (False, 0, 0)
        depth = len(records)
        fields = {"seq", "kernel", "op", "attrs", "prev", "digest", "ts"}
        prev = GENESIS
        for i, rec in enumerate(records):
            try:
                if (not isinstance(rec, dict) or set(rec) != fields
                        or type(rec["seq"]) is not int or rec["seq"] != i
                        or not isinstance(rec["kernel"], str)
                        or not isinstance(rec["op"], str)
                        or not isinstance(rec["attrs"], dict)
                        or not isinstance(rec["prev"], str)
                        or re.fullmatch(r"[0-9a-f]{64}", rec["prev"]) is None
                        or not isinstance(rec["digest"], str)
                        or re.fullmatch(r"[0-9a-f]{64}", rec["digest"]) is None
                        or type(rec["ts"]) not in (int, float)
                        or not math.isfinite(rec["ts"])):
                    return (False, depth, i)
                body = {key: rec[key] for key in ("seq", "kernel", "op", "attrs", "prev")}
                actual = UnifiedReceiptChain._digest_body(body)
                if rec["prev"] != prev or not hmac.compare_digest(rec["digest"], actual):
                    return (False, depth, i)
                prev = rec["digest"]
            except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
                return (False, depth, i)
        if expected_head is not None:
            if depth != expected_depth or not hmac.compare_digest(prev, expected_head):
                # The observed chain is internally consistent but not the
                # expected extent/history. Report the boundary, not a fake row.
                return (False, depth, min(depth, expected_depth))
        return (True, depth, -1)

    def verify(
        self,
        *,
        expected_head: Optional[str] = None,
        expected_depth: Optional[int] = None,
    ) -> Tuple[bool, int, int]:
        """Verify consistency, optionally against a separately retained checkpoint.

        Returns (ok, observed_depth, first_break). For checkpoint mismatch,
        first_break is min(observed_depth, expected_depth), an extent boundary.
        Unanchored success cannot detect valid-prefix truncation or rehashing.
        """
        with self._lock:
            return self._verify_records(
                self._records, expected_head=expected_head, expected_depth=expected_depth
            )

    def export_with_checkpoint(self) -> Tuple[str, Dict[str, Any]]:
        """Capture one export and matching checkpoint under the same lock.

        The checkpoint still needs separate authenticated custody. Keeping
        it only beside the export cannot prevent replacement of both.
        """
        with self._lock:
            checkpoint = self.checkpoint()
            return self.to_json(), checkpoint

    def to_json(self) -> str:
        """Serialize stored records, not evidence of authorship or execution.

        Preserve an independently trusted checkpoint to verify this export's
        expected head and length. A checkpoint bundled only with the export
        cannot protect against coordinated replacement of both.
        """
        with self._lock:
            return json.dumps(self._records, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @staticmethod
    def verify_json(
        blob: str,
        *,
        expected_head: Optional[str] = None,
        expected_depth: Optional[int] = None,
    ) -> Tuple[bool, int, int]:
        """Verify strict JSON, then consistency and an optional external checkpoint.

        Duplicate object keys, nonfinite numbers, malformed records and wrong
        sequence ordinals are rejected. Invalid JSON returns (False, 0, 0).
        No signature, origin, timestamp authenticity or execution claim follows.
        """
        UnifiedReceiptChain._validate_checkpoint(expected_head, expected_depth)

        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = value
            return result

        def reject_constant(value):
            raise ValueError("nonfinite JSON constant")

        if not isinstance(blob, str):
            return (False, 0, 0)
        try:
            records = json.loads(blob, object_pairs_hook=unique_object, parse_constant=reject_constant)
        except (ValueError, TypeError, OverflowError, RecursionError):
            return (False, 0, 0)
        return UnifiedReceiptChain._verify_records(
            records, expected_head=expected_head, expected_depth=expected_depth
        )
