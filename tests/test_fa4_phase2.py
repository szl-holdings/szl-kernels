# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""Tests for the FA4 Phase 2 conditional-rescale reference op (scripts/).

CPU-only, small shapes: the reference numerics and the skip logic are the
claim under test, not GPU speed. The receipt-producing bench runs on the
device lane; here we pin correctness of both arms and the technique's
branch behavior.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys

import torch
import torch.nn.functional as F

_SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
_SCRIPT = os.path.join(_SCRIPTS, "fa4_phase2_conditional_rescale.py")
# the harness imports its Phase 1 sibling as a plain module (same-directory
# import when run as a script); make that resolve under importlib loading too
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
_SPEC = importlib.util.spec_from_file_location("fa4_phase2_conditional_rescale", _SCRIPT)
ph2 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ph2)


def _mk(B, S, H, KVH, d, seed=7, dtype=torch.bfloat16, zero_q=False):
    torch.manual_seed(seed)
    q = torch.zeros(B, H, S, d, dtype=dtype) if zero_q else \
        torch.randn(B, H, S, d, dtype=dtype)
    k = torch.randn(B, KVH, S, d, dtype=dtype)
    v = torch.randn(B, KVH, S, d, dtype=dtype)
    return q, k, v


def _ref(q, k, v):
    return F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=True,
        enable_gqa=(k.shape[1] != q.shape[1]))


def test_both_arms_match_math_reference():
    q, k, v = _mk(1, 128, 4, 2, 32)
    ref = _ref(q, k, v)
    for flag in (False, True):
        got, stats = ph2.ref_tiled_attention(q, k, v, tile=32,
                                             conditional_rescale=flag,
                                             return_stats=True)
        diff = (got.float() - ref).abs().amax().item()
        assert math.isfinite(diff) and diff <= 0.05, (flag, diff)
        assert stats["tiles"] > 0


def test_zero_q_stable_max_skips_rescales():
    # all-zero queries -> every tile shares max 0: only the first tile
    # (running max -inf) must rescale; every later tile must skip.
    q, k, v = _mk(1, 128, 2, 2, 16, zero_q=True)
    out, stats = ph2.ref_tiled_attention(q, k, v, tile=32,
                                         conditional_rescale=True,
                                         return_stats=True)
    assert stats["tiles"] == 10, stats  # (1+2+3+4) causal m0 x n0 tile pairs
    assert stats["skipped_rescales"] == stats["tiles"] - 4, stats  # first tile per m-block rescales
    assert stats["skipped_fraction"] > 0.5
    # flag off never skips
    _, stats_off = ph2.ref_tiled_attention(q, k, v, tile=32,
                                            conditional_rescale=False,
                                            return_stats=True)
    assert stats_off["skipped_rescales"] == 0


def test_gqa_expansion_matches_reference():
    q, k, v = _mk(2, 96, 8, 2, 16)  # KVH != H exercises the GQA path
    ref = _ref(q, k, v)
    got = ph2.ref_tiled_attention(q, k, v, tile=32, conditional_rescale=True)
    diff = (got.float() - ref).abs().amax().item()
    assert math.isfinite(diff) and diff <= 0.05, diff


def test_correctness_gate_reports_both_arms():
    q, k, v = _mk(1, 64, 2, 1, 16)
    for flag in (False, True):
        ok, diff, stats = ph2.correctness_gate(q, k, v, flag, tile=32)
        assert ok, (flag, diff)
        assert diff <= 0.05
        assert stats["tiles"] > 0
