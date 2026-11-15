"""FA4 Phase 2 technique receipts (szl-kernels docs/flash-attention-4-study.md #3).

ONE transferable technique from FlashAttention-4 (arXiv:2603.05451):
conditional softmax rescaling. In online-softmax attention every K/V tile
forces a rescale of the running accumulator by exp(running_max - new_max).
When the running maximum is stable across tiles (the common case after the
first few tiles), that correction is ~1 and the multiply can be skipped.

Measured-or-null doctrine, mirroring Phase 1:
- The flag lives on the reference tiled-attention op in THIS harness. The
  estate kernel suite (torch-ext/szl_kernels) has no attention op yet, so
  there is no "estate attention kernel" to flag; that integration is
  follow-up work (issue #76), not a claim made here.
- Both arms (flag off = always rescale, flag on = conditional rescale) run
  the SAME code path with the SAME per-tile host sync for the tile-level
  branch decision, so the only delta measured is the technique itself.
  The per-tile host sync is a reference-harness artifact present in BOTH
  arms; a fused kernel decides this on-device without a sync. Absolute
  reference-arm speed is therefore NOT the claim - Phase 1 native SDPA
  baselines remain the estate's fast paths.
- Timing uses PH2_WARMUP/PH2_ITERS (2/5) instead of Phase 1's 5/25: the
  tiled loop's per-tile syncs make 25 iterations cost minutes per cell
  without changing the within-arm comparison. Recorded in params.
- Null results are receipts: if conditional rescaling does not help at
  the torch reference level, that is recorded, not hidden.
- Correctness gate vs an fp32 math reference runs BEFORE any benching, on
  a B=1 / S<=1024 slice of the cell (the fp32 reference materializes S^2
  scores and OOMs at full battery shapes on an 8 GB GPU); the technique's
  skip-fraction stats are computed at the FULL shape. A failed gate records
  a null for the cell, never a timed number.

Receipts: kernel id, shapes, dtype, TFLOP/s, joules per call (measured-or-
null), skipped-rescale fraction, correctness deltas. No claims beyond what
is measured.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
from functools import partial

import torch
import torch.nn.functional as F

from fa4_phase1_baselines import (
    BATCHES,
    DTYPES,
    FAMILIES,
    SEQ_LENS,
    bench_cell,
    flops_causal,
    io_bytes,
)

TILE = 128
# Skip the rescale when new_max - running_max <= DELTA_MAX. exp(-5e-4)
# ~ 0.9995: compounding ~1.6% over 32 tiles worst case, inside bf16
# rounding noise (bf16 eps ~0.0078). Fail-closed default.
DELTA_MAX = 5e-4
# Reduced iteration counts for the reference arms (see module docstring).
PH2_WARMUP = 2
PH2_ITERS = 5


def _causal_scores(q, k_tile, m0, n0, scale):
    # q: (B, H, tm, d) rows [m0, m0+tm); k_tile: (B, H, tn, d) cols [n0, n0+tn)
    s = torch.matmul(q, k_tile.transpose(-1, -2)) * scale
    rows = m0 + torch.arange(q.shape[-2], device=q.device).view(-1, 1)
    cols = n0 + torch.arange(k_tile.shape[-2], device=q.device).view(1, -1)
    s = s.masked_fill(cols > rows, float("-inf"))
    return s.float()


def ref_tiled_attention(q, k, v, *, conditional_rescale=False,
                        delta_max=DELTA_MAX, tile=TILE, return_stats=False):
    """Reference online-softmax tiled attention (torch, bf16 in / fp32 softmax).

    conditional_rescale=False: classic flash-style update, rescale every tile.
    conditional_rescale=True:  skip the accumulator/denominator rescale when
        the tile-level max delta is within delta_max (the FA4 technique).

    Both arms run the identical tile-level branch test (one host sync per
    tile) so the comparison isolates the rescale itself. Skip is tile-level
    (all rows stable), matching a real kernel's CTA/warp-uniform branch.
    """
    B, H, S, d = q.shape
    KVH = k.shape[1]
    scale = 1.0 / math.sqrt(d)
    # GQA: expand KV heads to query heads (reference numerics only)
    if KVH != H:
        rep = H // KVH
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    acc = torch.zeros(B, H, S, v.shape[-1], dtype=torch.float32, device=q.device)
    denom = torch.zeros(B, H, S, dtype=torch.float32, device=q.device)
    run_max = torch.full((B, H, S), float("-inf"), dtype=torch.float32, device=q.device)
    tiles = 0
    skipped = 0
    for n0 in range(0, S, tile):
        tn = min(tile, S - n0)
        k_tile = k[:, :, n0:n0 + tn, :]
        v_tile = v[:, :, n0:n0 + tn, :]
        for m0 in range(0, S, tile):
            tm = min(tile, S - m0)
            # causal: tile (m0..m0+tm) x (n0..n0+tn) only if n0 <= m0+tm-1
            if n0 > m0 + tm - 1:
                continue
            q_blk = q[:, :, m0:m0 + tm, :]
            s = _causal_scores(q_blk, k_tile, m0, n0, scale)  # (B,H,tm,tn) fp32
            tile_max = s.amax(dim=-1)  # (B,H,tm)
            old_max = run_max[:, :, m0:m0 + tm]
            new_max = torch.maximum(old_max, tile_max)
            # tile-level branch test: identical in BOTH arms (honest delta)
            delta = (new_max - old_max).amax().item()
            do_skip = bool(conditional_rescale
                           and math.isfinite(delta) and delta <= delta_max)
            if do_skip:
                # running max is stable: keep the OLD frame (defer the
                # exp-frame switch entirely); acc/denom stay unscaled and
                # run_max stays put. Since out = acc/denom is frame-
                # invariant and every skipped tile satisfies
                # tile_max <= run_max + delta_max, exp(s - run_max) is
                # bounded by exp(delta_max) ~ 1.0005: no overflow, no
                # compounding error (exact up to fp32 rounding).
                p = torch.exp(s - old_max.unsqueeze(-1))
                skipped += 1
            else:
                p = torch.exp(s - new_max.unsqueeze(-1))
                corr = torch.exp(old_max - new_max)
                acc[:, :, m0:m0 + tm, :] = acc[:, :, m0:m0 + tm, :] * corr.unsqueeze(-1)
                denom[:, :, m0:m0 + tm] = denom[:, :, m0:m0 + tm] * corr
                run_max[:, :, m0:m0 + tm] = new_max
            pv = torch.matmul(p.to(v_tile.dtype), v_tile).float()
            ps = p.sum(dim=-1)
            acc[:, :, m0:m0 + tm, :] = acc[:, :, m0:m0 + tm, :] + pv
            denom[:, :, m0:m0 + tm] = denom[:, :, m0:m0 + tm] + ps
            tiles += 1
    out = (acc / denom.unsqueeze(-1)).to(q.dtype)
    if return_stats:
        return out, {"tiles": tiles, "skipped_rescales": skipped,
                     "skipped_fraction": (skipped / tiles) if tiles else None}
    return out


def correctness_gate(q, k, v, conditional_rescale, tile=TILE):
    """fp32 math reference vs the tiled op; returns (ok, max_abs_diff, stats)."""
    ref = F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=True,
        enable_gqa=(k.shape[1] != q.shape[1]))
    got, stats = ref_tiled_attention(q, k, v, conditional_rescale=conditional_rescale,
                                     tile=tile, return_stats=True)
    diff = (got.float() - ref).abs().amax().item()
    ok = bool(math.isfinite(diff) and diff <= 0.05)
    return ok, diff, stats


def _cell_metrics(arm, fl, B, S, H, KVH, d):
    mean_ms, watts, peak = arm["mean_ms"], arm["watts"], arm["peak"]
    return {
        "mean_ms": round(mean_ms, 4),
        "tflops": round(fl / (mean_ms / 1000.0) / 1e12, 4),
        "io_gbps": round(io_bytes(B, S, H, KVH, d, 2) / (mean_ms / 1000.0) / 1e9, 2),
        "avg_watts": round(watts, 2) if watts else None,
        "joules_per_call": round((watts * mean_ms / 1000.0), 4) if watts else None,
        "peak_mem_mb": round(peak, 1),
    }


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device attached; fail-closed, no receipts")
    prop = torch.cuda.get_device_properties(0)
    receipt = {
        "kind": "szl.fa4-phase2/technique-receipt",
        "run": "fa4-phase2-conditional-rescale-20261115",
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "technique": "conditional softmax rescaling (FA4-inspired, arXiv:2603.05451)",
        "scope_note": (
            "reference torch-level tiled attention in this harness; the estate "
            "kernel suite (torch-ext/szl_kernels) has no attention op yet - "
            "Kernel Hub integration is follow-up work (issue #76), not a claim"
        ),
        "gpu": {
            "name": prop.name,
            "compute_capability": f"{prop.major}.{prop.minor}",
            "mem_gb": round(prop.total_memory / 2**30, 1),
            "driver_cuda": torch.version.cuda,
            "hardware_deviation": (
                "battery v2 receipts report NVIDIA GeForce RTX 3090, which is NOT "
                "attached to this device; numbers below are for the recorded GPU only"
            ),
        },
        "torch": torch.__version__,
        "params": {
            "seq_lens": SEQ_LENS, "batch_sizes": BATCHES, "dtypes": sorted(DTYPES),
            "warmup": PH2_WARMUP, "iters": PH2_ITERS, "causal": True,
            "gqa_enabled": True, "tile": TILE, "delta_max": DELTA_MAX,
            "input_layout": "torch sdpa 4D convention (B, H, S, d)",
            "warmup_iters_note": (
                "reference arms use 2/5 instead of Phase 1's 5/25: the tiled "
                "loop has a host sync per tile in BOTH arms; fewer iterations "
                "do not change the within-arm comparison"
            ),
            "arms": ["rescale-always (flag off)", "conditional-rescale (flag on)"],
            "same_branch_test_both_arms": True,
            "correctness_gate": "fp32 math reference, B=1/S<=1024 slice, max_abs_diff <= 0.05; skip stats at full shape",
        },
        "mfu_note": (
            "UNKNOWN: bf16 tensor-core peak TFLOPs for this laptop GPU class is "
            "not spec-pinned in this receipt; utilization is reported only as "
            "achieved TFLOP/s and ratios to MEASURED Phase 1 cells, never to an "
            "assumed silicon peak"
        ),
        "results": [],
        "nulls": [],
    }

    dev = torch.device("cuda:0")
    for fam_name, fam in FAMILIES.items():
        H, KVH, d = fam["H"], fam["KVH"], fam["d"]
        for dtype_name, dt in DTYPES.items():
            for B in BATCHES:
                for S in SEQ_LENS:
                    row = {"family": fam_name, "dtype": dtype_name, "B": B, "S": S,
                           "H": H, "KVH": KVH, "d": d,
                           "layers_per_model": fam["full_attn_layers"]}
                    try:
                        torch.manual_seed(20261115)
                        q = torch.randn(B, H, S, d, dtype=dt, device=dev)  # torch sdpa 4D is (N,H,L,E)
                        k = torch.randn(B, KVH, S, d, dtype=dt, device=dev)
                        v = torch.randn(B, KVH, S, d, dtype=dt, device=dev)
                        gates = {}
                        for flag in (False, True):
                            # numerics gate on a capped slice: the fp32 math
                            # reference materializes S^2 scores and OOMs at
                            # full battery shapes on an 8 GB GPU; the gate
                            # proves the math, not the VRAM headroom
                            gate_cap = 1024
                            gb = min(q.shape[2], gate_cap)
                            qs = q[:1, :, :gb, :]
                            ks = k[:1, :, :gb, :]
                            vs = v[:1, :, :gb, :]
                            ok, diff, _gate_stats = correctness_gate(qs, ks, vs, flag)
                            # skip-fraction stats at the FULL shape (the
                            # tiled op never materializes S^2; no OOM risk)
                            _out, stats = ref_tiled_attention(
                                q, k, v, conditional_rescale=flag,
                                return_stats=True)
                            gates[flag] = (ok, diff, stats)
                        if not all(g[0] for g in gates.values()):
                            raise RuntimeError(
                                f"correctness gate failed: off={gates[False][1]:.4g} "
                                f"on={gates[True][1]:.4g} (gate 0.05)")
                        fl = flops_causal(B, S, H, d)
                        arm_out = {}
                        for flag in (False, True):
                            fn_arm = partial(ref_tiled_attention, q, k, v,
                                             conditional_rescale=flag)
                            mean_ms, watts, peak = bench_cell(fn_arm, PH2_WARMUP, PH2_ITERS)
                            arm_out[flag] = {"mean_ms": mean_ms, "watts": watts,
                                             "peak": peak}
                        off, on = arm_out[False], arm_out[True]
                        row.update({
                            "supported": True,
                            "correctness_max_abs_diff": {
                                "flag_off": round(gates[False][1], 6),
                                "flag_on": round(gates[True][1], 6),
                            },
                            "flag_off": _cell_metrics(off, fl, B, S, H, KVH, d),
                            "flag_on": _cell_metrics(on, fl, B, S, H, KVH, d),
                            "skipped_rescale_fraction_on": (
                                round(gates[True][2]["skipped_fraction"], 4)
                                if gates[True][2]["skipped_fraction"] is not None else None),
                            "delta_ms_on_minus_off": round(on["mean_ms"] - off["mean_ms"], 4),
                            "delta_pct": (
                                round(100.0 * (on["mean_ms"] - off["mean_ms"]) / off["mean_ms"], 2)
                                if off["mean_ms"] > 0 else None),
                        })
                    except Exception as e:
                        row.update({"supported": False,
                                    "reason": f"{type(e).__name__}: {e}"[:300]})
                    receipt["results"].append(row)
                    torch.cuda.empty_cache()
        print(f"family {fam_name}: done", flush=True)

    outdir = os.path.join(os.path.dirname(__file__), "..", "receipts")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.abspath(os.path.join(outdir,
        "fa4-phase2-conditional-rescale-20261115.json"))
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, sort_keys=False)
    sup = sum(1 for r in receipt["results"] if r.get("supported"))
    print(f"RECEIPT_WRITTEN {outpath}")
    print(f"cells total={len(receipt['results'])} supported={sup} nulls={len(receipt['nulls'])}")


if __name__ == "__main__":
    main()
