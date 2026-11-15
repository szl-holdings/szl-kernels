"""FA4 Phase 2b: CuTe-DSL authoring-lane smoke (issue #76, study doc S3).

FlashAttention-4 is authored in CuTe-DSL embedded in Python (arXiv:2603.05451).
This script smoke-tests the CuTe-DSL authoring lane on THIS device, stage by
stage, measured-or-null:

  stage 1 toolchain : nvidia-cutlass-dsl importable, versions, GPU visibility
  stage 2 jit-smoke : compile + run a trivial CuTe-DSL kernel on the GPU
  stage 3 cr-op     : compile + run a conditional-rescale flash-tile update
                      (the FA4 Phase 2 technique) as a CuTe-DSL kernel

CuTe DSL 4.8.x conventions used here (all verified against the installed
4.8.1): device code is a @cute.kernel; it is launched from a @cute.jit host
wrapper via kernel(args).launch(grid=..., block=...); cute.arch.thread_idx()
returns a tuple; torch tensors cross the boundary via
cutlass.cute.runtime.from_dlpack; math via cute.exp2 / cute.max; the CUDA
context via cutlass.cuda.initialize_cuda_context(). The pre-4.x Cuda()
execution-target argument no longer exists.

Every stage records measured facts or the exact exception (null). No stage
is reported green without executing. The full FA-style attention kernel and
its Kernel Hub publication remain open work (issue #76); this receipt
records the lane's honest state, not a claim.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import traceback

import torch

try:
    import cutlass  # noqa: F401
    import cutlass.cute as cute  # noqa: F401
    from cutlass.cute.runtime import from_dlpack  # noqa: F401
    _CUTE_IMPORT_ERROR = None
except Exception as _e:  # measured-or-null: exact error is the receipt
    cutlass = None
    cute = None
    from_dlpack = None
    _CUTE_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"

RECEIPT = {
    "kind": "szl.fa4-phase2/cute-dsl-smoke-receipt",
    "run": "fa4-phase2-cute-dsl-smoke-20261115",
    "ts": None,
    "stages": {},
    "nulls": [],
}


def _stage(name):
    def deco(fn):
        def wrapped():
            try:
                out = fn()
                RECEIPT["stages"][name] = {"ok": True, **(out or {})}
                print(f"stage {name}: OK")
            except Exception as e:  # measured-or-null: exact error is the receipt
                tb = traceback.format_exc(limit=6)
                RECEIPT["stages"][name] = {
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}"[:500],
                    "traceback_tail": tb.strip().splitlines()[-1][:300],
                }
                RECEIPT["nulls"].append(name)
                print(f"stage {name}: NULL ({type(e).__name__})")
        return wrapped
    return deco


if cute is not None:
    @cute.kernel
    def scale_one_kernel(gx: cute.Tensor, alpha: cutlass.Float32):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            gx[0] = gx[0] * alpha

    @cute.jit
    def scale_one(gx: cute.Tensor, alpha: cutlass.Float32):
        cutlass.cuda.initialize_cuda_context()
        scale_one_kernel(gx, alpha).launch(grid=(1, 1, 1), block=(32, 1, 1))


if cute is not None:
    @cute.kernel
    def flash_tile_update_kernel(gs: cute.Tensor, g_old_max: cute.Tensor,
                                 g_tile_max: cute.Tensor, g_acc: cute.Tensor,
                                 g_p: cute.Tensor, delta_max: cutlass.Float32):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            old_max = g_old_max[0]
            tile_max = g_tile_max[0]
            new_max = cute.max(old_max, tile_max)
            delta = new_max - old_max
            if delta > delta_max:
                # rescale path: switch the exp frame
                corr = cute.exp2(delta * 1.4426950408889634)  # exp via exp2(ln2)
                g_acc[0] = g_acc[0] / corr
            g_p[0] = cute.exp2((gs[0] - cute.max(old_max, tile_max))
                               * 1.4426950408889634)

    @cute.jit
    def flash_tile_update(gs: cute.Tensor, g_old_max: cute.Tensor,
                          g_tile_max: cute.Tensor, g_acc: cute.Tensor,
                          g_p: cute.Tensor, delta_max: cutlass.Float32):
        cutlass.cuda.initialize_cuda_context()
        flash_tile_update_kernel(gs, g_old_max, g_tile_max, g_acc, g_p,
                                 delta_max).launch(grid=(1, 1, 1),
                                                   block=(32, 1, 1))


@_stage("toolchain")
def _toolchain():
    assert _CUTE_IMPORT_ERROR is None, _CUTE_IMPORT_ERROR
    assert torch.cuda.is_available(), "no CUDA device"
    prop = torch.cuda.get_device_properties(0)
    return {
        "cutlass_version": getattr(cutlass, "__version__", None),
        "cute_module": cutlass.cute.__name__,
        "gpu": prop.name,
        "compute_capability": f"{prop.major}.{prop.minor}",
        "torch": torch.__version__,
    }


@_stage("jit-smoke")
def _jit_smoke():
    # Trivial CuTe-DSL kernel: scale a tiny tensor in place on the GPU.
    x = torch.full((8,), 2.0, dtype=torch.float32, device="cuda")
    alpha = cutlass.Float32(3.0)
    scale_one(from_dlpack(x), alpha)
    torch.cuda.synchronize()
    got = x[0].item()
    assert abs(got - 6.0) < 1e-5, f"expected 6.0, got {got}"
    return {"kernel": "scale_one", "result": got}


@_stage("cr-op")
def _cr_op():
    # The FA4 Phase 2 technique as a CuTe-DSL kernel: one flash-tile update
    # with a conditional rescale (corr applied only when the max delta
    # exceeds delta_max; else the old exp-frame is kept). Correctness is
    # checked against a torch reference computed from the INPUT values
    # BEFORE launch (the kernel mutates acc/p in place).
    import math as _math
    s_v, old_v, tile_v, acc_v = 1.25, 1.0, 1.3, 4.0
    new_max = max(old_v, tile_v)
    corr = _math.exp(old_v - new_max)
    acc_ref = acc_v * corr
    p_ref = _math.exp(s_v - new_max)

    s = torch.tensor([s_v], dtype=torch.float32, device="cuda")
    old_max = torch.tensor([old_v], dtype=torch.float32, device="cuda")
    tile_max = torch.tensor([tile_v], dtype=torch.float32, device="cuda")
    acc = torch.tensor([acc_v], dtype=torch.float32, device="cuda")
    p = torch.zeros(1, dtype=torch.float32, device="cuda")
    delta_max = cutlass.Float32(5e-4)
    flash_tile_update(from_dlpack(s), from_dlpack(old_max),
                      from_dlpack(tile_max), from_dlpack(acc),
                      from_dlpack(p), delta_max)
    torch.cuda.synchronize()
    got_acc, got_p = acc.item(), p.item()
    assert abs(got_acc - acc_ref) < 1e-4, (got_acc, acc_ref)
    assert abs(got_p - p_ref) < 1e-4, (got_p, p_ref)
    return {"acc": got_acc, "p": got_p, "acc_ref": acc_ref, "p_ref": p_ref}


def main():
    RECEIPT["ts"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    _toolchain()
    if RECEIPT["stages"].get("toolchain", {}).get("ok"):
        _jit_smoke()
        if RECEIPT["stages"].get("jit-smoke", {}).get("ok"):
            _cr_op()
    outdir = os.path.join(os.path.dirname(__file__), "..", "receipts")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.abspath(os.path.join(outdir,
        "fa4-phase2-cute-dsl-smoke-20261115.json"))
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(RECEIPT, f, indent=2, sort_keys=False)
    print(f"RECEIPT_WRITTEN {outpath}")
    print(f"stages ok={sum(1 for s in RECEIPT['stages'].values() if s.get('ok'))} "
          f"nulls={len(RECEIPT['nulls'])}")


if __name__ == "__main__":
    main()
