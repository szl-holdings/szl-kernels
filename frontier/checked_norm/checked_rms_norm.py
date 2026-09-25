"""Original inference-only Triton RMSNorm with stable scaling and finite checks.

Implements RMSNorm mathematics; no novelty or estate deployment is claimed.
One row per GPU program. Output is only returned after a finite-status readback.
"""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _checked_rms(X, W, Y, Bad, N: tl.constexpr, Eps: tl.constexpr, Block: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, Block)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask, other=1).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), axis=0), tl.sqrt(Eps))
    normalized = x / scale
    inv = tl.rsqrt(tl.sum(normalized * normalized, axis=0) / N + (Eps / scale) / scale)
    y = (normalized * inv * w).to(Y.dtype.element_ty)
    finite = (x == x) & (tl.abs(x) != float("inf")) & (w == w) & (tl.abs(w) != float("inf"))
    finite = finite & (y == y) & (tl.abs(y.to(tl.float32)) != float("inf"))
    invalid = tl.sum(tl.where(mask & ~finite, 1, 0), axis=0)
    if invalid > 0:
        tl.atomic_or(Bad, 1)
    tl.store(Y + row * N + cols, y, mask)


def checked_rms_norm(x, weight, eps=1e-6):
    """Normalize a contiguous CUDA [rows, columns] tensor; reject invalid output.

    Supported dtypes: float16, bfloat16, float32; columns 1..8192.
    This function synchronizes to inspect a device error flag. No autograd support.
    """
    if type(eps) not in (int, float) or not math.isfinite(eps) or not 1e-30 <= eps <= 1e10:
        raise ValueError("eps must be finite and in [1e-30,1e10]")
    if x.ndim != 2 or x.shape[0] < 1 or not 1 <= x.shape[1] <= 8192:
        raise ValueError("expected nonempty [rows, columns], columns <=8192")
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("supported CUDA floating tensor required")
    if weight.shape != (x.shape[1],) or weight.device != x.device or weight.dtype != x.dtype:
        raise ValueError("weight must match columns, device and dtype")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("contiguous inputs required")
    if x.requires_grad or weight.requires_grad:
        raise ValueError("inference-only kernel has no backward implementation")
    with torch.cuda.device(x.device):
        output = torch.empty_like(x)
        bad = torch.zeros((), device=x.device, dtype=torch.int32)
        _checked_rms[(x.shape[0],)](x, weight, output, bad, x.shape[1], float(eps), triton.next_power_of_2(x.shape[1]),
                                    num_warps=4 if x.shape[1] <= 2048 else 8)
        if bad.item():
            raise ValueError("non-finite input, weight or output")
    return output


def checked_torch_reference(x, weight, eps=1e-6):
    """Stable eager GPU baseline with the same finite-rejection semantics."""
    values = x.float()
    scale = values.abs().amax(dim=-1, keepdim=True).clamp_min(math.sqrt(eps))
    normalized = values / scale
    output = (normalized * torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + (eps / scale) / scale) * weight.float()).to(x.dtype)
    if not (torch.isfinite(x).all() & torch.isfinite(weight).all() & torch.isfinite(output).all()).item():
        raise ValueError("non-finite input, weight or output")
    return output
