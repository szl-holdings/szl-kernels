"""FA4 Phase 1 baseline receipts (szl-kernels docs/flash-attention-4-study.md).

Benchmarks the attention paths actually exercised by the battery v2 model set
(estate-battery-v2-run-manifest.json, 13 models, on-chain receipts) across the
battery's real model shapes, on the GPU actually attached to this device.

HARDWARE DEVIATION (recorded per measured-or-null doctrine): the battery v2
receipts report an NVIDIA RTX 3090, which is NOT attached to this device.
This device's only attached GPU is recorded in the receipt as-is. Numbers are
valid for the recorded GPU only; they are NOT RTX 3090 numbers.

Receipts: kernel id, shapes, dtype, achieved TFLOP/s, IO bytes, joules per call
(measured-or-null), peak memory. No claims beyond what is measured.
"""
from __future__ import annotations

import datetime as _dt
from functools import partial
import json
import os
import statistics
import threading
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F

# --- battery v2 model families (from HF config.json of the actual repos) ---
FAMILIES = {
    # SZLHOLDINGS/SZL-GovernedDistill-0.8B-v1 (Qwen3.5 hybrid; full-attn layers only)
    "qwen35-0.8b-full-attn": {"H": 8, "KVH": 2, "d": 256, "full_attn_layers": 6, "src": "SZLHOLDINGS/SZL-GovernedDistill-0.8B-v1 config.json"},
    # SZLHOLDINGS/SZL-Nemotron-Mini-4B-Estate-v1
    "nemotron-mini-4b": {"H": 24, "KVH": 8, "d": 128, "full_attn_layers": 32, "src": "SZLHOLDINGS/SZL-Nemotron-Mini-4B-Estate-v1 config.json"},
}
SEQ_LENS = [512, 1024, 2048, 4096]
BATCHES = [1, 2, 4]
DTYPES = {"bf16": torch.bfloat16}
WARMUP, ITERS = 5, 25
BACKENDS = {
    "sdpa-flash": [SDPBackend.FLASH_ATTENTION],
    "sdpa-cudnn": [SDPBackend.CUDNN_ATTENTION],
    "sdpa-efficient": [SDPBackend.EFFICIENT_ATTENTION],
    "sdpa-math": [SDPBackend.MATH],
}

def flops_causal(B, S, H, d):
    # QK^T (2*B*H*S^2*d) + PV (2*B*H*S^2*d), causal ~ half of non-causal
    return 2.0 * B * H * S * S * d

def io_bytes(B, S, H, KVH, d, itemsize):
    # one read each of Q,K,V + one write of O (steady-state; scores not materialized)
    return B * S * d * itemsize * (2 * H + 2 * KVH)

class PowerSampler:
    """Background NVML power sampler; measured-or-null."""
    def __init__(self):
        self.ok = False
        self.samples = []
        self._stop = threading.Event()
        self._t = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._get = lambda: float(pynvml.nvmlDeviceGetPowerUsage(self._h)) / 1000.0
            self._get()  # probe
            self.ok = True
        except Exception:
            self._h = None

    def start(self):
        if not self.ok:
            return
        def loop():
            while not self._stop.is_set():
                try:
                    self.samples.append(self._get())
                except Exception:
                    pass
                time.sleep(0.05)
        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()

    def stop(self):
        if self._t is None:
            return None
        self._stop.set()
        self._t.join(timeout=2)
        self._t = None
        if not self.samples:
            return None
        return statistics.mean(self.samples)

def bench_cell(fn, warmup, iters):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    sampler = PowerSampler()
    sampler.start()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    wall = t0.elapsed_time(t1) / 1000.0  # seconds for all iters
    watts = sampler.stop()
    peak = torch.cuda.max_memory_allocated() / 2**20
    mean_ms = wall * 1000.0 / iters
    return mean_ms, watts, peak

def main():
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device attached; fail-closed, no receipts")
    prop = torch.cuda.get_device_properties(0)
    receipt = {
        "kind": "szl.fa4-phase1/baseline-receipt",
        "run": "fa4-phase1-baselines-20261115",
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
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
        "flash_attn_pkg": None,
        "flops_formula": "causal_prefill = 2*B*H*S^2*d (QK^T + PV, causal ~1/2 of full, mul+add counted)",
        "io_bytes_formula": "B*S*d*itemsize*(2*H + 2*KVH), Q/K/V reads + O write, scores not materialized",
        "params": {"seq_lens": SEQ_LENS, "batch_sizes": BATCHES, "dtypes": sorted(DTYPES), "warmup": WARMUP, "iters": ITERS, "causal": True, "gqa_enabled": True},
        "results": [],
        "nulls": [],
    }
    try:
        import flash_attn  # noqa: F401
        receipt["flash_attn_pkg"] = flash_attn.__version__
    except Exception:
        receipt["nulls"].append({"what": "flash_attn package", "reason": "not installed on this device; SDPA-native kernels only"})
        receipt["flash_attn_pkg"] = None

    dev = torch.device("cuda:0")
    for fam_name, fam in FAMILIES.items():
        H, KVH, d = fam["H"], fam["KVH"], fam["d"]
        for dtype_name, dt in DTYPES.items():
            for B in BATCHES:
                for S in SEQ_LENS:
                    torch.manual_seed(20261115)
                    q = torch.randn(B, S, H, d, dtype=dt, device=dev)
                    k = torch.randn(B, S, KVH, d, dtype=dt, device=dev)
                    v = torch.randn(B, S, KVH, d, dtype=dt, device=dev)
                    for bk, bs in BACKENDS.items():
                        row = {"family": fam_name, "dtype": dtype_name, "B": B, "S": S,
                               "backend": bk, "H": H, "KVH": KVH, "d": d,
                               "layers_per_model": fam["full_attn_layers"]}
                        try:
                            fn = partial(F.scaled_dot_product_attention, q, k, v, is_causal=True, enable_gqa=True)
                            with sdpa_kernel(bs):
                                F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
                                torch.cuda.synchronize()
                                mean_ms, watts, peak = bench_cell(fn, WARMUP, ITERS)
                            fl = flops_causal(B, S, H, d)
                            row.update({
                                "supported": True,
                                "mean_ms": round(mean_ms, 4),
                                "tflops": round(fl / (mean_ms / 1000.0) / 1e12, 2),
                                "io_gbps": round(io_bytes(B, S, H, KVH, d, 2) / (mean_ms / 1000.0) / 1e9, 1),
                                "avg_watts": round(watts, 2) if watts else None,
                                "joules_per_call": round((watts * mean_ms / 1000.0), 4) if watts else None,
                                "peak_mem_mb": round(peak, 1),
                            })
                        except Exception as e:
                            row.update({"supported": False, "reason": f"{type(e).__name__}: {e}"[:300]})
                        receipt["results"].append(row)
                        torch.cuda.empty_cache()
                    del q, k, v
        print(f"family {fam_name}: done", flush=True)

    outdir = os.path.join(os.path.dirname(__file__), "..", "receipts")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.abspath(os.path.join(outdir, "fa4-phase1-baselines-20261115.json"))
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, sort_keys=False)
    sup = sum(1 for r in receipt["results"] if r.get("supported"))
    print(f"RECEIPT_WRITTEN {outpath}")
    print(f"cells total={len(receipt['results'])} supported={sup} nulls={len(receipt['nulls'])}")

if __name__ == "__main__":
    main()
