"""Run independent FP64 correctness and measured CUDA/wall-time benchmarks."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
import torch
import triton
from checked_rms_norm import checked_rms_norm, checked_torch_reference


def reference64(x, weight, eps=1e-6):
    values = x.cpu().double()
    return (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps) * weight.cpu().double()).to(x.dtype)


def correctness():
    count = 0
    maximum = 0.0
    for seed in (712, 919):
        torch.manual_seed(seed)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for rows, cols in ((1, 1), (7, 31), (19, 128), (128, 768), (64, 1024), (7, 4096), (3, 8192)):
                x = torch.randn(rows, cols, device="cuda", dtype=dtype)
                w = torch.randn(cols, device="cuda", dtype=dtype)
                before_x, before_w = x.clone(), w.clone()
                actual = checked_rms_norm(x, w).cpu()
                expected = reference64(x, w)
                torch.testing.assert_close(actual, expected, rtol=0.02 if dtype == torch.bfloat16 else 0.003 if dtype == torch.float16 else 2e-5,
                                           atol=0.02 if dtype == torch.bfloat16 else 0.003 if dtype == torch.float16 else 2e-6)
                assert torch.equal(x, before_x) and torch.equal(w, before_w)
                maximum = max(maximum, float((actual.float() - expected.float()).abs().max()))
                count += 1
    for value in (0.0, 1e-30, 1e30, -1e30, 3e38):
        x = torch.full((3, 127), value, device="cuda", dtype=torch.float32)
        w = torch.ones(127, device="cuda")
        torch.testing.assert_close(checked_rms_norm(x, w).cpu(), reference64(x, w), rtol=2e-5, atol=2e-6)
        count += 1
    for where in ("input", "weight"):
        for value in (float("nan"), float("inf"), -float("inf")):
            x = torch.ones((3, 127), device="cuda")
            w = torch.ones(127, device="cuda")
            if where == "input":
                x[1, 126] = value
            else:
                w[126] = value
            try:
                checked_rms_norm(x, w)
                raise AssertionError("nonfinite value was not rejected")
            except ValueError:
                count += 1
    # Output overflow after casting back to fp16 must also reject.
    x = torch.zeros((1, 128), device="cuda", dtype=torch.float16)
    x[0, 0] = 1
    try:
        checked_rms_norm(x, torch.full((128,), 65504, device="cuda", dtype=torch.float16))
        raise AssertionError("output overflow was not rejected")
    except ValueError:
        count += 1
    for x, w, eps in [
        (torch.ones((2, 64), device="cuda", requires_grad=True), torch.ones(64, device="cuda"), 1e-6),
        (torch.ones((2, 128), device="cuda")[:, ::2], torch.ones(64, device="cuda"), 1e-6),
        (torch.ones((2, 64), device="cuda"), torch.ones(64, device="cuda"), float("nan")),
    ]:
        try:
            checked_rms_norm(x, w, eps)
            raise AssertionError("unsupported contract was accepted")
        except ValueError:
            count += 1
    return {"cases_passed": count, "max_absolute_error_across_dtypes": maximum, "oracle": "independent CPU float64 direct RMSNorm; tolerance varies by output dtype"}


def measure(fn, x, w, repeats=30):
    wall, gpu = [], []
    for _ in range(10):
        fn(x, w)
    torch.cuda.synchronize()
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        fn(x, w)
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - t0) * 1000)
        gpu.append(start.elapsed_time(end))
    return {"wall_ms_samples": wall, "cuda_ms_samples": gpu, "wall_ms_median": statistics.median(wall), "cuda_ms_median": statistics.median(gpu)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("evidence/rms_benchmark.json"))
    args = parser.parse_args()
    result = {"observed_at": datetime.now(timezone.utc).isoformat(), "torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda,
              "device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(), "source_sha256": hashlib.sha256(Path(__file__).with_name("checked_rms_norm.py").read_bytes()).hexdigest(),
              "status": "LOCAL_RESEARCH_PROTOTYPE", "scope": "Inference-only microbenchmark; eager stable baseline has same finite checks. Native RMSNorm lacks equivalent validation and is contextual only. No model-level speedup established."}
    result["correctness"] = correctness()
    result["benchmarks"] = []
    torch.manual_seed(717)
    for rows, cols in ((1, 128), (128, 1024), (1024, 4096)):
        x = torch.randn(rows, cols, device="cuda")
        w = torch.randn(cols, device="cuda")
        # Alternate the first implementation measured to reduce order bias.
        functions = [("checked_triton", checked_rms_norm), ("checked_eager", checked_torch_reference),
                     ("native_context_only", lambda a, b: torch.nn.functional.rms_norm(a, (a.shape[-1],), b, 1e-6))]
        timings = {name: [] for name, _ in functions}
        for round_id in range(3):
            order = functions[round_id:] + functions[:round_id]
            for name, fn in order:
                timings[name].append(measure(fn, x, w))
        merged = {}
        for name, runs in timings.items():
            wall = [sample for run in runs for sample in run["wall_ms_samples"]]
            gpu = [sample for run in runs for sample in run["cuda_ms_samples"]]
            merged[name] = {"wall_ms_samples": wall, "cuda_ms_samples": gpu, "wall_ms_median": statistics.median(wall), "cuda_ms_median": statistics.median(gpu)}
        speedup = merged["checked_eager"]["wall_ms_median"] / merged["checked_triton"]["wall_ms_median"]
        row = {"shape": [rows, cols], "dtype": "float32", "timings": merged, "wall_speedup_vs_checked_eager": speedup}
        result["benchmarks"].append(row)
        print(json.dumps({"shape": row["shape"], "speedup": speedup}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["correctness"]))


if __name__ == "__main__":
    main()
