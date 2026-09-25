# Checked RMSNorm research candidate

An inference-only Triton implementation fuses row RMS normalization, affine
scaling and non-finite input/output detection. Max-magnitude scaling avoids
overflow when squaring large finite FP32 values. A device status readback must
pass before the Python function returns output. The implementation synchronizes
for that check and deliberately rejects tensors requiring gradients.

This is an opt-in research module under `frontier/`, with no change to the
canonical runtime, the receipted corpus, package exports, publication manifests,
or remote model weights. It implements established RMSNorm mathematics; it does
not claim a novel normalization algorithm.

## Run

Use an environment with compatible PyTorch, Triton and a CUDA GPU:

```text
python frontier/checked_norm/benchmark_rms.py --output rms-benchmark.json
```

The benchmark checks an independent CPU FP64 reference across FP16, BF16 and
FP32, multiple seeds, zero/tiny/large inputs, non-power-of-two row widths,
non-finite input/weight values, output overflow, and unsupported API contracts.
It records raw CUDA-event and full-call wall-clock samples, three interleaved
measurement rounds, medians, software versions and the candidate source hash.

The stable eager baseline has equivalent finite checks. Native PyTorch RMSNorm
is reported as context: it has different validation and extreme-value behavior.
Compile/warm-up are excluded. The full-call wall measurement includes the
candidate's mandatory status readback and allocation costs. No confidence
interval, energy measurement, end-to-end model throughput or novelty is claimed.

## Local evidence, 2026-09-12

RTX 5050 Laptop, approximately 8 GiB; PyTorch 2.11.0+cu128, Triton 3.7.1.
57 correctness/rejection cases passed. Across the three measured FP32 shapes
the observed full-call median speedup versus checked eager was 1.52x to 5.09x.
Native PyTorch RMSNorm was faster than the checked candidate on ordinary random
inputs. These measurements are a local experiment, not a deployment decision.

The [raw dated report](evidence/rms-benchmark-20260912.json) contains the measured
samples and the candidate source SHA-256. It is historical evidence, not a claim
that the current checkout was GPU-tested on 2026-09-19. The available Python
environment on that date had CPU-only PyTorch 2.14.0; CUDA tests were therefore
explicitly skipped. CPU CI parses both research modules but does not establish
their CUDA correctness or performance.

Next experiment: integrate into one representative block with equal validation
and receipt overhead, evaluate output drift across a held-out input suite, and
compare against a compiled baseline and Liger on supported hardware. Only a
repeatable end-to-end gain would justify a runtime adoption proposal.

## Research references

- [RMSNorm](https://arxiv.org/abs/1910.07467), Zhang and Sennrich.
- [Triton](https://github.com/triton-lang/triton), blocked GPU programming.
- [Liger Kernel](https://arxiv.org/abs/2410.10989), fused training kernels.
- [KernelBench](https://arxiv.org/abs/2502.10517), correctness and speed evaluation.

The implementation here was written for this experiment. Future reuse of third
party code must retain the corresponding license and attribution.
