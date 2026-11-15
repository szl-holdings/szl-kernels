# FlashAttention-4 study and estate benchmark plan

Study of Zadouri, Hoehnerbach, Shah, Liu, Thakkar, Dao,
"FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric
Hardware Scaling" ([arXiv:2603.05451](https://arxiv.org/abs/2603.05451),
5 Mar 2026), scoped to what it means for szl-kernels and the Salad battery
fleet (RTX 3090, Ampere).

## 1. What FA4 does

Blackwell (B200/GB200) exhibits asymmetric hardware scaling: tensor-core
throughput doubles while shared-memory bandwidth and the exponential (SFU)
units scale slowly or not at all. Attention kernels stop being matmul-bound
and become rescale/exp/smem-bound. FA4's co-design answers with:

1. **Pipelines for fully asynchronous MMA** with larger tile sizes, so the
   tensor cores stay fed despite slower auxiliaries.
2. **Software-emulated exponential** (exp via matmul-friendly polynomial/
   table paths instead of SFU) and **conditional softmax rescaling** that
   skips most non-matmul rescale work per tile.
3. **Tensor Memory (TMEM) + 2-CTA MMA mode** to cut shared-memory traffic and
   backward-pass atomic adds.

Results on B200 (BF16): up to 1.3x over cuDNN 9.13, 2.7x over Triton,
1613 TFLOPs/s (71% utilization). Implemented in CuTe-DSL (Python-embedded),
20-30x faster compile than C++ template approaches.

## 2. What transfers to our hardware - honestly

The headline FA4 numbers are Blackwell-specific. The Salad battery fleet is
RTX 3090 (Ampere, GA102): no TMEM, no 2-CTA MMA, no async-MMA asymmetry of
the Blackwell kind, and FA3-style warp specialization is already Hopper-only.
We therefore make NO claim that FA4 yields speedups on the 3090. What DOES
transfer as engineering doctrine:

- **Non-matmul-op minimization**: the conditional-rescaling idea (skip
  rescales that cannot change the outcome under the running max) is
  architecture-relevant on Ampere too, where softmax bookkeeping is a known
  secondary bottleneck. A bounded, measured experiment is worthwhile.
- **Kernel/algorithm co-design discipline**: FA4's core lesson is to design
  the ALGORITHM around the hardware's asymmetry, not to tune a fixed
  algorithm. Our kernel work should state, per kernel, which resource is
  the bottleneck on the target GPU before optimizing.
- **Python-embedded DSL ergonomics**: CuTe-DSL-style development (fast
  compile, full expressivity) matches our preference for inspectable,
  receipt-generating kernels over opaque prebuilt binaries.

## 3. Benchmark plan (Salad, measured-or-null doctrine)

Phase 1 - baseline receipts (3090, Ampere):
- Pin and bench attention paths actually used by the battery v2 model set
  (13 models, receipts on-chain at SZLHOLDINGS/frontier-54-runs):
  PyTorch SDPA (flash backend), FA2 (via packaged kernels), and any
  estate-local attention kernels, across the battery's real sequence
  lengths and batch shapes.
- Emit szl receipts: kernel id, shapes, dtype, TFLOPs/s achieved, memory
  moved, joules per call (via the szl_energy_attest path when available).

Phase 2 - FA4-technique experiment (bounded, honest):
- Implement ONE transferable technique - conditional softmax rescaling -
  behind a flag on the estate attention kernel; measure against Phase 1
  baselines on the 3090. Report the delta with CI; if the 3090 shows no
  gain, record the null result and close the thread (null results are
  receipts too).

Phase 3 - Blackwell readiness (owner-gated):
- FA4 proper requires B200/GB200. Acquiring Blackwell time (cloud or
  colocation) is a budget decision for the owner. This phase stays a
  documented plan, not a claim, until hardware exists: port the Phase 1/2
  harness, then evaluate upstream FA4 (CuTe-DSL) against cuDNN and Triton
  baselines under the same receipt discipline.

## 4. Estate coupling

- Battery v2 (13 heavy models, 6 probes each, zero refusals) gives us the
  real workload shapes that Phase 1 must bench - no synthetic-only
  benchmarks.
- Every Phase 1-3 number lands in the receipts corpus with the same
  measured-or-null and hash-chain discipline as the rest of the estate
  (see szl_energy_attest SPEC).
- Λ note per estate doctrine: kernel speedup claims are measured facts with
  uncertainty, never trust-ceiling claims.

## References

- FA4: https://arxiv.org/abs/2603.05451 (doi:10.48550/arXiv.2603.05451)
- FA3 (context, Hopper): Shah et al., arXiv:2407.08608
- Salad battery v2 manifest: SZLHOLDINGS/frontier-54-runs,
  receipts/estate-battery-v2-run-manifest.json (sha c66d0904...)
