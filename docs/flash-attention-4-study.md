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


## 5. Phase 1 baseline receipts (measured 2026-11-15)

Receipt: `receipts/fa4-phase1-baselines-20261115.json` (schema
`szl.fa4-phase1/baseline-receipt`), harness `scripts/fa4_phase1_baselines.py`.
96 cells = 2 battery-v2 model families x {512,1024,2048,4096} x {1,2,4} x
4 SDPA backends, causal prefill, bf16, GQA, 5 warmup + 25 timed calls,
NVML-sampled watts (measured-or-null).

### Hardware deviation (recorded, not papered over)

The battery v2 receipts report an NVIDIA RTX 3090, which is **not attached
to the CTO-lane device**. The only attached GPU is the **RTX 5050 Laptop
GPU (8 GB, compute capability 12.0, Blackwell)**, and every number below is
measured on it. These are NOT 3090 numbers; the 3090 pass stays open until
that GPU is attached to a lane (owner/parallel-lane action). The 5050 point
is independently interesting: it is Blackwell, the FA4-generation
architecture class, albeit laptop-class.

### Measured facts (RTX 5050, torch 2.10.0+cu130)

| family (battery shapes) | backend | cells | best TFLOP/s | best shape |
|---|---|---|---|---|
| nemotron-mini-4b (H24 KVH8 d128) | sdpa-cudnn | 12/12 | 13.28 | B4 S4096, 31.0 ms |
| nemotron-mini-4b | sdpa-efficient | 12/12 | 4.75 | B4 S4096, 86.9 ms |
| nemotron-mini-4b | sdpa-math | 12/12 | 1.80 | B1 S4096, 57.1 ms |
| qwen35-0.8b full-attn (H8 KVH2 d256) | sdpa-efficient | 12/12 | 3.17 | B1 S4096, 21.7 ms |
| qwen35-0.8b full-attn | sdpa-math | 12/12 | 2.08 | B1 S4096, 33.1 ms |

- B1 S4096 head-to-head (nemotron): cuDNN 7.92 ms / 0.0816 J/call vs math
  57.13 ms / 0.5836 J/call -> cuDNN is ~7.2x faster and ~7.2x lower energy
  per call on this hardware. This is the baseline Phase 2 must beat or null.
- Joules per call were measured (NVML, ~50 ms sampling), not estimated:
  e.g. cuDNN B1 S4096 = 0.0816 J at 10.3 W.

### Nulls (honest)

- `sdpa-flash`: 0/24 cells - this torch build is **not compiled with flash
  attention** ("No available kernel"); recorded as null, not skipped.
- `flash_attn` package: absent on this device; FA2-via-package cells are
  therefore null. SDPA-native kernels only.
- RTX 3090 pass: null this run (hardware not attached) - see deviation above.

### Standing next steps

1. Phase 2 (conditional softmax rescaling, single technique, null-result-if-
   null) should target the cuDNN baseline measured here when run on Blackwell,
   and re-baseline on the 3090 when attached (Ampere) before comparison.
2. The battery's inference hosts should be receipt-audited for which SDPA
   backend actually serves each model family; this receipt measures the
   kernels, not the serving-path selection.
