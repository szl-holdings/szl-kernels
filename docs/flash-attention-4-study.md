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
NVML-sampled watts (measured-or-null). The receipt in this PR was produced by
the committed harness (re-run after the lint fixes; both runs: 96 cells,
60 supported, identical support profile).

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
| nemotron-mini-4b (H24 KVH8 d128) | sdpa-cudnn | 12/12 | 13.20 | B2 S4096, 15.6 ms |
| nemotron-mini-4b | sdpa-efficient | 12/12 | 4.74 | B2 S4096, 43.5 ms |
| nemotron-mini-4b | sdpa-math | 12/12 | 1.80 | B1 S4096, 57.3 ms |
| qwen35-0.8b full-attn (H8 KVH2 d256) | sdpa-efficient | 12/12 | 3.17 | B1 S4096, 21.7 ms |
| qwen35-0.8b full-attn | sdpa-math | 12/12 | 2.07 | B1 S4096, 33.2 ms |

- B1 S4096 head-to-head (nemotron): cuDNN 7.97 ms / 0.082 J/call vs math
  57.30 ms / 0.5992 J/call -> cuDNN is ~7.2x faster and ~7.3x lower energy
  per call on this hardware. This is the baseline Phase 2 must beat or null.
- Joules per call were measured (NVML, ~50 ms sampling), not estimated:
  e.g. cuDNN B1 S4096 = 0.082 J at 10.3 W.

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

### v2 correction (2026-11-15, later the same day)

The v1 Phase 1 receipt (PR #78) benched a scrambled workload: tensors were
created as (B, S, H, d) and handed to torch SDPA, whose 4D convention is
(N, H, L, E) - so the benched op had heads=S, q_len=H, kv_len=KVH, while
the receipt's tflops/io_gbps were computed from the intended causal-prefill
formula. All v1 TFLOP/s and IO GB/s numbers above are superseded by the
regenerated v2 receipt (same filename, git history preserves v1; see the
receipt's correction_20261115 field). v1 timings/energy measured the
scrambled op. v2 numbers below, same hardware, same harness discipline:

v2: 96 cells, 36 supported. Best per family/backend:

| family | backend | supported cells | best TFLOP/s | best shape (B, S) | mean_ms @ best |
|---|---|---|---|---|---|
| nemotron-mini-4b | sdpa-cudnn | 12/12 | 21.00 | (4, 512) | 0.3068 |
| nemotron-mini-4b | sdpa-efficient | 0/12 | - | - | -|
| nemotron-mini-4b | sdpa-flash | 0/12 | - | - | -|
| nemotron-mini-4b | sdpa-math | 12/12 | 0.78 | (1, 512) | 2.0614 |
| qwen35-0.8b-full-attn | sdpa-cudnn | 0/12 | - | - | -|
| qwen35-0.8b-full-attn | sdpa-efficient | 0/12 | - | - | -|
| qwen35-0.8b-full-attn | sdpa-flash | 0/12 | - | - | -|
| qwen35-0.8b-full-attn | sdpa-math | 12/12 | 1.06 | (1, 512) | 1.011 |

Null cells in v2 are recorded with reasons in the receipt (an 8 GB laptop
GPU cannot materialize math-backend scores at the largest battery shapes;
these are honest nulls, not skips).

## 6. Phase 2 - conditional softmax rescaling (measured 2026-11-15)

Technique: the ONE transferable FA4 idea selected by the study plan (S3):
skip the online-softmax accumulator rescale when a tile's max delta stays
within delta_max = 5e-4, keeping the old exp-frame (frame-invariant output,
bounded by exp(delta_max) - exact up to fp32 rounding, no compounding).
Harness: scripts/fa4_phase2_conditional_rescale.py; receipt:
receipts/fa4-phase2-conditional-rescale-20261115.json (schema
szl.fa4-phase2/technique-receipt). The flag lives on the reference
tiled-attention op; the estate kernel suite has no attention op yet, so
Kernel Hub integration remains follow-up work (issue #76).

Arms: rescale-always vs conditional-rescale, identical code path and
identical per-tile branch test in BOTH arms (host sync included equally),
2 warmup + 5 timed calls per arm per cell, 24 cells, 24 supported.

| family | B | S | skip fraction (flag on) | delta ms | delta % | TFLOP/s off | TFLOP/s on |
|---|---|---|---|---|---|---|---|
| qwen35-0.8b-full-attn | 1 | 512 | 0.0 | -1.0116 | -5.53 | 0.0587 | 0.0621 |
| qwen35-0.8b-full-attn | 1 | 1024 | 0.0 | -5.1822 | -10.61 | 0.088 | 0.0984 |
| qwen35-0.8b-full-attn | 1 | 2048 | 0.0 | 41.9609 | 23.72 | 0.0971 | 0.0785 |
| qwen35-0.8b-full-attn | 1 | 4096 | 0.0 | 145.0975 | 18.63 | 0.0882 | 0.0744 |
| qwen35-0.8b-full-attn | 2 | 512 | 0.0 | 1.432 | 9.76 | 0.1463 | 0.1333 |
| qwen35-0.8b-full-attn | 2 | 1024 | 0.0 | 2.5399 | 4.94 | 0.167 | 0.1591 |
| qwen35-0.8b-full-attn | 2 | 2048 | 0.0 | -16.5775 | -7.68 | 0.1592 | 0.1725 |
| qwen35-0.8b-full-attn | 2 | 4096 | 0.0 | 14.8616 | 2.03 | 0.1873 | 0.1836 |
| qwen35-0.8b-full-attn | 4 | 512 | 0.0 | -0.9213 | -5.78 | 0.2695 | 0.286 |
| qwen35-0.8b-full-attn | 4 | 1024 | 0.0 | -6.7312 | -11.29 | 0.2882 | 0.3249 |
| qwen35-0.8b-full-attn | 4 | 2048 | 0.0 | -69.2569 | -29.13 | 0.289 | 0.4078 |
| qwen35-0.8b-full-attn | 4 | 4096 | 0.0 | -5.0913 | -0.72 | 0.3868 | 0.3896 |
| nemotron-mini-4b | 1 | 512 | 0.0 | 4.6 | 41.28 | 0.1446 | 0.1023 |
| nemotron-mini-4b | 1 | 1024 | 0.0 | 4.6621 | 10.33 | 0.1428 | 0.1294 |
| nemotron-mini-4b | 1 | 2048 | 0.0 | 4.033 | 3.04 | 0.1939 | 0.1882 |
| nemotron-mini-4b | 1 | 4096 | 0.0 | 55.1286 | 8.14 | 0.1523 | 0.1408 |
| nemotron-mini-4b | 2 | 512 | 0.0 | -0.7825 | -5.9 | 0.2427 | 0.2579 |
| nemotron-mini-4b | 2 | 1024 | 0.0 | -15.159 | -23.35 | 0.1985 | 0.2589 |
| nemotron-mini-4b | 2 | 2048 | 0.0 | 27.0819 | 16.36 | 0.3113 | 0.2676 |
| nemotron-mini-4b | 2 | 4096 | 0.0 | 116.3319 | 16.62 | 0.2945 | 0.2525 |
| nemotron-mini-4b | 4 | 512 | 0.0 | 3.3939 | 29.93 | 0.5681 | 0.4372 |
| nemotron-mini-4b | 4 | 1024 | 0.0 | -7.682 | -13.13 | 0.4406 | 0.5072 |
| nemotron-mini-4b | 4 | 2048 | 0.0 | 5.758 | 2.9 | 0.519 | 0.5044 |
| nemotron-mini-4b | 4 | 4096 | 0.0 | 1.5854 | 0.21 | 0.5351 | 0.534 |

**Verdict (measured, fail-closed):** skip fraction is 0.0 in every supported
cell - the tile-level (all-rows) branch NEVER fires at battery shapes
with random inputs (some row's max delta always exceeds 5e-4), so both
arms executed identical code; the per-cell deltas (mean +3.12%) are
timing noise of the sync-heavy reference loop, not a technique effect.
Null result at the torch reference level, recorded as such. The technique
is only meaningful at per-CTA kernel granularity, which a host-synced
reference loop cannot express: the real evaluation is CuTe-DSL lane work
(smoke below, full kernel under issue #76).

### CuTe-DSL authoring-lane smoke (measured-or-null)

Receipt: receipts/fa4-phase2-cute-dsl-smoke-20261115.json (schema
szl.fa4-phase2/cute-dsl-smoke-receipt). nvidia-cutlass 4.2.0.0 installed on
the device lane. Stages:

- toolchain: OK (cutlass_version=4.8.1, cute_module=cutlass.cute, gpu=NVIDIA GeForce RTX 5050 Laptop GPU, compute_capability=12.0, torch=2.10.0+cu130)
- jit-smoke: OK (kernel=scale_one, result=6.0)
- cr-op: OK (acc=2.963273048400879, p=0.951229453086853, acc_ref=2.9632728827268715, p_ref=0.951229424500714)

Cross-links: szl-kernels issue #76 (FA-4-class kernels on the Kernel Hub,
Python-native CuTe-DSL authoring lane); yarqa issue #110 (adopt FA4 numbers
as the comparison target for yarqa attention kernels - this section and the
receipts above are the measured 5050-class data points for that table).
