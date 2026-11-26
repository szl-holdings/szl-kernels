# SZL Governed-Kernel Portability Reference v1

Status: PROPOSE-ONLY leader absorb (Ouroboros Cluster C). No merge without explicit owner GO. This document adds a reference; it changes no code, no CI, no published kernel, and makes no performance claim.

## Purpose

One kernel, N backends, identical numeric tolerances, one signed witness receipt per backend. This is the estate portability reference inspired by the ThunderKittens 2.0 tile-level abstraction (tile-level kernel DSL with clean portability across GPU generations; verified observable: hazyresearch.stanford.edu/blog/2026-02-19-tk-2, together.ai Vera Rubin port) and the FlashAttention-4 asymmetric-hardware posture already banked in round 1 (backend-specialization with a portable reference fallback; never receipt-attest speed as correctness).

## The reference contract (per kernel)

1. A governed kernel declares an operator spec: tile shape, dtype, tolerance gate (max_abs_error target), and the algorithm-declared digest (szl.canon/v1 canonical bytes).
2. The torch implementation is the reference backend. Every candidate backend (torch-cpu, torch-cuda, triton, rocm, webgpu per frontier/kernel_benchmark_manifest.json) specializes the SAME operator spec; none may change numerics.
3. Fail-closed to reference: any backend that cannot prove parity falls back to the reference path; denial is the default, acceleration is never an excuse for divergence.
4. Parity measurement produces one witness per backend per target, shaped by szl.kernel-witness/v1 (frontier/kernel_witness_schema.json): operator, source_revision, kernel_revision, dtype, shape, max_abs_error, correctness_parity_pass, compile_mode, hardware_fingerprint, receipt_sha256. The schema rule is verbatim law: a witness without max_abs_error and correctness_parity_pass cannot support an acceleration claim.
5. Identical tolerances: all backends of one kernel share one tolerance gate; a backend may be stricter, never looser, and any stricter setting must be recorded in the witness.

## Conformance treadmill

For each manifest target, in priority order (YARQA-ATTN, szl-receipt-attn, szl-block-kv, then the rest): run candidate backend vs reference on identical seeded inputs -> record max_abs_error -> witness PASS only within the tolerance gate -> land the witness in receipts/cross-gpu/<hardware-dir>/ beside the existing FA4 phase1/phase2 layout. Speed may be reported ONLY beside a parity witness (benchmark receipts first; speed never substitutes for correctness).

## Baseline clause (DeepGEMM)

For GEMM-class estate benchmarks, adopt DeepGEMM (github.com/deepseek-ai/DeepGEMM) as the comparison BASELINE-CLASS reference, consistent with prior-absorb posture. Watch-mark: no estate receipt exists yet, so no DeepGEMM number is claimed anywhere in this proposal; a baseline becomes real only when a measured cross-GPU receipt lands.

## Honesty gates (carried verbatim)

The honesty matrix doctrine stands: published test failures stay failed; blocked tokens are not a product; excluded_from_promotion packages (frontier/kernel_benchmark_manifest.json) stay excluded until their own CI repairs land. This reference adds no exception.

## Sources

- ThunderKittens 2.0 (Hazy Research, 2026-02-19): tile-level abstraction reference - VERIFIED observable.
- FlashAttention-4 (arXiv 2603.05451; tridao.me; pytorch.org low-precision FA4 post): asymmetric-hardware posture, round-1 absorbed.
- DeepGEMM (DeepSeek): BLAS baseline reference - WATCH, unmeasured by the estate.
- Estate grounding: frontier/kernel_witness_schema.json, frontier/kernel_honesty_matrix.json, frontier/kernel_benchmark_manifest.json, receipts/cross-gpu/ (FA4 phase1/phase2 receipts).