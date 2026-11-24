# Leader-Study Tranche-4 Absorb: kernel portability + benchmark receipts (propose-only)

Lap: ouroboros tranche-4 leader study (Vibe dc lane, Kimi GO OURO-FRNT1-GOV-3fa2b8 attempt-2, order A,B,D,C; condition D = propose-only, no merges). No code changes in this PR.

## Leaders studied (cluster C: kernels/ML/attention, consumer-GPU branch)

1. FlashAttention-4 (asymmetric-hardware co-design, CuTe-DSL, software-emulated exp/softmax).
   - Sources: arXiv 2603.05451 ; https://tridao.me/blog/2026/flash4/ ; https://pytorch.org/blog/low-precision-flash-attention-4-end-to-end-block-scaled-attention-for-blackwell/
   - Absorb posture (round-1 holds): backend-specialization with a portable reference fallback; signed benchmark receipts (GPU/clocks/dtype/shape/tolerances); fail-closed to the reference kernel.

2. ThunderKittens 2.0 (Hazy Research; tile-level kernel DSL, cross-generation portability incl. Vera Rubin NVL72 port).
   - Sources: https://hazyresearch.stanford.edu/blog/2026-02-19-tk ; https://www.together.ai/blog/to-infinity-and-beyond-thunderkittens-now-on-nvidia-vera-rubin-nvl72
   - Absorb action: TK-style tile abstraction as the portability reference for the receipt-attn / maskmod / block-kv / specdecode consumer-GPU branch.

3. DeepGEMM (DeepSeek) as the clean BLAS kernel-library reference for benchmark receipts. Source: https://github.com/deepseek-ai/DeepGEMM

## Applicability tests (measured-only, never fabricated)

- One kernel, two backends, identical numeric tolerances, one signed receipt per backend.
- Benchmark receipts record GPU model, clocks, dtype, shapes, and tolerances; speed is never receipt-attested as correctness.

## Proposed next steps (each gated, none executed in this PR)

- Portability-reference skeleton doc for the four consumer-GPU kernel repos.
- DeepGEMM baselines recorded as benchmark-receipt seeds on the next owner-gated benchmark lap.
