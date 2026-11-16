# Cross-GPU FA4 battery receipts (2026-11-15/16)

FA4 Phase 1 (SDPA backend baselines) + Phase 2 (conditional-rescale) measured across estate and Salad-pool GPUs, battery v2, fail-closed per-cell (supported=false + reason recorded, never dropped).

| GPU | torch build | P1 supported | Best TFLOP/s | P2 cells | P2 skip fraction | P2 delta mean |
| --- | --- | --- | --- | --- | --- | --- |
| NVIDIA RTX A6000 (szl-gate-1) | 2.14.1+cu126 | 84/96 | 115.10 (cuDNN, nemotron B1 S4096 d128 bf16) | 24/24 | 0.0 | -0.28% |
| Salad RTX 3090 24GB pool | 2.8.0+cu129 | 60/96 | 65.84 (cuDNN, nemotron B2 S2048 d128 bf16) | 24/24 | 0.0 | +1.73% |
| NVIDIA RTX 5050 (BETTERWITHAGE) | 2.10.0+cu130 | 36/96 | 21.0 (cuDNN) | 24/24 | 0.0 | +3.12% |
| Salad RTX 5090 pool | 2.4.1+cu121 | honest null (no sm_120 kernels) | - | - | - | - |

5050 device receipts live at receipts/fa4-phase1-baselines-20261115.json and receipts/fa4-phase2-conditional-rescale-20261115.json (unchanged).

## Honest notes
- Coverage is build-dependent: sdpa-efficient is 24/24 on A6000/cu126 but 0/24 ("No available kernel") on 3090/cu129; sdpa-flash is 24/24 on A6000 and 3090 (sm_86) but was 0/24 on the 5050/cu130 build; cuDNN covers nemotron (head_dim<=128) on all three but nulls on qwen d=256. Comparison targets must pin GPU + torch build.
- salad-rtx3090-torch2.4.1-allnull/ preserves the first Salad dispatch's honest all-null receipts: torch 2.4.1 SDPA predates the enable_gqa kwarg (added in torch 2.5), so every GQA cell raised TypeError before benchmarking. Not-supported records, not supported=0 measurements.
- Salad RTX 5090 (gpu-class 851399fb) produced an honest all-unsupported phase-2 receipt on torch 2.4.1+cu121 (no sm_120 kernels; sm_120 needs cu12.8+ builds). Evidence: dataset git history of SZLHOLDINGS/frontier-54-runs mesh/salad-fa4-bench-r1/ plus gate logs (fr90_fetch2.log) and the estate ledger.
- Phase 2 verdict repeats on every GPU measured: skipped_rescale_fraction = 0.0 on all 24 cells (A6000, 3090, 5050); correctness gate (fp32 math reference, max_abs_diff <= 0.05) passed everywhere it ran.
- Provenance: Salad receipts uploaded by the container to dataset SZLHOLDINGS/frontier-54-runs at mesh/salad-fa4-bench-r1/ (rc.txt PHASE1_RC=0 PHASE2_RC=0, torch 2.8.0+cu129, ts 2026-11-16T00:42Z); local preservation on BETTERWITHAGE C:\SZL\estate-auditeceipts-20261115-mesh\; A6000 run executed on szl-gate-1.
- No flash_attn package in any run: SDPA-native kernels only (recorded in receipt nulls).
