---
tags:
- kernel
library_name: kernels
license: apache-2.0
szl-governance:
  verdict: ADVISORY
  lambda: "Conjecture 1 (open) — uniqueness unproven; advisory only"
  energy: MEASURED-only (real NVML delta; None when unavailable)
  provenance: UnifiedReceiptChain (SHA3-256, op-agnostic, cross-kernel)
  honest_blocked: "a failed check stays failed — never faked green"
---

# szl-kernels — the unified governed-kernel suite

**The first kernel that governs provenance *across* ops, not just within one.** A `get_kernel`-discoverable suite that ties SZL Holdings' three governed kernels — [`szl-governed-norm`](https://huggingface.co/SZLHOLDINGS/szl-governed-norm), [`szl-lambda-gate`](https://huggingface.co/SZLHOLDINGS/szl-lambda-gate), and [`governed-inference-meter`](https://huggingface.co/SZLHOLDINGS/governed-inference-meter) — into **one shared, hash-chained `UnifiedReceiptChain`**, and anchors a two-kernel governance layer on top: [`szl-govsign`](https://huggingface.co/SZLHOLDINGS/szl-govsign) (signs the verdict) and [`szl-blocked`](https://huggingface.co/SZLHOLDINGS/szl-blocked) (refuses honestly + derives EU AI Act Annex IV).

> Every Kernel Hub leader competes on **FLOPs per op**. They do not sign their build artifacts, they do not surface an honest BLOCKED verdict, and they do not stitch provenance *across* ops. `szl-kernels` opens that lane: a real forward pass touching norm + an advisory Λ gate + an energy reading produces **one auditable, tamper-evident log** — not three disconnected ones.

## The governed-kernel series

Five independently published, `get_kernel`-discoverable kernels that share one `UnifiedReceiptChain`. The first three are the **numeric core**; the last two are the **governance layer**.

| Kernel | Lane | Live hologram |
|---|---|---|
| [`szl-governed-norm`](https://huggingface.co/SZLHOLDINGS/szl-governed-norm) | RMSNorm/LayerNorm + SHA3-256 receipts | [governed-norm-holo](https://huggingface.co/spaces/SZLHOLDINGS/governed-norm-holo) |
| [`szl-lambda-gate`](https://huggingface.co/SZLHOLDINGS/szl-lambda-gate) | advisory Λ gate (Conjecture 1, OPEN) | [lambda-gate-holo](https://huggingface.co/spaces/SZLHOLDINGS/lambda-gate-holo) |
| [`governed-inference-meter`](https://huggingface.co/SZLHOLDINGS/governed-inference-meter) | MEASURED-joule energy accounting | [energy-attest-holo](https://huggingface.co/spaces/SZLHOLDINGS/energy-attest-holo) |
| [`szl-govsign`](https://huggingface.co/SZLHOLDINGS/szl-govsign) | signed governance attestation (DSSE / in-toto, ECDSA P-256) | [szl-govsign-live](https://huggingface.co/spaces/SZLHOLDINGS/szl-govsign-live) |
| [`szl-blocked`](https://huggingface.co/SZLHOLDINGS/szl-blocked) | honest-BLOCKED first-class state + EU AI Act Annex IV DRAFT | [szl-blocked-live](https://huggingface.co/spaces/SZLHOLDINGS/szl-blocked-live) |
| **`szl-kernels`** (this repo) | **unified suite — cross-kernel `UnifiedReceiptChain`** | [szl-kernels-live](https://huggingface.co/spaces/SZLHOLDINGS/szl-kernels-live) |

`suite.list_kernels()` returns the numeric core; `suite.list_series()` returns the govsign + blocked governance layer.

## The gap this closes

Today even SZL's own governance is fragmented: `szl-governed-norm` keeps its own receipt chain, the energy meter keeps its own ledger, and `szl-lambda-gate` keeps none. So a single forward pass yields three logs no third party can re-walk as one ordered, tamper-evident sequence. `UnifiedReceiptChain` is that missing artifact — op-agnostic SHA3-256 receipts that hash-chain norm, Λ, and energy calls into **one** verifiable stream, in call order. `szl-govsign` then makes that chain head third-party-verifiable; `szl-blocked` makes a refusal a recorded, first-class state and derives the compliance paperwork from it.

## Quickstart

```python
import torch
from kernels import get_kernel

# Current `kernels` (>=0.15) requires an explicit revision/version + trust flag for org kernels:
suite = get_kernel("SZLHOLDINGS/szl-kernels", revision="main", trust_remote_code=True)

print(suite.list_kernels())     # the 3 numeric suite members + honest roles
print(suite.list_series())      # the 2 governance-layer companions (govsign, blocked)
print(suite.selfcheck())        # one-shot CPU health: ALL checks pass

# ONE shared chain spanning multiple ops:
chain = suite.UnifiedReceiptChain()
x = torch.randn(4, 64)
y    = suite.governed_rms_norm(chain, x, eps=1e-6)                      # governed_norm
gate = suite.governed_lambda_gate(chain, torch.tensor([0.9,0.8,0.95]))  # lambda_gate (advisory)
e    = suite.governed_measure_energy(chain)                            # energy_core (MEASURED-only)

ok, depth, brk = chain.verify()       # the WHOLE pass verifies as ONE chain
print(ok, depth, chain.kernels_touched())   # True 3 ['governed_norm','lambda_gate','energy_core']
print(chain.to_json())                # export for offline third-party re-verification
```

### Flagship — a governed transformer sub-block

```python
blk = suite.GovernedBlock()
res = blk.forward(x, gov_axes=torch.tensor([0.95, 0.9, 0.92]))
print(res["chain_ok"], res["chain_depth"], res["kernels_touched"])
# norm + advisory Λ gate + energy + binding receipt = 4 ops, one verifiable chain.
# The Λ gate is ADVISORY: it is recorded for audit, it does NOT alter the numerics.
```

## API

| Symbol | What it does |
|---|---|
| `UnifiedReceiptChain` | Op-agnostic SHA3-256 hash chain. `emit`, `verify() -> (ok, depth, first_break)`, `kernels_touched()`, `to_json()`, `verify_json()` (offline). |
| `governed_rms_norm(chain, x, weight=None, eps=1e-6)` | RMSNorm + a receipt into the shared chain. Numerics match `szl-governed-norm`. |
| `governed_layer_norm(chain, x, ...)` | LayerNorm + receipt. |
| `governed_lambda_gate(chain, axes, weights=None, threshold=0.5)` | **Advisory** Λ gate; records an advisory receipt (`advisory=True`, never proven trust). |
| `governed_measure_energy(chain, measurement=None)` | Records an energy reading **verbatim** — `joules=None` + `UNAVAILABLE_NO_NVML` when no GPU. Never fabricated. |
| `GovernedBlock` | Pre-norm sub-block composing all three + a binding receipt into one auditable pass. |
| `list_kernels()`, `list_series()`, `get_member()`, `selfcheck()` | Numeric registry + governance-layer series + one-shot CPU health check. |

## Honesty (SZL doctrine)

- **Λ is advisory.** Its uniqueness is **Conjecture 1 — OPEN**. A recorded gate "pass" is a non-compensatory advisory signal, **never proven trust**.
- **Energy is MEASURED-only.** Real NVML cumulative-energy delta when a GPU is present; otherwise `joules=None`, labeled `UNAVAILABLE_NO_NVML`. **No joule is ever fabricated.**
- **The digest is an integrity fingerprint, not a signature.** SHA3-256 over a canonical receipt body proves tamper-evidence + ordering — not authorship. Signing is a separate, out-of-band layer — see [`szl-govsign`](https://huggingface.co/SZLHOLDINGS/szl-govsign) for DSSE / in-toto attestation.
- **Honest BLOCKED beats fake green.** A failed verification stays failed — see [`szl-blocked`](https://huggingface.co/SZLHOLDINGS/szl-blocked) for refusal as a first-class, provenanced state.
- **Universal (pure-Python) suite — a correctness + provenance reference, not a CUDA speed record. No fabricated benchmarks.** Suite tests: 7/7 passing.

## Provenance

Backed by the Lean 4 formalization [szl-holdings/lutar-lean](https://github.com/szl-holdings/lutar-lean) (749 declarations / 14 axioms / 163 tracked sorries), DOI [10.5281/zenodo.20434308](https://doi.org/10.5281/zenodo.20434308). Λ uniqueness = Conjecture 1 (open).

## See it live

- 🔮 Suite: [szl-kernels-live](https://huggingface.co/spaces/SZLHOLDINGS/szl-kernels-live) — holographic cross-kernel provenance graph with in-browser SHA3-256 + tamper / honest-BLOCKED demo.
- 🔮 Members: [governed-norm-holo](https://huggingface.co/spaces/SZLHOLDINGS/governed-norm-holo) · [lambda-gate-holo](https://huggingface.co/spaces/SZLHOLDINGS/lambda-gate-holo) · [energy-attest-holo](https://huggingface.co/spaces/SZLHOLDINGS/energy-attest-holo) · [receipt-chain-live](https://huggingface.co/spaces/SZLHOLDINGS/receipt-chain-live)
- 🔮 Governance layer: [szl-govsign-live](https://huggingface.co/spaces/SZLHOLDINGS/szl-govsign-live) · [szl-blocked-live](https://huggingface.co/spaces/SZLHOLDINGS/szl-blocked-live)
- 🔮 [szl-substrate](https://huggingface.co/spaces/SZLHOLDINGS/szl-substrate) — the hub tying the whole governed-compute substrate together.

## Compatibility

Python 3.9+, `torch>=2.5`, standard library + torch only. Runs on CPU and CUDA.

## License

Apache-2.0. Copyright 2026 SZL Holdings.

---

<sub><b>SZL Holdings</b> · unified governed-kernel suite · cross-kernel provenance · Λ advisory (Conjecture 1) · energy MEASURED-only · <a href="https://a11oy.net">a11oy.net</a> · <a href="https://github.com/szl-holdings">github.com/szl-holdings</a> · <a href="https://huggingface.co/SZLHOLDINGS">huggingface.co/SZLHOLDINGS</a></sub>
