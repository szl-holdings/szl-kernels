---
thumbnail: https://huggingface.co/SZLHOLDINGS/szl-kernels/resolve/main/og-card.png
tags:
- kernel
- governance
- provenance
- suite
- sklearn
- embeddings
- word-embeddings
- feature-extraction
- doi:10.5281/zenodo.19944926
library_name: kernels
license: apache-2.0
szl-governance:
  verdict: ADVISORY
  lambda: "Conjecture 1 (open) — uniqueness unproven; advisory only"
  energy: MEASURED-only (real NVML delta; None when unavailable)
  provenance: UnifiedReceiptChain (SHA3-256, op-agnostic, cross-kernel)
  honest_blocked: "a failed check stays failed — never faked green"
---

<!-- SZL-ESTATE-CARD:v2:START -->
<p align="center"><a href="https://a-11-oy.com/"><img src="https://huggingface.co/spaces/SZLHOLDINGS/README/resolve/main/assets/estate-banner-v2.svg" alt="SZL Holdings — governed, receipted, verifiable" width="100%"></a></p>
<p align="center">
  <a href="https://github.com/szl-holdings/.github/tree/main/doctrine"><img src="https://img.shields.io/badge/doctrine-v11%20LOCKED-0B1F3A?style=flat-square" alt="doctrine v11"></a>
  <a href="https://a-11-oy.com/"><img src="https://img.shields.io/badge/evidence%20wall-LIVE%20%C2%B7%20verify%20in%20browser-3AF4C8?style=flat-square" alt="live evidence wall"></a>
  <a href="https://huggingface.co/datasets/SZLHOLDINGS/szl-lake"><img src="https://img.shields.io/badge/szl--lake-offline%20verifiable-C9B787?style=flat-square" alt="szl-lake offline verifiable"></a>
  <a href="https://huggingface.co/spaces/SZLHOLDINGS/holographic"><img src="https://img.shields.io/badge/estate%20map-holographic-5B8DEE?style=flat-square" alt="holographic estate map"></a>
</p>
<p align="center"><sub>Part of the <a href="https://huggingface.co/SZLHOLDINGS">SZL Holdings</a> governed estate — claims are designed to carry checkable receipts. Verification proves integrity &amp; origin, never accuracy or performance.</sub></p>
<!-- SZL-ESTATE-CARD:v2:END -->

<!-- SZL-ARTIFACT-NOTICE:v1:START — honesty plate: repo semantics, no fake model tags. -->
> **🟩 Kernel suite + REAL trained SZL-MiniEmbed.** The canonical governed-kernel runtime now rejects non-finite or out-of-range Λ thresholds before emitting a Lambda receipt. The receipted `corpus/kernels` snapshot remains byte-for-byte preserved for SZL-MiniEmbed replay, so the canonical runtime and receipted corpus intentionally differ at those files. Since **SZL-MiniEmbed v1** this repo ALSO ships real trained word embeddings — `vectors.npz` + `vocab.json` + `config.json` — built with **no gensim**: a distance-weighted term–term co-occurrence matrix over the SZL text estate (doctrine v10/v11 + rag-corpus-v1 + thesis-corpus-v18 + kernel-family READMEs), PPMI-weighted, reduced with sklearn `TruncatedSVD` to **128-dim** over a **3290**-term vocabulary. Evidence is **INTRINSIC SANITY ONLY** — receipted nearest-neighbour lists on 20 doctrine terms; **NO downstream/benchmark score is claimed**. The kernel suite stays authoritative. Λ = Conjecture 1 · ADVISORY.
<!-- SZL-ARTIFACT-NOTICE:v1:END -->


<p align="center">
  <a href="https://huggingface.co/SZLHOLDINGS/szl-kernels/tree/main/build/torch-universal/szl_kernels"><img src="https://img.shields.io/badge/kernel%20hub-torch--universal-5b8dee?style=flat-square" alt="kernel hub"></a>
  <a href="https://huggingface.co/SZLHOLDINGS/szl-kernels/blob/main/MODEL_PROVENANCE.json"><img src="https://img.shields.io/badge/provenance-MODEL_PROVENANCE.json-3af4c8?style=flat-square" alt="provenance"></a>
  <a href="https://huggingface.co/SZLHOLDINGS/szl-kernels/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-7e8aa3?style=flat-square" alt="license"></a>
</p>

# szl-kernels — the unified governed-kernel suite

**Canonical source:** [`szl-holdings/szl-kernels`](https://github.com/szl-holdings/szl-kernels).
The protected `main` workflow verifies the kernel suite, replays the MiniEmbed
artifact within its declared tolerance, and checks the source-binding contract
against the public Hugging Face artifact. Exact publication is performed by the
authorized release gateway in
[`szl-holdings/szl-forge`](https://github.com/szl-holdings/szl-forge/blob/main/.github/workflows/publish-szl-kernels.yml).
That gateway checks out an exact protected Git revision, publishes the declared
file set, and reads every byte back at the resulting immutable Hub revision.
The exact Git source revision is written to `publication.json`.

This is a governed kernel suite with a receipted word-embedding companion. It is
not a general-purpose language model, and its intrinsic nearest-neighbor replay
is not a downstream quality benchmark.

## Artifact truth card

| Lane | Classification | Evidence available here | Limitation |
|---|---|---|---|
| Governed kernels | Executable software | Source, manifests, tests, receipt-chain verifier, and `selfcheck()` | A successful self-check covers the exercised implementation path; it is not a safety, performance, or deployment claim. |
| SZL-MiniEmbed v1 | Trained embedding weights | `vectors.npz`, vocabulary/config files, bundled corpus, `TRAINING_RECEIPT.json`, and deterministic replay tooling | Small in-domain co-occurrence embedding with intrinsic sanity evidence only; no downstream benchmark or general-purpose capability claim. |
| Hub model-style repository | Distribution and presentation surface | Source-bound publication metadata and immutable-revision readback in the authorized release flow | A model API listing does not make every file trained weights, and Hub reachability is not runtime readiness. |

**Investor value.** The repository combines an auditable governed-compute
reference with a small, receipted learned artifact, while keeping software,
weights, and evidence visibly separate.

**Developer/evaluator path.** Review `publication.json` and
`MODEL_PROVENANCE.json`, run `suite.selfcheck()` for the software path, and run
`python scripts/eval.py` for the MiniEmbed replay. Treat returned results as
observations from that run; do not infer a green status from this card.

> **Kernel Hub migration (verified 2026-07-15):** `get_kernel(...)` now resolves
> the matching first-class [Kernel Hub repository](https://huggingface.co/kernels/SZLHOLDINGS/szl-kernels).
> Its `main` and stable `v1` refs both pin verified revision
> `06cc46f9733a844ee1c4cab558b06b3bd2d377ea`. This model-type repository is
> retained as the legacy source/card mirror.

**A kernel suite for governing provenance across operations.** This `get_kernel`-discoverable suite ties SZL Holdings' three governed kernels — [`szl-governed-norm`](https://huggingface.co/SZLHOLDINGS/szl-governed-norm), [`szl-lambda-gate`](https://huggingface.co/SZLHOLDINGS/szl-lambda-gate), and [`governed-inference-meter`](https://huggingface.co/SZLHOLDINGS/governed-inference-meter) — into **one shared, hash-chained `UnifiedReceiptChain`**, and anchors a governance/interop layer on top: [`szl-govsign`](https://huggingface.co/SZLHOLDINGS/szl-govsign) (signs the verdict), [`szl-blocked`](https://huggingface.co/SZLHOLDINGS/szl-blocked) (refuses honestly + derives an EU AI Act Annex IV draft), and [`szl-provctl`](https://huggingface.co/SZLHOLDINGS/szl-provctl) (verifies the provenance DAG + bridges to in-toto/SLSA).

> **Evidence boundary:** no ecosystem-wide novelty claim is made. Within this
> published suite, a forward pass touching norm + an advisory Λ gate + an energy
> reading can produce one auditable, tamper-evident log instead of three
> disconnected logs. Verify that bounded behavior with `selfcheck()` and the
> exported chain verifier before relying on it.

## Quickstart

```bash
pip install kernels torch
```

```python
import torch
from kernels import get_kernel

# Current `kernels` (>=0.15) requires an explicit revision/version + trust flag for org kernels:
suite = get_kernel("SZLHOLDINGS/szl-kernels", revision="main", trust_remote_code=True)

print(suite.list_kernels())     # the 3 numeric suite members + honest roles
print(suite.list_series())      # the governance/interop companions (govsign, blocked, provctl)
print(suite.selfcheck())        # inspect returned checks; this card assumes no pass

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

## Cookbook

Three copy-paste recipes spanning the governed-kernel series. Every printed value is
labeled **expected shape (not executed here)** — the shapes are transcribed from each
kernel's committed API, not from a run on this card (SZL doctrine: never self-download to
inflate counters, never fabricate an output). Λ stays **Conjecture 1 (OPEN)**; energy stays
**MEASURED-only**; a BLOCKED verdict stays BLOCKED.

### 1 — One receipt chain across three ops (suite)

```python
import torch
from kernels import get_kernel

suite = get_kernel("SZLHOLDINGS/szl-kernels", revision="main", trust_remote_code=True)

chain = suite.UnifiedReceiptChain()
x = torch.randn(4, 64)
suite.governed_rms_norm(chain, x, eps=1e-6)                        # op 1: governed_norm
suite.governed_lambda_gate(chain, torch.tensor([0.9, 0.8, 0.95]))  # op 2: lambda_gate (ADVISORY)
suite.governed_measure_energy(chain)                              # op 3: energy_core (MEASURED-only)

ok, depth, first_break = chain.verify()
print(ok, depth, chain.kernels_touched())
# expected shape (not executed here):
#   True 3 ['governed_norm', 'lambda_gate', 'energy_core']
#   -> one hash-chain, three ops, verifies as ONE ordered sequence.
#   The Λ gate receipt is ADVISORY (Conjecture 1, OPEN): recorded, never proven trust.
#   energy_core reports joules=None + UNAVAILABLE_NO_NVML on CPU — never a fabricated joule.
```

### 2 — honest-BLOCKED, not fake-green (szl-blocked)

```python
from kernels import get_kernel

blk = get_kernel("SZLHOLDINGS/szl-blocked", revision="main", trust_remote_code=True)

chain  = blk.UnifiedReceiptChain()
policy = blk.deny_if_action_in({"exfiltrate", "delete_all"})
work   = lambda v: v * 2

allowed = blk.governed_call(work, policy, chain, request={"action": "summarize"},  args=(21,))
blocked = blk.governed_call(work, policy, chain, request={"action": "exfiltrate"}, args=(21,))

print(allowed.blocked, allowed.output)
print(blocked.blocked, blocked.output)
# expected shape (not executed here):
#   False 42     -> ALLOWED path ran work(21); an ALLOW receipt is on the chain.
#   True None    -> BLOCKED path: work was NEVER called, output is None,
#                   a BLOCK receipt is recorded. Honest-BLOCKED, never faked green.
```

### 3 — Sign then verify a governance verdict (szl-govsign / DSSE)

```python
from kernels import get_kernel

gs = get_kernel("SZLHOLDINGS/szl-govsign", revision="main", trust_remote_code=True)

priv = gs.generate_ephemeral_keypair()   # production: Sigstore keyless / cosign key, out-of-band
pred = gs.build_governance_predicate(
    lambda_verdict = gs.LambdaVerdict(score=0.92, notes="advisory only — Conjecture 1 (OPEN)"),
    energy         = gs.EnergyLabel(value=12.5, unit="joules"),   # MEASURED-only
    decision       = gs.GovernanceDecision(status="ALLOWED", reason="passed gates"),
    honest_blocked = False,
)
subjects = [gs.Subject(name="szl_kernels/UnifiedReceiptChain", digest={"sha256": "<chain-head>"})]
envelope = gs.attest(subjects, pred, priv)

print(gs.verify(envelope, priv.public_key()))
# expected shape (not executed here):
#   True   -> DSSE envelope (ECDSA P-256) verifies: authorship + integrity of the verdict.
#            Any tamper -> verify() returns False (fails closed).
#            The signature does NOT upgrade Λ to proven trust: proven_trust is locked False.
```

> These recipes chain across three separately published, `get_kernel`-discoverable kernels.
> See [`szl-provctl`](https://huggingface.co/SZLHOLDINGS/szl-provctl) to turn any of these
> chains into documented in-toto v1 / SLSA v1 shapes for external compatibility testing.

## The governed-kernel series

Independently published, `get_kernel`-discoverable kernels that share one `UnifiedReceiptChain`. The first three are the **numeric core**; govsign + blocked + provctl are the **governance / interop layer**.

| Kernel | Lane | Live hologram |
|---|---|---|
| [`szl-governed-norm`](https://huggingface.co/SZLHOLDINGS/szl-governed-norm) | RMSNorm/LayerNorm + SHA3-256 receipts | [`governed-norm-holo`](https://szlholdings-governed-norm-holo.static.hf.space) ✅ **live** |
| [`szl-lambda-gate`](https://huggingface.co/SZLHOLDINGS/szl-lambda-gate) | advisory Λ gate (Conjecture 1, OPEN) | [`lambda-gate-holo`](https://szlholdings-lambda-gate-holo.static.hf.space) ✅ **live** |
| [`governed-inference-meter`](https://huggingface.co/SZLHOLDINGS/governed-inference-meter) | MEASURED-joule energy accounting | [`energy-attest-holo`](https://szlholdings-energy-attest-holo.static.hf.space) ✅ **live** |
| [`szl-govsign`](https://huggingface.co/SZLHOLDINGS/szl-govsign) | signed governance attestation (DSSE / in-toto, ECDSA P-256) | [`szl-govsign-live`](https://szlholdings-szl-govsign-live.static.hf.space) ✅ **live** |
| [`szl-blocked`](https://huggingface.co/SZLHOLDINGS/szl-blocked) | honest-BLOCKED first-class state + EU AI Act Annex IV DRAFT | [`szl-blocked-live`](https://szlholdings-szl-blocked-live.static.hf.space) ✅ **live** |
| [`szl-provctl`](https://huggingface.co/SZLHOLDINGS/szl-provctl) | provenance-DAG verify + in-toto v1 / SLSA v1 interop + per-kernel MEASURED energy | [`szl-provctl-live`](https://szlholdings-szl-provctl-live.static.hf.space) ✅ **live** |
| **`szl-kernels`** (this repo) | **unified suite — cross-kernel `UnifiedReceiptChain`** | [`szl-kernels-live`](https://szlholdings-szl-kernels-live.static.hf.space) ✅ **live** |

`suite.list_kernels()` returns the numeric core; `suite.list_series()` returns the govsign + blocked + provctl governance/interop layer.

### The honest-model trio (offline replays of the live Alloy surface)

Published as HF **model** repos (NOT trained models, NO weights — pure-Python, stdlib-only offline replays). Each ships a `library_name: kernels` card and MEASURED local test counts:

| Model | Lane | Tests (MEASURED) |
|---|---|---|
| [`szl-invariants`](https://huggingface.co/SZLHOLDINGS/szl-invariants) | 8 falsifiable receipt/ledger invariants, offline | 14/14 |
| [`szl-ouroboros`](https://huggingface.co/SZLHOLDINGS/szl-ouroboros) | bounded-loop trace + MEASURED/DERIVED loop-tax accounting | 13/13 |
| [`szl-formulas`](https://huggingface.co/SZLHOLDINGS/szl-formulas) | the 21 canonical formulas + governed-loop composer, PROOF-STATUS mirrored verbatim (locked-proven = exactly 8) | 17/17 |

## The gap this closes

The standalone SZL kernels keep separate receipt state. A single forward pass through them therefore yields logs that are not one ordered stream. `UnifiedReceiptChain` adds op-agnostic SHA3-256 receipts that hash-chain norm, Λ, and energy calls into **one** verifiable stream, in call order. `szl-govsign` can sign that chain head for verification against a separately trusted public key; `szl-blocked` records refusal as a first-class state and derives a draft documentation skeleton; `szl-provctl` verifies supplied multi-run provenance records and serializes them into documented in-toto/SLSA shapes for compatibility testing.

## API

| Symbol | What it does |
|---|---|
| `UnifiedReceiptChain` | Op-agnostic SHA3-256 hash chain. `emit`, `verify() -> (ok, depth, first_break)`, `kernels_touched()`, `to_json()`, `verify_json()` (offline). |
| `governed_rms_norm(chain, x, weight=None, eps=1e-6)` | RMSNorm + a receipt into the shared chain. Numerics match `szl-governed-norm`. |
| `governed_layer_norm(chain, x, ...)` | LayerNorm + receipt. |
| `governed_lambda_gate(chain, axes, weights=None, threshold=0.5)` | **Advisory** Λ gate; rejects non-finite or out-of-range thresholds before emitting a receipt, then records an advisory result (`advisory=True`, never proven trust). |
| `governed_measure_energy(chain, measurement=None)` | Records an energy reading **verbatim** — `joules=None` + `UNAVAILABLE_NO_NVML` when no GPU. Never fabricated. |
| `GovernedBlock` | Pre-norm sub-block composing all three + a binding receipt into one auditable pass. |
| `list_kernels()`, `list_series()`, `get_member()`, `selfcheck()` | Numeric registry + governance-layer series + one-shot CPU health check. |

## Honesty (SZL doctrine)

- **Λ is advisory.** Its uniqueness is **Conjecture 1 — OPEN**. A recorded gate "pass" is a non-compensatory advisory signal, **never proven trust**.
- **Energy is MEASURED-only.** Real NVML cumulative-energy delta when a GPU is present; otherwise `joules=None`, labeled `UNAVAILABLE_NO_NVML`. **No joule is ever fabricated.**
- **The digest is an integrity fingerprint, not a signature.** SHA3-256 over a canonical receipt body proves tamper-evidence + ordering — not authorship. Signing is a separate, out-of-band layer — see [`szl-govsign`](https://huggingface.co/SZLHOLDINGS/szl-govsign) for DSSE / in-toto attestation.
- **Honest BLOCKED beats fake green.** A failed verification stays failed — see [`szl-blocked`](https://huggingface.co/SZLHOLDINGS/szl-blocked) for refusal as a first-class, provenanced state.
- **Universal (pure-Python) suite: a correctness and provenance reference, not a CUDA speed record. No performance result or current test status is asserted by this card.**

## Provenance

Backed by the Lean 4 formalization [szl-holdings/lutar-lean](https://github.com/szl-holdings/lutar-lean) (749 declarations / 14 axioms / 163 tracked sorries), DOI [10.5281/zenodo.20434308](https://doi.org/10.5281/zenodo.20434308). Λ uniqueness = Conjecture 1 (open).

## Presentation and verification surfaces

- [Canonical Hugging Face card](https://huggingface.co/SZLHOLDINGS/szl-kernels)
- [Kernel Hub repository](https://huggingface.co/kernels/SZLHOLDINGS/szl-kernels)
- [Governed Kernels & Verifiers collection](https://huggingface.co/collections/SZLHOLDINGS/governed-kernels-and-verifiers-6a542ad83a4b75151bf5eae3)
- [SZL organization inventory](https://huggingface.co/SZLHOLDINGS)
- [Canonical GitHub source](https://github.com/szl-holdings/szl-kernels)

These links are navigation, not status badges. Availability, deployment state,
and current revision must be checked at evaluation time; a reachable page does
not establish correctness, performance, or runtime readiness.## Compatibility

Python 3.9+, `torch>=2.5`, standard library + torch only. Runs on CPU and CUDA.

## License

Apache-2.0. Copyright 2026 SZL Holdings.


## Trained SZL-MiniEmbed v1 (MEASURED — see `TRAINING_RECEIPT.json`)

Real word embeddings over the SZL text estate, produced **without gensim**: a distance-weighted
term–term co-occurrence matrix (window 5) → PPMI → sklearn `TruncatedSVD` → L2-normalized
vectors. Corpus = **26 documents / 26 source files** (every file's sha256 is recorded in
the receipt): `doctrine-v10-v11`, `rag-corpus-v1` (`corpus.jsonl`), `thesis-corpus-v18`, and the
kernel-family READMEs + `build/*.py`. Seed 20260721; the exact corpus text is bundled under
`corpus/` so the build is reproducible offline.

| property | value |
|---|---|
| vocabulary size | **3290** |
| embedding dim | **128** |
| co-occurrence window | 5 |
| SVD explained-variance ratio (MEASURED) | **0.3146** |
| doctrine probe terms in vocab | **20** / 20 |

### Intrinsic nearest-neighbour sanity (MEASURED, receipted)

Cosine nearest neighbours for doctrine terms — the **only** evidence claimed. This is intrinsic
sanity, **not a benchmark**: no analogy/retrieval score is asserted.

| term | top neighbours |
|---|---|
| `ouroboros` | substrate, replit, custodian, ouroboros-arch, payload, subsystems |
| `governance` | formal, score, first, layer, itself, system |
| `receipt` | chain, receipts, hash, emits, emitted, every |
| `provenance` | openmdw, chain, lineage, dags, composes, order |
| `lambda` | min, lam, float, emit, action, compute |
| `kernel` | discoverable, kernels, get, szl-kernels, hub, governed-kernel |
| `invariant` | learned, operator, knowledge, th11, reidemeister, knot |
| `tamper` | touched, kernels, break, verifies, verify, detected |
| `verify` | offline, break, tamper, depth, touched, brk |
| `conjecture` | uniqueness, depends, unproven, open, cauchy, honest |

```python
import numpy as np, json
V = np.load("vectors.npz")["vectors"]              # float32 [vocab, dim], L2-normalized
vocab = json.load(open("vocab.json"))["index"]     # {term: row}
def nn(term, k=6):
    v = V[vocab[term]]; s = V @ v
    return [(list(vocab)[i], float(s[i])) for i in np.argsort(-s)[1:k+1]]
print(nn("receipt"))
```

**Honest scope / blind spot:** these are distributional co-occurrence embeddings over a small
in-domain corpus (3290 terms). They capture SZL-doctrine term neighbourhoods; they are **not**
a general-purpose embedding model and carry **no** benchmark claim. Rare/out-of-vocab terms are
simply absent. The kernel suite remains the primary, authoritative artifact.

Re-verify everything: `python scripts/eval.py` (sha256-checks `vectors.npz` + `vocab.json`
against the receipt, regenerates the embeddings from the bundled corpus, and compares the
nearest-neighbour sets — mean Jaccard overlap ≥ 0.90 — and SVD variance within ±0.02).

---

<sub><b>SZL Holdings</b> · unified governed-kernel suite · cross-kernel provenance · Λ advisory (Conjecture 1) · energy MEASURED-only · <a href="https://a-11-oy.com">a-11-oy.com</a> · <a href="https://github.com/szl-holdings">github.com/szl-holdings</a> · <a href="https://huggingface.co/SZLHOLDINGS">huggingface.co/SZLHOLDINGS</a></sub>

---

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19944926.svg)](https://doi.org/10.5281/zenodo.19944926)

## Citation


**Cite this.** Part of the SZL Holdings *Ouroboros Thesis* (Governed Post-Determinism).  
Concept DOI (always-latest): [10.5281/zenodo.19944926](https://doi.org/10.5281/zenodo.19944926).  
Author: Stephen P. Lutar Jr. · [ORCID 0009-0001-0110-4173](https://orcid.org/0009-0001-0110-4173) · License CC-BY-4.0.  
Full DOI-pinned lineage (v1→v26) + the 8 papers: [szl-papers PAPERS_INDEX](https://github.com/szl-holdings/szl-papers/blob/main/PAPERS_INDEX.md).  
No artifact-specific DOI is minted for this model; the concept DOI above covers the program.

Honesty (Doctrine v11): Λ unconditional uniqueness is **Conjecture 1** (machine-checked FALSE as stated) — never a theorem; conditional uniqueness is **Theorem U** (axiom-free). Locked-proven formulas = **exactly 8** {F1,F4,F7,F11,F12,F18,F19,F22}; ~185 experimental theorems are a separate CI-green tier; Khipu BFT safety = Conjecture 2. Trust never 100%.

```bibtex
@misc{lutar_szl_ouroboros,
  author    = {Lutar, Stephen P., Jr.},
  title     = {SZL Holdings --- The Ouroboros Thesis (Governed Post-Determinism)},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19944926},
  url       = {https://doi.org/10.5281/zenodo.19944926},
  note      = {Concept DOI --- always resolves to the latest version. ORCID 0009-0001-0110-4173. CC-BY-4.0.}
}
```

*Signed-off-by: Stephen Lutar <stephenlutar2@gmail.com>*

## Files in this repo

| Path | What it is |
|---|---|
| `build/torch-universal/szl_kernels/__init__.py` | public API — suite entry points + `selfcheck()` |
| `build/torch-universal/szl_kernels/_chain.py` | cross-kernel `UnifiedReceiptChain` (SHA3-256) |
| `build/torch-universal/szl_kernels/_ops.py` | the governed op set |
| `tests/test_suite.py` | suite test |
| `build.toml` · `metadata.json` | Kernel Hub build/metadata manifests |
| `LICENSE` · `SECURITY.md` | Apache-2.0 · security policy |

---

<p align="center">
  <a href="https://huggingface.co/SZLHOLDINGS">SZL Holdings</a> ·
  <a href="https://a-11-oy.com">a-11-oy.com</a> ·
  <a href="https://huggingface.co/SZLHOLDINGS/a11oy-v19-substrate">a11oy-v19-substrate</a>
</p>

<p align="center"><sub>SLSA: L1 honest · L2 attested · L3 roadmap. Λ = Conjecture 1. Trust ceiling 0.97.</sub></p>
