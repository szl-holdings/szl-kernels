---
license: apache-2.0
library_name: kernels
tags:
- kernel
- szl-holdings
szl:
  source_repo: szl-holdings/szl-kernels
  proof_url: https://github.com/szl-holdings/szl-kernels
---


<p><a href="https://huggingface.co/spaces/SZLHOLDINGS/szl-command-lab"><img src="https://raw.githubusercontent.com/szl-holdings/szl-brand/27d12e77fc0f9f1a0e5fae8b34adf2380d51d070/kit/logos/orbit-v2/suite/szl-icon-32.svg" alt="SZL Holdings" width="40" height="40" /></a></p>

# SZL Kernels · Native Kernel Hub

Inspect the closed Torch kernel API for receipted operations and cosine retrieval, with an immutable publication and loading contract.

**Artifact:** Native software kernel · Torch CPU API · **Stage:** Reference implementation · publication requires verified source

[Explore in Command Lab](https://huggingface.co/spaces/SZLHOLDINGS/szl-command-lab) · [Build](https://github.com/szl-holdings/szl-kernels) · [Evidence](https://github.com/szl-holdings/szl-kernels/blob/331b259cf6b6dde17c4bdacc245e61c7addff506/KERNEL_HUB.md)

## Before you use it

- Review the exact immutable provider revision and source binding before executing remote kernel code.
- Receipts are unsigned integrity records; they do not prove authorship, retrieval quality or a measured performance advantage.
- This repository is written only by the `hf-kernel-publish` workflow of [szl-holdings/szl-kernels](https://github.com/szl-holdings/szl-kernels); `SZL_KERNEL_SOURCE.json` names the GitHub commit each build came from.

<details>
<summary>Technical details and original evidence</summary>

The retained source below is exact and may contain historical observations. Its dates, use restrictions, licenses and evidence boundaries continue to apply.

<!-- SZL-PRESERVED-TECHNICAL-BODY:START -->
# Kernel Hub API and publication boundary

The first-class Hugging Face kernel and the Python distribution have different
entrypoints. The Kernel Hub API is the source-controlled
`torch-ext/szl_kernels/_kernel_api.py`, mirrored in
`build/torch-universal/szl_kernels/_kernel_api.py`. The `hf-kernel-publish`
workflow stages this file as `build/torch-cpu/szl_kernels/__init__.py` and
`build/torch-universal/szl_kernels/__init__.py`, with `_chain.py`, `_ops.py`, and
`retrieval.py` beside it. The two variants carry identical bytes. Every import in
this closure is standard-library, Torch, or relative to those files.

The broader installed `szl_kernels` distribution retains its existing entrypoint
and separate MiniEmbed and estate companions. The fitted offline navigator and
its scikit-learn dependency are not part of this kernel publication. Their
evaluation/publication holds are unchanged.

## Compatibility and identity

The first `v1` builds (July and 2026-10-03, kernel-builder) contained only a
`torch-cpu` variant, so `version=1` did not resolve on CUDA hosts. Their
entrypoints had SHA-256
`96caac81dd719c785c9cb1458ac835352a8b45cbce7a5603a205b3a88319bbf0`.
Their fifteen public exports are preserved by the dedicated entrypoint, which
adds `governed_cosine_topk`. MiniEmbed and estate exports were never part of
that published v1 API, and they are not published to this kernel repository.

The Python package version is `0.2.0`; the compatible Kernel Hub major API
version is integer `1`. Every release of szl-holdings/szl-kernels republishes
`main` and `v1` together, with both `build/torch-cpu` and `build/torch-universal`,
and the workflow then loads the result anonymously through `get_kernel` and
compares the served bytes with the GitHub source before it reports success.
The kernel-builder signature file of the earlier v1 builds is not carried
forward: these builds are unsigned.

## Audited consumer example

Use an isolated environment with `kernels>=0.17.2` and compatible Torch. The
`v1` branch follows each release:

```python
from kernels import get_kernel

suite = get_kernel("SZLHOLDINGS/szl-kernels", version=1, trust_remote_code=True)
```

For production, pin an immutable commit of this kernel repository. Review the
published `SZL_KERNEL_SOURCE.json` and the provider files at that commit before
importing remote code:

```python
import torch
from huggingface_hub import HfApi
from kernels import get_kernel

refs = HfApi().list_repo_refs("SZLHOLDINGS/szl-kernels", repo_type="kernel")
revision = next(b.target_commit for b in refs.branches if b.name == "v1")

suite = get_kernel(
    "SZLHOLDINGS/szl-kernels",
    revision=revision,
    trust_remote_code=["SZLHOLDINGS/szl-kernels"],  # Opt in to this repository's code only.
)
assert suite.__version__ == "0.2.0"
chain = suite.UnifiedReceiptChain()
result = suite.governed_cosine_topk(
    chain,
    torch.tensor([1.0, 0.0]),
    torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    k=1,
)
assert result["indices"].tolist() == [[1]]
assert chain.verify() == (True, 1, -1)
```

`version=1` follows the mutable v1 branch. A branch reference is not immutable
provenance; record the commit in your lockfile. Commits of the model repository
`SZLHOLDINGS/szl-kernels` (which also hosts the SZL-MiniEmbed table) are not
kernel revisions. Do not pass `version` and `revision` together.
The explicit remote-code opt-in does not authenticate the publisher or certify
the code; it acknowledges execution of source the consumer has separately
reviewed.

## Numerical and claim boundaries

Retrieval hashes logical C-order raw tensor bytes in native byte order. A Torch
uint8 view preserves float32 bit patterns, including negative zero, and int64
index values without a NumPy dependency. Row-bounded tensor copies feed Python
byte buffers of at most 65,536 bytes. That bound is not an allocator, BLAS, sorting
workspace, or total tensor-memory bound.

The operation is a correctness/provenance reference, not a speed claim. Receipts
are unsigned hash-chain records, not proof of authorship or retrieval quality.
The test suite checks legacy exports, closed imports, CPU numerical results,
known raw-byte hashes, byte-buffer bounds, and operation with
`Tensor.numpy` disabled. No model-promotion, independent evaluation, clinical
fitness, or general-world usefulness claim follows from these tests.

<!-- SZL-PRESERVED-TECHNICAL-BODY:END -->

</details>


Artwork: SZL Orbit · © 2026 SZL Holdings / Stephen Lutar · [CC BY 4.0](https://github.com/szl-holdings/szl-brand/blob/27d12e77fc0f9f1a0e5fae8b34adf2380d51d070/kit/LICENSE).
