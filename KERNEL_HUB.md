# Kernel Hub API and publication boundary

The first-class Hugging Face kernel and the Python distribution have different
entrypoints. The Kernel Hub API is the source-controlled
`torch-ext/szl_kernels/_kernel_api.py`, mirrored in
`build/torch-universal/szl_kernels/_kernel_api.py`. The release builder stages
this file as `build/torch-cpu/__init__.py` and as its compatibility package
entrypoint. It stages `_chain.py`, `_ops.py`, and `retrieval.py` beside both
entrypoints. Every import in this closure is standard-library, Torch, or relative
to those files.

The broader installed `szl_kernels` distribution retains its existing entrypoint
and separate MiniEmbed and estate companions. The fitted offline navigator and
its scikit-learn dependency are not part of this kernel publication. Their
evaluation/publication holds are unchanged.

## Compatibility and identity

The existing first-class `v1` branch at
`09818b62d683c33d200fca32e2ebfd95c64c65c7` contains only a `torch-cpu` build.
Its root and compatibility entrypoints have SHA-256
`96caac81dd719c785c9cb1458ac835352a8b45cbce7a5603a205b3a88319bbf0`.
Their fifteen public exports are preserved by the dedicated entrypoint, which
adds `governed_cosine_topk`. MiniEmbed and estate exports were never part of
that published v1 API.

The Python package version is `0.2.0`; the compatible Kernel Hub major API
version is integer `1`. All existing v1 variants must be updated together.
This document does not assert that a new provider revision is already published.
Publication requires the existing exact-source authorization, generated metadata
and digests, protected-source checks, and authoritative provider readback.

## Audited consumer example

Use an isolated environment with a supported first-class loader, such as
`kernels==0.16.1`, and compatible Torch. Review the published source binding and
verify the provider files before importing remote code. Prefer the immutable
provider commit from that verified binding:

```python
import os
import re

import torch
from kernels import get_kernel

# Set this to the actual 40-character HF kernel commit from verified readback.
# No placeholder is a real published revision.
revision = os.environ["SZL_REVIEWED_KERNEL_REVISION"]
if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
    raise ValueError("an audited immutable HF kernel revision is required")

suite = get_kernel(
    "SZLHOLDINGS/szl-kernels",
    revision=revision,
    trust_remote_code=True,  # Explicit opt-in to execute the audited SZL source.
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

After verified publication, `get_kernel("SZLHOLDINGS/szl-kernels", version=1,
trust_remote_code=True)` follows the mutable v1 branch. A branch reference is
not immutable provenance. Do not pass `version` and `revision` together.
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
