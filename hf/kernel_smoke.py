# SPDX-License-Identifier: Apache-2.0
"""Smoke hook for hf-kernel-publish (szl-kernels Kernel Hub API).

`smoke(kernel)` receives the module that the `kernels` loader imported, from a local
build (check-local) or from the Hub (verify). Raise, or return False, to fail the run.
CPU-only, deterministic and fast. The result's repr is written to the receipt.

The Hub entrypoint is torch-ext/szl_kernels/_kernel_api.py (staged as __init__.py):
the fifteen published v1 exports plus governed_cosine_topk, with no MiniEmbed or
estate companions (those need numpy and the root vectors.npz / vocab.json).
"""
from __future__ import annotations

V1_EXPORTS = {
    "UnifiedReceiptChain", "tensor_digest", "GENESIS",
    "governed_rms_norm", "governed_layer_norm", "governed_lambda_gate",
    "governed_measure_energy", "GovernedBlock", "list_kernels", "list_series",
    "get_member", "selfcheck", "DOCTRINE_FOOTER", "PROVENANCE", "__version__",
}


def smoke(kernel):
    import torch

    exports = set(kernel.__all__)
    if exports != V1_EXPORTS | {"governed_cosine_topk"}:
        raise AssertionError(f"Kernel Hub API exports changed: {sorted(exports ^ (V1_EXPORTS | {'governed_cosine_topk'}))}")
    if hasattr(kernel, "MiniEmbed") or hasattr(kernel, "list_estate"):
        raise AssertionError("the pip distribution entrypoint was published instead of the Kernel Hub API")
    report = kernel.selfcheck()
    if not report["ok"]:
        raise AssertionError(f"selfcheck failed: {report}")
    chain = kernel.UnifiedReceiptChain()
    result = kernel.governed_cosine_topk(
        chain, torch.tensor([1.0, 0.0]), torch.tensor([[0.0, 1.0], [1.0, 0.0]]), k=1)
    if result["indices"].tolist() != [[1]] or chain.verify() != (True, 1, -1):
        raise AssertionError("governed_cosine_topk returned an unexpected result or receipt")
    return {"version": kernel.__version__, "selfcheck": report["ok"],
            "kernels_touched": report["kernels_touched"], "chain_head": report["chain_head"]}
