# SPDX-License-Identifier: Apache-2.0
"""Bounded float32 cosine retrieval with exact-byte input/output receipts.

Pure PyTorch reference implementation; no acceleration or retrieval-quality claim.
Inputs must remain unchanged until this inference-only call returns. Hashes bind
logical C-order raw tensor bytes, shape and dtype; they are not signatures.
Zero document vectors receive similarity zero. Zero query vectors are rejected.
"""
from __future__ import annotations

import hashlib
import sys
from typing import Any, Dict

import torch

from ._chain import UnifiedReceiptChain


def _update_raw_hash(digest: Any, tensor: torch.Tensor, rows: int) -> None:
    """Transfer/hash at most rows logical rows per chunk, not the whole table."""
    matrix = tensor.unsqueeze(0) if tensor.ndim == 1 else tensor
    for start in range(0, matrix.shape[0], rows):
        chunk = matrix[start:start + rows].detach().contiguous().cpu()
        digest.update(chunk.numpy().tobytes(order="C"))


def _raw_sha256(tensor: torch.Tensor, rows: int) -> str:
    digest = hashlib.sha256()
    _update_raw_hash(digest, tensor, rows)
    return digest.hexdigest()


def _unit_rows(matrix: torch.Tensor, *, reject_zero: bool) -> torch.Tensor:
    """Scale first so very small/large finite float32 rows normalize safely."""
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("inputs must contain only finite values")
    scale = matrix.abs().amax(dim=1, keepdim=True)
    zero = scale == 0
    if reject_zero and bool(zero.any()):
        raise ValueError("query vectors must have nonzero norm")
    scaled = matrix / torch.where(zero, torch.ones_like(scale), scale)
    norms = torch.linalg.vector_norm(scaled, ord=2, dim=1, keepdim=True)
    return scaled / torch.where(zero, torch.ones_like(norms), norms)


def _merge_topk(
    scores: torch.Tensor, indices: torch.Tensor,
    other_scores: torch.Tensor, other_indices: torch.Tensor, k: int,
):
    scores = torch.cat((scores, other_scores), dim=1)
    indices = torch.cat((indices, other_indices), dim=1)
    # Primary descending score, secondary ascending absolute document index.
    by_index = torch.argsort(indices, dim=1, stable=True)
    indices = torch.gather(indices, 1, by_index)
    scores = torch.gather(scores, 1, by_index)
    by_score = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]
    return torch.gather(scores, 1, by_score), torch.gather(indices, 1, by_score)


def governed_cosine_topk(
    chain: UnifiedReceiptChain,
    query: torch.Tensor,
    documents: torch.Tensor,
    *,
    k: int = 5,
    block_rows: int = 512,
) -> Dict[str, Any]:
    """Return indices/scores of shape [B,k] and one UnifiedReceiptChain receipt.

    query: dense float32 [D] or [B,D]; documents: dense float32 [N,D].
    Both must use the same CPU or CUDA device. 1 <= k <= N, block_rows > 0.
    Ties use ascending document index, including ties across block boundaries.

    Similarity matrices contain at most B*min(block_rows,N) elements. Normalized
    document storage is one [block_rows,D] chunk, and candidate merging uses
    at most [B,2*k]. This is not a bound on allocator, BLAS or sorting workspace.
    Exact hashes use raw logical row-major bytes without decimal rounding.
    No receipt is emitted unless all input chunks and output scores are finite.
    """
    if not isinstance(chain, UnifiedReceiptChain):
        raise TypeError("chain must be a UnifiedReceiptChain")
    for name, tensor in (("query", query), ("documents", documents)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.layout != torch.strided or tensor.dtype != torch.float32:
            raise TypeError(f"{name} must be a dense strided float32 tensor")
        if tensor.device.type not in ("cpu", "cuda"):
            raise ValueError(f"{name} must be on CPU or CUDA")
    if query.ndim not in (1, 2) or documents.ndim != 2:
        raise ValueError("query must have shape [D] or [B,D]; documents [N,D]")
    if documents.shape[0] < 1 or documents.shape[1] < 1 or query.numel() < 1:
        raise ValueError("batch, document count, and vector dimensions must be positive")
    if query.shape[-1] != documents.shape[1]:
        raise ValueError("query and document dimensions must agree")
    if query.device != documents.device:
        raise ValueError("query and documents must be on the same device")
    if type(k) is not int or not 1 <= k <= documents.shape[0]:
        raise ValueError("k must be an integer in [1, document count]")
    if type(block_rows) is not int or block_rows < 1:
        raise ValueError("block_rows must be a positive integer")
    actual_rows = min(block_rows, documents.shape[0])
    hash_rows = min(actual_rows, 512)
    with torch.no_grad(), torch.autocast(device_type=query.device.type, enabled=False):
        query_snapshot = query.detach().clone(memory_format=torch.contiguous_format)
        q = query_snapshot.unsqueeze(0) if query.ndim == 1 else query_snapshot
        unit_query = _unit_rows(q, reject_zero=True)
        query_hash = _raw_sha256(query_snapshot, hash_rows)
        document_hash = hashlib.sha256()
        best_scores = torch.empty((q.shape[0], 0), dtype=torch.float32, device=q.device)
        best_indices = torch.empty((q.shape[0], 0), dtype=torch.int64, device=q.device)
        zero_documents = 0
        for start in range(0, documents.shape[0], actual_rows):
            chunk = documents[start:start + actual_rows].detach().clone(
                memory_format=torch.contiguous_format
            )
            normalized = _unit_rows(chunk, reject_zero=False)
            zero_documents += int((chunk.abs().amax(dim=1) == 0).sum())
            _update_raw_hash(document_hash, chunk, hash_rows)
            similarities = torch.matmul(unit_query, normalized.T).clamp(-1.0, 1.0)
            if not bool(torch.isfinite(similarities).all()):
                raise ValueError("cosine computation produced nonfinite scores")
            local_k = min(k, chunk.shape[0])
            local_order = torch.argsort(
                similarities, dim=1, descending=True, stable=True
            )[:, :local_k]
            local_scores = torch.gather(similarities, 1, local_order)
            local_indices = local_order + start
            best_scores, best_indices = _merge_topk(
                best_scores, best_indices, local_scores, local_indices, k
            )
        if not bool(torch.isfinite(best_scores).all()):
            raise ValueError("top-k computation produced nonfinite scores")
        attrs = {
            "schema": "szl.governed-cosine-topk/v1",
            "query_shape": list(query.shape),
            "documents_shape": list(documents.shape),
            "output_shape": list(best_scores.shape),
            "dtype": "float32",
            "index_dtype": "int64",
            "device": str(query.device),
            "byte_order": sys.byteorder,
            "input_hash_format": "logical_c_order_raw_bytes",
            "hash_algorithm": "sha256",
            "query_sha256": query_hash,
            "documents_sha256": document_hash.hexdigest(),
            "scores_sha256": _raw_sha256(best_scores, hash_rows),
            "indices_sha256": _raw_sha256(best_indices, hash_rows),
            "k": k,
            "block_rows": block_rows,
            "actual_block_rows": actual_rows,
            "max_similarity_elements": q.shape[0] * actual_rows,
            "zero_document_count": zero_documents,
            "zero_document_policy": "score_zero",
            "tie_break": "ascending_document_index",
            "implementation": "pytorch_float32_blocked_reference",
            "torch_version": str(torch.__version__),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_tf32_allowed": bool(torch.backends.cuda.matmul.allow_tf32)
            if query.device.type == "cuda" else None,
            "receipt_authenticity": "UNSIGNED",
            "retrieval_quality": "NOT_MEASURED",
            "acceleration_claim": False,
        }
        receipt = chain.emit("governed_retrieval", "cosine_topk", attrs)
    return {"indices": best_indices, "scores": best_scores, "receipt": receipt}
