# SPDX-License-Identifier: Apache-2.0
"""Cosine reference parity, ordering, input binding, and fail-closed tests."""
import hashlib
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "torch-ext"))
from szl_kernels import UnifiedReceiptChain, governed_cosine_topk


def reference(query, documents, k):
    q = query.reshape(-1, query.shape[-1]).double()
    d = documents.double()
    q = q / torch.linalg.vector_norm(q, dim=1, keepdim=True)
    norm = torch.linalg.vector_norm(d, dim=1, keepdim=True)
    d = d / torch.where(norm == 0, torch.ones_like(norm), norm)
    all_scores = (q @ d.T).clamp(-1, 1)
    indices = torch.argsort(all_scores, dim=1, descending=True, stable=True)[:, :k]
    return indices, torch.gather(all_scores, 1, indices).float()


@pytest.mark.parametrize("block_rows", [1, 4, 17, 512])
@pytest.mark.parametrize("batch", [1, 3])
def test_numerical_reference_parity(block_rows, batch):
    gen = torch.Generator().manual_seed(192)
    q = torch.randn((batch, 13), generator=gen)
    docs = torch.randn((37, 13), generator=gen)
    chain = UnifiedReceiptChain()
    result = governed_cosine_topk(chain, q, docs, k=7, block_rows=block_rows)
    idx, scores = reference(q, docs, 7)
    assert torch.equal(result["indices"], idx)
    torch.testing.assert_close(result["scores"], scores, atol=2e-7, rtol=2e-6)
    assert chain.verify() == (True, 1, -1)
    assert result["receipt"]["attrs"]["max_similarity_elements"] == batch * min(block_rows, 37)


@pytest.mark.parametrize("block_rows", [1, 2, 3, 6])
def test_ties_cross_blocks_and_zero_documents(block_rows):
    q = torch.tensor([1.0, 0.0])
    docs = torch.tensor([[1., 0.], [0., 0.], [1., 0.], [-1., 0.], [0., 1.], [1., 0.]])
    out = governed_cosine_topk(UnifiedReceiptChain(), q, docs, k=6, block_rows=block_rows)
    assert out["indices"].tolist() == [[0, 2, 5, 1, 4, 3]]
    assert out["scores"].tolist() == [[1., 1., 1., 0., 0., -1.]]
    assert out["receipt"]["attrs"]["zero_document_count"] == 1


def test_raw_input_hashes_include_all_bytes_and_support_noncontiguous():
    q = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4).T
    docs = torch.arange(1, 34, dtype=torch.float32).reshape(3, 11).T
    assert not q.is_contiguous() and not docs.is_contiguous()
    out = governed_cosine_topk(UnifiedReceiptChain(), q, docs, k=4, block_rows=3)
    attrs = out["receipt"]["attrs"]
    assert attrs["query_sha256"] == hashlib.sha256(q.contiguous().numpy().tobytes()).hexdigest()
    assert attrs["documents_sha256"] == hashlib.sha256(docs.contiguous().numpy().tobytes()).hexdigest()
    assert attrs["scores_sha256"] == hashlib.sha256(out["scores"].numpy().tobytes()).hexdigest()
    assert attrs["indices_sha256"] == hashlib.sha256(out["indices"].numpy().tobytes()).hexdigest()
    changed = docs.clone()
    changed[-1, -1] = torch.nextafter(changed[-1, -1], torch.tensor(float("inf")))
    second = governed_cosine_topk(UnifiedReceiptChain(), q, changed, k=4, block_rows=3)
    assert second["receipt"]["attrs"]["documents_sha256"] != attrs["documents_sha256"]


def test_numerically_extreme_finite_rows():
    small = torch.finfo(torch.float32).tiny
    large = torch.finfo(torch.float32).max
    q = torch.tensor([[small, small], [large, large]])
    docs = torch.tensor([[large, large], [small, small], [-large, large], [0., 0.]])
    out = governed_cosine_topk(UnifiedReceiptChain(), q, docs, k=4, block_rows=1)
    assert out["indices"].tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert torch.isfinite(out["scores"]).all()
    torch.testing.assert_close(out["scores"], torch.tensor([[1., 1., 0., 0.], [1., 1., 0., 0.]]))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("side", ["query", "documents"])
def test_nonfinite_values_never_emit_partial_receipt(bad, side):
    q, docs = torch.ones((2, 3)), torch.ones((10, 3))
    (q if side == "query" else docs)[-1, -1] = bad
    chain = UnifiedReceiptChain()
    with pytest.raises(ValueError, match="finite"):
        governed_cosine_topk(chain, q, docs, block_rows=2)
    assert chain.count() == 0


@pytest.mark.parametrize("q,docs", [
    (torch.zeros(3), torch.ones((5, 3))),
    (torch.ones((0, 3)), torch.ones((5, 3))),
    (torch.ones(3), torch.ones((0, 3))),
    (torch.ones(0), torch.ones((5, 0))),
    (torch.ones((1, 1, 3)), torch.ones((5, 3))),
    (torch.ones(4), torch.ones((5, 3))),
    (torch.ones(3), torch.ones(3)),
])
def test_invalid_shapes_and_zero_queries(q, docs):
    chain = UnifiedReceiptChain()
    with pytest.raises(ValueError):
        governed_cosine_topk(chain, q, docs)
    assert chain.count() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.int64])
def test_dtype_rejected(dtype):
    chain = UnifiedReceiptChain()
    with pytest.raises(TypeError, match="float32"):
        governed_cosine_topk(chain, torch.ones(3, dtype=dtype), torch.ones((5, 3)))
    assert chain.count() == 0


@pytest.mark.parametrize("kw", [{"k": 0}, {"k": 6}, {"k": True}, {"k": 1.5},
                                  {"block_rows": 0}, {"block_rows": -1},
                                  {"block_rows": True}, {"block_rows": 2.5}])
def test_invalid_bounds(kw):
    chain = UnifiedReceiptChain()
    with pytest.raises(ValueError):
        governed_cosine_topk(chain, torch.ones(3), torch.ones((5, 3)), **kw)
    assert chain.count() == 0


def test_chunked_matrix_and_host_transfers(monkeypatch):
    shapes, transfers = [], []
    real_matmul, real_cpu = torch.matmul, torch.Tensor.cpu
    def tracked_matmul(a, b):
        shapes.append((tuple(a.shape), tuple(b.shape)))
        return real_matmul(a, b)
    def tracked_cpu(tensor, *args, **kwargs):
        transfers.append(tuple(tensor.shape))
        return real_cpu(tensor, *args, **kwargs)
    monkeypatch.setattr(torch, "matmul", tracked_matmul)
    monkeypatch.setattr(torch.Tensor, "cpu", tracked_cpu)
    governed_cosine_topk(UnifiedReceiptChain(), torch.ones((2, 9)), torch.ones((1031, 9)), k=3, block_rows=13)
    assert len(shapes) == 80
    assert all(a == (2, 9) and b[0] == 9 and b[1] <= 13 for a, b in shapes)
    assert all(shape[0] <= 13 for shape in transfers)


def test_no_input_mutation_or_gradient_and_existing_chain_continuity():
    q = torch.tensor([1., 2.], requires_grad=True)
    docs = torch.tensor([[1., 2.], [2., 1.]], requires_grad=True)
    before = docs.detach().clone()
    chain = UnifiedReceiptChain()
    prior = chain.emit("test", "before", {"purpose": "composition"})
    out = governed_cosine_topk(chain, q, docs, k=1)
    assert torch.equal(before, docs)
    assert not out["scores"].requires_grad
    assert out["receipt"]["prev"] == prior["digest"]
    assert chain.verify() == (True, 2, -1)
    assert out["receipt"]["attrs"]["receipt_authenticity"] == "UNSIGNED"
    assert not out["receipt"]["attrs"]["acceleration_claim"]


def test_source_mirror_and_export_wiring():
    for name in ("retrieval.py", "__init__.py"):
        assert (ROOT / "torch-ext/szl_kernels" / name).read_bytes() == (ROOT / "build/torch-universal/szl_kernels" / name).read_bytes()
    import szl_kernels
    assert "governed_cosine_topk" in szl_kernels.__all__
    assert szl_kernels.__version__ == "0.2.0"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; no GPU performance claim")
def test_cuda_reference_and_device_mismatch():
    q, docs = torch.randn((2, 7)), torch.randn((19, 7))
    expected = governed_cosine_topk(UnifiedReceiptChain(), q, docs, k=4)
    result = governed_cosine_topk(UnifiedReceiptChain(), q.cuda(), docs.cuda(), k=4, block_rows=3)
    assert torch.equal(result["indices"].cpu(), expected["indices"])
    torch.testing.assert_close(result["scores"].cpu(), expected["scores"], atol=2e-6, rtol=2e-5)
    with pytest.raises(ValueError, match="same device"):
        governed_cosine_topk(UnifiedReceiptChain(), q.cuda(), docs)
