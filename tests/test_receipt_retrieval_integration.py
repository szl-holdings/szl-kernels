"""CPU integration of exact-source retrieval with detached/checkpointed receipts.

These import real local source, not Hugging Face or a simulated kernel client.
The small synthetic vectors test numerical behavior, not retrieval quality.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
from types import ModuleType
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "build/torch-universal/szl_kernels"
PACKAGE = "_szl_receipt_retrieval_integration"
package = ModuleType(PACKAGE)
package.__path__ = [str(SOURCE)]
sys.modules[PACKAGE] = package


def load_source(name):
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", SOURCE / f"{name}.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("source spec unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


chain_module = load_source("_chain")
retrieval = load_source("retrieval")
Chain = chain_module.UnifiedReceiptChain


def raw_hash(tensor):
    """Independent small-fixture byte oracle, not retrieval's hashing helper."""
    code = "f" if tensor.dtype == torch.float32 else "q"
    order = "<" if sys.byteorder == "little" else ">"
    values = tensor.detach().reshape(-1).cpu().tolist()
    return hashlib.sha256(struct.pack(f"{order}{len(values)}{code}", *values)).hexdigest()


def reference(q, d, k):
    q = q.reshape(-1, q.shape[-1]).double()
    d = d.double()
    q = q / torch.linalg.vector_norm(q, dim=1, keepdim=True)
    norms = torch.linalg.vector_norm(d, dim=1, keepdim=True)
    d = d / torch.where(norms == 0, torch.ones_like(norms), norms)
    similarities = (q @ d.T).clamp(-1, 1)
    indices = torch.argsort(similarities, dim=1, descending=True, stable=True)[:, :k]
    return indices, torch.gather(similarities, 1, indices)


class RetrievalIntegrationTests(unittest.TestCase):
    def test_dense_reference_across_block_sizes(self):
        generator = torch.Generator(device="cpu").manual_seed(519)
        for batch in (1, 3):
            q = torch.randn(batch, 13, generator=generator)
            d = torch.randn(37, 13, generator=generator)
            expected_i, expected_s = reference(q, d, 7)
            for rows in (1, 4, 17, 512):
                with self.subTest(batch=batch, rows=rows):
                    c = Chain()
                    actual = retrieval.governed_cosine_topk(c, q, d, k=7, block_rows=rows)
                    self.assertTrue(torch.equal(actual["indices"], expected_i))
                    torch.testing.assert_close(actual["scores"].double(), expected_s, rtol=1e-5, atol=1e-6)
                    cp = c.checkpoint()
                    self.assertEqual(c.verify(expected_head=cp["head"], expected_depth=cp["depth"]), (True, 1, -1))

    def test_tie_order_across_blocks_and_zero_documents(self):
        q = torch.tensor([1.0, 0.0])
        d = torch.tensor([[0., 1.], [1., 0.], [1., -0.], [-1., 0.], [0., 0.]])
        for rows in (1, 2, 4, 10):
            with self.subTest(rows=rows):
                c = Chain(); out = retrieval.governed_cosine_topk(c, q, d, k=5, block_rows=rows)
                self.assertEqual(out["indices"].tolist(), [[1, 2, 0, 4, 3]])
                self.assertEqual(out["scores"].tolist(), [[1., 1., 0., 0., -1.]])
                self.assertEqual(out["receipt"]["attrs"]["zero_document_count"], 1)

    def test_raw_hashes_match_independent_struct_bytes(self):
        q = torch.tensor([1.0, -0.0])
        d = torch.tensor([[0., 1.], [1., 0.], [-1., -0.]])
        c = Chain(); out = retrieval.governed_cosine_topk(c, q, d, k=3, block_rows=1)
        attrs = out["receipt"]["attrs"]
        for field, tensor in (("query", q), ("documents", d), ("scores", out["scores"]), ("indices", out["indices"])):
            self.assertEqual(attrs[f"{field}_sha256"], raw_hash(tensor))

    def test_noncontiguous_vectors(self):
        generator = torch.Generator(device="cpu").manual_seed(71)
        q = torch.randn(3, 18, generator=generator)[:, ::2]
        d = torch.randn(17, 18, generator=generator)[:, ::2]
        self.assertFalse(d.is_contiguous())
        c = Chain(); out = retrieval.governed_cosine_topk(c, q, d, k=4, block_rows=3)
        indices, scores = reference(q, d, 4)
        self.assertTrue(torch.equal(indices, out["indices"]))
        torch.testing.assert_close(out["scores"].double(), scores, rtol=1e-5, atol=1e-6)
        self.assertEqual(out["receipt"]["attrs"]["documents_sha256"], raw_hash(d))

    def test_returned_retrieval_receipt_cannot_rewrite_history(self):
        c = Chain(); out = retrieval.governed_cosine_topk(c, torch.tensor([1., 0.]), torch.eye(2), k=1)
        cp = c.checkpoint(); original = c.to_json()
        out["receipt"]["attrs"]["query_shape"][0] = 999
        out["receipt"]["digest"] = "f" * 64
        self.assertEqual(c.to_json(), original)
        self.assertEqual(c.verify(expected_head=cp["head"], expected_depth=cp["depth"]), (True, 1, -1))

    def test_multi_operation_checkpoint_rejects_retrieval_omission(self):
        c = Chain()
        c.emit_norm("rms_norm", torch.tensor([2., 1.]), torch.tensor([1., 0.5]), 1e-6)
        retrieval.governed_cosine_topk(c, torch.tensor([1., 0.]), torch.eye(2), k=1)
        cp = c.checkpoint(); truncated = json.dumps(json.loads(c.to_json())[:-1])
        self.assertEqual(Chain.verify_json(truncated), (True, 1, -1))
        self.assertFalse(Chain.verify_json(truncated, expected_head=cp["head"], expected_depth=cp["depth"])[0])

    def test_zero_query_rejected_without_receipt(self):
        c = Chain()
        with self.assertRaises(ValueError):
            retrieval.governed_cosine_topk(c, torch.zeros(2), torch.eye(2), k=1)
        self.assertEqual(c.count(), 0)

    def test_late_nonfinite_document_rejected_without_partial_receipt(self):
        c = Chain(); d = torch.tensor([[1., 0.], [0., 1.], [float("nan"), 0.]])
        with self.assertRaises(ValueError):
            retrieval.governed_cosine_topk(c, torch.tensor([1., 0.]), d, k=1, block_rows=1)
        self.assertEqual(c.count(), 0)

    def test_source_copies_are_identical(self):
        other = ROOT / "torch-ext/szl_kernels/_chain.py"
        self.assertEqual((SOURCE / "_chain.py").read_bytes(), other.read_bytes())

    def test_output_keeps_unsigned_unmeasured_claims(self):
        c = Chain(); out = retrieval.governed_cosine_topk(c, torch.tensor([1., 0.]), torch.eye(2), k=1)
        attrs = out["receipt"]["attrs"]
        self.assertEqual(attrs["receipt_authenticity"], "UNSIGNED")
        self.assertEqual(attrs["retrieval_quality"], "NOT_MEASURED")
        self.assertIs(attrs["acceleration_claim"], False)


if __name__ == "__main__":
    unittest.main()
