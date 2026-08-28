# SPDX-License-Identifier: Apache-2.0
"""CPU tests for MiniEmbed — lookup/encode, UnifiedReceiptChain, honest LFS selfcheck."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build" / "torch-universal"))

import szl_kernels as sk  # noqa: E402
from szl_kernels.miniembed import MiniEmbed, PUBLISHED_SHA256, _is_lfs_pointer  # noqa: E402

SUITE_SELFCHECK_KEYS = (
    "norm_correct",
    "lambda_advisory",
    "energy_honest",
    "cross_kernel_verify",
    "spans_three_kernels",
    "offline_reverify",
    "tamper_detected",
    "block_forward",
)


def _write_tiny_table(tmp_path: Path, *, lfs_vectors: bool = False) -> Path:
    vocab = ["receipt", "chain", "lambda"]
    index = {term: i for i, term in enumerate(vocab)}
    rng = np.random.RandomState(0)
    table = rng.randn(3, 8).astype(np.float32)
    table /= np.linalg.norm(table, axis=1, keepdims=True)
    if lfs_vectors:
        (tmp_path / "vectors.npz").write_bytes(
            b"version https://git-lfs.github.com/spec/v1\n"
            b"oid sha256:" + b"ab" * 32 + b"\n"
            b"size 99\n"
        )
    else:
        np.savez_compressed(tmp_path / "vectors.npz", vectors=table)
    (tmp_path / "vocab.json").write_text(
        json.dumps({"vocab": vocab, "index": index}),
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model": "SZL-MiniEmbed",
                "method": "PPMI+TruncatedSVD",
                "dim": 8,
                "vocab_size": 3,
                "files": {
                    "vectors": "vectors.npz (key 'vectors', float32 [V,dim])",
                    "vocab": "vocab.json ({'vocab':[term...], 'index':{term:i}})",
                },
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_miniembed_exported_from_package() -> None:
    assert sk.MiniEmbed is MiniEmbed
    assert "MiniEmbed" in sk.__all__


def test_suite_selfcheck_still_eight_of_eight() -> None:
    result = sk.selfcheck()
    assert result["ok"] is True, result
    assert set(result["checks"]) == set(SUITE_SELFCHECK_KEYS)
    assert all(result["checks"].values()), result["checks"]
    assert result["kernels_touched"] == ["governed_norm", "lambda_gate", "energy_core"]


def test_torch_ext_mirrors_build_package() -> None:
    build = ROOT / "build" / "torch-universal" / "szl_kernels"
    ext = ROOT / "torch-ext" / "szl_kernels"
    names = sorted(path.name for path in build.iterdir() if path.is_file())
    assert "miniembed.py" in names
    for name in names:
        assert (build / name).read_bytes() == (ext / name).read_bytes(), name


def test_lookup_encode_match_npz_vocab_contract() -> None:
    embed = MiniEmbed(root=ROOT)
    if not embed.available:
        pytest.skip(f"MiniEmbed table unavailable: {embed.unavailable_label}")
    table = np.load(ROOT / "vectors.npz", allow_pickle=False)["vectors"]
    vocab = json.loads((ROOT / "vocab.json").read_text(encoding="utf-8"))
    term = "receipt"
    row = embed.lookup(term)
    assert row.dtype == np.float32
    assert row.shape == (128,)
    np.testing.assert_allclose(row, table[vocab["index"][term]], rtol=0, atol=0)
    matrix = embed.encode(["receipt", "chain"])
    assert matrix.shape == (2, 128)
    np.testing.assert_allclose(matrix[0], row, rtol=0, atol=0)
    assert embed.encode("lambda").shape == (128,)


def test_oov_raises_keyerror() -> None:
    embed = MiniEmbed(root=ROOT)
    if not embed.available:
        pytest.skip(f"MiniEmbed table unavailable: {embed.unavailable_label}")
    with pytest.raises(KeyError):
        embed.lookup("this-term-is-not-in-the-szl-miniembed-vocab")
    with pytest.raises(KeyError):
        embed.encode(["receipt", "this-term-is-not-in-the-szl-miniembed-vocab"])
    assert embed.neighbors("this-term-is-not-in-the-szl-miniembed-vocab") is None


def test_neighbors_match_training_receipt_when_loaded() -> None:
    embed = MiniEmbed(root=ROOT)
    if not embed.available:
        pytest.skip(f"MiniEmbed table unavailable: {embed.unavailable_label}")
    receipt = json.loads((ROOT / "TRAINING_RECEIPT.json").read_text(encoding="utf-8"))
    listed = receipt["metrics_MEASURED"]["intrinsic_nearest_neighbors"]
    for term, expected in listed.items():
        got = embed.neighbors(term, k=6)
        assert got is not None, term
        assert [word for word, _ in got] == [word for word, _ in expected], term


def test_selfcheck_matches_published_hashes_when_table_is_real() -> None:
    embed = MiniEmbed(root=ROOT)
    result = embed.selfcheck()
    if _is_lfs_pointer(ROOT / "vectors.npz"):
        assert result["ok"] is False
        assert result["label"] == "UNAVAILABLE_LFS"
        assert result["checks"]["honest_lfs"] is True
        assert result["checks"]["vectors.npz_sha256_match"] is not True
        return
    assert embed.available is True
    assert result["ok"] is True, result
    assert result["file_sha256"]["vectors.npz"] == PUBLISHED_SHA256["vectors.npz"]
    assert result["file_sha256"]["vocab.json"] == PUBLISHED_SHA256["vocab.json"]
    receipt = json.loads((ROOT / "TRAINING_RECEIPT.json").read_text(encoding="utf-8"))
    assert result["file_sha256"]["vectors.npz"] == receipt["model"]["files"]["vectors.npz"]
    assert result["file_sha256"]["vocab.json"] == receipt["model"]["files"]["vocab.json"]


def test_selfcheck_lfs_pointer_is_unavailable_not_a_pass(tmp_path: Path) -> None:
    root = _write_tiny_table(tmp_path, lfs_vectors=True)
    embed = MiniEmbed(root=root)
    assert embed.available is False
    assert embed.unavailable_label == "UNAVAILABLE_LFS"
    with pytest.raises(RuntimeError, match="UNAVAILABLE_LFS"):
        embed.lookup("receipt")
    result = embed.selfcheck()
    assert result["ok"] is False
    assert result["label"] == "UNAVAILABLE_LFS"
    assert result["checks"]["honest_lfs"] is True
    assert result["checks"]["vectors.npz_sha256_match"] is not True
    # Pointer bytes must not be reported as the published object hash.
    pointer_hash = hashlib.sha256((root / "vectors.npz").read_bytes()).hexdigest()
    assert pointer_hash != PUBLISHED_SHA256["vectors.npz"]
    assert result["file_sha256"]["vectors.npz"] is None


def test_table_and_lookup_receipts_use_unified_chain(tmp_path: Path) -> None:
    root = _write_tiny_table(tmp_path, lfs_vectors=False)
    chain = sk.UnifiedReceiptChain()
    embed = MiniEmbed(root=root, chain=chain)
    table_rec = embed.receipt_table()
    row = embed.lookup("receipt")
    assert row.shape == (8,)
    ok, depth, brk = chain.verify()
    assert ok is True and brk == -1 and depth == 2
    assert chain.kernels_touched() == ["miniembed"]
    assert table_rec["kernel"] == "miniembed" and table_rec["op"] == "table"
    lookup_rec = chain.tail(1)[0]
    assert lookup_rec["op"] == "lookup"
    assert lookup_rec["attrs"]["not_a_transformer_lm"] is True
    assert "Conjecture 1" in lookup_rec["attrs"]["lambda_status"]
    blob = chain.to_json()
    ok2, _, _ = sk.UnifiedReceiptChain.verify_json(blob)
    assert ok2 is True
    mutated = json.loads(blob)
    mutated[0]["attrs"]["vocab_size"] = 0
    ok3, _, brk3 = sk.UnifiedReceiptChain.verify_json(json.dumps(mutated))
    assert ok3 is False and brk3 == 0


def test_fixture_encode_does_not_require_gpu(tmp_path: Path) -> None:
    embed = MiniEmbed(root=_write_tiny_table(tmp_path))
    matrix = embed.encode(["chain", "lambda"])
    assert matrix.dtype == np.float32
    assert matrix.shape == (2, 8)
    assert "receipt" in embed
    assert len(embed) == 3


def test_source_states_not_transformer_chaski_khipu() -> None:
    source = (ROOT / "build" / "torch-universal" / "szl_kernels" / "miniembed.py").read_text(
        encoding="utf-8"
    )
    collapsed = " ".join(source.split())
    assert "NOT a transformer language model" in collapsed
    assert "NOT Chaski" in collapsed
    assert "NOT Khipu" in collapsed
    lowered = collapsed.lower()
    assert "tokens/s" not in lowered
    assert "tokens per second" not in lowered
    assert "tokens/sec" not in lowered


def test_metadata_name_is_legal_dashed_kernel_id() -> None:
    meta = json.loads(
        (ROOT / "build" / "torch-universal" / "szl_kernels" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert meta["name"] == "szl-kernels"
    assert meta["miniembed"]["kind"] == "distributional_word_embedding_table"
