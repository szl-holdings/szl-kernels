# SPDX-License-Identifier: Apache-2.0
"""Local ingestion, split separation, calibration and inference receipt contracts."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="install the frontier/dev extra to test offline retrieval")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "frontier/offline_retrieval"))
sys.path.insert(0, str(ROOT / "torch-ext"))
import common
from common import RetrievalBuildError, calibrated_threshold, load_query_splits, sha256_file, write_json_atomic
from navigator import OfflineNavigator
from evaluate_retrieval import evaluate, source_ranking
from szl_kernels import UnifiedReceiptChain


def query_file(path, split, rows):
    write_json_atomic(path, {"schema_version": "szl.offline-retrieval-queries/v1", "split": split,
                            "independent_adjudication": False, "queries": rows})


def row(identifier, text, relevant):
    return {"id": identifier, "text": text, "relevant_paths": relevant}


def test_calibration_reports_distinct_recall_and_specificity():
    result = calibrated_threshold([(0.5, True), (0.6, True), (0.7, True),
                                   (0.1, False), (0.2, False), (0.3, False), (0.8, False)])
    assert result["answerable_recall"] == 1.0
    assert result["unanswerable_specificity"] == 0.75
    assert result["balanced_accuracy"] == 0.875


def test_calibration_handles_bm25_range_and_rejects_nonfinite():
    result = calibrated_threshold([(20.0, True), (20.0, False)])
    assert result["threshold"] > 20.0
    assert result["answerable_recall"] == 0.0
    assert result["unanswerable_specificity"] == 1.0
    with pytest.raises(RetrievalBuildError, match="finite"):
        calibrated_threshold([(float("nan"), True), (1., False)])


@pytest.mark.parametrize("problem", ["same_id", "same_text", "wrong_split"])
def test_calibration_evaluation_separation(tmp_path, problem):
    cal, ev = tmp_path / "cal.json", tmp_path / "eval.json"
    query_file(cal, "calibration", [row("a", "ONE QUERY", ["a.md"])])
    query_file(ev, "calibration" if problem == "wrong_split" else "evaluation", [
        row("a" if problem == "same_id" else "b", " one   query " if problem == "same_text" else "different", [])])
    with pytest.raises(RetrievalBuildError):
        load_query_splits(cal, ev)


def test_ingestion_deduplicates_bytes_and_preserves_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "REPO_ROOT", tmp_path)
    sources = []
    for i in range(8):
        name = f"source{i}.md"
        (tmp_path / name).write_text(f"# Topic {i}\n\nUseful source material number {i}.", encoding="utf-8")
        sources.append({"path": name, "mode": "markdown"})
    (tmp_path / "copy.md").write_bytes((tmp_path / "source0.md").read_bytes())
    sources.append({"path": "copy.md", "mode": "markdown"})
    manifest = tmp_path / "manifest.json"
    write_json_atomic(manifest, {"schema_version": "szl.offline-retrieval-corpus/v1", "sources": sources})
    documents, duplicates = common.build_documents(manifest)
    assert len(documents) == 8 and len(duplicates) == 1
    for doc in documents:
        assert doc["source_sha256"] == sha256_file(tmp_path / doc["source_path"])
        assert doc["chunk_sha256"] == hashlib.sha256(doc["text"].encode()).hexdigest()
    assert duplicates[0]["duplicates"] == "source0.md"


def test_ingestion_rejects_escape_and_does_not_execute_python(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "REPO_ROOT", tmp_path)
    with pytest.raises(RetrievalBuildError, match="escapes"):
        common.safe_repo_file("../outside.md")
    path = tmp_path / "documented.py"
    path.write_text('"""Safe documentation."""\nraise RuntimeError("must not execute")\n', encoding="utf-8")
    assert common.python_docstrings(path) == "Module: Safe documentation."


@pytest.fixture
def artifact_bundle(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    cal, ev = tmp_path / "cal.json", tmp_path / "eval.json"
    query_file(cal, "calibration", [row("cal-1", "alpha evidence", ["a.md"]), row("cal-2", "absentwords", [])])
    query_file(ev, "evaluation", [row("eval-1", "alpha", ["a.md"]), row("eval-2", "unrepresentedtokens", [])])
    vectors = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    for name in ("tfidf.npz", "lsa.npz"):
        common.save_npz_atomic(artifacts / name, vectors=vectors)
    common.save_npz_atomic(artifacts / "components.npz", components=np.eye(2, dtype=np.float32))
    docs = []
    for i, (path, text) in enumerate((("a.md", "alpha evidence"), ("b.md", "beta notes"))):
        digest = hashlib.sha256(text.encode()).hexdigest()
        docs.append({"document_id": str(i), "source_path": path, "source_sha256": digest,
                     "chunk_sha256": digest, "title": path, "text": text})
    write_json_atomic(artifacts / "documents.json", {"schema_version": "szl.offline-retrieval-documents/v1", "documents": docs})
    write_json_atomic(artifacts / "vectorizer.json", {"schema_version": "szl.tfidf-vectorizer/v1",
        "vocabulary": {"alpha": 0, "beta": 1}, "idf": [1., 1.], "lowercase": True,
        "token_pattern": r"(?u)\b\w\w+\b", "ngram_range": [1, 1], "sublinear_tf": True, "norm": "l2"})
    calibration = {"threshold": 0.2, "balanced_accuracy": 1., "answerable_recall": 1., "unanswerable_specificity": 1.}
    manifest = {"schema_version": "szl.offline-source-navigator/v1", "document_count": 2,
        "vectorizer": "vectorizer.json", "calibration_queries_sha256": sha256_file(cal),
        "claim_boundary": {"evaluation_independently_adjudicated": False, "network_used": False},
        "models": {"tfidf-v1": {"kind": "classical_tfidf_retrieval", "document_vectors": "tfidf.npz", "dimension": 2, "calibration": calibration},
                   "lsa-v1": {"kind": "classical_lsa_retrieval", "document_vectors": "lsa.npz", "components": "components.npz", "dimension": 2, "calibration": calibration}},
        "artifact_sha256": {p.name: sha256_file(p) for p in artifacts.iterdir()}}
    write_json_atomic(artifacts / "model_manifest.json", manifest)
    rebind_receipt(artifacts)
    return artifacts, cal, ev


def rebind_receipt(artifacts):
    chain = UnifiedReceiptChain()
    chain.emit("synthetic_test_fixture", "fit", {"purpose": "test-only fixture"})
    write_json_atomic(artifacts / "training_receipt.json", {
        "model_manifest_sha256": sha256_file(artifacts / "model_manifest.json"),
        "chain": json.loads(chain.to_json()), "chain_depth": chain.count(), "chain_head": chain.head()})


@pytest.mark.parametrize("model", ["tfidf-v1", "lsa-v1"])
def test_navigator_returns_cited_results_and_receipted_zero_query(artifact_bundle, model):
    artifacts, _, _ = artifact_bundle
    navigator = OfflineNavigator(artifacts)
    result = navigator.rank(model, "alpha", top_k=2)
    assert not result["abstained"] and result["results"][0]["source_path"] == "a.md"
    assert result["receipt"]["chain_depth"] == 2
    zero = navigator.rank(model, "unrepresentedtokens", top_k=2)
    assert zero["abstained"] and zero["top_score"] == 0
    assert zero["results"] == [] and zero["candidate_results"] == []
    assert zero["abstention_reason"] == "NO_MODEL_VOCABULARY"
    assert zero["receipt"]["chain_depth"] == 1
    assert zero["receipt"]["chain"][0]["attrs"]["kernel_called"] is False
    assert UnifiedReceiptChain.verify_json(json.dumps(zero["receipt"]["chain"])) == (True, 1, -1)


def test_artifact_hash_mismatch_rejected(artifact_bundle):
    artifacts, _, _ = artifact_bundle
    (artifacts / "tfidf.npz").write_bytes(b"different artifact")
    with pytest.raises(RetrievalBuildError, match="hash mismatch"):
        OfflineNavigator(artifacts)


@pytest.mark.parametrize("model", ["tfidf-v1", "lsa-v1"])
def test_navigator_binds_snapshot_across_artifact_replacement(artifact_bundle, model):
    artifacts, _, _ = artifact_bundle
    navigator = OfflineNavigator(artifacts)
    original_hash = sha256_file(artifacts / "model_manifest.json")
    # No model arrays have been loaded yet: replacing the files must not affect
    # the bytes selected by this already-verified navigator instance.
    for name in ("tfidf.npz", "lsa.npz", "components.npz"):
        (artifacts / name).write_bytes(b"a different generation")
    (artifacts / "model_manifest.json").write_text("{}", encoding="utf-8")
    (artifacts / "training_receipt.json").write_text("{}", encoding="utf-8")
    result = navigator.rank(model, "alpha", top_k=2)
    assert result["results"][0]["source_path"] == "a.md"
    assert result["receipt"]["model_manifest_sha256"] == original_hash
    assert result["receipt"]["chain"][-1]["attrs"]["model_manifest_sha256"] == original_hash
    with pytest.raises(RetrievalBuildError):
        OfflineNavigator(artifacts)


def test_model_artifact_must_belong_to_hash_allowlist(artifact_bundle):
    artifacts, _, _ = artifact_bundle
    manifest = common.load_json(artifacts / "model_manifest.json")
    manifest["models"]["tfidf-v1"]["document_vectors"] = "unlisted.npz"
    write_json_atomic(artifacts / "model_manifest.json", manifest)
    rebind_receipt(artifacts)
    with pytest.raises(RetrievalBuildError, match="verified manifest"):
        OfflineNavigator(artifacts).rank("tfidf-v1", "alpha")


def test_evaluation_uses_benchmark_metrics_and_exact_inputs(artifact_bundle):
    artifacts, cal, ev = artifact_bundle
    bench = ROOT.parent / "szl-retrieval-bench"
    if not bench.is_dir():
        pytest.skip("sibling benchmark checkout not present; evaluator refuses downloads")
    report = evaluate(artifacts, bench, calibration_path=cal, evaluation_path=ev)
    assert report["status"] == "HOLD_EXTERNAL_EVALUATION_REQUIRED"
    assert report["input_sha256"]["evaluation_queries.json"] == sha256_file(ev)
    assert set(report["methods"]) == {"tfidf-v1", "lsa-v1", "bm25"}
    for result in report["methods"].values():
        assert result["ranking_before_abstention"]["recall@5"] == 1.
        assert result["ranking_before_abstention"]["P@5"] == 0.2
        assert result["abstention"]["unanswerable_specificity"] == 1.
    reported_hash = report.pop("report_content_sha256")
    assert reported_hash == hashlib.sha256(common.canonical_json_bytes(report)).hexdigest()
    query_file(cal, "calibration", [row("new1", "alpha different", ["a.md"]), row("new2", "unseen", [])])
    with pytest.raises(RetrievalBuildError, match="calibration query hash"):
        evaluate(artifacts, bench, calibration_path=cal, evaluation_path=ev)


def test_rank_unit_is_unique_source_path():
    assert source_ranking([{"source_path": "b"}, {"source_path": "b"}, {"source_path": "a"}]) == ["b", "a"]


def test_evaluation_binds_captured_query_bytes(artifact_bundle, monkeypatch):
    import evaluate_retrieval as evaluator
    artifacts, cal, ev = artifact_bundle
    bench = ROOT.parent / "szl-retrieval-bench"
    if not bench.is_dir():
        pytest.skip("sibling benchmark checkout not present; evaluator refuses downloads")
    expected_cal, expected_ev = sha256_file(cal), sha256_file(ev)
    original_navigator = evaluator.OfflineNavigator

    def replace_queries_after_capture(*args, **kwargs):
        navigator = original_navigator(*args, **kwargs)
        cal.write_text("{}", encoding="utf-8")
        ev.write_text("{}", encoding="utf-8")
        return navigator

    monkeypatch.setattr(evaluator, "OfflineNavigator", replace_queries_after_capture)
    report = evaluator.evaluate(artifacts, bench, calibration_path=cal, evaluation_path=ev)
    assert report["input_sha256"]["calibration_queries.json"] == expected_cal
    assert report["input_sha256"]["evaluation_queries.json"] == expected_ev
    assert report["evaluation_queries"] == 2
    assert report["methods"]["tfidf-v1"]["per_query"][0]["query_id"] == "eval-1"


def test_npz_serialization_is_byte_reproducible(tmp_path):
    import zipfile
    arrays = {"vectors": np.arange(84, dtype=np.float32).reshape(12, 7)}
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    common.save_npz_atomic(first, **arrays)
    common.save_npz_atomic(second, **arrays)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as bundle:
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in bundle.infolist())
