#!/usr/bin/env python3
"""Fit deterministic TF-IDF and LSA retrieval models and write safe artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import platform
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from common import (
    ARTIFACTS,
    HERE,
    REPO_ROOT,
    RetrievalBuildError,
    build_documents,
    calibrated_threshold,
    load_query_splits,
    save_npz_atomic,
    sha256_file,
    write_json_atomic,
)


SEED = 20260905
TOKEN_PATTERN = r"(?u)\b[a-zA-Zλ][a-zA-Z0-9_\-λ]{1,}\b"


def top_scores(matrix: np.ndarray, queries: np.ndarray) -> list[float]:
    if matrix.ndim != 2 or queries.ndim != 2 or matrix.shape[1] != queries.shape[1]:
        raise RetrievalBuildError("calibration matrix shapes are incompatible")
    if not np.isfinite(matrix).all() or not np.isfinite(queries).all():
        raise RetrievalBuildError("calibration arrays contain non-finite values")
    return [float(value) for value in np.max(queries @ matrix.T, axis=1)]


def stripped_chain(chain: Any) -> list[dict[str, Any]]:
    records = json.loads(chain.to_json())
    for record in records:
        record.pop("ts", None)
    return records


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT / "build" / "torch-universal"))
    from szl_kernels import UnifiedReceiptChain

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    corpus_manifest = HERE / "corpus_manifest.json"
    calibration_path = HERE / "calibration_queries.json"
    documents, duplicate_sources = build_documents(corpus_manifest)
    calibration, _ = load_query_splits(calibration_path, HERE / "evaluation_queries.json")
    known_paths = {doc["source_path"] for doc in documents}
    for query in calibration:
        missing = set(query["relevant_paths"]) - known_paths
        if missing:
            raise RetrievalBuildError(f"calibration query {query['id']} references missing sources: {sorted(missing)}")

    texts = [document["text"] for document in documents]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=8192,
        norm="l2",
        dtype=np.float32,
    )
    sparse_tfidf = vectorizer.fit_transform(texts)
    if sparse_tfidf.shape[0] != len(documents) or sparse_tfidf.shape[1] < 32:
        raise RetrievalBuildError(f"invalid TF-IDF shape: {sparse_tfidf.shape}")
    tfidf_documents = np.asarray(sparse_tfidf.toarray(), dtype=np.float32)
    tfidf_documents = normalize(tfidf_documents, norm="l2", copy=False).astype(np.float32)

    lsa_dim = min(128, sparse_tfidf.shape[0] - 1, sparse_tfidf.shape[1] - 1)
    if lsa_dim < 2:
        raise RetrievalBuildError(f"cannot fit LSA with dimension {lsa_dim}")
    svd = TruncatedSVD(n_components=lsa_dim, n_iter=10, random_state=SEED)
    lsa_documents = svd.fit_transform(sparse_tfidf)
    lsa_documents = normalize(lsa_documents, norm="l2", copy=False).astype(np.float32)
    lsa_components = np.asarray(svd.components_, dtype=np.float32)

    calibration_sparse = vectorizer.transform([row["text"] for row in calibration])
    tfidf_queries = normalize(
        np.asarray(calibration_sparse.toarray(), dtype=np.float32), norm="l2", copy=False
    ).astype(np.float32)
    lsa_queries = normalize(
        np.asarray(svd.transform(calibration_sparse), dtype=np.float32), norm="l2", copy=False
    ).astype(np.float32)
    labels = [bool(row["relevant_paths"]) for row in calibration]
    tfidf_calibration = calibrated_threshold(list(zip(top_scores(tfidf_documents, tfidf_queries), labels)))
    lsa_calibration = calibrated_threshold(list(zip(top_scores(lsa_documents, lsa_queries), labels)))

    documents_path = ARTIFACTS / "documents.json"
    vectorizer_path = ARTIFACTS / "vectorizer.json"
    tfidf_path = ARTIFACTS / "tfidf-documents.npz"
    lsa_documents_path = ARTIFACTS / "lsa-documents.npz"
    lsa_components_path = ARTIFACTS / "lsa-components.npz"
    write_json_atomic(documents_path, {"schema_version": "szl.offline-retrieval-documents/v1", "documents": documents})
    ordered_vocab = sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1])
    write_json_atomic(
        vectorizer_path,
        {
            "schema_version": "szl.tfidf-vectorizer/v1",
            "vocabulary": {term: int(index) for term, index in ordered_vocab},
            "idf": [float(value) for value in vectorizer.idf_],
            "token_pattern": TOKEN_PATTERN,
            "ngram_range": [1, 2],
            "lowercase": True,
            "sublinear_tf": True,
            "norm": "l2",
        },
    )
    save_npz_atomic(tfidf_path, vectors=tfidf_documents)
    save_npz_atomic(lsa_documents_path, vectors=lsa_documents)
    save_npz_atomic(lsa_components_path, components=lsa_components)

    artifact_files = [documents_path, vectorizer_path, tfidf_path, lsa_documents_path, lsa_components_path]
    file_hashes = {path.name: sha256_file(path) for path in artifact_files}
    source_rows: list[dict[str, Any]] = []
    for document in documents:
        row = {
            "source_path": document["source_path"],
            "source_sha256": document["source_sha256"],
            "license": document["license"],
        }
        if row not in source_rows:
            source_rows.append(row)
    source_rows.sort(key=lambda item: item["source_path"])

    models = {
        "tfidf-v1": {
            "kind": "classical_tfidf_retrieval",
            "trained": True,
            "document_vectors": tfidf_path.name,
            "dimension": int(tfidf_documents.shape[1]),
            "calibration": tfidf_calibration,
        },
        "lsa-v1": {
            "kind": "classical_lsa_retrieval",
            "trained": True,
            "document_vectors": lsa_documents_path.name,
            "components": lsa_components_path.name,
            "dimension": int(lsa_documents.shape[1]),
            "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
            "calibration": lsa_calibration,
        },
    }
    manifest = {
        "schema_version": "szl.offline-source-navigator/v1",
        "seed": SEED,
        "document_count": len(documents),
        "source_count": len(source_rows),
        "duplicate_sources_excluded": duplicate_sources,
        "sources": source_rows,
        "corpus_manifest_sha256": sha256_file(corpus_manifest),
        "calibration_queries_sha256": sha256_file(calibration_path),
        "thread_lineage_sha256": sha256_file(HERE / "thread_lineage.md"),
        "vectorizer": vectorizer_path.name,
        "models": models,
        "artifact_sha256": file_hashes,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": __import__("sklearn").__version__,
        },
        "claim_boundary": {
            "publication_eligible": False,
            "evaluation_independently_adjudicated": False,
            "general_retrieval_quality": False,
            "clinical_validity": False,
            "production_runtime_witness": False,
            "network_used": False,
        },
    }
    manifest_path = ARTIFACTS / "model_manifest.json"
    write_json_atomic(manifest_path, manifest)

    chain = UnifiedReceiptChain()
    chain.emit(
        "offline_source_navigator",
        "admit_sources",
        {
            "corpus_manifest_sha256": manifest["corpus_manifest_sha256"],
            "document_count": len(documents),
            "source_count": len(source_rows),
            "duplicate_source_count": len(duplicate_sources),
        },
    )
    for model_id, model in models.items():
        chain.emit(
            "offline_source_navigator",
            "fit_model",
            {
                "model_id": model_id,
                "kind": model["kind"],
                "dimension": model["dimension"],
                "trained": True,
                "document_vectors_sha256": file_hashes[model["document_vectors"]],
            },
        )
        chain.emit(
            "offline_source_navigator",
            "calibrate_abstention",
            {
                "model_id": model_id,
                "threshold": model["calibration"]["threshold"],
                "balanced_accuracy": model["calibration"]["balanced_accuracy"],
                "answerable_recall": model["calibration"]["answerable_recall"],
                "unanswerable_specificity": model["calibration"]["unanswerable_specificity"],
                "query_set_sha256": manifest["calibration_queries_sha256"],
                "independent_adjudication": False,
            },
        )
    ok, depth, first_break = chain.verify()
    if not ok or first_break != -1:
        raise RetrievalBuildError("training receipt chain failed verification")
    receipt = {
        "schema_version": "szl.offline-retrieval-training-receipt/v1",
        "status": "MEASURED_LOCAL",
        "model_manifest_sha256": sha256_file(manifest_path),
        "chain_depth": depth,
        "chain_head": chain.head(),
        "chain": stripped_chain(chain),
        "signature_status": "UNSIGNED_HONEST",
        "claim_boundary": manifest["claim_boundary"],
    }
    write_json_atomic(ARTIFACTS / "training_receipt.json", receipt)
    print(
        json.dumps(
            {
                "ok": True,
                "operation": "train-offline-source-navigator",
                "models": sorted(models),
                "documents": len(documents),
                "sources": len(source_rows),
                "duplicates_excluded": len(duplicate_sources),
                "artifacts": str(ARTIFACTS),
                "training_receipt_chain_head": chain.head(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RetrievalBuildError as exc:
        print(json.dumps({"ok": False, "error": {"code": "RETRIEVAL_BUILD_FAILED", "message": str(exc)}}))
        raise SystemExit(2)
