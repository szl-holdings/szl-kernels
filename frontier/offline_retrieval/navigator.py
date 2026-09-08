#!/usr/bin/env python3
"""Offline source-bound retrieval over locally fitted TF-IDF or LSA models."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import torch

from common import (
    ARTIFACTS,
    REPO_ROOT,
    RetrievalBuildError,
    parse_json_bytes,
)


def _safe_npz(encoded: bytes, name: str, key: str) -> np.ndarray:
    try:
        with np.load(io.BytesIO(encoded), allow_pickle=False) as payload:
            if payload.files != [key]:
                raise RetrievalBuildError(f"unexpected arrays in {name}: {payload.files}")
            value = np.asarray(payload[key], dtype=np.float32)
    except (OSError, ValueError, KeyError) as exc:
        raise RetrievalBuildError(f"cannot load safe NumPy artifact {name}: {exc}") from exc
    if value.dtype == object or not np.isfinite(value).all():
        raise RetrievalBuildError(f"unsafe or non-finite array in {name}")
    return np.ascontiguousarray(value)


class OfflineNavigator:
    """A small in-memory snapshot; retraining on disk cannot mix model generations."""

    def __init__(self, artifacts: Path = ARTIFACTS, *, device: str = "cpu") -> None:
        self.artifacts = artifacts.resolve()
        self.device = self._resolve_device(device)
        self.manifest_path = self.artifacts / "model_manifest.json"
        manifest_bytes = self._read_bytes(self.manifest_path)
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        self.manifest = parse_json_bytes(manifest_bytes, label="model_manifest.json")
        if not isinstance(self.manifest, dict) or self.manifest.get("schema_version") != "szl.offline-source-navigator/v1":
            raise RetrievalBuildError("unsupported model manifest schema")
        self._artifact_bytes: dict[str, bytes] = {}
        self._verify_artifacts()
        document_payload = parse_json_bytes(self._snapshot_bytes("documents.json"), label="documents.json")
        if not isinstance(document_payload, dict) or document_payload.get("schema_version") != "szl.offline-retrieval-documents/v1":
            raise RetrievalBuildError("unsupported document artifact schema")
        self.documents = document_payload.get("documents")
        if not isinstance(self.documents, list) or len(self.documents) != self.manifest.get("document_count"):
            raise RetrievalBuildError("document artifact count does not match manifest")
        for document in self.documents:
            if not isinstance(document, dict) or not isinstance(document.get("text"), str):
                raise RetrievalBuildError("invalid document record")
            if hashlib.sha256(document["text"].encode("utf-8")).hexdigest() != document.get("chunk_sha256"):
                raise RetrievalBuildError("document text does not match chunk hash")
        self.vectorizer = self._load_vectorizer()
        self._model_cache: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}

    @staticmethod
    def _read_bytes(path: Path) -> bytes:
        try:
            return path.read_bytes()
        except OSError as exc:
            raise RetrievalBuildError(f"cannot read artifact {path.name}: {exc}") from exc

    def _snapshot_bytes(self, name: Any) -> bytes:
        if not isinstance(name, str) or name not in self._artifact_bytes:
            raise RetrievalBuildError("artifact reference is not in the verified manifest")
        return self._artifact_bytes[name]

    def _artifact_path(self, name: Any) -> Path:
        if not isinstance(name, str) or Path(name).name != name or name not in self.manifest["artifact_sha256"]:
            raise RetrievalBuildError("artifact reference is not in the verified manifest")
        candidate = (self.artifacts / name).resolve()
        if candidate.parent != self.artifacts:
            raise RetrievalBuildError("artifact reference escapes artifact directory")
        return candidate

    @staticmethod
    def _resolve_device(value: str) -> torch.device:
        if value not in {"cpu", "cuda", "auto"}:
            raise RetrievalBuildError(f"unsupported device: {value}")
        if value == "auto":
            value = "cuda" if torch.cuda.is_available() else "cpu"
        if value == "cuda" and not torch.cuda.is_available():
            raise RetrievalBuildError("CUDA requested but torch.cuda.is_available() is false")
        return torch.device(value)

    def _verify_artifacts(self) -> None:
        expected = self.manifest.get("artifact_sha256")
        if not isinstance(expected, dict) or not expected:
            raise RetrievalBuildError("model manifest has no artifact hashes")
        for name, digest in expected.items():
            if not isinstance(name, str) or Path(name).name != name:
                raise RetrievalBuildError(f"unsafe artifact name in manifest: {name!r}")
            if not isinstance(digest, str) or len(digest) != 64:
                raise RetrievalBuildError(f"invalid artifact digest for {name}")
            encoded = self._read_bytes(self._artifact_path(name))
            observed = hashlib.sha256(encoded).hexdigest()
            if observed != digest:
                raise RetrievalBuildError(f"artifact hash mismatch: {name}")
            self._artifact_bytes[name] = encoded
        receipt_path = self.artifacts / "training_receipt.json"
        receipt_bytes = self._read_bytes(receipt_path)
        self.training_receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
        receipt = parse_json_bytes(receipt_bytes, label="training_receipt.json")
        if not isinstance(receipt, dict) or receipt.get("model_manifest_sha256") != self.manifest_sha256:
            raise RetrievalBuildError("training receipt does not bind the current model manifest")
        chain = receipt.get("chain")
        if not isinstance(chain, list):
            raise RetrievalBuildError("training receipt chain is missing")
        sys.path.insert(0, str(REPO_ROOT / "build" / "torch-universal"))
        from szl_kernels import UnifiedReceiptChain

        ok, depth, first_break = UnifiedReceiptChain.verify_json(
            json.dumps(chain, sort_keys=True, separators=(",", ":"))
        )
        if not ok or first_break != -1 or depth != receipt.get("chain_depth") or not chain:
            raise RetrievalBuildError("training receipt chain verification failed")
        if chain[-1].get("digest") != receipt.get("chain_head") or any(row.get("seq") != i for i, row in enumerate(chain)):
            raise RetrievalBuildError("training receipt head or sequence mismatch")
        self._snapshot_bytes("documents.json")

    def _load_vectorizer(self) -> TfidfVectorizer:
        payload = parse_json_bytes(self._snapshot_bytes(self.manifest.get("vectorizer")), label="vectorizer.json")
        if not isinstance(payload, dict) or payload.get("schema_version") != "szl.tfidf-vectorizer/v1":
            raise RetrievalBuildError("unsupported vectorizer schema")
        vocabulary = payload.get("vocabulary")
        idf = payload.get("idf")
        if not isinstance(vocabulary, dict) or not isinstance(idf, list):
            raise RetrievalBuildError("vectorizer vocabulary or idf is missing")
        indices = sorted(vocabulary.values())
        if indices != list(range(len(vocabulary))) or len(idf) != len(vocabulary):
            raise RetrievalBuildError("vectorizer vocabulary indices are not canonical")
        idf_array = np.asarray(idf, dtype=np.float32)
        if not np.isfinite(idf_array).all() or np.any(idf_array <= 0):
            raise RetrievalBuildError("vectorizer idf is invalid")
        vectorizer = TfidfVectorizer(
            vocabulary={str(term): int(index) for term, index in vocabulary.items()},
            lowercase=payload.get("lowercase") is True,
            token_pattern=str(payload.get("token_pattern")),
            ngram_range=tuple(payload.get("ngram_range")),
            sublinear_tf=payload.get("sublinear_tf") is True,
            norm=str(payload.get("norm")),
            dtype=np.float32,
        )
        vectorizer.fit([" ".join(vocabulary)])
        vectorizer.idf_ = idf_array
        return vectorizer

    def model_ids(self) -> list[str]:
        models = self.manifest.get("models")
        if not isinstance(models, dict):
            raise RetrievalBuildError("manifest models are missing")
        return sorted(models)

    def _load_model(self, model_id: str) -> tuple[np.ndarray, np.ndarray | None]:
        if model_id in self._model_cache:
            return self._model_cache[model_id]
        models = self.manifest.get("models")
        if not isinstance(models, dict) or model_id not in models:
            raise RetrievalBuildError(f"unknown model {model_id!r}; available: {self.model_ids()}")
        model = models[model_id]
        documents = _safe_npz(self._snapshot_bytes(model["document_vectors"]), model["document_vectors"], "vectors")
        components = None
        if model_id == "lsa-v1":
            components = _safe_npz(self._snapshot_bytes(model["components"]), model["components"], "components")
            if documents.ndim != 2 or components.ndim != 2 or components.shape[0] != documents.shape[1] or components.shape[1] != len(self.vectorizer.vocabulary_):
                raise RetrievalBuildError("LSA components and document vectors are incompatible")
        if documents.ndim != 2 or documents.shape[0] != len(self.documents):
            raise RetrievalBuildError(f"invalid document matrix for {model_id}")
        if documents.shape[1] != int(model.get("dimension") or 0):
            raise RetrievalBuildError(f"document matrix dimension mismatch for {model_id}")
        self._model_cache[model_id] = (documents, components)
        return documents, components

    def query_vector(self, model_id: str, query: str) -> np.ndarray:
        if not isinstance(query, str) or not query.strip():
            raise RetrievalBuildError("query must be a non-empty string")
        documents, components = self._load_model(model_id)
        sparse = self.vectorizer.transform([query.strip()])
        if model_id == "tfidf-v1":
            vector = np.asarray(sparse.toarray(), dtype=np.float32)
        elif model_id == "lsa-v1":
            assert components is not None
            vector = np.asarray(sparse @ components.T, dtype=np.float32)
        else:
            raise RetrievalBuildError(f"unsupported model: {model_id}")
        vector = normalize(vector, norm="l2", copy=False).astype(np.float32)
        if not np.isfinite(vector).all() or vector.shape != (1, documents.shape[1]):
            raise RetrievalBuildError("query vector is invalid")
        return np.ascontiguousarray(vector)

    def rank(self, model_id: str, query: str, *, top_k: int = 5) -> dict[str, Any]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise RetrievalBuildError("top_k must be a positive integer")
        documents, _ = self._load_model(model_id)
        vector = self.query_vector(model_id, query)
        k = min(top_k, len(self.documents))
        sys.path.insert(0, str(REPO_ROOT / "build" / "torch-universal"))
        from szl_kernels import UnifiedReceiptChain, governed_cosine_topk

        chain = UnifiedReceiptChain()
        zero_query = not bool(np.any(vector))
        result = governed_cosine_topk(
            chain,
            torch.from_numpy(vector).to(self.device),
            torch.from_numpy(documents).to(self.device),
            k=k,
            block_rows=512,
        ) if not zero_query else {
            "indices": torch.empty((1, 0), dtype=torch.int64),
            "scores": torch.empty((1, 0), dtype=torch.float32),
        }
        indices = result.get("indices")
        scores = result.get("scores")
        if not isinstance(indices, torch.Tensor) or not isinstance(scores, torch.Tensor):
            raise RetrievalBuildError("retrieval kernel returned an invalid result")
        index_values = indices.detach().cpu().reshape(-1).tolist()
        score_values = scores.detach().cpu().reshape(-1).tolist()
        if not zero_query and (len(index_values) != k or len(score_values) != k):
            raise RetrievalBuildError("retrieval kernel returned the wrong number of rows")
        ranked: list[dict[str, Any]] = []
        for rank, (index, score) in enumerate(zip(index_values, score_values), start=1):
            document = self.documents[int(index)]
            ranked.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "document_id": document["document_id"],
                    "source_path": document["source_path"],
                    "source_sha256": document["source_sha256"],
                    "chunk_sha256": document["chunk_sha256"],
                    "title": document["title"],
                    "excerpt": document["text"][:500],
                }
            )
        model = self.manifest["models"][model_id]
        threshold = float(model["calibration"]["threshold"])
        if not math.isfinite(threshold):
            raise RetrievalBuildError("calibration threshold must be finite")
        top_score = ranked[0]["score"] if ranked else 0.0
        abstained = zero_query or top_score < threshold
        chain.emit("offline_source_navigator", "abstain" if abstained else "retrieve", {
            "model_id": model_id,
            "model_manifest_sha256": self.manifest_sha256,
            "query_text_sha256": hashlib.sha256(query.strip().encode("utf-8")).hexdigest(),
            "query_vector_sha256": hashlib.sha256(vector.tobytes(order="C")).hexdigest(),
            "threshold": threshold, "top_score": top_score, "top_k": top_k,
            "abstained": abstained, "kernel_called": not zero_query,
            "reason": "NO_MODEL_VOCABULARY" if zero_query else "CALIBRATED_SCORE",
        })
        ok, depth, first_break = chain.verify()
        if not ok or first_break != -1:
            raise RetrievalBuildError("retrieval receipt chain failed verification")
        receipt_chain = json.loads(chain.to_json())
        for record in receipt_chain:
            record.pop("ts", None)
        return {
            "ok": True,
            "operation": "offline-source-navigator-query",
            "model_id": model_id,
            "model_kind": model["kind"],
            "status": "ABSTAIN_INSUFFICIENT_LOCAL_EVIDENCE" if abstained else "RETRIEVED_LOCAL_EVIDENCE",
            "abstained": abstained,
            "abstention_reason": "NO_MODEL_VOCABULARY" if zero_query else ("BELOW_CALIBRATED_THRESHOLD" if abstained else None),
            "threshold": threshold,
            "top_score": top_score,
            "device": str(self.device),
            "results": [] if abstained else ranked,
            "candidate_results": ranked if abstained else [],
            "receipt": {
                "signature_status": "UNSIGNED_HONEST",
                "chain_head": chain.head(),
                "chain_depth": depth,
                "chain": receipt_chain,
                "model_manifest_sha256": self.manifest_sha256,
            },
            "claim_boundary": self.manifest["claim_boundary"],
        }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--query", required=True)
    value.add_argument("--model", default="lsa-v1", choices=("tfidf-v1", "lsa-v1"))
    value.add_argument("--top-k", type=int, default=5)
    value.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    value.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    navigator = OfflineNavigator(args.artifacts, device=args.device)
    print(json.dumps(navigator.rank(args.model, args.query, top_k=args.top_k), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RetrievalBuildError as exc:
        print(json.dumps({"ok": False, "error": {"code": "RETRIEVAL_QUERY_FAILED", "message": str(exc)}}))
        raise SystemExit(2)
