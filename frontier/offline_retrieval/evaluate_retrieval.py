#!/usr/bin/env python3
"""Evaluate local retrieval models and BM25 on distinct, author-created queries."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (ARTIFACTS, HERE, REPO_ROOT, RetrievalBuildError, calibrated_threshold,
                    canonical_json_bytes, load_query_splits, sha256_file, write_json_atomic)
from navigator import OfflineNavigator


def load_benchmark(bench_root: Path):
    source_root = bench_root.resolve() / "src"
    expected = source_root / "szl_retrieval_bench"
    for name in ("__init__.py", "bm25.py", "metrics.py"):
        if not (expected / name).is_file():
            raise RetrievalBuildError(f"local benchmark implementation missing: {name}")
    sys.path.insert(0, str(source_root))
    modules = {name: importlib.import_module(f"szl_retrieval_bench.{name}") for name in ("bm25", "metrics")}
    for name, module in modules.items():
        if Path(module.__file__).resolve() != (expected / f"{name}.py").resolve():
            raise RetrievalBuildError("loaded benchmark implementation differs from --bench-root")
    hashes = {name: sha256_file(expected / name) for name in ("__init__.py", "bm25.py", "metrics.py")}
    return modules["bm25"].BM25, modules["metrics"], hashes


def source_ranking(chunks: list[dict[str, Any]]) -> list[str]:
    """Collapse chunk hits to unique source paths without changing first-hit order."""
    return list(dict.fromkeys(row["source_path"] for row in chunks))


def ranking_metrics(ranked: list[str], relevant: list[str], metrics) -> dict[str, float]:
    qrels = {path: 1 for path in relevant}
    return {
        "ndcg@5": float(metrics.ndcg(ranked, qrels, k=5)),
        "recall@5": float(metrics.recall_at(ranked, qrels, 5)),
        "P@5": float(metrics.precision_at_k(ranked, relevant, 5)["P@5"]),
        "R_precision": float(metrics.r_precision(ranked, relevant)["R_precision"]),
        "MRR": float(metrics.mrr(ranked, qrels)),
        "MAP": float(metrics.average_precision(ranked, qrels)),
    }


def evaluate(artifacts: Path, bench_root: Path, *, calibration_path: Path | None = None,
             evaluation_path: Path | None = None, device: str = "cpu") -> dict[str, Any]:
    calibration_path = calibration_path or HERE / "calibration_queries.json"
    evaluation_path = evaluation_path or HERE / "evaluation_queries.json"
    try:
        calibration_bytes = calibration_path.read_bytes()
        evaluation_bytes = evaluation_path.read_bytes()
    except OSError as exc:
        raise RetrievalBuildError(f"cannot capture evaluation inputs: {exc}") from exc
    calibration_hash = hashlib.sha256(calibration_bytes).hexdigest()
    evaluation_hash = hashlib.sha256(evaluation_bytes).hexdigest()
    calibration, evaluation = load_query_splits(
        calibration_path, evaluation_path, calibration_bytes=calibration_bytes,
        evaluation_bytes=evaluation_bytes,
    )
    navigator = OfflineNavigator(artifacts, device=device)
    if calibration_hash != navigator.manifest["calibration_queries_sha256"]:
        raise RetrievalBuildError("calibration query hash does not match trained model manifest")
    known = {d["source_path"] for d in navigator.documents}
    for row in calibration + evaluation:
        if set(row["relevant_paths"]) - known:
            raise RetrievalBuildError(f"query {row['id']} names a source outside the trained corpus")
    if not any(q["relevant_paths"] for q in evaluation) or not any(not q["relevant_paths"] for q in evaluation):
        raise RetrievalBuildError("evaluation requires both answerable and unanswerable queries")
    BM25, metrics, bench_hashes = load_benchmark(bench_root)
    baseline = BM25([d["text"] for d in navigator.documents])
    baseline_calibration = calibrated_threshold([
        (max(baseline.score(q["text"])), bool(q["relevant_paths"])) for q in calibration
    ])
    methods = {}
    for model_id in ("tfidf-v1", "lsa-v1", "bm25"):
        calibration_info = baseline_calibration if model_id == "bm25" else navigator.manifest["models"][model_id]["calibration"]
        threshold = float(calibration_info["threshold"])
        query_records = []
        metric_rows = []
        counts = {"answerable_retrieved": 0, "answerable_abstained": 0,
                  "unanswerable_abstained": 0, "unanswerable_retrieved": 0}
        for query in evaluation:
            if model_id == "bm25":
                scores = baseline.score(query["text"])
                if any(not math.isfinite(score) for score in scores):
                    raise RetrievalBuildError("BM25 produced nonfinite scores")
                order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
                ranked = source_ranking([navigator.documents[i] for i in order])
                top_score = scores[order[0]]
                abstained = top_score <= 0.0 or top_score < threshold
                if top_score <= 0.0:
                    ranked = []
                receipt_head = None
            else:
                result = navigator.rank(model_id, query["text"], top_k=len(navigator.documents))
                ranked = source_ranking(result["results"] or result["candidate_results"])
                top_score, abstained = result["top_score"], result["abstained"]
                receipt_head = result["receipt"]["chain_head"]
            answerable = bool(query["relevant_paths"])
            counts[("answerable" if answerable else "unanswerable") + ("_abstained" if abstained else "_retrieved")] += 1
            measured = ranking_metrics(ranked, query["relevant_paths"], metrics) if answerable else None
            if measured is not None:
                metric_rows.append(measured)
            query_records.append({
                "query_id": query["id"], "query_sha256": hashlib.sha256(query["text"].encode("utf-8")).hexdigest(),
                "answerable": answerable, "relevant_paths": query["relevant_paths"],
                "ranked_source_paths": ranked, "top_score": top_score, "abstained": abstained,
                "ranking_metrics_before_abstention": measured, "retrieval_chain_head": receipt_head,
            })
        tp, fn = counts["answerable_retrieved"], counts["answerable_abstained"]
        tn, fp = counts["unanswerable_abstained"], counts["unanswerable_retrieved"]
        sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
        methods[model_id] = {
            "calibration": calibration_info,
            "ranking_before_abstention": {key: sum(row[key] for row in metric_rows) / len(metric_rows) for key in metric_rows[0]},
            "abstention": {**counts, "answerable_recall": sensitivity,
                "unanswerable_specificity": specificity,
                "balanced_accuracy": (sensitivity + specificity) / 2,
                "coverage": (tp + fp) / len(evaluation)},
            "per_query": query_records,
        }
    report = {
        "schema_version": "szl.offline-retrieval-evaluation/v1",
        "status": "HOLD_EXTERNAL_EVALUATION_REQUIRED",
        "measurement": "MEASURED_LOCAL_AUTHOR_CREATED_QUERY_SET",
        "ranking_unit": "unique_source_path_first_chunk_hit",
        "ranking_scope": "answerable_queries_before_abstention",
        "ood_scope": "author_created_unanswerable_queries_not_general_ood_coverage",
        "calibration_queries": len(calibration), "evaluation_queries": len(evaluation),
        "answerable_queries": sum(bool(q["relevant_paths"]) for q in evaluation),
        "unanswerable_queries": sum(not q["relevant_paths"] for q in evaluation),
        "independent_adjudication": False, "device": str(navigator.device),
        "query_splits_disjoint_by_id_and_normalized_text": True,
        "input_sha256": {
            "calibration_queries.json": calibration_hash,
            "evaluation_queries.json": evaluation_hash,
            "model_manifest.json": navigator.manifest_sha256,
            "training_receipt.json": navigator.training_receipt_sha256,
            "benchmark_implementation": bench_hashes,
            "local_implementation": {name: sha256_file(HERE / name) for name in
                ("common.py", "train_retrieval.py", "navigator.py", "evaluate_retrieval.py")},
            "retrieval_kernel.py": sha256_file(REPO_ROOT / "build/torch-universal/szl_kernels/retrieval.py"),
        },
        "methods": methods,
        "claim_boundary": {**navigator.manifest["claim_boundary"], "acceleration_claim": False,
                           "general_ood_detection": False, "independent_benchmark_superiority": False},
        "signature_status": "UNSIGNED_HONEST",
    }
    report["report_content_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--bench-root", type=Path, default=REPO_ROOT.parent / "szl-retrieval-bench")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args(argv)
    report = evaluate(args.artifacts, args.bench_root, device=args.device)
    output = args.artifacts / "evaluation_report.json"
    write_json_atomic(output, report)
    print(json.dumps({"ok": True, "status": report["status"], "output": str(output),
        "methods": {name: {key: value[key] for key in ("ranking_before_abstention", "abstention")}
                    for name, value in report["methods"].items()}}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RetrievalBuildError as exc:
        print(json.dumps({"ok": False, "error": {"code": "RETRIEVAL_EVALUATION_FAILED", "message": str(exc)}}))
        raise SystemExit(2)
