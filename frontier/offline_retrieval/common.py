"""Shared, fail-closed helpers for the offline retrieval build."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
ARTIFACTS = HERE / "artifacts"


class RetrievalBuildError(RuntimeError):
    """A stable, expected build or validation failure."""


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RetrievalBuildError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json_bytes(encoded: bytes, *, label: str) -> Any:
    """Parse the same immutable bytes that callers bind with a content hash."""
    try:
        return json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetrievalBuildError(f"cannot parse JSON {label}: {exc}") from exc


def load_json(path: Path) -> Any:
    try:
        encoded = path.read_bytes()
    except OSError as exc:
        raise RetrievalBuildError(f"cannot read JSON {path}: {exc}") from exc
    return parse_json_bytes(encoded, label=str(path))


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            Path(temporary).unlink(missing_ok=True)
        except OSError:
            pass


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    for name, array in arrays.items():
        if array.dtype == object:
            raise RetrievalBuildError(f"unsafe object array refused: {name}")
        if not np.isfinite(array).all():
            raise RetrievalBuildError(f"non-finite array refused: {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp.npz"
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise RetrievalBuildError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def safe_repo_file(relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise RetrievalBuildError(f"invalid repository-relative path: {relative!r}")
    candidate = (REPO_ROOT / relative).resolve()
    try:
        candidate.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise RetrievalBuildError(f"path escapes repository: {relative}") from exc
    if not candidate.is_file():
        raise RetrievalBuildError(f"source file is missing: {relative}")
    return candidate


_SPACE = re.compile(r"[ \t\f\v]+")
_BLANKS = re.compile(r"\n{3,}")


def normalize_text(value: str) -> str:
    lines = [_SPACE.sub(" ", line).strip() for line in value.replace("\r", "").split("\n")]
    return _BLANKS.sub("\n\n", "\n".join(lines)).strip()


def python_docstrings(path: Path) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise RetrievalBuildError(f"cannot parse Python source {path}: {exc}") from exc
    pieces: list[str] = []
    module_doc = ast.get_docstring(tree, clean=True)
    if module_doc:
        pieces.append(f"Module: {module_doc}")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        doc = ast.get_docstring(node, clean=True)
        if doc:
            kind = "Class" if isinstance(node, ast.ClassDef) else "Function"
            pieces.append(f"{kind} {node.name}: {doc}")
    return normalize_text("\n\n".join(pieces))


def chunk_text(text: str, *, maximum_chars: int = 1400) -> list[tuple[str, str]]:
    text = normalize_text(text)
    if not text:
        return []
    sections: list[tuple[str, list[str]]] = []
    title = "Document"
    paragraphs: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        heading = re.match(r"^#{1,6}\s+(.+)$", paragraph.splitlines()[0])
        if heading:
            if paragraphs:
                sections.append((title, paragraphs))
                paragraphs = []
            title = heading.group(1).strip()
            remaining = "\n".join(paragraph.splitlines()[1:]).strip()
            if remaining:
                paragraphs.append(remaining)
        else:
            paragraphs.append(paragraph)
    if paragraphs:
        sections.append((title, paragraphs))

    chunks: list[tuple[str, str]] = []
    for section_title, section_paragraphs in sections:
        current = ""
        part = 1
        for paragraph in section_paragraphs:
            words = paragraph.split()
            pieces: list[str] = []
            active = ""
            for word in words:
                proposed = f"{active} {word}".strip()
                if active and len(proposed) > maximum_chars:
                    pieces.append(active)
                    active = word
                else:
                    active = proposed
            if active:
                pieces.append(active)
            for piece in pieces:
                proposed = f"{current}\n\n{piece}".strip()
                if current and len(proposed) > maximum_chars:
                    chunks.append((f"{section_title} / {part}", current))
                    part += 1
                    current = piece
                else:
                    current = proposed
        if current:
            chunks.append((f"{section_title} / {part}", current))
    return chunks


def build_documents(manifest_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "szl.offline-retrieval-corpus/v1":
        raise RetrievalBuildError("unsupported corpus manifest schema")
    entries = manifest.get("sources")
    if not isinstance(entries, list) or not entries:
        raise RetrievalBuildError("corpus manifest has no sources")
    seen_paths: set[str] = set()
    seen_bytes: dict[str, str] = {}
    duplicates: list[dict[str, str]] = []
    documents: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise RetrievalBuildError("corpus source entry must be an object")
        relative = str(entry.get("path") or "")
        if relative in seen_paths:
            raise RetrievalBuildError(f"duplicate source path: {relative}")
        seen_paths.add(relative)
        path = safe_repo_file(relative)
        source_sha = sha256_file(path)
        if source_sha in seen_bytes:
            duplicates.append({"path": relative, "duplicates": seen_bytes[source_sha], "sha256": source_sha})
            continue
        seen_bytes[source_sha] = relative
        mode = entry.get("mode")
        if mode == "python_docstrings":
            text = python_docstrings(path)
        elif mode == "markdown":
            try:
                text = normalize_text(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError) as exc:
                raise RetrievalBuildError(f"cannot read source {relative}: {exc}") from exc
        else:
            raise RetrievalBuildError(f"unsupported source mode for {relative}: {mode}")
        for number, (title, body) in enumerate(chunk_text(text)):
            chunk_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
            doc_id = hashlib.sha256(
                f"{relative}\0{number}\0{chunk_sha}".encode("utf-8")
            ).hexdigest()[:24]
            documents.append(
                {
                    "document_id": doc_id,
                    "source_path": relative,
                    "source_sha256": source_sha,
                    "chunk_sha256": chunk_sha,
                    "chunk_number": number,
                    "title": title,
                    "text": body,
                    "license": str(entry.get("license") or manifest.get("default_license") or "UNKNOWN"),
                }
            )
    if len(documents) < 8:
        raise RetrievalBuildError(f"too few training chunks: {len(documents)}")
    return documents, duplicates


def load_queries(path: Path, *, expected_split: str | None = None,
                 encoded: bytes | None = None) -> list[dict[str, Any]]:
    payload = load_json(path) if encoded is None else parse_json_bytes(encoded, label=str(path))
    if not isinstance(payload, dict) or payload.get("schema_version") != "szl.offline-retrieval-queries/v1":
        raise RetrievalBuildError(f"unsupported query schema: {path}")
    if expected_split is not None and payload.get("split") != expected_split:
        raise RetrievalBuildError(f"query split must be {expected_split}: {path}")
    queries = payload.get("queries")
    if not isinstance(queries, list) or not queries:
        raise RetrievalBuildError(f"query file is empty: {path}")
    ids: set[str] = set()
    clean: list[dict[str, Any]] = []
    for row in queries:
        if not isinstance(row, dict):
            raise RetrievalBuildError("query row must be an object")
        query_id = row.get("id")
        text = row.get("text")
        relevant = row.get("relevant_paths")
        if not isinstance(query_id, str) or not query_id or query_id in ids:
            raise RetrievalBuildError(f"invalid or duplicate query id: {query_id!r}")
        if not isinstance(text, str) or not text.strip():
            raise RetrievalBuildError(f"query {query_id} has no text")
        if not isinstance(relevant, list) or any(not isinstance(item, str) for item in relevant):
            raise RetrievalBuildError(f"query {query_id} has invalid relevant_paths")
        ids.add(query_id)
        clean.append({"id": query_id, "text": text.strip(), "relevant_paths": sorted(set(relevant))})
    return clean


def load_query_splits(calibration_path: Path, evaluation_path: Path, *,
                      calibration_bytes: bytes | None = None,
                      evaluation_bytes: bytes | None = None):
    calibration = load_queries(calibration_path, expected_split="calibration", encoded=calibration_bytes)
    evaluation = load_queries(evaluation_path, expected_split="evaluation", encoded=evaluation_bytes)
    if {row["id"] for row in calibration} & {row["id"] for row in evaluation}:
        raise RetrievalBuildError("calibration and evaluation query identifiers overlap")
    normalized = lambda text: " ".join(text.casefold().split())
    if {normalized(row["text"]) for row in calibration} & {normalized(row["text"]) for row in evaluation}:
        raise RetrievalBuildError("calibration and evaluation query texts overlap")
    return calibration, evaluation


def calibrated_threshold(scores: list[tuple[float, bool]]) -> dict[str, Any]:
    if not scores or not any(label for _, label in scores) or not any(not label for _, label in scores):
        raise RetrievalBuildError("calibration requires answerable and unanswerable queries")
    if any(not math.isfinite(float(score)) or type(label) is not bool for score, label in scores):
        raise RetrievalBuildError("calibration requires finite scores and boolean labels")
    observed = sorted(set(float(score) for score, _ in scores))
    candidates = [value for value in (math.nextafter(observed[0], -math.inf),
                                     math.nextafter(observed[-1], math.inf)) if math.isfinite(value)]
    candidates.extend(observed)
    candidates.extend((left + right) / 2.0 for left, right in zip(observed, observed[1:]))
    best: tuple[float, float, float, float, float] | None = None
    for threshold in candidates:
        positive = [(score >= threshold) for score, label in scores if label]
        negative = [(score < threshold) for score, label in scores if not label]
        tpr = sum(positive) / len(positive)
        tnr = sum(negative) / len(negative)
        balanced = (tpr + tnr) / 2.0
        candidate = (balanced, min(tpr, tnr), tnr, threshold, tpr)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return {
        "threshold": float(best[3]),
        "balanced_accuracy": float(best[0]),
        "answerable_recall": float(best[4]),
        "unanswerable_specificity": float(best[2]),
        "calibration_scores": [
            {"top_score": float(score), "answerable": bool(label)} for score, label in scores
        ],
    }
