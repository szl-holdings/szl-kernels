#!/usr/bin/env python3
"""Verify and publish an exact GitHub-to-Hugging-Face source binding."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "publishing" / "source-binding.json"
DEFAULT_REPORT = ROOT / "reports" / "source-binding.json"
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class BindingError(RuntimeError):
    """Raised when evidence is insufficient for publication."""


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_file(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if path == ROOT or ROOT not in path.parents or not path.is_file():
        raise BindingError(f"artifact file is missing or outside the repository: {relative}")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "szl.kernel-source-binding/v1":
        raise BindingError("unsupported source-binding schema")
    repo_id = payload.get("repo_id")
    source_repository = payload.get("source_repository")
    if not isinstance(repo_id, str) or repo_id.count("/") != 1:
        raise BindingError("repo_id must be an owner/repository name")
    if not isinstance(source_repository, str) or source_repository.count("/") != 1:
        raise BindingError("source_repository must be an owner/repository name")
    artifact_files = payload.get("artifact_files")
    if not isinstance(artifact_files, list) or not artifact_files:
        raise BindingError("artifact_files must be a non-empty list")
    if any(not isinstance(item, str) or not item for item in artifact_files):
        raise BindingError("artifact_files entries must be non-empty strings")
    if len(artifact_files) != len(set(artifact_files)):
        raise BindingError("artifact_files contains duplicate paths")
    expected = payload.get("expected_artifact_sha256")
    if not isinstance(expected, dict) or not expected:
        raise BindingError("expected_artifact_sha256 must be a non-empty object")
    if not set(expected).issubset(set(artifact_files)):
        raise BindingError("every expected hash must name a declared artifact file")
    return payload


def local_evidence(contract: dict[str, Any]) -> list[dict[str, Any]]:
    expected = contract["expected_artifact_sha256"]
    evidence: list[dict[str, Any]] = []
    for relative in contract["artifact_files"]:
        path = safe_file(relative)
        observed = file_sha256(path)
        wanted = expected.get(relative)
        if wanted is not None and observed != wanted:
            raise BindingError(
                f"{relative} SHA-256 drifted (expected {wanted}, observed {observed})"
            )
        evidence.append(
            {
                "path": Path(relative).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": observed,
            }
        )
    return evidence


def tree_sha256(evidence: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(evidence, key=lambda value: value["path"]):
        digest.update(item["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(item["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_checkout(source_revision: str) -> dict[str, Any]:
    try:
        observed = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip().lower()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise BindingError(f"unable to verify the Git checkout: {error}") from error
    if observed != source_revision:
        raise BindingError(
            f"source revision does not match checkout HEAD "
            f"(supplied {source_revision}, observed {observed})"
        )
    if status:
        raise BindingError("tracked worktree is dirty; refuse source binding")
    return {"head_revision": observed, "tracked_worktree": "CLEAN"}


def hub_evidence(
    api: HfApi,
    contract: dict[str, Any],
    *,
    token: str | None,
    download_fn: Callable[..., str] = hf_hub_download,
) -> dict[str, Any]:
    info = api.model_info(contract["repo_id"], files_metadata=True, token=token)
    observed_files = {sibling.rfilename for sibling in info.siblings or []}
    missing = sorted(set(contract["artifact_files"]) - observed_files)
    missing_critical = sorted(
        set(contract["expected_artifact_sha256"]) - observed_files
    )
    if missing_critical:
        raise BindingError(
            f"Hub artifact is missing critical files: {missing_critical}"
        )

    critical: list[dict[str, Any]] = []
    for relative, wanted in sorted(contract["expected_artifact_sha256"].items()):
        downloaded = Path(
            download_fn(
                contract["repo_id"],
                relative,
                repo_type="model",
                revision=info.sha,
                token=token,
            )
        )
        observed = file_sha256(downloaded)
        if observed != wanted:
            raise BindingError(
                f"Hub {relative} SHA-256 drifted "
                f"(expected {wanted}, observed {observed})"
            )
        critical.append({"path": relative, "sha256": observed})
    return {
        "revision": info.sha,
        "declared_files_present": len(set(contract["artifact_files"]) & observed_files),
        "declared_files_pending_publication": missing,
        "critical_artifacts": critical,
    }


def publication_payload(
    contract: dict[str, Any],
    *,
    source_revision: str,
    local_files: list[dict[str, Any]],
    hub_before: dict[str, Any],
    checkout: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "szl.hf-kernel-source-binding/v1",
        "artifact": {
            "repo_id": contract["repo_id"],
            "repo_type": "model",
            "kind": "governed_kernel_suite_with_receipted_word_embeddings",
        },
        "source_repository": contract["source_repository"],
        "source_revision": source_revision,
        "source": {
            "url": f"https://github.com/{contract['source_repository']}",
            "revision": source_revision,
            "checkout": checkout,
            "artifact_tree_sha256": tree_sha256(local_files),
            "declared_file_count": len(local_files),
            "files": local_files,
        },
        "observed_hub_before_publication": hub_before,
        "claims": contract["claims"],
        "limitations": contract["limitations"],
    }


def verify_readback(
    contract: dict[str, Any],
    publication_bytes: bytes,
    *,
    revision: str,
    token: str,
    download_fn: Callable[..., str] = hf_hub_download,
) -> None:
    expected_paths = list(contract["artifact_files"]) + ["publication.json"]
    for relative in expected_paths:
        downloaded = Path(
            download_fn(
                contract["repo_id"],
                relative,
                repo_type="model",
                revision=revision,
                token=token,
            )
        )
        expected = (
            publication_bytes if relative == "publication.json" else safe_file(relative).read_bytes()
        )
        if downloaded.read_bytes() != expected:
            raise BindingError(f"readback mismatch at {relative}")


def run(
    *,
    contract_path: Path,
    report_path: Path,
    source_revision: str,
    publish: bool,
    token: str | None,
    api: HfApi | None = None,
    download_fn: Callable[..., str] = hf_hub_download,
    checkout_verifier: Callable[[str], dict[str, Any]] = verify_checkout,
) -> dict[str, Any]:
    source_revision = source_revision.strip().lower()
    if FULL_SHA_RE.fullmatch(source_revision) is None:
        raise BindingError("source revision must be an exact 40-character Git SHA")
    checkout = checkout_verifier(source_revision)
    contract = load_contract(contract_path)
    api = api or HfApi(token=token)
    local_files = local_evidence(contract)
    hub_before = hub_evidence(api, contract, token=token, download_fn=download_fn)
    publication = publication_payload(
        contract,
        source_revision=source_revision,
        local_files=local_files,
        hub_before=hub_before,
        checkout=checkout,
    )
    publication_bytes = canonical_json(publication).encode("utf-8")
    result: dict[str, Any] = {
        "schema": "szl.kernel-source-binding-report/v1",
        "mode": "PUBLISH" if publish else "DRY_RUN",
        "repo_id": contract["repo_id"],
        "source_repository": contract["source_repository"],
        "source_revision": source_revision,
        "checkout": checkout,
        "artifact_tree_sha256": tree_sha256(local_files),
        "declared_file_count": len(local_files),
        "hub_revision_before": hub_before["revision"],
        "publication_sha256": hashlib.sha256(publication_bytes).hexdigest(),
        "status": "VERIFIED_DRY_RUN",
    }
    if publish:
        if not token:
            raise BindingError("HF_TOKEN is required when --publish is used")
        operations = [
            CommitOperationAdd(
                path_in_repo=relative,
                path_or_fileobj=str(safe_file(relative)),
            )
            for relative in contract["artifact_files"]
        ]
        operations.append(
            CommitOperationAdd(
                path_in_repo="publication.json",
                path_or_fileobj=io.BytesIO(publication_bytes),
            )
        )
        commit = api.create_commit(
            repo_id=contract["repo_id"],
            repo_type="model",
            operations=operations,
            commit_message=f"Publish canonical source {source_revision[:12]}",
            token=token,
        )
        revision = getattr(commit, "oid", None)
        if not revision:
            revision = api.model_info(contract["repo_id"], token=token).sha
        verify_readback(
            contract,
            publication_bytes,
            revision=revision,
            token=token,
            download_fn=download_fn,
        )
        result.update(
            {
                "hub_revision_after": revision,
                "status": "PUBLISHED_AND_EXACT_READBACK_VERIFIED",
            }
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(canonical_json(result), encoding="utf-8")
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--source-revision",
        default=os.getenv("GITHUB_SHA", ""),
        help="Exact canonical Git revision. Defaults to GITHUB_SHA.",
    )
    parser.add_argument("--publish", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(
        contract_path=args.contract,
        report_path=args.report,
        source_revision=args.source_revision,
        publish=args.publish,
        token=os.getenv("HF_TOKEN"),
    )
    print(canonical_json(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
