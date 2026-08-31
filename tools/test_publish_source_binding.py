from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import publish_source_binding as binding


class FakeApi:
    def __init__(self, files: list[str]) -> None:
        self.files = files

    def model_info(
        self,
        repo_id: str,
        files_metadata: bool = False,
        token: str | None = None,
    ) -> SimpleNamespace:
        del repo_id, files_metadata, token
        return SimpleNamespace(
            sha="a" * 40,
            siblings=[SimpleNamespace(rfilename=path) for path in self.files],
        )


class PublishSourceBindingTests(unittest.TestCase):
    @staticmethod
    def _verified_checkout(revision: str) -> dict[str, str]:
        return {"head_revision": revision, "tracked_worktree": "CLEAN"}

    def _fixture(self, root: Path) -> tuple[Path, dict[str, Path]]:
        artifacts = {
            "README.md": root / "README.md",
            "vectors.npz": root / "vectors.npz",
        }
        artifacts["README.md"].write_text("kernel\n", encoding="utf-8")
        artifacts["vectors.npz"].write_bytes(b"weights")
        expected = hashlib.sha256(b"weights").hexdigest()
        contract = root / "contract.json"
        contract.write_text(
            json.dumps(
                {
                    "schema": "szl.kernel-source-binding/v1",
                    "repo_id": "SZLHOLDINGS/szl-kernels",
                    "source_repository": "szl-holdings/szl-kernels",
                    "artifact_files": list(artifacts),
                    "expected_artifact_sha256": {"vectors.npz": expected},
                    "claims": {
                        "artifact_equivalence": "BYTE_IDENTICAL_DECLARED_FILE_SET",
                        "reproducible_build": "BOUNDED_REPLAY_ONLY",
                    },
                    "limitations": ["test fixture"],
                }
            ),
            encoding="utf-8",
        )
        return contract, artifacts

    def test_dry_run_verifies_local_and_hub_artifact_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract, artifacts = self._fixture(root)

            def download(
                repo_id: str,
                filename: str,
                **_: object,
            ) -> str:
                self.assertEqual(repo_id, "SZLHOLDINGS/szl-kernels")
                return str(artifacts[filename])

            report = root / "report.json"
            with mock.patch.object(binding, "ROOT", root):
                result = binding.run(
                    contract_path=contract,
                    report_path=report,
                    source_revision="b" * 40,
                    publish=False,
                    token=None,
                    api=FakeApi(list(artifacts)),
                    download_fn=download,
                    checkout_verifier=self._verified_checkout,
                )
            self.assertEqual(result["status"], "VERIFIED_DRY_RUN")
            self.assertEqual(result["declared_file_count"], 2)
            self.assertTrue(report.is_file())

    def test_local_artifact_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract, artifacts = self._fixture(root)
            artifacts["vectors.npz"].write_bytes(b"changed")
            with mock.patch.object(binding, "ROOT", root):
                payload = binding.load_contract(contract)
                with self.assertRaisesRegex(binding.BindingError, "drifted"):
                    binding.local_evidence(payload)

    def test_publish_requires_token(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract, artifacts = self._fixture(root)

            def download(
                repo_id: str,
                filename: str,
                **_: object,
            ) -> str:
                del repo_id
                return str(artifacts[filename])

            with mock.patch.object(binding, "ROOT", root):
                with self.assertRaisesRegex(binding.BindingError, "HF_TOKEN"):
                    binding.run(
                        contract_path=contract,
                        report_path=root / "report.json",
                        source_revision="c" * 40,
                        publish=True,
                        token=None,
                        api=FakeApi(list(artifacts)),
                        download_fn=download,
                        checkout_verifier=self._verified_checkout,
                    )

    def test_cli_requires_an_explicit_non_ambiguous_mode(self) -> None:
        with self.assertRaises(SystemExit):
            binding.parse_args([])
        dry_run = binding.parse_args(["--dry-run"])
        self.assertTrue(dry_run.dry_run)
        self.assertFalse(dry_run.publish)
        publish = binding.parse_args(["--publish"])
        self.assertFalse(publish.dry_run)
        self.assertTrue(publish.publish)
        with self.assertRaises(SystemExit):
            binding.parse_args(["--dry-run", "--publish"])

    def test_new_noncritical_file_can_wait_for_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract, artifacts = self._fixture(root)

            def download(
                repo_id: str,
                filename: str,
                **_: object,
            ) -> str:
                del repo_id
                return str(artifacts[filename])

            with mock.patch.object(binding, "ROOT", root):
                payload = binding.load_contract(contract)
                observed = binding.hub_evidence(
                    FakeApi(["vectors.npz"]),
                    payload,
                    token=None,
                    download_fn=download,
                )
            self.assertEqual(
                observed["declared_files_pending_publication"],
                ["README.md"],
            )

    def test_missing_critical_hub_artifact_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract, artifacts = self._fixture(root)
            del artifacts
            with mock.patch.object(binding, "ROOT", root):
                payload = binding.load_contract(contract)
                with self.assertRaisesRegex(binding.BindingError, "critical"):
                    binding.hub_evidence(
                        FakeApi(["README.md"]),
                        payload,
                        token=None,
                    )

    def test_source_revision_must_be_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract, artifacts = self._fixture(root)
            del artifacts
            with self.assertRaisesRegex(binding.BindingError, "40-character"):
                binding.run(
                    contract_path=contract,
                    report_path=root / "report.json",
                    source_revision="main",
                    publish=False,
                    token=None,
                    api=FakeApi(["README.md", "vectors.npz"]),
                    checkout_verifier=self._verified_checkout,
                )

    def test_checkout_revision_mismatch_fails_closed(self) -> None:
        with mock.patch.object(
            binding.subprocess,
            "check_output",
            side_effect=["d" * 40 + "\n", ""],
        ):
            with self.assertRaisesRegex(binding.BindingError, "does not match"):
                binding.verify_checkout("e" * 40)


if __name__ == "__main__":
    unittest.main()
