from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from corpus_inputs import (  # noqa: E402
    CorpusInputError,
    InputGroup,
    collect_corpus_inputs,
)


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _snapshot(root: Path) -> list[tuple[str, str]]:
    inputs = collect_corpus_inputs(
        [InputGroup("fixture", root, ("*.md", "**/*.md", "**/*.py"))]
    )
    return [
        (item.source_id, hashlib.sha256(item.path.read_bytes()).hexdigest())
        for item in inputs
    ]


def test_overlapping_globs_include_each_physical_file_once(tmp_path: Path) -> None:
    _write(tmp_path, "README.md", "root")
    _write(tmp_path, "nested/guide.md", "guide")

    inputs = collect_corpus_inputs(
        [InputGroup("docs", tmp_path, ("*.md", "**/*.md"))]
    )

    assert [item.source_id for item in inputs] == [
        "docs/README.md",
        "docs/nested/guide.md",
    ]
    assert len({item.path.resolve() for item in inputs}) == 2


def test_source_identifiers_and_digests_survive_checkout_relocation(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first-checkout"
    second = tmp_path / "different" / "checkout"
    for root in (first, second):
        _write(root, "README.md", "same bytes")
        _write(root, "pkg/kernel.py", "VALUE = 1\n")

    assert _snapshot(first) == _snapshot(second)
    assert all(str(tmp_path) not in source_id for source_id, _ in _snapshot(first))


def test_one_physical_file_cannot_claim_two_logical_sources(tmp_path: Path) -> None:
    _write(tmp_path, "shared.md", "one file")

    with pytest.raises(CorpusInputError, match="multiple logical identifiers"):
        collect_corpus_inputs(
            [
                InputGroup("first", tmp_path, ("*.md",)),
                InputGroup("second", tmp_path, ("*.md",)),
            ]
        )


def test_duplicate_logical_source_name_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "one.md", "one")

    with pytest.raises(CorpusInputError, match="duplicate logical source name"):
        collect_corpus_inputs(
            [
                InputGroup("same", tmp_path, ("one.md",)),
                InputGroup("same", tmp_path, ("*.md",)),
            ]
        )


def test_shipped_forge_inputs_are_unique_and_path_independent() -> None:
    inputs = collect_corpus_inputs(
        [
            InputGroup(
                "doctrine-v10-v11",
                ROOT / "corpus/doctrine-v10-v11",
                ("*.md", "**/*.md"),
            ),
            InputGroup(
                "rag-corpus-v1",
                ROOT / "corpus/rag-corpus-v1",
                ("corpus.jsonl",),
            ),
            InputGroup(
                "thesis-corpus-v18",
                ROOT / "corpus/thesis-corpus-v18",
                ("**/*.tex",),
            ),
            InputGroup(
                "kernels",
                ROOT / "corpus/kernels",
                ("README.md", "**/*.md", "**/*.py"),
            ),
        ]
    )

    assert len(inputs) == 20
    assert len({item.source_id for item in inputs}) == 20
    assert len({item.path.resolve() for item in inputs}) == 20
    assert all(not Path(item.source_id).is_absolute() for item in inputs)
