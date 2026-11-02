"""Deterministic, path-independent corpus input discovery.

The training receipt identifies a material by a logical source name and its
path relative to that source root.  Absolute checkout paths are never part of
the identifier.  Multiple glob patterns may select the same file; that file is
included once.  A file claimed by two logical sources is ambiguous and fails
closed instead of being silently double weighted.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, Sequence


class CorpusInputError(ValueError):
    """The configured source roots do not define an unambiguous input set."""


@dataclass(frozen=True)
class InputGroup:
    """One logical corpus source and the globs evaluated below its root."""

    name: str
    root: Path
    patterns: tuple[str, ...]


@dataclass(frozen=True)
class CorpusInput:
    """A discovered source file with a stable receipt identifier."""

    group: str
    source_id: str
    path: Path


def _physical_key(path: Path) -> str:
    """Return a platform-appropriate identity for a resolved filesystem path."""

    return os.path.normcase(str(path.resolve(strict=True)))


def _validate_group(group: InputGroup) -> tuple[str, Path, tuple[str, ...]]:
    name = group.name.strip()
    if not name or "/" in name or "\\" in name:
        raise CorpusInputError(f"invalid logical source name: {group.name!r}")

    root = group.root.resolve(strict=True)
    if not root.is_dir():
        raise CorpusInputError(f"logical source root is not a directory: {root}")

    patterns = tuple(pattern.strip() for pattern in group.patterns)
    if not patterns or any(not pattern for pattern in patterns):
        raise CorpusInputError(f"logical source {name!r} needs non-empty patterns")
    return name, root, patterns


def collect_corpus_inputs(groups: Sequence[InputGroup]) -> list[CorpusInput]:
    """Expand ``groups`` into one deterministic, unambiguous input list.

    Repeated matches from overlapping patterns in the same group are
    deduplicated by resolved physical path.  A resolved path assigned to more
    than one logical group, or a logical identifier assigned to two physical
    files, is rejected.  Returned records are sorted by logical identifier so
    checkout location and glob traversal order cannot affect corpus order.
    """

    group_names: set[str] = set()
    physical_to_source: dict[str, str] = {}
    source_to_physical: dict[str, str] = {}
    records: dict[str, CorpusInput] = {}

    for raw_group in groups:
        name, root, patterns = _validate_group(raw_group)
        if name in group_names:
            raise CorpusInputError(f"duplicate logical source name: {name!r}")
        group_names.add(name)

        for pattern in patterns:
            for candidate in root.glob(pattern):
                if not candidate.is_file():
                    continue

                resolved = candidate.resolve(strict=True)
                try:
                    relative = resolved.relative_to(root)
                except ValueError as exc:
                    raise CorpusInputError(
                        f"input escapes logical source root {name!r}: {candidate}"
                    ) from exc

                source_id = f"{name}/{relative.as_posix()}"
                physical = _physical_key(resolved)

                previous_source = physical_to_source.get(physical)
                if previous_source is not None and previous_source != source_id:
                    raise CorpusInputError(
                        "one physical input has multiple logical identifiers: "
                        f"{previous_source!r} and {source_id!r}"
                    )

                previous_physical = source_to_physical.get(source_id)
                if previous_physical is not None and previous_physical != physical:
                    raise CorpusInputError(
                        f"logical identifier maps to multiple files: {source_id!r}"
                    )

                physical_to_source[physical] = source_id
                source_to_physical[source_id] = physical
                records[source_id] = CorpusInput(name, source_id, resolved)

    return [records[source_id] for source_id in sorted(records)]


def source_ids(inputs: Iterable[CorpusInput]) -> list[str]:
    """Return identifiers after rejecting duplicates in caller-provided input."""

    identifiers = [item.source_id for item in inputs]
    if len(identifiers) != len(set(identifiers)):
        raise CorpusInputError("duplicate logical identifiers in corpus inputs")
    return identifiers
