# SPDX-License-Identifier: Apache-2.0
"""Refuse pickle/joblib loaders in the approved kernel path."""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
CALL = re.compile(r"\b(?:joblib\.(?:load|dump)|pickle\.loads?|dill\.load)\s*\(")
APPROVED = ("torch-ext", "tests", "scripts")


def test_no_model_joblib_in_tree():
    assert not (ROOT / "model.joblib").exists()


def test_no_forbidden_loader_calls():
    hits = []
    for rel in APPROVED:
        base = ROOT / rel
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for i, line in enumerate(text.splitlines(), 1):
                if CALL.search(line):
                    hits.append(f"{path.relative_to(ROOT)}:{i}:{line.strip()}")
    assert hits == []
