# SPDX-License-Identifier: Apache-2.0
"""Build a clean wheel and verify the installed release identity is complete."""
import ast
from email.parser import BytesParser
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def test_clean_wheel_ships_matching_runtime_metadata(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("pyproject.toml", "README.md"):
        shutil.copyfile(ROOT / name, source / name)
    shutil.copytree(
        ROOT / "torch-ext/szl_kernels",
        source / "torch-ext/szl_kernels",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    wheel_dir = tmp_path / "wheels"
    result = subprocess.run(
        [
            sys.executable, "-I", "-B", "-m", "pip", "wheel",
            "--disable-pip-version-check", "--no-deps", "--no-build-isolation",
            "--no-index", "--wheel-dir", str(wheel_dir), str(source),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = list(wheel_dir.glob("szl_kernels-*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        metadata_bytes = archive.read("szl_kernels/metadata.json")
        assert metadata_bytes == (source / "torch-ext/szl_kernels/metadata.json").read_bytes()
        runtime_metadata = json.loads(metadata_bytes)
        module = ast.parse(archive.read("szl_kernels/__init__.py").decode("utf-8"))
        module_versions = [
            ast.literal_eval(node.value)
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)
        ]
        distribution_paths = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        assert len(distribution_paths) == 1
        distribution = BytesParser().parsebytes(archive.read(distribution_paths[0]))
        assert module_versions == [distribution["Version"]]
        assert runtime_metadata["version"] == distribution["Version"] == "0.2.0"
