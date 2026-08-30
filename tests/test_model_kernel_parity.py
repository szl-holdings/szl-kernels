from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCE = ROOT / "build" / "torch-universal" / "szl_kernels" / "_chain.py"
KERNEL_SOURCE = ROOT / "torch-ext" / "szl_kernels" / "_chain.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_model_and_kernel_chain_sources_match_after_line_normalization() -> None:
    assert _source(MODEL_SOURCE) == _source(KERNEL_SOURCE)


def test_tensor_digest_is_chunked_and_explicitly_little_endian() -> None:
    for path in (MODEL_SOURCE, KERNEL_SOURCE):
        source = _source(path)
        tree = ast.parse(source, filename=str(path))

        assert ".numpy()" not in source
        assert ".tobytes()" not in source
        assert any(
            isinstance(node, ast.Import)
            and any(alias.name == "struct" for alias in node.names)
            for node in tree.body
        )
        assert "_DIGEST_CHUNK_ELEMENTS = 4096" in source
        assert 'struct.pack(f"<{len(values)}q", *values)' in source
