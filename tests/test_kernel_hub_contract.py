# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for the separately staged, pure-Torch Kernel Hub API."""
import ast
import builtins
import hashlib
import importlib.util
from pathlib import Path
import struct
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "build/torch-universal/szl_kernels"
STAGED_FILES = {
    "__init__.py": "_kernel_api.py",
    "_chain.py": "_chain.py",
    "_ops.py": "_ops.py",
    "retrieval.py": "retrieval.py",
}
# AST-extracted from both first-class v1 entrypoints at the immutable revision
# 09818b62d683c33d200fca32e2ebfd95c64c65c7 (SHA-256 below).
PUBLISHED_V1_INIT_SHA256 = "96caac81dd719c785c9cb1458ac835352a8b45cbce7a5603a205b3a88319bbf0"
PUBLISHED_V1_EXPORTS = {
    "UnifiedReceiptChain", "tensor_digest", "GENESIS",
    "governed_rms_norm", "governed_layer_norm", "governed_lambda_gate",
    "governed_measure_energy", "GovernedBlock", "list_kernels", "list_series",
    "get_member", "selfcheck", "DOCTRINE_FOOTER", "PROVENANCE", "__version__",
}


@pytest.fixture
def staged_kernel(tmp_path, monkeypatch):
    """Import only the staged closure, never the broader pip entrypoint."""
    package_path = tmp_path / "staged_kernel"
    package_path.mkdir()
    for target, source in STAGED_FILES.items():
        (package_path / target).write_bytes((PACKAGE / source).read_bytes())
    module_name = "_szl_kernel_contract_" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12]
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"numpy", "sklearn"}:
            raise AssertionError("Kernel Hub entrypoint must not import model dependencies")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    spec = importlib.util.spec_from_file_location(module_name, package_path / "__init__.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        for name in list(sys.modules):
            if name == module_name or name.startswith(module_name + "."):
                del sys.modules[name]


def test_hub_api_preserves_published_v1_exports(staged_kernel):
    assert len(PUBLISHED_V1_EXPORTS) == 15
    assert set(staged_kernel.__all__) == PUBLISHED_V1_EXPORTS | {"governed_cosine_topk"}
    assert all(hasattr(staged_kernel, name) for name in staged_kernel.__all__)
    assert staged_kernel.__version__ == "0.2.0"
    assert not hasattr(staged_kernel, "MiniEmbed")
    assert not hasattr(staged_kernel, "list_estate")
    assert "miniembed" not in staged_kernel.PROVENANCE
    assert set(staged_kernel.list_kernels()) == {"governed_norm", "lambda_gate", "energy_core"}
    assert set(staged_kernel.list_series()) == {"govsign", "blocked", "provctl"}
    member = staged_kernel.get_member("governed_norm")
    member["hub_id"] = "changed"
    assert staged_kernel.get_member("governed_norm")["hub_id"] == "SZLHOLDINGS/szl-governed-norm"
    with pytest.raises(KeyError):
        staged_kernel.get_member("MiniEmbed")


def test_hub_import_closure_is_complete_stdlib_and_torch_only():
    pending = ["__init__.py"]
    visited = set()
    allowed = set(sys.stdlib_module_names) | {"torch"}
    while pending:
        target = pending.pop()
        if target in visited:
            continue
        visited.add(target)
        assert target in STAGED_FILES
        tree = ast.parse((PACKAGE / STAGED_FILES[target]).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] in allowed for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    assert node.level == 1 and node.module is not None
                    dependency = node.module.replace(".", "/") + ".py"
                    assert dependency in STAGED_FILES
                    pending.append(dependency)
                else:
                    assert node.module is not None and node.module.split(".")[0] in allowed
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"eval", "exec"}
                if node.func.id == "__import__":
                    # Existing v1 _ops uses literal hashlib/json imports.
                    # Admit only statically resolvable allowed module names.
                    assert node.args and isinstance(node.args[0], ast.Constant)
                    imported = node.args[0].value
                    assert isinstance(imported, str) and imported.split(".")[0] in allowed
            elif isinstance(node, ast.Attribute):
                assert node.attr not in {"numpy", "import_module"}
    assert visited == set(STAGED_FILES)


def test_hub_selfcheck_is_independent_of_caller_default_dtype(staged_kernel):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        report = staged_kernel.selfcheck()
        assert report["ok"], report
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(previous)


def test_kernel_source_mirrors_match():
    for source in ("_kernel_api.py", "retrieval.py", "_chain.py", "_ops.py"):
        assert (ROOT / "torch-ext/szl_kernels" / source).read_bytes() == (PACKAGE / source).read_bytes()


@pytest.mark.parametrize("rows", [1, 3, 512])
@pytest.mark.parametrize("kind", ["float32", "int64"])
def test_raw_hash_matches_known_native_bytes(staged_kernel, rows, kind):
    if kind == "float32":
        values = [0.0, -0.0, 1.0, -1.0, float.fromhex("0x1.fffffep+127"), -float.fromhex("0x1.fffffep+127"),
                  float.fromhex("0x1p-149"), -float.fromhex("0x1p-149"), 1.25, -3.5, 65536.0, -65536.0]
        tensor = torch.tensor(values, dtype=torch.float32).reshape(3, 4).T
        code = "f"
    else:
        values = [0, -1, 1, -(2 ** 63), 2 ** 63 - 1, 2 ** 53 + 1,
                  -(2 ** 53 + 1), 256, -256, 32768, -32768, 65536]
        tensor = torch.tensor(values, dtype=torch.int64).reshape(3, 4).T
        code = "q"
    assert not tensor.is_contiguous()
    logical_values = [value for row in tensor.tolist() for value in row]
    expected = struct.pack("=" + str(len(logical_values)) + code, *logical_values)
    retrieval = sys.modules[staged_kernel.__name__ + ".retrieval"]
    assert retrieval._raw_sha256(tensor, rows) == hashlib.sha256(expected).hexdigest()
    one_dimensional = tensor.contiguous().reshape(-1)
    assert retrieval._raw_sha256(one_dimensional, rows) == hashlib.sha256(expected).hexdigest()


def test_raw_hash_python_buffers_are_bounded(staged_kernel):
    class RecordingDigest:
        def __init__(self):
            self.digest = hashlib.sha256()
            self.sizes = []

        def update(self, data):
            assert isinstance(data, bytes)
            self.sizes.append(len(data))
            self.digest.update(data)

    tensor = torch.arange(40000, dtype=torch.float32).reshape(1, -1)
    digest = RecordingDigest()
    retrieval = sys.modules[staged_kernel.__name__ + ".retrieval"]
    retrieval._update_raw_hash(digest, tensor, rows=512)
    assert digest.sizes == [65536, 65536, 28928]
    expected = struct.pack("=40000f", *range(40000))
    assert digest.digest.hexdigest() == hashlib.sha256(expected).hexdigest()


def test_retrieval_and_v1_selfcheck_never_use_tensor_numpy(staged_kernel, monkeypatch):
    def fail_numpy(*args, **kwargs):
        raise AssertionError("Tensor.numpy is outside the pure-Torch kernel contract")

    monkeypatch.setattr(torch.Tensor, "numpy", fail_numpy)
    query = torch.tensor([1.0, -0.0])
    documents = torch.tensor([[0.0, 1.0], [1.0, -0.0], [1.0, 0.0], [0.0, 0.0]])
    chain = staged_kernel.UnifiedReceiptChain()
    result = staged_kernel.governed_cosine_topk(chain, query, documents, k=4, block_rows=1)
    assert result["indices"].tolist() == [[1, 2, 0, 3]]
    assert result["scores"].tolist() == [[1.0, 1.0, 0.0, 0.0]]
    attrs = result["receipt"]["attrs"]
    assert attrs["query_sha256"] == hashlib.sha256(struct.pack("=2f", 1.0, -0.0)).hexdigest()
    assert attrs["indices_sha256"] == hashlib.sha256(struct.pack("=4q", 1, 2, 0, 3)).hexdigest()
    assert attrs["receipt_authenticity"] == "UNSIGNED"
    assert attrs["acceleration_claim"] is False
    assert chain.verify() == (True, 1, -1)
    report = staged_kernel.selfcheck()
    assert report["ok"], report
    assert report["tamper_detected"]
    assert report["version"] == "0.2.0"
    assert report["kernels_touched"] == ["governed_norm", "lambda_gate", "energy_core"]
