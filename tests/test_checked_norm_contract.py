"""Always validate source; execute CUDA tests only with supported dependencies."""
import ast
import importlib.util
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "frontier" / "checked_norm"


def test_candidate_sources_parse_without_importing_gpu_dependencies():
    for name in ("checked_rms_norm.py", "benchmark_rms.py"):
        ast.parse((HERE / name).read_text(encoding="utf-8"))


def test_checked_norm_cuda_contract():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or importlib.util.find_spec("triton") is None:
        pytest.skip("CUDA and Triton are required for the research kernel")
    previous = sys.path[:]
    try:
        sys.path.insert(0, str(HERE))
        import benchmark_rms
        assert benchmark_rms.correctness()["cases_passed"] == 57
    finally:
        sys.path[:] = previous
