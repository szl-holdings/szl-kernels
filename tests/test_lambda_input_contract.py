"""Advisory receipts must describe the complete accepted input on both builds."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["build/torch-universal", "torch-ext"])
def kernel(request):
    path = ROOT / request.param / "szl_kernels" / "__init__.py"
    name = "_lambda_contract_" + request.param.replace("/", "_").replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("axes", [
    torch.tensor([[1., 1.], [0., 0.]]),
    torch.tensor(0.5),
    torch.tensor([]),
    torch.tensor([1, 1]),
    torch.tensor([True, False]),
    torch.tensor([1 + 0j]),
])
def test_invalid_axes_emit_no_advisory_receipt(kernel, axes):
    chain = kernel.UnifiedReceiptChain()
    with pytest.raises(ValueError, match="non-empty one-dimensional floating"):
        kernel.governed_lambda_gate(chain, axes)
    assert chain.tail(1) == []


@pytest.mark.parametrize("weights", [
    torch.tensor([1.]),
    torch.tensor([[1., 1.]]),
    torch.tensor([1, 1]),
    torch.tensor([0., 0.]),
    torch.tensor([-1., 2.]),
    torch.tensor([float("nan"), 1.]),
    torch.tensor([float("inf"), 1.]),
    torch.tensor([1e300, 1e300], dtype=torch.float64),
])
def test_invalid_weights_emit_no_advisory_receipt(kernel, weights):
    chain = kernel.UnifiedReceiptChain()
    with pytest.raises(ValueError, match="weights"):
        kernel.governed_lambda_gate(chain, torch.tensor([0.25, 1.]), weights)
    assert chain.tail(1) == []


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_weighted_score_matches_geometric_mean_and_receipt(kernel, dtype):
    chain = kernel.UnifiedReceiptChain()
    result = kernel.governed_lambda_gate(
        chain, torch.tensor([0.25, 1.], dtype=dtype),
        torch.tensor([1., 1.], dtype=dtype), threshold=0.6,
    )
    assert result["score"] == pytest.approx(0.5, abs=0.004)
    assert result["passed"] is False
    assert result["advisory"] is True
    assert chain.verify() == (True, 1, -1)
    assert chain.tail(1)[0]["attrs"]["score"] == result["score"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_finite_weights_cannot_overflow_to_a_perfect_score(kernel, dtype):
    chain = kernel.UnifiedReceiptChain()
    largest = torch.finfo(dtype).max
    result = kernel.governed_lambda_gate(
        chain, torch.tensor([0.25, 1.], dtype=dtype),
        torch.tensor([largest, largest], dtype=dtype), threshold=0.9,
    )
    assert result["score"] == pytest.approx(0.5)
    assert result["passed"] is False


@pytest.mark.parametrize("bad_axis", [0., -1., float("nan"), float("inf"), float("-inf")])
def test_zero_weight_cannot_compensate_for_an_invalid_axis(kernel, bad_axis):
    result = kernel.governed_lambda_gate(
        kernel.UnifiedReceiptChain(), torch.tensor([bad_axis, 1.]),
        torch.tensor([0., 1.]),
    )
    assert result["score"] == 0.
    assert result["passed"] is False


def test_positive_weight_scaling_preserves_score(kernel):
    axes = torch.tensor([0.125, 1.], dtype=torch.float64)
    scores = [kernel.governed_lambda_gate(
        kernel.UnifiedReceiptChain(), axes,
        torch.tensor([scale, 2 * scale], dtype=torch.float64),
    )["score"] for scale in (1e-300, 1., 1e300)]
    assert scores == pytest.approx([0.5, 0.5, 0.5])
