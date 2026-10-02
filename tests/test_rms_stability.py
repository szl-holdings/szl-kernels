"""Independent float64 oracle and autograd regressions for governed RMSNorm."""
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build" / "torch-universal"))
import szl_kernels as sk
from szl_kernels._ops import _rms_norm


def oracle(x, weight, eps):
    values = x.double()
    output = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    return output if weight is None else output * weight.double()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("value,eps", [(0., 1e-6), (1e-20, 1e-40), (1e-30, 1e-60), (1., 1e-6), (1e4, 1e-6), (1e30, 1e-6)])
def test_rms_matches_float64_oracle(dtype, value, eps):
    if value > torch.finfo(dtype).max:
        pytest.skip("value not representable in input dtype")
    x = torch.tensor([[value, -value, value / 2], [-value / 4, value, 0]], dtype=dtype)
    weight = torch.tensor([0.5, 1., -2.], dtype=dtype)
    before_x, before_weight = x.clone(), weight.clone()
    actual = _rms_norm(x, weight, eps)
    expected = oracle(x, weight, eps).to(dtype)
    tol = 0.02 if dtype == torch.bfloat16 else 0.003 if dtype == torch.float16 else 2e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
    assert actual.dtype == dtype
    assert torch.equal(x, before_x) and torch.equal(weight, before_weight)


def test_large_finite_output_is_receipted_correctly():
    x = torch.full((2, 3), 1e30)
    chain = sk.UnifiedReceiptChain()
    output = sk.governed_rms_norm(chain, x)
    torch.testing.assert_close(output, torch.ones_like(x))
    assert chain.verify() == (True, 1, -1)
    assert chain.tail(1)[0]["attrs"]["out_digest"] == sk.tensor_digest(output)


def test_rms_noncontiguous_and_unweighted():
    x = torch.tensor([[1e30, -2e30, 3e30], [4e30, -5e30, 6e30]]).T
    assert not x.is_contiguous()
    torch.testing.assert_close(_rms_norm(x, None, 1e-6), oracle(x, None, 1e-6).float())


def test_rms_preserves_float64_gradients():
    x = torch.tensor([[0.3, -0.7, 1.2], [1.4, 0.4, -0.8]], dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([0.7, -0.2, 1.3], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: _rms_norm(a, b, 1e-6), (x, weight))
    assert torch.autograd.gradgradcheck(lambda a, b: _rms_norm(a, b, 1e-6), (x, weight))


def test_rms_large_float32_gradients_match_oracle():
    x = torch.tensor([[1e20, -2e20, 3e20]], requires_grad=True)
    weight = torch.tensor([0.7, -0.2, 1.3], requires_grad=True)
    expected = oracle(x, weight, 1e-6)
    actual = _rms_norm(x, weight, 1e-6)
    ga = torch.autograd.grad(actual.sum(), (x, weight))
    ge = torch.autograd.grad(expected.sum(), (x, weight))
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, rtol=2e-5, atol=1e-30)


def test_both_distribution_sources_match():
    root = Path(__file__).resolve().parents[1]
    assert (root / "build/torch-universal/szl_kernels/_ops.py").read_text(encoding="utf-8") == (root / "torch-ext/szl_kernels/_ops.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("eps", [-1., float("inf"), float("nan")])
def test_invalid_epsilon_emits_no_receipt(eps):
    chain = sk.UnifiedReceiptChain()
    with pytest.raises(ValueError, match="eps must be finite and non-negative"):
        sk.governed_rms_norm(chain, torch.ones(2, 3), eps=eps)
    assert chain.tail(1) == []


def test_zero_epsilon_nonzero_rows_and_undefined_zero_rows():
    x = torch.tensor([[1e30, -1e30]])
    torch.testing.assert_close(_rms_norm(x, None, 0), torch.tensor([[1., -1.]]))
    assert torch.isnan(_rms_norm(torch.zeros(1, 3), None, 0)).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_largest_finite_uniform_rows(dtype):
    x = torch.full((2, 3), torch.finfo(dtype).max, dtype=dtype)
    torch.testing.assert_close(_rms_norm(x, None, 1e-6), torch.ones_like(x))


@pytest.mark.parametrize("eps", [5e-324, 1e-300, 1e300])
def test_extreme_positive_epsilon_zero_rows(eps):
    x = torch.zeros((2, 3))
    torch.testing.assert_close(_rms_norm(x, None, eps), x)
