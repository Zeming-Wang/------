import pytest

torch = pytest.importorskip("torch")

from maas_reproduction.training.batch_accumulator import BatchAccumulator


def test_full_batch_uses_mean_policy_loss_and_steps_optimizer():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    accumulator = BatchAccumulator(optimizer, batch_size=2)

    first = accumulator.add(policy_log_prob=parameter * 1.0, utility=1.0)
    assert not first["update_performed"]
    result = accumulator.add(policy_log_prob=parameter * 2.0, utility=3.0)

    # loss = -mean([1 * 1, 2 * 3]) = -3.5; d(loss)/d(parameter) = -3.5
    assert result["update_performed"] is True
    assert result["loss_value"] == pytest.approx(-3.5)
    assert parameter.item() == pytest.approx(1.35)
    assert accumulator.pending_count == 0
    assert not accumulator.has_live_tensors


def test_partial_batch_flushes_and_clears_pending_tensors():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    accumulator = BatchAccumulator(optimizer, batch_size=4)
    accumulator.add(policy_log_prob=parameter, utility=2.0)

    result = accumulator.flush_partial()

    assert result["update_performed"] is True
    assert result["batch_size"] == 1
    assert accumulator.pending_count == 0
    assert not accumulator.has_live_tensors


def test_non_grad_logprob_is_skipped_without_retaining_values():
    parameter = torch.tensor(1.0)
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.0))], lr=0.1)
    accumulator = BatchAccumulator(optimizer, batch_size=1)

    result = accumulator.add(policy_log_prob=parameter, utility=1.0)

    assert result["update_performed"] is False
    assert result["skip_reason"] == "loss_does_not_require_grad"
    assert not accumulator.has_live_tensors


def test_invalid_values_are_rejected():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.0))], lr=0.1)
    accumulator = BatchAccumulator(optimizer, batch_size=2)
    with pytest.raises(ValueError):
        accumulator.add(policy_log_prob=torch.tensor(1.0), utility=float("nan"))
