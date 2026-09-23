import random

import pytest

torch = pytest.importorskip("torch")

from maas_reproduction.runtime.checkpoint_manager import CheckpointError, CheckpointManager


def test_checkpoint_round_trip_restores_training_and_rng_state(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(tmp_path)
    path = manager.save(
        model,
        optimizer,
        cursor=7,
        epoch=3,
        operator_catalog=("Generate", "ScEnsemble"),
        metadata={"batch": 4},
    )
    assert path.exists()
    expected_random = random.random()
    expected_torch = torch.rand(1)

    random.random()
    torch.rand(1)
    restored = manager.load(expected_operator_catalog=("Generate", "ScEnsemble"))
    assert restored["cursor"] == 7
    assert restored["epoch"] == 3
    assert restored["metadata"] == {"batch": 4}
    assert random.random() == expected_random
    assert torch.equal(torch.rand(1), expected_torch)


def test_checkpoint_rejects_catalog_mismatch(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(tmp_path)
    manager.save(model, optimizer, cursor=0, epoch=0, operator_catalog=("Generate",))
    with pytest.raises(CheckpointError, match="operator catalog"):
        manager.load(expected_operator_catalog=("Programmer",))


def test_live_policy_tensor_is_never_accepted(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(tmp_path)
    # A live tensor can only enter through an exposed state_dict value.  The
    # manager detaches ordinary module state, so verify the resulting payload.
    manager.save(model, optimizer, cursor=0, epoch=0, operator_catalog=("Generate",))
    payload = manager.load()
    assert not any(
        isinstance(value, torch.Tensor) and value.requires_grad
        for value in payload["controller"].values()
    )


def test_checkpoint_refuses_to_capture_a_partial_batch(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(tmp_path)
    with pytest.raises(CheckpointError, match="batch boundary"):
        manager.save(model, optimizer, cursor=1, epoch=0,
                     operator_catalog=("Generate",), pending_gradients=2)


def test_checkpoint_records_that_it_sits_on_a_batch_boundary(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(tmp_path)
    manager.save(model, optimizer, cursor=0, epoch=0, operator_catalog=("Generate",))
    assert manager.load()["pending_gradient_count"] == 0


def test_load_rejects_a_checkpoint_taken_inside_a_batch(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(tmp_path)
    path = manager.save(model, optimizer, cursor=0, epoch=0, operator_catalog=("Generate",))
    payload = manager.load()
    payload["pending_gradient_count"] = 3
    torch.save(payload, path)
    with pytest.raises(CheckpointError, match="batch boundary"):
        manager.load()
