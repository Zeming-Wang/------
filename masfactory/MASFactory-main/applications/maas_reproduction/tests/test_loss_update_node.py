from types import SimpleNamespace
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is None:
    loss_update_forward = None
else:
    from maas_reproduction.nodes.loss_update_node import loss_update_forward


class OptimizerSpy:
    def __init__(self) -> None:
        self.steps = 0
        self.zeroes = 0

    def step(self) -> None:
        self.steps += 1

    def zero_grad(self) -> None:
        self.zeroes += 1


class LossUpdateNodeTest(unittest.TestCase):
    def setUp(self) -> None:
        if torch is None:
            raise unittest.SkipTest("torch is required for MaAS loss update tests")

    def test_accumulates_until_batch_is_full_then_updates(self) -> None:
        optimizer = OptimizerSpy()
        attrs = {
            "settings": SimpleNamespace(mode="Graph"),
            "batch_size": 2,
            "batch_logprobs": [],
            "batch_scores": [],
            "batch_costs": [],
            "all_scores": [],
            "previous_cost": 0.0,
            "optimizer": optimizer,
            "device": torch.device("cpu"),
        }

        first = loss_update_forward(
            {"score": 1.0, "cost": 1.0, "logprob": torch.tensor(0.2, requires_grad=True), "problem_index": 0},
            attrs,
        )
        second = loss_update_forward(
            {"score": 0.0, "cost": 1.5, "logprob": torch.tensor(0.3, requires_grad=True), "problem_index": 1},
            attrs,
        )

        self.assertFalse(first["result_update_performed"])
        self.assertTrue(second["result_update_performed"])
        self.assertEqual(optimizer.steps, 1)
        self.assertEqual(optimizer.zeroes, 1)
        self.assertEqual(attrs["batch_logprobs"], [])
        self.assertEqual(attrs["all_scores"], [1.0, 0.0])

    def test_python_float_logprob_records_loss_without_update(self) -> None:
        optimizer = OptimizerSpy()
        attrs = {
            "settings": SimpleNamespace(mode="Graph"),
            "batch_size": 1,
            "batch_logprobs": [],
            "batch_scores": [],
            "batch_costs": [],
            "all_scores": [],
            "previous_cost": 0.0,
            "optimizer": optimizer,
            "device": torch.device("cpu"),
        }

        result = loss_update_forward(
            {"score": 1.0, "cost": 0.1, "logprob": 0.2, "problem_index": 0},
            attrs,
        )

        self.assertIsNotNone(result["result_loss"])
        self.assertFalse(result["result_update_performed"])
        self.assertEqual(optimizer.steps, 0)

    def test_incomplete_batch_only_accumulates(self) -> None:
        optimizer = OptimizerSpy()
        attrs = {
            "settings": SimpleNamespace(mode="Graph"),
            "batch_size": 4,
            "batch_logprobs": [],
            "batch_scores": [],
            "batch_costs": [],
            "all_scores": [],
            "previous_cost": 0.0,
            "optimizer": optimizer,
            "device": torch.device("cpu"),
        }

        result = loss_update_forward(
            {"score": 1.0, "cost": 0.1, "logprob": torch.tensor(0.2, requires_grad=True), "problem_index": 0},
            attrs,
        )

        self.assertFalse(result["result_update_performed"])
        self.assertEqual(len(attrs["batch_logprobs"]), 1)
        self.assertEqual(optimizer.steps, 0)

    def test_resolves_device_from_controller_parameters(self) -> None:
        from maas_reproduction.nodes.loss_update_node import _resolve_device

        controller = torch.nn.Linear(1, 1)

        self.assertEqual(_resolve_device({"controller": controller}), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
