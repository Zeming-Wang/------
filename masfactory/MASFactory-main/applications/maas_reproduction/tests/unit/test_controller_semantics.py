from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from applications.maas_reproduction.maas_reproduction.models import controller as controller_module
from applications.maas_reproduction.maas_reproduction.models.controller import (
    MultiLayerController,
    sample_operators,
)
from applications.maas_reproduction.maas_reproduction.models.embeddings import FakeEmbeddingProvider


def test_sample_operators_matches_source_cumulative_without_replacement(monkeypatch):
    probabilities = torch.tensor([0.1, 0.2, 0.3, 0.4])

    def choose_first_remaining(weights, num_samples):
        assert num_samples == 1
        return torch.tensor([0], device=weights.device)

    monkeypatch.setattr(torch, "multinomial", choose_first_remaining)

    selected = sample_operators(probabilities, threshold=0.3)

    assert selected.tolist() == [0, 1]


def test_first_layer_generate_reorder_drives_next_layer_previous_embedding(monkeypatch):
    sampled = iter((torch.tensor([0, 1]), torch.tensor([3])))
    monkeypatch.setattr(controller_module, "sample_operators", lambda *_args, **_kwargs: next(sampled))

    captured: dict[str, object] = {}

    class FixedFirst(torch.nn.Module):
        def forward(self, query, operators, previous=None):
            probabilities = torch.tensor([[0.4, 0.35, 0.15, 0.1]], device=operators.device)
            return probabilities.log(), probabilities

    class CaptureSecond(torch.nn.Module):
        def forward(self, query, operators, previous=None):
            captured["previous"] = previous.detach().cpu()
            probabilities = torch.tensor([[0.1, 0.1, 0.1, 0.7]], device=operators.device)
            return probabilities.log(), probabilities

    controller = MultiLayerController(
        input_dim=2,
        hidden_dim=2,
        num_layers=2,
        device=torch.device("cpu"),
        embedding_provider=FakeEmbeddingProvider(2),
    )
    controller.layers = torch.nn.ModuleList([FixedFirst(), CaptureSecond()])
    embeddings = torch.tensor([[10.0, 0.0], [20.0, 0.0], [30.0, 0.0], [40.0, 0.0]])

    log_probs, selected_names = controller.forward(
        "query", embeddings, ("SelfRefine", "Generate", "Programmer", "EarlyStop")
    )

    assert selected_names[0] == ["Generate", "SelfRefine"]
    assert torch.equal(captured["previous"], embeddings[[1, 0]])
    assert torch.isclose(log_probs[0], torch.log(torch.tensor(0.35)) + torch.log(torch.tensor(0.4)))


def test_controller_rejects_catalog_without_exact_generate():
    controller = MultiLayerController(
        input_dim=2,
        hidden_dim=2,
        num_layers=1,
        device=torch.device("cpu"),
        embedding_provider=FakeEmbeddingProvider(2),
    )

    with pytest.raises(ValueError, match="must contain Generate"):
        controller.forward("query", torch.zeros((2, 2)), ("SelfRefine", "EarlyStop"))


def test_controller_moves_policy_parameters_after_layers_are_created():
    controller = MultiLayerController(
        input_dim=2,
        hidden_dim=2,
        num_layers=1,
        device=torch.device("meta"),
        embedding_provider=FakeEmbeddingProvider(2),
    )

    assert controller.device.type == "meta"
    assert {parameter.device.type for parameter in controller.parameters()} == {"meta"}
