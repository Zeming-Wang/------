import importlib.util
import json
from pathlib import Path

import torch


SCRIPT = Path(__file__).with_name("export_source_controller_bundle.py")


def _load_exporter():
    spec = importlib.util.spec_from_file_location("export_source_controller_bundle", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _controller_state_dict():
    state = {}
    for layer in range(4):
        operator_input = 384 if layer == 0 else 768
        state[f"layers.{layer}.operator_encoder.weight"] = torch.zeros(32, operator_input)
        state[f"layers.{layer}.operator_encoder.bias"] = torch.zeros(32)
        state[f"layers.{layer}.query_encoder.weight"] = torch.zeros(32, 384)
        state[f"layers.{layer}.query_encoder.bias"] = torch.zeros(32)
    return state


def test_exporter_freezes_source_operator_text_and_embeddings(tmp_path):
    exporter = _load_exporter()
    controller_path = tmp_path / "GSM8K_controller_sample4.pth"
    torch.save(_controller_state_dict(), controller_path)
    operator_path = tmp_path / "operator.json"
    operator_path.write_text(
        json.dumps(
            {
                name: {"description": f"description {name}", "interface": f"{name}()"}
                for name in exporter.OPERATOR_CATALOGS["GSM8K"]
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "source_controller_bundle.pt"

    def fake_embedding(text):
        return torch.full((384,), float(text.split(".", 1)[0]))

    exporter.export_source_controller_bundle(
        dataset="GSM8K",
        controller_path=controller_path,
        operator_json_path=operator_path,
        output_path=output_path,
        embedding_fn=fake_embedding,
    )

    bundle = torch.load(output_path, map_location="cpu")
    assert bundle["operator_catalog"] == exporter.OPERATOR_CATALOGS["GSM8K"]
    assert bundle["metadata"]["operator_descriptions"][0] == (
        "0. Generate: description Generate, with interface Generate()."
    )
    assert bundle["operator_embeddings"].shape == (7, 384)
    assert bundle["operator_embeddings"][:, 0].tolist() == list(range(7))


def test_humaneval_v2_exports_full_controller_distribution_fixtures(tmp_path):
    exporter = _load_exporter()
    catalog = exporter.OPERATOR_CATALOGS["HumanEval"]
    controller_path = tmp_path / "HumanEval_controller.pth"
    torch.save(_controller_state_dict(), controller_path)
    operator_path = tmp_path / "operator.json"
    operator_path.write_text(
        json.dumps({name: {"description": name, "interface": f"{name}()"} for name in catalog}),
        encoding="utf-8",
    )
    query_path = tmp_path / "queries.json"
    query_path.write_text(json.dumps([f"problem {index}" for index in range(5)]), encoding="utf-8")
    output_path = tmp_path / "humaneval-v2.pt"

    class FakeLayer(torch.nn.Module):
        def __init__(self, first):
            super().__init__()
            self.operator_encoder = torch.nn.Linear(384 if first else 768, 32)
            self.query_encoder = torch.nn.Linear(384, 32)

        def forward(self, query, operators, previous=None):
            logits = torch.arange(len(catalog), dtype=torch.float32).unsqueeze(0)
            return torch.log_softmax(logits, dim=1), torch.softmax(logits, dim=1)

    class FakeController(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(FakeLayer(index == 0) for index in range(4))

        def forward(self, _text, operators, names):
            logs, selected = [], []
            previous = None
            for layer in self.layers:
                log_probs, _ = layer(torch.zeros(384), operators, previous)
                logs.append(log_probs[0, 0])
                selected.append([names[0]])
                previous = operators[[0]]
            return logs, selected

    exporter.export_source_controller_bundle(
        dataset="HumanEval",
        controller_path=controller_path,
        operator_json_path=operator_path,
        output_path=output_path,
        query_fixture_path=query_path,
        embedding_fn=lambda _text: torch.zeros(384),
        controller_factory=FakeController,
    )

    bundle = torch.load(output_path, map_location="cpu")
    assert bundle["format_version"] == 2
    assert bundle["operator_catalog"] == catalog
    assert len(bundle["metadata"]["controller_parity_fixtures"]) == 5
    first = bundle["metadata"]["controller_parity_fixtures"][0]
    assert len(first["layers"]) == 4
    assert all(tuple(row["probs"].shape) == (len(catalog),) for row in first["layers"])
    assert all(tuple(row["log_probs"].shape) == (len(catalog),) for row in first["layers"])
