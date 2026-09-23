from pathlib import Path
import json
import os
import subprocess
import sys

import pytest


torch = pytest.importorskip("torch")

from applications.maas_reproduction.maas_reproduction import contracts
from applications.maas_reproduction.maas_reproduction.models.controller import MultiLayerController
from applications.maas_reproduction.maas_reproduction.models.embeddings import FakeEmbeddingProvider
from applications.maas_reproduction.maas_reproduction.runtime.bootstrap import build_runtime
from applications.maas_reproduction.maas_reproduction.runtime.settings import RuntimeSettings
from applications.maas_reproduction.main import main


def _write_bundle(path: Path) -> torch.Tensor:
    catalog = contracts.operator_catalog_for("GSM8K")
    controller = MultiLayerController(
        embedding_provider=FakeEmbeddingProvider(),
        device="cpu",
    )
    embeddings = torch.arange(len(catalog) * 384, dtype=torch.float32).reshape(len(catalog), 384)
    torch.save(
        {
            "artifact_type": "maas_source_controller_bundle",
            "format_version": 1,
            "dataset": "GSM8K",
            "controller_state_dict": controller.state_dict(),
            "operator_catalog": catalog,
            "operator_embeddings": embeddings,
            "embedding_spec": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384,
            },
            "controller_spec": {"input_dim": 384, "hidden_dim": 32, "num_layers": 4},
            "metadata": {"source": "integration fixture"},
        },
        path,
    )
    return embeddings


def test_source_bundle_runs_test_without_optimizer_or_backward(tmp_path, monkeypatch):
    bundle_path = tmp_path / "source_controller_bundle.pt"
    expected_embeddings = _write_bundle(bundle_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("test mode must not train")

    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    settings = RuntimeSettings(dataset="GSM8K", split="test", mode="test", sample=1)

    runtime = build_runtime(
        settings,
        fake=True,
        output_root_override=tmp_path / "output",
        subset=1,
        source_bundle_path=bundle_path,
    )

    assert runtime.optimizer is None
    assert runtime.batch_accumulator is None
    assert all(not parameter.requires_grad for parameter in runtime.policy_controller.parameters())
    assert runtime.operator_catalog == contracts.operator_catalog_for("GSM8K")
    assert torch.equal(runtime.operator_embeddings.cpu(), expected_embeddings)
    assert len(list(runtime.dataset_runner.iter_run(limit=1))) == 1


def test_source_bundle_cli_ignores_latest_training_checkpoint(tmp_path, monkeypatch):
    bundle_path = tmp_path / "source_controller_bundle.pt"
    _write_bundle(bundle_path)
    output_root = tmp_path / "output"
    stale_checkpoint = output_root / "run_stale" / "checkpoints" / "latest.pt"
    stale_checkpoint.parent.mkdir(parents=True)
    stale_checkpoint.write_bytes(b"this must never be loaded")
    config_root = tmp_path / "config"
    config_root.mkdir()
    (config_root / "models.json").write_text("{}", encoding="utf-8")
    (config_root / "experiments.json").write_text(
        json.dumps(
            {
                "dataset": "GSM8K",
                "split": "test",
                "mode": "test",
                "sample": 1,
                "output_root": str(output_root),
            }
        ),
        encoding="utf-8",
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("test mode must not train")

    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)

    exit_code = main(
        [
            "--mode",
            "test",
            "--config-root",
            str(config_root),
            "--source-bundle",
            str(bundle_path),
            "--fake-model",
            "--subset",
            "1",
        ]
    )

    assert exit_code == 0


def test_run_test_script_forwards_command_line_arguments():
    repository_root = Path(__file__).parents[4]
    script = repository_root / "applications" / "maas_reproduction" / "scripts" / "run_test.py"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository_root)

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0
    assert "--source-bundle" in completed.stdout
