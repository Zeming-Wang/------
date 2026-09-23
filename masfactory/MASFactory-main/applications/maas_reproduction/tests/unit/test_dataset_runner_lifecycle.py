import json

import pytest

from applications.maas_reproduction.maas_reproduction.adapters.artifact_store import ArtifactStore
from applications.maas_reproduction.maas_reproduction.runtime.dataset_runner import DatasetRunner
from applications.maas_reproduction.maas_reproduction.runtime.run_logger import RunLogger


class Graph:
    def __init__(self):
        self.calls = []

    def invoke(self, message):
        sample = message["sample"]
        self.calls.append(sample["problem_index"])
        return {"sample_result": {
            "problem_index": sample["problem_index"],
            "status": "success",
            "score": 1.0,
            "cost_delta": 0.25,
            "cost_reliable": True,
            "evaluation_reliable": True,
            "update_performed": False,
        }}


class CheckpointSpy:
    def __init__(self, directory):
        self.directory = directory
        self.calls = []

    def save(self, controller, optimizer, **kwargs):
        self.calls.append(kwargs)


def test_runner_executes_each_sample_once_per_epoch_and_writes_public_artifacts(tmp_path):
    graph = Graph()
    artifacts = ArtifactStore(tmp_path, results_root=tmp_path / "results")
    runner = DatasetRunner(
        graph=graph,
        dataset=[{"problem": "a"}, {"problem": "b"}],
        epochs=3,
        artifact_store=artifacts,
        run_config={"dataset": "GSM8K", "run_id": "run_test"},
    )

    results = runner.run()

    assert len(results) == 6
    assert graph.calls == [0, 1, 0, 1, 0, 1]
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "results" / "sample_0.json").exists()
    assert json.loads((tmp_path / "metrics.json").read_text()) == {
        "sample_count": 6,
        "success_count": 6,
        "failure_count": 0,
        "average_score": 1.0,
        "average_cost": 0.25,
        "update_count": 0,
        "completed_epochs": 3,
    }
    assert runner.epoch == 3
    assert runner.cursor == 0


def test_runner_checkpoint_position_resumes_without_repeating_samples(tmp_path):
    graph = Graph()
    checkpoints = CheckpointSpy(tmp_path / "checkpoints")
    runner = DatasetRunner(
        graph=graph,
        dataset=[{"problem": "a"}, {"problem": "b"}],
        epochs=3,
        checkpoint_manager=checkpoints,
        controller=object(),
        optimizer=object(),
        operator_catalog=("Generate",),
    )

    runner.restore_checkpoint({"cursor": 1, "epoch": 1, "metadata": {}})
    runner.run()

    assert graph.calls == [1, 0, 1]
    assert runner.epoch == 3
    assert runner.cursor == 0
    assert any(call.get("path") and call["path"].name == "checkpoint_epoch_002.pt" for call in checkpoints.calls)


def test_runner_logs_runtime_errors_without_private_payload(tmp_path):
    class BrokenGraph:
        def invoke(self, _message):
            raise RuntimeError("api_key=secret prompt=full private prompt")

    logger = RunLogger(tmp_path, "run_test")
    runner = DatasetRunner(
        graph=BrokenGraph(),
        dataset=[{"problem": "a"}],
        run_logger=logger,
    )

    with pytest.raises(RuntimeError):
        runner.run()

    error_text = (tmp_path / "logs" / "error.log").read_text(encoding="utf-8")
    assert '"stage": "graph_invoke"' in error_text
    assert "secret" not in error_text
    assert "full private prompt" not in error_text


class FakeAccumulator:
    """Stand-in for the BatchAccumulator boundary protocol."""

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.pending_count = 0
        self.partial_flushes = 0

    def record(self):
        self.pending_count += 1
        if self.pending_count >= self.batch_size:
            self.pending_count = 0
            return True
        return False

    def flush_partial(self):
        self.partial_flushes += 1
        performed = self.pending_count > 0
        self.pending_count = 0
        return {"update_performed": performed, "pending_count": 0, "loss_value": 0.5}


class BatchGraph:
    """Drive a FakeAccumulator the way LossUpdateNode drives the real one."""

    def __init__(self, accumulator, *, fail_at=None):
        self.accumulator = accumulator
        self.fail_at = fail_at
        self.calls = []

    def invoke(self, message):
        index = message["sample"]["problem_index"]
        self.calls.append(index)
        if self.fail_at is not None and index == self.fail_at:
            raise RuntimeError("simulated crash")
        return {"sample_result": {
            "problem_index": index,
            "status": "success",
            "score": 1.0,
            "cost_delta": 0.25,
            "cost_reliable": True,
            "evaluation_reliable": True,
            "update_performed": self.accumulator.record(),
        }}


def _batch_runner(tmp_path, *, batch_size, dataset_size=6, epochs=1, fail_at=None):
    accumulator = FakeAccumulator(batch_size)
    graph = BatchGraph(accumulator, fail_at=fail_at)
    checkpoints = CheckpointSpy(tmp_path / "checkpoints")
    runner = DatasetRunner(
        graph=graph,
        dataset=[{"problem": str(index)} for index in range(dataset_size)],
        epochs=epochs,
        batch_accumulator=accumulator,
        checkpoint_manager=checkpoints,
        controller=object(),
        optimizer=object(),
        operator_catalog=("Generate",),
        artifact_store=ArtifactStore(tmp_path, results_root=tmp_path / "results"),
    )
    return runner, graph, accumulator, checkpoints


def test_runner_checkpoints_only_at_batch_boundaries(tmp_path):
    runner, _graph, accumulator, checkpoints = _batch_runner(tmp_path, batch_size=4)

    runner.run()

    # 6 samples, batch_size=4: only offset 3 completed a batch, so cursor=4 is
    # the only mid-epoch resumable position; the trailing two samples stay
    # pending.  Cursors 1, 2, 3, 5 and 6 must never reach a checkpoint.
    assert {call["cursor"] for call in checkpoints.calls} == {0, 4}
    history = [call for call in checkpoints.calls if call.get("path") is not None]
    assert [call["path"].name for call in history] == ["checkpoint_epoch_001.pt"]
    assert accumulator.partial_flushes == 1
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["sample_count"] == 6
    # One full batch plus the epoch-end partial flush.
    assert metrics["update_count"] == 2
    assert metrics["completed_epochs"] == 1


def test_runner_crash_inside_a_batch_resumes_from_the_last_boundary(tmp_path):
    runner, graph, _accumulator, checkpoints = _batch_runner(tmp_path, batch_size=4, fail_at=5)

    with pytest.raises(RuntimeError):
        runner.run()

    assert graph.calls == [0, 1, 2, 3, 4, 5]
    saved = [call for call in checkpoints.calls if call.get("path") is None]
    # The crash landed two samples into the second batch: the checkpoint must
    # still describe cursor=4, never the mid-batch cursor=5.
    assert [call["cursor"] for call in saved] == [4]
    assert saved[-1]["pending_gradients"] == 0

    resumed, resumed_graph, _resumed_accumulator, _ = _batch_runner(tmp_path, batch_size=4)
    resumed.restore_checkpoint({
        "cursor": 4,
        "epoch": 0,
        "pending_gradient_count": 0,
        "metadata": saved[-1]["metadata"],
    })
    resumed.run()

    assert resumed_graph.calls == [4, 5]


def test_restore_rejects_metrics_that_disagree_with_the_position(tmp_path):
    runner, _graph, _accumulator, _checkpoints = _batch_runner(tmp_path, batch_size=2, dataset_size=4)

    with pytest.raises(ValueError, match="disagree"):
        runner.restore_checkpoint({
            "cursor": 2,
            "epoch": 0,
            "metadata": {"runner_metrics": {"sample_count": 3, "completed_epochs": 0}},
        })

    with pytest.raises(ValueError, match="non-negative int"):
        runner.restore_checkpoint({
            "cursor": 2,
            "epoch": 0,
            "metadata": {"runner_metrics": {"sample_count": True, "completed_epochs": 0}},
        })

    with pytest.raises(ValueError, match="completed_epochs"):
        runner.restore_checkpoint({
            "cursor": 0,
            "epoch": 1,
            "metadata": {"runner_metrics": {"sample_count": 4, "completed_epochs": 0}},
        })


def test_truncated_run_does_not_record_a_completed_epoch(tmp_path):
    runner, _graph, _accumulator, checkpoints = _batch_runner(tmp_path, batch_size=4, dataset_size=6)

    runner.run(limit=3)

    assert runner.epoch == 0
    assert runner.cursor == 3
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["completed_epochs"] == 0
    assert metrics["sample_count"] == 3
    assert not any(call.get("path") for call in checkpoints.calls)
