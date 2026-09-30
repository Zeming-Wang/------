from __future__ import annotations

import pytest

from masfactory.components.graphs.graph import Graph

from applications.maas_reproduction.components.architecture_exec_graph.workflow import ArchitectureExecGraph


def test_humaneval_retries_the_whole_architecture_at_most_three_times(monkeypatch) -> None:
    calls = []

    def fake_graph_forward(self, input):
        calls.append(dict(input))
        if len(calls) < 3:
            raise RuntimeError("transient graph failure")
        return {"architecture_result": "ok"}

    monkeypatch.setattr(Graph, "_forward", fake_graph_forward)
    graph = ArchitectureExecGraph(
        dataset="HumanEval",
        operator_catalog=("Generate",),
        graph_max_attempts=3,
    )

    assert graph._forward({"architecture_request": "request"}) == {"architecture_result": "ok"}
    assert len(calls) == 3


def test_humaneval_raises_after_three_whole_graph_failures(monkeypatch) -> None:
    calls = []

    def fake_graph_forward(self, input):
        calls.append(dict(input))
        raise RuntimeError("persistent graph failure")

    monkeypatch.setattr(Graph, "_forward", fake_graph_forward)
    graph = ArchitectureExecGraph(
        dataset="HumanEval",
        operator_catalog=("Generate",),
        graph_max_attempts=3,
    )

    with pytest.raises(RuntimeError, match="persistent"):
        graph._forward({"architecture_request": "request"})
    assert len(calls) == 3
