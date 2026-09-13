"""Sequential dataset execution lifecycle for Task 2."""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any


class DatasetRunner:
    """Invoke a RootGraph once per sample with isolated invocation state."""

    def __init__(self, *, graph: object, dataset: Iterable[Mapping[str, Any]], start_index: int = 0,
                 epochs: int = 1, batch_accumulator: object | None = None,
                 checkpoint_manager: object | None = None, controller: object | None = None,
                 optimizer: object | None = None, operator_catalog: Iterable[str] = (),
                 artifact_store: object | None = None) -> None:
        if not hasattr(graph, "invoke"):
            raise TypeError("graph must expose invoke")
        if start_index < 0:
            raise ValueError("start_index must be non-negative")
        self.graph = graph
        self.dataset = list(dataset)
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        self.epochs = epochs
        self.batch_accumulator = batch_accumulator
        self.checkpoint_manager = checkpoint_manager
        self.controller, self.optimizer = controller, optimizer
        self.operator_catalog = tuple(operator_catalog)
        self.artifact_store = artifact_store
        self.cursor = start_index
        self.epoch = 0

    def run(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Run samples in order and retain only detached public ``sample_result`` records."""
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        results: list[dict[str, Any]] = []
        for offset, sample in enumerate(self.dataset):
            if offset < self.cursor:
                continue
            if limit is not None and len(results) >= limit:
                break
            if not isinstance(sample, Mapping):
                raise TypeError("dataset samples must be mappings")
            # Copy the envelope so a graph cannot mutate dataset-owned state.
            payload = dict(sample)
            payload.setdefault("problem_index", offset)
            output = self.graph.invoke({"sample": payload})
            if isinstance(output, tuple):
                output = output[0]
            if not isinstance(output, Mapping) or "sample_result" not in output:
                raise ValueError("RootGraph output must contain sample_result")
            record = output["sample_result"]
            if hasattr(record, "to_dict"):
                record = record.to_dict()
            if not isinstance(record, Mapping):
                raise TypeError("sample_result must be a mapping")
            results.append(dict(record))
            if self.artifact_store is not None:
                self.artifact_store.write_sample_result(results[-1])
            self.cursor = offset + 1
            if self.checkpoint_manager is not None and self.controller is not None and self.optimizer is not None:
                self.checkpoint_manager.save(self.controller, self.optimizer, cursor=self.cursor,
                                             epoch=self.epoch, operator_catalog=self.operator_catalog)
        if self.batch_accumulator is not None and getattr(self.batch_accumulator, "pending_count", 0):
            self.batch_accumulator.flush_partial()
        self.epoch += 1
        return results

    def iter_run(self) -> Iterator[dict[str, Any]]:
        for result in self.run():
            yield result


__all__ = ["DatasetRunner"]
