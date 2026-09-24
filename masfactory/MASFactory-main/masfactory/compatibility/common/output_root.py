from __future__ import annotations

from typing import ClassVar

from masfactory.components.graphs.root_graph import RootGraph


class WorkflowOutputRootGraph(RootGraph):
    """Pick a node payload from a compatibility outputs map after invoke."""

    _outputs_key: ClassVar[str] = ""
    _preferred_suffixes: ClassVar[tuple[str, ...]] = ()
    _node_id_hints: ClassVar[tuple[str, ...]] = ()

    def invoke(self, input: dict, attributes: dict[str, object] | None = None):
        output, attrs = super().invoke(input, attributes)
        outputs = attrs.get(self._outputs_key)
        if not isinstance(outputs, dict) or not outputs:
            return output, attrs
        for payload in reversed(list(outputs.values())):
            if isinstance(payload, dict) and (
                payload.get("text") or payload.get("content") or payload.get("message")
            ):
                return payload, attrs
        for suffix in self._preferred_suffixes:
            for node_id, payload in outputs.items():
                if node_id == suffix or node_id.endswith(f"__{suffix}") or node_id.endswith(suffix):
                    if isinstance(payload, dict):
                        return payload, attrs
        for hint in self._node_id_hints:
            for node_id, payload in outputs.items():
                if hint in node_id and isinstance(payload, dict):
                    return payload, attrs
        last = next(reversed(outputs.values()))
        if isinstance(last, dict):
            return last, attrs
        return output, attrs
