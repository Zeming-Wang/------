from __future__ import annotations

from masfactory.components.graphs.root_graph import RootGraph

from masfactory.compatibility.dify.runtime import DifyRuntimeState, init_dify_state


class DifyRootGraph(RootGraph):
    """Executable root graph for compiled Dify workflows."""

    def invoke(self, input: dict, attributes: dict[str, object] | None = None):
        if attributes:
            self._attributes_store = {**self._attributes_store, **attributes}
        self._attributes_store.update(input)
        metadata = self._attributes_store.get("compatibility")
        if isinstance(metadata, dict):
            init_dify_state(metadata, self._attributes_store).attach(self._attributes_store)
        output, attrs = super().invoke(input, attributes)
        state = DifyRuntimeState.from_store(self._attributes_store)
        if state.workflow_result is not None:
            output = state.workflow_result
            self._output = output
        return output, attrs
