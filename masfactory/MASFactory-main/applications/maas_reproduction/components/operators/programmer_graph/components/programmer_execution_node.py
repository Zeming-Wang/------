"""Injected execution adapter node."""
from __future__ import annotations
from typing import Any
from masfactory.components.custom_node import CustomNode
from ..workflow_helpers import call_adapter


class ProgrammerExecutionNode(CustomNode):
    def __init__(self, name: str = "ProgrammerExecutionNode", *, executor: Any = None, **kwargs: Any) -> None:
        self.executor = executor
        super().__init__(name, forward=self._forward, pull_keys={}, push_keys={}, **kwargs)

    def _forward(self, data: dict[str, Any]) -> dict[str, Any]:
        if data.get("parse_error"):
            return {**data, "execution_success": False, "execution_output": data["parse_error"],
                    "feedback": data["parse_error"]}
        try:
            if self.executor is None:
                raise RuntimeError("programmer executor is not configured")
            output = call_adapter(self.executor, data)
            if isinstance(output, dict) and output.get("success", True) is False:
                error = str(output.get("error", "execution failed"))
                return {**data, "execution_success": False, "execution_output": error, "feedback": error}
            text = output if isinstance(output, str) else str(output)
            return {**data, "execution_success": True, "execution_output": text, "feedback": ""}
        except Exception as exc:
            return {**data, "execution_success": False, "execution_output": str(exc), "feedback": str(exc)}


__all__ = ["ProgrammerExecutionNode"]
