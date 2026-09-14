"""Shared Programmer primitives used by dynamic and bootstrap entry points."""
from __future__ import annotations

from typing import Any

from .workflow_helpers import call_adapter, extract_code


class ProgrammerCore:
    """Single implementation of code generation, parsing, execution and retry."""

    def __init__(self, *, generator: Any, executor: Any, max_attempts: int = 3) -> None:
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        self.generator = generator
        self.executor = executor
        self.max_attempts = max_attempts

    @staticmethod
    def parse(value: Any) -> str:
        code = extract_code(value)
        if not isinstance(code, str) or not code.strip():
            raise ValueError("programmer did not return code")
        if "solve" not in code:
            raise ValueError("generated code has no solve function")
        return code

    def execute(self, payload: dict[str, Any], code: str) -> tuple[bool, str, str]:
        if self.executor is None:
            raise RuntimeError("programmer executor is not configured")
        output = call_adapter(self.executor, {**payload, "code": code})
        if isinstance(output, dict) and output.get("success", True) is False:
            error = str(output.get("error", "execution failed"))
            return False, error, error
        text = output if isinstance(output, str) else str(output)
        return True, text, ""

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        feedback = payload.get("feedback", "")
        last_code: str | None = None
        last_output = ""
        last_error = ""
        for attempt in range(1, self.max_attempts + 1):
            generated = call_adapter(self.generator, {**payload, "feedback": feedback, "attempt": attempt})
            code = self.parse(generated)
            last_code = code
            success, output, error = self.execute(payload, code)
            last_output, last_error = output, error
            if success:
                return {"code": code, "output": output, "attempts": attempt}
            feedback = error
        return {"code": last_code, "output": last_output, "error": last_error, "attempts": self.max_attempts}


__all__ = ["ProgrammerCore"]
