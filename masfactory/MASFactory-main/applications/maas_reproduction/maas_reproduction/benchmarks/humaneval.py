from __future__ import annotations
from typing import Any
from .base import BaseBenchmark


class HumanEvalBenchmark(BaseBenchmark):
    """Execute the supplied public ``check`` function in a restricted scope."""
    def score(self, prediction: str | None, expected_answer: Any, *, context: dict[str, Any] | None = None) -> float:
        if context is None or not context.get("entry_point") or not context.get("test"):
            raise ValueError("HumanEval requires entry_point and test context")
        namespace: dict[str, Any] = {"__builtins__": __builtins__}
        exec(str(prediction), namespace)
        entry = context["entry_point"]
        if entry not in namespace:
            raise ValueError(f"Function {entry} is not defined")
        exec(str(context["test"]), namespace)
        check = namespace.get("check")
        if not callable(check):
            raise ValueError("test must define check")
        result = check(namespace[entry])
        return 1.0 if result is None or result is True else 0.0


HumanEvalScorer = HumanEvalBenchmark
__all__ = ["HumanEvalBenchmark", "HumanEvalScorer"]
