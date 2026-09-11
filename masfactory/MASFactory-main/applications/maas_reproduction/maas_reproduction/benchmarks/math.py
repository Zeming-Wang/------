from __future__ import annotations
import re
from math import isclose
from typing import Any
from .base import BaseBenchmark


class MATHBenchmark(BaseBenchmark):
    @staticmethod
    def extract_model_answer(text: Any) -> str:
        matches = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", str(text), re.DOTALL)
        if matches:
            return matches[-1].strip()
        sentences = [s.strip() for s in re.split(r"(?<!\d)[.!?]\s+", str(text)) if s.strip()]
        return sentences[-1] if sentences else ""

    @staticmethod
    def _number(value: Any) -> float | None:
        text = str(value).replace(",", "").strip()
        try:
            if text.endswith("%"):
                return float(text[:-1].rstrip("\\")) / 100
            return float(text)
        except (TypeError, ValueError):
            return None

    def score(self, prediction: str | None, expected_answer: Any, *, context: dict[str, Any] | None = None) -> float:
        actual = self.extract_model_answer(prediction)
        expected = self.extract_model_answer(expected_answer)
        if actual == expected:
            return 1.0
        a, b = self._number(actual), self._number(expected)
        if a is not None and b is not None and isclose(a, b, abs_tol=1e-3):
            return 1.0
        # Keep the adapter dependency-light; symbolic equality is optional.
        try:
            from sympy import simplify
            from sympy.parsing.sympy_parser import parse_expr
            if simplify(parse_expr(actual) - parse_expr(expected)) == 0:
                return 1.0
        except Exception:
            pass
        return 0.0


MATHScorer = MATHBenchmark
__all__ = ["MATHBenchmark", "MATHScorer"]
