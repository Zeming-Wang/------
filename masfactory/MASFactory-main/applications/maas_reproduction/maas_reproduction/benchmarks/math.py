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
        # Keep imports lazy, while preserving the source benchmark's parser
        # order and numerical fallback.
        try:
            from sympy import N, simplify
            from sympy.parsing.latex import parse_latex
            from sympy.parsing.sympy_parser import parse_expr
        except ImportError as exc:
            raise RuntimeError(
                "MATH benchmark requires sympy==1.13.1; install the project dependencies."
            ) from exc

        def parse(value: str):
            for parser in (parse_latex, parse_expr):
                try:
                    return parser(value)
                except Exception:
                    pass
            return value

        parsed_actual, parsed_expected = parse(actual), parse(expected)
        try:
            if simplify(parsed_actual - parsed_expected) == 0:
                return 1.0
        except Exception:
            pass
        try:
            if isclose(float(N(parsed_actual)), float(N(parsed_expected)), abs_tol=1e-3):
                return 1.0
        except Exception:
            pass
        return 0.0


MATHScorer = MATHBenchmark
__all__ = ["MATHBenchmark", "MATHScorer"]
