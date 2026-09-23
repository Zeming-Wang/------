"""MATH answer extraction and scoring."""

from __future__ import annotations

import re
from math import isclose
from typing import Any

#判断math两个结果是否相同，有可能存在的问题，比如失败的原因不明确就记为0分
class MATHScorer:
    def extract_model_answer(self, text: str) -> str:
        pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
        boxed_matches = re.findall(pattern, text, re.DOTALL)
        if boxed_matches:
            return boxed_matches[-1].strip()

        sentence_end_pattern = r"(?<!\d)[.!?]\s+"
        sentences = re.split(sentence_end_pattern, text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        return sentences[-1] if sentences else ""

    def calculate_score(self, expected_output: str, prediction: str) -> tuple[int, str]:
        expected_answer = self.extract_model_answer(expected_output)
        predicted_answer = self.extract_model_answer(prediction)

        if self.math_equal(predicted_answer, expected_answer):
            return 1, predicted_answer
        return 0, predicted_answer

    def math_equal(self, prediction: Any, reference: Any) -> bool:
        if str(prediction) == str(reference):
            return True

        try:
            if self.is_digit(prediction) and self.is_digit(reference):
                prediction = self.parse_digits(prediction)
                reference = self.parse_digits(reference)
                return isclose(prediction, reference, abs_tol=1e-3)
        except Exception:
            pass

        try:
            return self.symbolic_equal(prediction, reference)
        except Exception:
            pass

        return False

    def is_digit(self, num):
        return self.parse_digits(num) is not None

    def parse_digits(self, num):
        num = str(num).replace(",", "")
        try:
            return float(num)
        except Exception:
            if num.endswith("%"):
                num = num[:-1]
                if num.endswith("\\"):
                    num = num[:-1]
                try:
                    return float(num) / 100
                except Exception:
                    pass
        return None

    def symbolic_equal(self, a, b):
        from sympy import N, simplify
        from sympy.parsing.latex import parse_latex
        from sympy.parsing.sympy_parser import parse_expr

        def _parse(s):
            for parser in [parse_latex, parse_expr]:
                try:
                    return parser(s)
                except Exception:
                    pass
            return s

        a = _parse(a)
        b = _parse(b)

        try:
            if simplify(a - b) == 0:
                return True
        except Exception:
            pass

        try:
            if isclose(N(a), N(b), abs_tol=1e-3):
                return True
        except Exception:
            pass
        return False
