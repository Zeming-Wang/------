"""HumanEval solution execution and scoring."""

from __future__ import annotations

import time
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple


class HumanEvalScorer:
    PASS = "PASS"
    FAIL = "FAIL"

    def __init__(self, log_path: str | Path = ".") -> None:
        self.log_path = Path(log_path)

    class TimeoutError(Exception):
        pass

    def run_with_timeout(self, func, args, timeout):
        result = []
        stop_event = threading.Event()

        def target():
            try:
                result.append(func(*args))
            except Exception as exc:
                result.append(exc)
            finally:
                stop_event.set()

        thread = threading.Thread(target=target)
        thread.start()
        is_timeout = not stop_event.wait(timeout)

        if is_timeout:
            raise self.TimeoutError("Function execution timed out")

        if not result:
            return None
        if isinstance(result[0], Exception):
            raise result[0]
        return result[0]

    def check_solution(self, solution, test, entry_point):
        solution = self._with_special_case_helpers(solution, entry_point)
        try:
            global_dict = {
                "math": __import__("math"),
                "hashlib": __import__("hashlib"),
                "re": __import__("re"),
                "List": List,
                "Dict": Dict,
                "Tuple": Tuple,
                "Optional": Optional,
                "Any": Any,
            }

            exec(solution, global_dict)

            if entry_point not in global_dict:
                raise ValueError(f"Function {entry_point} is not defined in the solution.")

            exec(test, global_dict)

            check = global_dict["check"]
            result = self.run_with_timeout(check, (global_dict[entry_point],), 15)

            if result is None:
                result = (self.PASS, "The solution passed all test cases.")

        except self.TimeoutError:
            result = (
                self.FAIL,
                "Execution timed out. Please check if your solution contains infinite loops or overly time-consuming operations.",
            )
        except Exception as exc:
            error_message = f"Error: {str(exc)}.\n Solution: {solution}.\n Test: {test}"
            self.log_path.mkdir(parents=True, exist_ok=True)
            with (self.log_path / "error.log").open("a", encoding="utf-8") as log_file:
                log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_message}\n")
            result = (self.FAIL, error_message)

        return result

    def calculate_score(self, expected_output: str, prediction: str) -> tuple[float, str]:
        return 0.0, prediction

    def _with_special_case_helpers(self, solution: str, entry_point: str) -> str:
        if entry_point == "decode_cyclic":
            return (
                '\n\ndef encode_cyclic(s: str):\n    """\n    returns encoded string by cycling groups of three characters.\n    """\n'
                "    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]\n"
                "    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]\n"
                '    return "".join(groups)'
                + "\n\n"
                + solution
            )
        if entry_point == "decode_shift":
            return (
                '\n\ndef encode_shift(s: str):\n    """\n    returns encoded string by shifting every character by 5 in the alphabet.\n    """\n'
                '    return "".join([chr(((ord(ch) + 5 - ord("a")) % 26) + ord("a")) for ch in s])\n\n\n'
                + solution
            )
        if entry_point == "find_zero":
            return "\n\ndef poly(xs: list, x: float):\n    return sum(coeff * (x ** i) for i, coeff in enumerate(xs))\n\n" + solution
        return solution
