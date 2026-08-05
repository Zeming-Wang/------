"""
Helper utilities for the LATS HumanEval executor.

This module provides only:
- set_print_code_attempts (for --print-code)
- run_humaneval_forward (used by CustomNode in the workflow)

No custom Agent or CustomNode subclasses; the graph uses system Agent and CustomNode via NodeTemplate.
"""

from ..humaneval.load import extract_python_code, parse_internal_tests_from_test
from ..humaneval.executor import run_internal_tests, full_evaluate
from ..utils.tee import tee, get_log_file

_print_code_attempts = False


def set_print_code_attempts(value: bool) -> None:
    """Enable or disable printing each generated code attempt."""
    global _print_code_attempts
    _print_code_attempts = value


def _print_generated_func_body(func_body: str, problem_name: str = "") -> None:
    """Print generated code to terminal and log (if --log and --print-code)."""
    if not _print_code_attempts:
        return
    title = "GENERATED FUNC BODY"
    if problem_name:
        title += f" [{problem_name}]"
    tee(f"\n--------------------- {title} ---------------------", get_log_file())
    tee(func_body, get_log_file())
    tee("------------------------------------------\n", get_log_file())


def run_humaneval_forward(input_dict: dict, attrs: dict | None = None) -> dict:
    """HumanEval execution (originally HumanEvalEnvironment._forward).

    Accept both edge-passed fields (message) and attribute-based fields (attrs),
    and prefer explicit message values when present.
    """
    attrs = attrs or {}
    content = (
        input_dict.get("action", "")
        or input_dict.get("content", "")
        or (attrs.get("content", "") if isinstance(attrs, dict) else "")
    )
    raw = str(content).strip()

    problem = input_dict.get("problem")
    if not isinstance(problem, dict):
        candidate = attrs.get("problem") if isinstance(attrs, dict) else None
        problem = candidate if isinstance(candidate, dict) else {}

    internal_tests = input_dict.get("internal_tests")
    if not isinstance(internal_tests, list):
        candidate_tests = attrs.get("internal_tests") if isinstance(attrs, dict) else None
        internal_tests = candidate_tests if isinstance(candidate_tests, list) else []
    entry_point = problem.get("entry_point", "")
    test = problem.get("test", "")
    prompt = problem.get("prompt", "")

    fail_safe = {
        "observation": "Error: No valid Python code.",
        "reward": 0.0,
        "reward_internal": 0.0,
        "reward_real": 0.0,
        "full_passed": False,
        "action": raw,
        "problem": problem,
        "internal_tests": internal_tests,
    }

    code = extract_python_code(raw)
    if not code:
        fail_safe["observation"] = "Error: Use a ```python ... ``` block or full function."
        return fail_safe
    if "def " not in code and prompt:
        code = prompt.rstrip() + "\n" + code

    if _print_code_attempts:
        _print_generated_func_body(code, problem.get("name", ""))

    if not internal_tests:
        internal_tests = parse_internal_tests_from_test(test, max_tests=6)

    is_passing_internal, feedback, reward_internal = run_internal_tests(
        code, internal_tests, timeout=5
    )
    reward_real = 1.0 if full_evaluate(entry_point, code, test, timeout=10) else 0.0
    reward = reward_internal + reward_real

    return {
        "observation": feedback,
        "reward": reward,
        "reward_internal": reward_internal,
        "reward_real": reward_real,
        "full_passed": reward_real >= 1.0,
        "action": code,
        "problem": problem,
        "internal_tests": internal_tests,
    }
