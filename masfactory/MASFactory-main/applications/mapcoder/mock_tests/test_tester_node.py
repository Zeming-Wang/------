"""Local tests for the Tester CustomNode (no API key required).

Run from repo root:

    python -m applications.mapcoder.tests.test_tester_node
"""

from applications.mapcoder.components.tester import (
    extract_code,
    run_sample_io_tests,
)


def _check(name: str, ok: bool, detail: str = "") -> None:
    mark = "OK  " if ok else "FAIL"
    print(f"[{mark}] {name}{(': ' + detail) if detail else ''}")
    if not ok:
        raise SystemExit(1)


def test_extract_code_with_python_fence() -> None:
    text = "Here is my answer:\n\n```python\ndef f(x):\n    return x + 1\n```\n"
    code = extract_code(text)
    _check("extract_code_with_python_fence", "def f(x):" in code and "return x + 1" in code)


def test_extract_code_with_generic_fence() -> None:
    text = "```\nprint(2 + 2)\n```"
    code = extract_code(text)
    _check("extract_code_with_generic_fence", code.strip() == "print(2 + 2)")


def test_extract_code_without_fence() -> None:
    text = "def g(x):\n    return x * 2\n"
    code = extract_code(text)
    _check("extract_code_without_fence", "def g(x):" in code)


def test_run_passed() -> None:
    candidate = "def square(x):\n    return x * x\n"
    sample_io = [
        {"call": "square(2)", "expected": "4"},
        {"call": "square(-3)", "expected": "9"},
    ]
    out = run_sample_io_tests(
        {"code": "```python\n" + candidate + "```", "sample_io": sample_io},
        {},
    )
    _check(
        "run_passed",
        out["full_passed"] is True and "passed" in out["observation"].lower(),
        detail=str(out),
    )


def test_run_failed_assertion() -> None:
    candidate = "def square(x):\n    return x + x  # wrong\n"
    sample_io = [
        {"call": "square(3)", "expected": "9"},
    ]
    out = run_sample_io_tests({"code": candidate, "sample_io": sample_io}, {})
    _check(
        "run_failed_assertion",
        out["full_passed"] is False and ("FAIL_CASE_1" in out["observation"] or "expected 9" in out["observation"]),
        detail=str(out),
    )


def test_run_runtime_error() -> None:
    candidate = "def f(x):\n    return x / 0\n"
    sample_io = [
        {"call": "f(1)", "expected": "0"},
    ]
    out = run_sample_io_tests({"code": candidate, "sample_io": sample_io}, {})
    _check(
        "run_runtime_error",
        out["full_passed"] is False and "ZeroDivisionError" in out["observation"],
        detail=str(out),
    )


def test_run_timeout() -> None:
    # Use an explicit short timeout via env-config trick: re-import after patch.
    candidate = "def loop():\n    while True:\n        pass\n"
    sample_io = [{"call": "loop()", "expected": "0"}]
    # Lower timeout via monkeypatch on the module's constant.
    import applications.mapcoder.components.tester as tester_mod
    orig = tester_mod.SANDBOX_TIMEOUT_S
    tester_mod.SANDBOX_TIMEOUT_S = 2
    try:
        out = run_sample_io_tests({"code": candidate, "sample_io": sample_io}, {})
    finally:
        tester_mod.SANDBOX_TIMEOUT_S = orig
    _check(
        "run_timeout",
        out["full_passed"] is False and "TIMEOUT" in out["observation"],
        detail=str(out),
    )


def test_run_no_code() -> None:
    out = run_sample_io_tests({"code": "", "sample_io": [{"call": "1", "expected": "1"}]}, {})
    _check("run_no_code", out["full_passed"] is False and "no code" in out["observation"].lower())


def main() -> None:
    test_extract_code_with_python_fence()
    test_extract_code_with_generic_fence()
    test_extract_code_without_fence()
    test_run_passed()
    test_run_failed_assertion()
    test_run_runtime_error()
    test_run_timeout()
    test_run_no_code()
    print("\nAll Tester tests passed.")


if __name__ == "__main__":
    main()
