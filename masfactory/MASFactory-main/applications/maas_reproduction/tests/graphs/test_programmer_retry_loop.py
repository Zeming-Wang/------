
from __future__ import annotations

import pytest

from applications.maas_reproduction.components.operators.programmer_graph.programmer_core import (
    ProgrammerCore,
)


def test_parse_failure_is_retried_with_feedback() -> None:
    calls: list[dict[str, object]] = []

    def generator(payload: dict[str, object]) -> dict[str, str]:
        calls.append(dict(payload))
        if len(calls) == 1:
            return {"code": "The answer is 42."}
        return {"code": "def solve():\n    return 42\n\nprint(solve())\n"}

    core = ProgrammerCore(generator=generator, executor=lambda _payload: "42\n", max_attempts=3)

    result = core.run({"problem": "What is six times seven?"})

    assert result["output"] == "42\n"
    assert result["attempts"] == 2
    assert len(calls) == 2
    assert calls[0]["feedback"] == ""
    assert "Code format error" in str(calls[1]["feedback"])
    assert "zero-argument function definition named solve" in str(calls[1]["feedback"])
    assert calls[1]["attempt"] == 2


def test_parse_extracts_a_fenced_python_program() -> None:
    code = ProgrammerCore.parse(
        {"code": "Here is the program:\n```python\ndef solve():\n    return 7\n\nprint(solve())\n```"}
    )

    assert code.startswith("def solve()")
    assert code.endswith("print(solve())")


def test_parse_rejects_solve_mentioned_only_in_a_comment() -> None:
    with pytest.raises(ValueError, match="no solve function"):
        ProgrammerCore.parse({"code": "# Remember to solve the problem.\nanswer = 42\n"})


def test_final_failure_keeps_bounded_attempt_diagnostics() -> None:
    response = "explanation " + ("x" * 2500)
    core = ProgrammerCore(
        generator=lambda _payload: {"code": response},
        executor=lambda _payload: "unused",
        max_attempts=3,
    )

    result = core.run({"problem": "example"})

    assert result["attempts"] == 3
    assert result["error"].startswith("Code format error:")
    assert len(result["attempt_diagnostics"]) == 3
    assert all(item["stage"] == "parse" for item in result["attempt_diagnostics"])
    assert all(len(item["raw_response_preview"]) <= 2000 for item in result["attempt_diagnostics"])
