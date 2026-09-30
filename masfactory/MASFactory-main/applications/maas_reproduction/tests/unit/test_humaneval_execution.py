from __future__ import annotations

import time

from applications.maas_reproduction.maas_reproduction.benchmarks.base import ScoreResult
from applications.maas_reproduction.maas_reproduction.benchmarks.humaneval import HumanEvalBenchmark


def context(entry_point: str = "add", test: str = "") -> dict[str, str]:
    return {"entry_point": entry_point, "test": test}


def test_humaneval_accepts_source_style_special_helper() -> None:
    scorer = HumanEvalBenchmark()
    result = scorer.evaluate(
        "def decode_shift(s):\n    return encode_shift(s)",
        None,
        context=context(
            "decode_shift",
            "def check(fn):\n    assert fn('abc') == 'fgh'",
        ),
    )

    assert isinstance(result, ScoreResult)
    assert result.reliable is True
    assert result.score == 1.0


def test_humaneval_execution_error_is_a_reliable_zero_score() -> None:
    scorer = HumanEvalBenchmark()
    result = scorer.evaluate(
        "def add(a, b):\n    raise RuntimeError('bad candidate')",
        None,
        context=context(
            test="def check(fn):\n    fn(1, 2)",
        ),
    )

    assert result == ScoreResult(0.0, True)


def test_humaneval_timeout_is_bounded_and_does_not_escape() -> None:
    scorer = HumanEvalBenchmark(timeout_seconds=0.2)
    started = time.monotonic()
    result = scorer.evaluate(
        "def add(a, b):\n    return a + b",
        None,
        context=context(test="def check(fn):\n    while True:\n        pass"),
    )

    assert time.monotonic() - started < 5.0
    assert result == ScoreResult(0.0, True)


def test_humaneval_preserves_source_standard_import_permissions() -> None:
    scorer = HumanEvalBenchmark()
    result = scorer.evaluate(
        "import os\ndef add(a, b):\n    return os.getcwd()",
        None,
        context=context(test="def check(fn):\n    fn(1, 2)"),
    )

    assert result == ScoreResult(1.0, True)


def test_humaneval_accepts_decimal_import_like_source() -> None:
    scorer = HumanEvalBenchmark()
    result = scorer.evaluate(
        "from decimal import Decimal\ndef add(a, b):\n    return int(Decimal(a) + Decimal(b))",
        None,
        context=context(test="def check(fn):\n    assert fn(1, 2) == 3"),
    )

    assert result == ScoreResult(1.0, True)
