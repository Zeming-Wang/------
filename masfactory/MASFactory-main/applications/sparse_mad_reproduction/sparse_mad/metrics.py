from __future__ import annotations

import re
from collections.abc import Iterable
from decimal import Decimal, InvalidOperation


_NORMALIZE_RE = re.compile(r"[^a-zA-Z0-9.\-]+")
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_TIME_RE = re.compile(r"\d{1,2}:\d{2}")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def estimate_messages_tokens(messages: list[dict[str, str]], *, model_name: str = "gpt-4o-mini") -> int:
    text = "\n".join(f"{message.get('role', '')}: {message.get('content', '')}" for message in messages)
    return max(1, len(_TOKEN_RE.findall(text)))


def normalize_answer(answer: str) -> str:
    text = str(answer).lower().strip().strip("`").strip()
    text = text.rstrip(".")
    if _TIME_RE.fullmatch(text):
        return text

    number_matches = _NUMBER_RE.findall(text.replace("$", ""))
    if number_matches:
        return _normalize_number(number_matches[-1])

    return _NORMALIZE_RE.sub("", text).strip().rstrip(".")


def _normalize_number(value: str) -> str:
    value = value.replace(",", "")
    try:
        number = Decimal(value)
    except InvalidOperation:
        return value
    if number == number.to_integral_value():
        return str(int(number))
    return format(number.normalize(), "f").rstrip("0").rstrip(".")


def is_correct(predicted: str, expected: str) -> bool:
    return normalize_answer(predicted) == normalize_answer(expected)


def sum_token_cost(runs: Iterable[dict]) -> int:
    return sum(int(run.get("token_cost") or 0) for run in runs)
