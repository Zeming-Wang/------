from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import random
import sys
import time
from typing import TypeVar


T = TypeVar("T")

RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})


def call_with_openai_retry(
    operation: Callable[[], T],
    *,
    operation_name: str = "llm_request",
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_unknown_status: bool = False,
) -> T:
    """Run an OpenAI-compatible request with visible bounded retry/backoff."""

    attempts = max(1, int(max_attempts))
    last_exc: Exception | None = None

    for attempt in range(attempts):
        try:
            result = operation()
            if attempt > 0:
                _print_event(
                    f"{operation_name} recovered on attempt {attempt + 1}/{attempts}"
                )
            return result
        except Exception as exc:
            last_exc = exc
            retryable = _is_retryable_error(exc, retry_unknown_status=retry_unknown_status)
            attempt_number = attempt + 1
            status_code = _get_status_code(exc)
            error_summary = _format_error_summary(exc)

            if not retryable:
                _print_event(
                    f"{operation_name} failed on attempt {attempt_number}/{attempts}; "
                    f"not retryable; status={status_code}; error={error_summary}"
                )
                raise

            if attempt_number == attempts:
                _print_event(
                    f"{operation_name} failed on attempt {attempt_number}/{attempts}; "
                    f"retry attempts exhausted; status={status_code}; error={error_summary}"
                )
                raise

            delay = _get_retry_delay_seconds(
                exc,
                base_delay=base_delay,
                max_delay=max_delay,
                attempt=attempt,
            )
            _print_event(
                f"{operation_name} failed on attempt {attempt_number}/{attempts}; "
                f"retrying in {delay:.2f}s; status={status_code}; error={error_summary}"
            )
            time.sleep(delay)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("OpenAI-compatible request retry loop exited unexpectedly")


def _is_retryable_error(exc: Exception, *, retry_unknown_status: bool) -> bool:
    status_code = _get_status_code(exc)
    if status_code in RETRYABLE_STATUS_CODES or (
        status_code is not None and status_code >= 500
    ):
        return True
    if status_code is not None:
        return False

    message = str(exc).lower()
    network_markers = (
        "too many requests",
        "rate limit",
        "connection",
        "connect",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "network",
    )
    if any(marker in message for marker in network_markers):
        return True

    return retry_unknown_status


def _get_status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return status_code if isinstance(status_code, int) else None


def _get_retry_delay_seconds(
    exc: Exception,
    *,
    base_delay: float,
    max_delay: float,
    attempt: int,
) -> float:
    delay_cap = max(0.0, float(max_delay))
    retry_after = _get_retry_after_seconds(exc)
    if retry_after is not None:
        jittered_delay = retry_after + random.uniform(
            0.0,
            min(1.0, max(0.0, retry_after) * 0.1),
        )
        return min(delay_cap, jittered_delay)

    exponential_delay = max(0.0, float(base_delay)) * (2**attempt)
    jittered_delay = exponential_delay * random.uniform(0.8, 1.2)
    return min(delay_cap, jittered_delay)


def _get_retry_after_seconds(exc: Exception) -> float | None:
    headers = _get_headers(exc)
    if headers is None:
        return None

    value = _get_header(headers, "retry-after")
    if value is None:
        return None

    text = str(value).strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass

    try:
        retry_at = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None

    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def _get_headers(exc: Exception):
    response = getattr(exc, "response", None)
    for source in (response, exc):
        headers = getattr(source, "headers", None)
        if headers is not None:
            return headers
    return None


def _get_header(headers, name: str):
    if hasattr(headers, "get"):
        value = headers.get(name)
        if value is not None:
            return value
        value = headers.get(name.title())
        if value is not None:
            return value

    lower_name = name.lower()
    try:
        items = headers.items()
    except AttributeError:
        return None

    for key, value in items:
        if str(key).lower() == lower_name:
            return value
    return None


def _format_error_summary(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    if not message:
        message = repr(exc)
    if len(message) > 300:
        message = f"{message[:297]}..."
    return f"{type(exc).__name__}: {message}"


def _print_event(message: str) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[llm-retry] {timestamp} {message}", file=sys.stderr, flush=True)
