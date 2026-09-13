"""Async bridge for invoking MaAS workflows from MASFactory synchronous nodes."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


class AsyncRunnerContextError(RuntimeError):
    """Raised when the MaAS sync bridge is called from an async context."""


def run_async_once(coro: Coroutine[Any, Any, T]) -> T:
    """Run one coroutine from a synchronous MASFactory workflow entrypoint.

    MASFactory nodes execute synchronously. MaAS architecture workflows are async. This
    bridge allows a sync node to run one complete MaAS async call, while rejecting nested
    calls from an already running event loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    coro.close()
    raise AsyncRunnerContextError(
        "MaAS MASFactory workflow must be invoked from a synchronous context."
    )
