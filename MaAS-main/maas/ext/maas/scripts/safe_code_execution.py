"""Run generated callables in disposable processes with hard cleanup.

This module contains no training or evaluation logic. It only owns the
lifecycle of a worker process so timed-out generated code cannot keep using
CPU after its caller has moved on.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import signal
import time
import traceback
from multiprocessing.connection import Connection
from typing import Any, Callable


class ProcessExecutionTimeout(TimeoutError):
    """Raised when a worker exceeds its wall-clock timeout."""


class ProcessExecutionError(RuntimeError):
    """Raised when a worker exits without returning a normal result."""


def _worker_entrypoint(
    target: Callable[..., Any],
    args: tuple[Any, ...],
    send_connection: Connection,
) -> None:
    # Give the worker its own process group so cleanup also removes any
    # descendants created by generated code.
    if os.name == "posix":
        try:
            os.setsid()
        except OSError:
            pass

    try:
        result = target(*args)
        send_connection.send(("result", result))
    except BaseException as error:
        payload = (
            f"{type(error).__name__}: {error}\n"
            f"{''.join(traceback.format_exception(type(error), error, error.__traceback__))}"
        )
        try:
            send_connection.send(("error", payload))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        send_connection.close()


def _process_group_exists(process_id: int) -> bool:
    if os.name != "posix":
        return False
    try:
        os.killpg(process_id, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _signal_worker(process: mp.Process, sig: signal.Signals) -> None:
    if process.pid is None:
        return

    if os.name == "posix":
        try:
            os.killpg(process.pid, sig)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass

    if not process.is_alive():
        return
    try:
        if sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    except ProcessLookupError:
        pass


def _reap_worker(process: mp.Process) -> None:
    """Stop and join one exact worker without touching unrelated processes."""
    if process.pid is None:
        return

    process_id = process.pid
    process.join(timeout=0.1)
    if process.is_alive() or _process_group_exists(process_id):
        _signal_worker(process, signal.SIGTERM)
        process.join(timeout=1.0)

        deadline = time.monotonic() + 1.0
        while _process_group_exists(process_id) and time.monotonic() < deadline:
            time.sleep(0.02)

    if process.is_alive() or _process_group_exists(process_id):
        _signal_worker(process, signal.SIGKILL)
        process.join(timeout=1.0)
    if not process.is_alive():
        process.close()


async def run_in_disposable_process(
    target: Callable[..., Any],
    *args: Any,
    timeout: float,
) -> Any:
    """Run ``target`` in a child and guarantee cleanup on every exit path."""
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")

    start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    context = mp.get_context(start_method)
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker_entrypoint,
        args=(target, args, send_connection),
        daemon=True,
    )
    process.start()
    send_connection.close()

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    try:
        while True:
            if receive_connection.poll():
                try:
                    message_type, payload = receive_connection.recv()
                except EOFError as error:
                    raise ProcessExecutionError(
                        "Worker exited without returning a result"
                    ) from error

                if message_type == "result":
                    return payload
                raise ProcessExecutionError(payload)

            if not process.is_alive():
                if receive_connection.poll(0.05):
                    continue
                raise ProcessExecutionError(
                    f"Worker exited with code {process.exitcode} without returning a result"
                )

            remaining = deadline - loop.time()
            if remaining <= 0:
                raise ProcessExecutionTimeout(
                    f"Worker exceeded the {timeout:g}-second timeout"
                )

            await asyncio.sleep(min(0.05, remaining))
    finally:
        receive_connection.close()
        _reap_worker(process)


def run_in_disposable_process_sync(
    target: Callable[..., Any],
    *args: Any,
    timeout: float,
) -> Any:
    """Synchronous counterpart for benchmark and test-runner call sites."""
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")

    start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    context = mp.get_context(start_method)
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker_entrypoint,
        args=(target, args, send_connection),
        daemon=True,
    )
    process.start()
    send_connection.close()

    deadline = time.monotonic() + timeout

    try:
        while True:
            if receive_connection.poll():
                try:
                    message_type, payload = receive_connection.recv()
                except EOFError as error:
                    raise ProcessExecutionError(
                        "Worker exited without returning a result"
                    ) from error

                if message_type == "result":
                    return payload
                raise ProcessExecutionError(payload)

            if not process.is_alive():
                if receive_connection.poll(0.05):
                    continue
                raise ProcessExecutionError(
                    f"Worker exited with code {process.exitcode} without returning a result"
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProcessExecutionTimeout(
                    f"Worker exceeded the {timeout:g}-second timeout"
                )

            time.sleep(min(0.05, remaining))
    finally:
        receive_connection.close()
        _reap_worker(process)
