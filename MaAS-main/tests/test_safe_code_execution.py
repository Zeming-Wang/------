import asyncio
import multiprocessing as mp
import os
import subprocess
import sys
import time

import pytest

from maas.ext.maas.scripts.safe_code_execution import (
    ProcessExecutionTimeout,
    run_in_disposable_process,
    run_in_disposable_process_sync,
)
from maas.actions import run_code as run_code_module


def _return_value(value):
    return value


def _busy_forever():
    while True:
        pass


def _run(coroutine):
    return asyncio.run(coroutine)


def _active_child_pids():
    return {process.pid for process in mp.active_children()}


def test_normal_result_is_unchanged():
    result = _run(run_in_disposable_process(_return_value, 42, timeout=1.0))
    assert result == 42


def test_timeout_kills_and_reaps_busy_worker():
    baseline = _active_child_pids()
    started = time.monotonic()

    with pytest.raises(ProcessExecutionTimeout):
        _run(run_in_disposable_process(_busy_forever, timeout=0.2))

    assert time.monotonic() - started < 2.0
    assert _active_child_pids() == baseline


def test_repeated_timeouts_do_not_accumulate_workers():
    baseline = _active_child_pids()

    for _ in range(5):
        with pytest.raises(ProcessExecutionTimeout):
            _run(run_in_disposable_process(_busy_forever, timeout=0.1))

    assert _active_child_pids() == baseline


def test_synchronous_timeout_kills_and_reaps_busy_worker():
    baseline = _active_child_pids()

    with pytest.raises(ProcessExecutionTimeout):
        run_in_disposable_process_sync(_busy_forever, timeout=0.2)

    assert _active_child_pids() == baseline


def test_run_text_timeout_does_not_leave_worker(monkeypatch):
    baseline = _active_child_pids()
    monkeypatch.setattr(run_code_module, "TEXT_CODE_TIMEOUT_SECONDS", 0.2)

    result = _run(
        run_code_module.RunCode.run_text("while True:\n    pass")
    )

    assert result == ("", "Code execution timed out")
    assert _active_child_pids() == baseline


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
def test_script_cleanup_kills_descendant_process_group():
    child_script = (
        "import subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', 'while True: pass']); "
        "time.sleep(60)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", child_script],
        start_new_session=True,
    )
    process_group = process.pid
    time.sleep(0.2)

    run_code_module._stop_subprocess_group(process)

    assert process.poll() is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(process_group, 0)
