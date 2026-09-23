from __future__ import annotations

import subprocess
import sys


def test_token_tracker_import_does_not_require_tiktoken_at_collection_time() -> None:
    script = r'''
import builtins

real_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == "tiktoken":
        raise ModuleNotFoundError("simulated missing tiktoken")
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
import masfactory.adapters.token_usage_tracker
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_token_counter_reports_installation_requirement_when_tiktoken_is_missing() -> None:
    script = r'''
import builtins

real_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == "tiktoken":
        raise ModuleNotFoundError("simulated missing tiktoken")
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
from masfactory.adapters.token_usage_tracker import DefaultTokenCounter
try:
    DefaultTokenCounter("offline")
except RuntimeError as exc:
    assert "tiktoken" in str(exc)
else:
    raise AssertionError("missing tiktoken was not reported")
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
