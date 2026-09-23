"""HumanEval scorer with source-compatible helpers and bounded execution."""
from __future__ import annotations

import ast
import base64
import builtins
import hashlib
import math
import re
import typing
from typing import Any

from ..adapters.code_executor import CodeExecutor
from .base import BaseBenchmark, ScoreResult


_EXECUTION_TIMEOUT_SECONDS = 15.0
_PASS_MARKER = "__MAAS_HUMANEVAL_PASS__"
_FAIL_MARKER = "__MAAS_HUMANEVAL_FAIL__"
_ALLOWED_IMPORTS = {
    "bisect", "collections", "functools", "heapq", "itertools", "math", "re",
    "statistics", "string", "typing",
}
_BLOCKED_NAMES = {
    "__builtins__", "__code__", "__globals__", "__loader__", "__subclasses__",
    "breakpoint", "compile", "eval", "exec", "globals", "input", "locals",
    "open", "vars",
}


def _restricted_import(name: str, globals: dict | None = None,
                       locals: dict | None = None, fromlist: tuple = (),
                       level: int = 0):
    root = name.split(".", 1)[0]
    if level or root not in _ALLOWED_IMPORTS:
        raise ImportError(f"import of {name!r} is not allowed")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _safe_builtins() -> dict[str, object]:
    names = (
        "ArithmeticError", "AssertionError", "AttributeError", "Exception",
        "IndexError", "KeyError", "NotImplemented", "RuntimeError",
        "StopIteration", "TypeError", "ValueError", "ZeroDivisionError", "abs",
        "all", "any", "bool", "chr", "dict", "divmod", "enumerate", "filter", "float",
        "frozenset", "hasattr", "hash", "int", "isinstance", "issubclass", "len",
        "list", "map", "max", "min", "next", "ord", "pow", "range", "repr", "reversed",
        "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip",
    )
    safe = {name: getattr(builtins, name) for name in names}
    safe.update({"__build_class__": builtins.__build_class__, "__import__": _restricted_import})
    return safe


class _SafetyVisitor(ast.NodeVisitor):
    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _BLOCKED_NAMES:
            raise ValueError(f"disallowed name: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in _BLOCKED_NAMES:
            raise ValueError(f"disallowed attribute: {node.attr}")
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".", 1)[0] not in _ALLOWED_IMPORTS:
                raise ValueError(f"import of {alias.name!r} is not allowed")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level or not node.module or node.module.split(".", 1)[0] not in _ALLOWED_IMPORTS:
            raise ValueError(f"import from {node.module!r} is not allowed")
        self.generic_visit(node)


def _parse_code(code: str) -> tuple[str, ast.Module]:
    text = str(code).replace("```python", "").replace("```py", "").replace("```", "").strip()
    try:
        return text, ast.parse(text)
    except SyntaxError:
        lines = text.splitlines()
        starts = [
            index for index, line in enumerate(lines)
            if line.lstrip().startswith(("def ", "async def ", "class ", "import ", "from "))
        ]
        for start in starts:
            for end in range(len(lines), start, -1):
                candidate = "\n".join(lines[start:end]).strip()
                try:
                    return candidate, ast.parse(candidate)
                except SyntaxError:
                    continue
        raise


def _introduced_names(node: ast.AST) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return {
            name.id for target in targets for name in ast.walk(target)
            if isinstance(name, ast.Name)
        }
    return set()


def _dependencies(node: ast.AST) -> set[str]:
    return {
        child.id for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _sanitize(code: str, entrypoint: str) -> str:
    """Keep imports and definitions reachable from the HumanEval entrypoint."""
    source, tree = _parse_code(code)
    _SafetyVisitor().visit(tree)
    imports: list[ast.AST] = []
    definitions: list[tuple[set[str], ast.AST]] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                               ast.Assign, ast.AnnAssign)):
            definitions.append((_introduced_names(node), node))

    by_name = {name: node for names, node in definitions for name in names}
    reachable = {entrypoint}
    pending = [entrypoint]
    while pending:
        name = pending.pop()
        node = by_name.get(name)
        if node is None:
            continue
        for dependency in _dependencies(node):
            if dependency in by_name and dependency not in reachable:
                reachable.add(dependency)
                pending.append(dependency)

    selected = [node for names, node in definitions if names & reachable]
    if entrypoint not in by_name:
        selected = [node for _, node in definitions]
    fragments = [ast.get_source_segment(source, node) for node in (*imports, *selected)]
    return "\n\n".join(fragment for fragment in fragments if fragment)


def _special_helpers(entrypoint: str) -> str:
    if entrypoint == "decode_cyclic":
        return """
def encode_cyclic(s: str):
    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]
    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]
    return "".join(groups)
"""
    if entrypoint == "decode_shift":
        return """
def encode_shift(s: str):
    return "".join([chr(((ord(ch) + 5 - ord("a")) % 26) + ord("a")) for ch in s])
"""
    if entrypoint == "find_zero":
        return """
def poly(xs: list, x: float):
    return sum(coeff * (x ** i) for i, coeff in enumerate(xs))
"""
    return ""


def _runner_source(solution: str, test: str, entrypoint: str) -> str:
    encoded_solution = base64.b64encode(solution.encode("utf-8")).decode("ascii")
    encoded_test = base64.b64encode(test.encode("utf-8")).decode("ascii")
    encoded_entrypoint = base64.b64encode(entrypoint.encode("utf-8")).decode("ascii")
    allowed = sorted(_ALLOWED_IMPORTS)
    safe_names = sorted(_safe_builtins())
    return f"""
import base64
import hashlib
import math
import re
import typing
import builtins

ALLOWED = {allowed!r}
def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split('.', 1)[0] not in ALLOWED:
        raise ImportError('import is not allowed')
    return builtins.__import__(name, globals, locals, fromlist, level)

safe_builtins = {{}}
for name in {safe_names!r}:
    if hasattr(builtins, name):
        safe_builtins[name] = getattr(builtins, name)
safe_builtins['__build_class__'] = builtins.__build_class__
safe_builtins['__import__'] = restricted_import
namespace = {{
    '__builtins__': safe_builtins,
    '__name__': '__main__',
    'math': math, 'hashlib': hashlib, 're': re,
    'List': typing.List, 'Dict': typing.Dict, 'Tuple': typing.Tuple,
    'Optional': typing.Optional, 'Any': typing.Any,
}}
solution = base64.b64decode({encoded_solution!r}).decode('utf-8')
test = base64.b64decode({encoded_test!r}).decode('utf-8')
entrypoint = base64.b64decode({encoded_entrypoint!r}).decode('utf-8')
exec(solution, namespace)
if entrypoint not in namespace or not callable(namespace[entrypoint]):
    raise ValueError('entrypoint is not defined')
exec(test, namespace)
check = namespace.get('check')
if not callable(check):
    raise ValueError('test must define check')
result = check(namespace[entrypoint])
print({_PASS_MARKER!r} if result is None else {_FAIL_MARKER!r})
"""


class HumanEvalBenchmark(BaseBenchmark):
    """Execute HumanEval candidates in an isolated, bounded subprocess."""

    PASS = "PASS"
    FAIL = "FAIL"

    def __init__(self, *args: Any, timeout_seconds: float = _EXECUTION_TIMEOUT_SECONDS,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = float(timeout_seconds)
        self._executor = CodeExecutor(max_output_chars=10000)

    def check_solution(self, solution: str | None, test: str, entry_point: str) -> tuple[str, str]:
        try:
            sanitized = _sanitize(str(solution or ""), entry_point)
            code = _special_helpers(entry_point) + "\n" + sanitized
            result = self._executor.execute(
                _runner_source(code, str(test), entry_point),
                timeout_seconds=self.timeout_seconds,
            )
            if result.timed_out:
                return self.FAIL, "Execution timed out."
            if not result.success:
                detail = result.stderr.strip() or result.error or "execution failed"
                return self.FAIL, f"Error: {detail}"
            if _PASS_MARKER in result.stdout:
                return self.PASS, "The solution passed all test cases."
            return self.FAIL, "The solution failed one or more test cases."
        except Exception as exc:
            return self.FAIL, f"Error: {type(exc).__name__}: {exc}"

    def evaluate(self, prediction: str | None, expected_answer: Any, *,
                 context: dict[str, Any] | None = None) -> ScoreResult:
        # The original benchmark treats an empty/invalid candidate as a failed
        # sample with score 0. Keep it trainable instead of making it an
        # evaluator-infrastructure failure.
        if prediction is None or not str(prediction).strip():
            return ScoreResult(0.0, True)
        return super().evaluate(prediction, expected_answer, context=context)

    def score(self, prediction: str | None, expected_answer: Any, *,
              context: dict[str, Any] | None = None) -> float:
        if context is None or not context.get("entry_point") or not context.get("test"):
            raise ValueError("HumanEval requires entry_point and test context")
        status, _ = self.check_solution(
            prediction, str(context["test"]), str(context["entry_point"])
        )
        return 1.0 if status == self.PASS else 0.0


HumanEvalScorer = HumanEvalBenchmark
__all__ = ["HumanEvalBenchmark", "HumanEvalScorer"]
