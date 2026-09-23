"""Compare MaAS reproduction and legacy MaAS GenerateCoT prompts.

This is an intentionally narrow diagnostic harness.  It runs one MATH test
problem through the reproduction ``GenerateCoTGraph`` twice:

1. with the prompt currently loaded by MASFactory reproduction; and
2. with the long few-shot ``GENERATE_COT_PROMPT`` read from the original MaAS
   source tree.

The legacy template is rendered with literal replacement instead of
``str.format``.  That preserves the LaTeX braces in the examples and avoids
turning the known legacy ``KeyError`` into a false cost comparison.

The script does not modify either project.  It writes the rendered prompts,
model responses, and a JSON report below ``assets/output``.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping
import warnings


APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parents[1]
# ``MaAS-main`` and ``masfactory`` are sibling directories under the user's
# workspace root, while ``PROJECT_ROOT`` points to ``masfactory/MASFactory-main``.
LEGACY_ROOT = PROJECT_ROOT.parent.parent / "MaAS-main"
DEFAULT_CONFIG_ROOT = APP_ROOT / "assets" / "config"
DEFAULT_DATA_ROOT = APP_ROOT / "assets" / "data"
DEFAULT_LEGACY_OP_PROMPT = (
    LEGACY_ROOT
    / "maas"
    / "ext"
    / "maas"
    / "scripts"
    / "optimized"
    / "MATH"
    / "train"
    / "template"
    / "op_prompt.py"
)
DEFAULT_LEGACY_INSTRUCTION_PROMPT = (
    LEGACY_ROOT
    / "maas"
    / "ext"
    / "maas"
    / "scripts"
    / "optimized"
    / "MATH"
    / "test"
    / "template"
    / "prompt.py"
)


def _redact(value: Any) -> str:
    """Redact credentials that could appear in provider exceptions."""

    text = str(value)
    for env_name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "BASE_URL"):
        secret = os.getenv(env_name)
        if secret:
            text = text.replace(secret, "<REDACTED>")
    return text


def _load_env() -> None:
    env_path = APP_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path, override=False)


def _read_string_assignment(path: Path, name: str) -> str:
    """Read one string assignment without importing the legacy MaAS package."""

    source = path.read_text(encoding="utf-8")
    # The legacy source contains LaTeX escapes such as ``\ge`` and ``\frac``.
    # They are intentionally parsed with Python's normal string-literal rules,
    # but their SyntaxWarning is irrelevant to this diagnostic output.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return value.value
            raise TypeError(f"{path}:{name} is not a string literal")
    raise KeyError(f"{name!r} was not found in {path}")


def _render_legacy_prompt(op_prompt_path: Path, instruction_prompt_path: Path, problem: str) -> str:
    """Render the original long prompt without interpreting LaTeX braces."""

    template = _read_string_assignment(op_prompt_path, "GENERATE_COT_PROMPT")
    instruction = _read_string_assignment(instruction_prompt_path, "GENERATE_SOLUTION_PROMPT")
    rendered = template.replace("{instruction}", instruction).replace("{input}", problem)
    required_markers = ("Demonstration Examples", "Solution Protocol", "Step-by-Step Analysis")
    missing = [marker for marker in required_markers if marker not in rendered]
    if missing:
        raise ValueError(f"legacy GenerateCoT prompt is missing markers: {missing}")
    return rendered


def _load_problem(data_root: Path, problem_index: int) -> Mapping[str, Any]:
    from applications.maas_reproduction.maas_reproduction.adapters.dataset_loader import DatasetLoader

    samples = DatasetLoader(data_root).load("MATH", "test", indices=[problem_index])
    sample = samples[0]
    return {
        "problem_index": sample.problem_index,
        "problem": sample.problem,
        "expected_answer": sample.expected_answer,
    }


def _tracker_counts(model: Any) -> tuple[int, int]:
    tracker = getattr(model, "token_tracker", None)
    return (
        int(getattr(tracker, "total_input_usage", 0) or 0),
        int(getattr(tracker, "total_output_usage", 0) or 0),
    )


def _response_usage(response: Any) -> tuple[int | None, int | None]:
    usage = response.get("usage") if isinstance(response, Mapping) else None
    if not isinstance(usage, Mapping):
        return None, None
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    return (
        int(prompt) if prompt is not None else None,
        int(completion) if completion is not None else None,
    )


def _response_content(response: Any) -> str:
    if isinstance(response, Mapping):
        return str(response.get("content", ""))
    return str(getattr(response, "content", ""))


def _message_stats(messages: list[dict[str, str]]) -> dict[str, Any]:
    contents = [str(message.get("content", "")) for message in messages]
    return {
        "message_count": len(messages),
        "total_chars": sum(len(content) for content in contents),
        "system_chars": sum(
            len(str(message.get("content", "")))
            for message in messages
            if message.get("role") == "system"
        ),
        "user_chars": sum(
            len(str(message.get("content", "")))
            for message in messages
            if message.get("role") == "user"
        ),
        "sha256": hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def _messages_for_case(case_name: str, prompt: str, problem: str) -> list[dict[str, str]]:
    if case_name == "current":
        # This is the exact message layout used by the reproduction runtime's
        # GenerateCoT adapter: current prompt as system, problem as user.
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": problem},
        ]
    if case_name == "legacy":
        # The original ActionNode.aask path uses its default system message
        # and sends the fully rendered GenerateCoT prompt as user content.
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    raise ValueError(f"unknown case: {case_name}")


def _run_generate_cot_case(
    *,
    case_name: str,
    prompt: str,
    problem: str,
    model: Any,
    usage_manager: Any,
    input_rate: float,
    output_rate: float,
) -> dict[str, Any]:
    from applications.maas_reproduction.components.operators.native_agent_operator_graph import GenerateCoTGraph
    from applications.maas_reproduction.maas_reproduction.schemas import OperatorInvocation

    messages = _messages_for_case(case_name, prompt, problem)
    message_stats = _message_stats(messages)
    before_input, before_output = _tracker_counts(model)
    before_cost = float(getattr(usage_manager, "total_cost", 0.0))
    call_details: dict[str, Any] = {}
    started = time.perf_counter()

    def adapter(_payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = model.invoke(messages, tools=None)
        except Exception as exc:
            call_details["error_type"] = type(exc).__name__
            call_details["error"] = _redact(exc)
            raise

        explicit_input, explicit_output = _response_usage(response)
        after_input, after_output = _tracker_counts(model)
        prompt_tokens = (
            explicit_input
            if explicit_input is not None
            else max(after_input - before_input, 0)
        )
        completion_tokens = (
            explicit_output
            if explicit_output is not None
            else max(after_output - before_output, 0)
        )
        prompt_tokens = int(prompt_tokens)
        completion_tokens = int(completion_tokens)
        usage_manager.record_usage(prompt_tokens, completion_tokens)
        content = _response_content(response)
        call_details.update(
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "content_chars": len(content),
            }
        )
        return {
            "solution": content,
            "metadata": {"usage": dict(call_details)},
        }

    graph = GenerateCoTGraph(operator=adapter, instructions=prompt)
    graph.build()
    invocation = OperatorInvocation(
        sequence_index=0,
        layer_index=0,
        position=0,
        operator_name="GenerateCoT",
        problem=problem,
    )
    operator_result = graph._forward({"operator_invocation": invocation})["operator_result"]

    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    after_cost = float(getattr(usage_manager, "total_cost", 0.0))
    cost_delta = after_cost - before_cost
    usage = dict(call_details)
    result = {
        "case": case_name,
        "status": operator_result.status,
        "solution": operator_result.solution,
        "solution_chars": len(operator_result.solution or ""),
        "prompt": {
            "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "chars": len(prompt),
            "lines": len(prompt.splitlines()),
        },
        "messages": message_stats,
        "usage": usage,
        "cost": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_cost": round(cost_delta, 10),
            "input_rate_per_1k": input_rate,
            "output_rate_per_1k": output_rate,
            "reliable": bool(getattr(usage_manager, "cost_reliable", False)),
        },
        "elapsed_ms": elapsed_ms,
        "call": call_details,
    }
    if operator_result.metadata:
        result["operator_metadata"] = dict(operator_result.metadata)
    return result


def _compare(current: Mapping[str, Any], legacy: Mapping[str, Any]) -> dict[str, Any]:
    def number(section: Mapping[str, Any], key: str) -> float | None:
        value = section.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    current_prompt_tokens = number(current.get("usage", {}), "prompt_tokens")
    legacy_prompt_tokens = number(legacy.get("usage", {}), "prompt_tokens")
    current_completion_tokens = number(current.get("usage", {}), "completion_tokens")
    legacy_completion_tokens = number(legacy.get("usage", {}), "completion_tokens")
    current_cost = number(current.get("cost", {}), "total_cost")
    legacy_cost = number(legacy.get("cost", {}), "total_cost")

    def diff(left: float | None, right: float | None) -> float | None:
        return round(right - left, 10) if left is not None and right is not None else None

    return {
        "legacy_minus_current": {
            "prompt_chars": diff(
                number(current.get("messages", {}), "total_chars"),
                number(legacy.get("messages", {}), "total_chars"),
            ),
            "prompt_tokens": diff(current_prompt_tokens, legacy_prompt_tokens),
            "completion_tokens": diff(current_completion_tokens, legacy_completion_tokens),
            "total_cost": diff(current_cost, legacy_cost),
            "solution_chars": diff(
                number(current, "solution_chars"), number(legacy, "solution_chars")
            ),
            "elapsed_ms": diff(number(current, "elapsed_ms"), number(legacy, "elapsed_ms")),
        },
        "cost_ratio_legacy_over_current": (
            round(legacy_cost / current_cost, 6)
            if current_cost is not None and legacy_cost is not None and current_cost > 0
            else None
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one MATH problem through reproduction GenerateCoT with two prompts"
    )
    parser.add_argument("--problem-index", type=int, default=3, help="MATH test problem index; default: 3")
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--legacy-op-prompt", type=Path, default=DEFAULT_LEGACY_OP_PROMPT)
    parser.add_argument("--legacy-instruction-prompt", type=Path, default=DEFAULT_LEGACY_INSTRUCTION_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fake-model", action="store_true", help="Run offline with the deterministic fake model")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.problem_index < 0:
        raise ValueError("--problem-index must be non-negative")

    os.chdir(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    _load_env()

    from applications.maas_reproduction.maas_reproduction.models.model_factory import create_shared_model
    from applications.maas_reproduction.maas_reproduction.runtime.bootstrap import _UsageManager
    from applications.maas_reproduction.maas_reproduction.runtime.prompt_loader import PromptLoader
    from applications.maas_reproduction.maas_reproduction.runtime.settings import load_settings

    sample = _load_problem(args.data_root, args.problem_index)
    problem = str(sample["problem"])
    settings = load_settings(
        args.config_root,
        mode_override="test",
        dataset_override="MATH",
        split_override="test",
        sample_override=1,
    )
    model = create_shared_model(settings.model, fake=args.fake_model)
    usage_manager = _UsageManager(
        input_rate=settings.model.input_cost_per_1k_tokens,
        output_rate=settings.model.output_cost_per_1k_tokens,
    )

    prompt_root = APP_ROOT / "assets" / "prompts"
    current_prompt_path = prompt_root / "math" / "GenerateCoT.txt"
    if not current_prompt_path.exists():
        current_prompt_path = prompt_root / "shared" / "GenerateCoT.txt"
    current_prompt = PromptLoader(prompt_root).load("GenerateCoT", "MATH")
    legacy_prompt = _render_legacy_prompt(
        args.legacy_op_prompt,
        args.legacy_instruction_prompt,
        problem,
    )

    output_dir = args.output_dir or (
        APP_ROOT
        / "assets"
        / "output"
        / f"compare_generate_cot_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    current = _run_generate_cot_case(
        case_name="current",
        prompt=current_prompt,
        problem=problem,
        model=model,
        usage_manager=usage_manager,
        input_rate=float(settings.model.input_cost_per_1k_tokens or 0.0),
        output_rate=float(settings.model.output_cost_per_1k_tokens or 0.0),
    )
    legacy = _run_generate_cot_case(
        case_name="legacy",
        prompt=legacy_prompt,
        problem=problem,
        model=model,
        usage_manager=usage_manager,
        input_rate=float(settings.model.input_cost_per_1k_tokens or 0.0),
        output_rate=float(settings.model.output_cost_per_1k_tokens or 0.0),
    )

    (output_dir / "current_prompt.txt").write_text(current_prompt, encoding="utf-8")
    (output_dir / "legacy_prompt.txt").write_text(legacy_prompt, encoding="utf-8")
    (output_dir / "current_response.txt").write_text(current.get("solution") or "", encoding="utf-8")
    (output_dir / "legacy_response.txt").write_text(legacy.get("solution") or "", encoding="utf-8")

    report = {
        "problem": {
            "problem_index": sample["problem_index"],
            "expected_answer": sample.get("expected_answer"),
            "problem_chars": len(problem),
            "problem_sha256": hashlib.sha256(problem.encode("utf-8")).hexdigest(),
        },
        "model": {
            "provider": settings.model.provider,
            "model_name": settings.model.model_name,
            "temperature": settings.model.temperature,
            "max_tokens": settings.model.max_tokens,
            "fake_model": bool(args.fake_model),
        },
        "prompt_sources": {
            "current": str(current_prompt_path.resolve()),
            "legacy_operator_template": str(args.legacy_op_prompt.resolve()),
            "legacy_instruction_template": str(args.legacy_instruction_prompt.resolve()),
            "legacy_rendering": "literal replacement of {instruction} and {input}; no str.format",
        },
        "current": current,
        "legacy": legacy,
        "comparison": _compare(current, legacy),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"[GENCOT-COMPARE] output_dir={output_dir}")
    for result in (current, legacy):
        cost = result["cost"]
        usage = result["usage"]
        print(
            f"[GENCOT-COMPARE] case={result['case']} status={result['status']} "
            f"prompt_chars={result['messages']['total_chars']} "
            f"prompt_tokens={usage.get('prompt_tokens')} "
            f"completion_tokens={usage.get('completion_tokens')} "
            f"cost=${float(cost['total_cost'] or 0.0):.8f} "
            f"elapsed_ms={result['elapsed_ms']}"
        )
    print(f"[GENCOT-COMPARE] report={report_path}")
    print(
        "[GENCOT-COMPARE] legacy_minus_current="
        + json.dumps(report["comparison"]["legacy_minus_current"], ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
