"""Trace one MATH test sample in the native MaAS reproduction.

The script instruments the runtime in-process, so production/reproduction
source files remain unchanged.  It records policy selection, the normalized
route, actual dispatch order, model invocations, low-level API attempts,
token usage, and cost.  It writes a JSON trace and a self-contained HTML
timeline below ``applications/maas_reproduction/assets/output``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import html
import json
import os
from pathlib import Path
import sys
import time
from functools import wraps
from typing import Any, Mapping


APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parents[1]
DEFAULT_CONFIG_ROOT = APP_ROOT / "assets" / "config"
DEFAULT_CHECKPOINT = (
    APP_ROOT
    / "assets"
    / "output"
    / "run_20260917_135036_948611"
    / "checkpoints"
    / "latest.pt"
)
INPUT_RATE_PER_1K = 0.00015
OUTPUT_RATE_PER_1K = 0.0006


def _redact(value: Any) -> str:
    text = str(value)
    lowered = text.lower()
    for marker in ("api_key=", "api-key=", "authorization=", "bearer "):
        index = lowered.find(marker)
        if index >= 0:
            end = index + len(marker)
            while end < len(text) and not text[end].isspace():
                end += 1
            text = text[: index + len(marker)] + "<REDACTED>" + text[end:]
            lowered = text.lower()
    return text[:500]


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


class TraceRecorder:
    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.events: list[dict[str, Any]] = []
        self.current_operator: str | None = None
        self.api_attempts = 0
        self.model_invocations = 0

    def event(self, kind: str, **fields: Any) -> None:
        record: dict[str, Any] = {
            "event_index": len(self.events),
            "elapsed_ms": round((time.perf_counter() - self.started) * 1000.0, 3),
            "kind": kind,
        }
        for key, value in fields.items():
            if key in {"error", "message"}:
                record[key] = _redact(value)
            else:
                record[key] = value
        self.events.append(record)


def _patch_execution_seams(trace: TraceRecorder) -> None:
    """Patch only function references used to construct this test graph."""

    from applications.maas_reproduction.components.architecture_exec_graph.components import (
        route_planner_node,
    )
    from applications.maas_reproduction.components.native_bootstrap_graph.components import (
        bootstrap_operator_nodes,
    )
    from applications.maas_reproduction.components.operator_dispatch_loop.components import (
        route_cursor_node,
        state_reducer_node,
    )

    original_plan = route_planner_node.RoutePlannerNode._plan

    @wraps(original_plan)
    def traced_plan(self, *args: Any, **kwargs: Any):
        result = original_plan(self, *args, **kwargs)
        route = result.get("route_plan") if isinstance(result, Mapping) else None
        items = []
        for item in _field(route, "items", ()) or ():
            items.append(
                {
                    "sequence_index": _field(item, "sequence_index"),
                    "layer_index": _field(item, "layer_index"),
                    "position": _field(item, "position"),
                    "operator_name": _field(item, "operator_name"),
                    "is_control_marker": bool(_field(item, "is_control_marker", False)),
                }
            )
        trace.event(
            "route_plan",
            route_length=len(items),
            items=items,
            planning_ok=bool(route),
        )
        return result

    route_planner_node.RoutePlannerNode._plan = traced_plan

    original_cursor = route_cursor_node.route_cursor_forward

    @wraps(original_cursor)
    def traced_cursor(message: dict, attributes: dict):
        result = original_cursor(message, attributes)
        invocation = result.get("operator_invocation", {})
        operator_name = _field(invocation, "operator_name", "unknown")
        trace.current_operator = str(operator_name)
        trace.event(
            "dispatch_enter",
            operator_name=str(operator_name),
            sequence_index=_field(invocation, "sequence_index"),
            layer_index=_field(invocation, "layer_index"),
            position=_field(invocation, "position"),
            current_solution_chars=len(str(_field(invocation, "current_solution", "") or "")),
            candidate_count=len(_field(invocation, "candidates", ()) or ()),
        )
        return result

    route_cursor_node.route_cursor_forward = traced_cursor

    original_reduce = state_reducer_node.reduce_operator_result

    @wraps(original_reduce)
    def traced_reduce(message: dict, attributes: dict):
        operator_result = message.get("operator_result") if isinstance(message, Mapping) else None
        try:
            result = original_reduce(message, attributes)
            trace.event(
                "dispatch_exit",
                operator_name=str(_field(operator_result, "operator_name", trace.current_operator)),
                status=str(_field(operator_result, "status", "unknown")),
                solution_chars=len(str(_field(operator_result, "solution", "") or "")),
                candidate_count=len(_field(operator_result, "candidates", ()) or ()),
                metadata=_public_metadata(_field(operator_result, "metadata", {})),
            )
            return result
        except Exception as exc:
            trace.event(
                "dispatch_error",
                operator_name=str(_field(operator_result, "operator_name", trace.current_operator)),
                error_type=type(exc).__name__,
                error=_redact(exc),
            )
            raise
        finally:
            trace.current_operator = None

    state_reducer_node.reduce_operator_result = traced_reduce

    def wrap_bootstrap(cls: Any, method_name: str, operator_name: str) -> None:
        original = getattr(cls, method_name)

        @wraps(original)
        def wrapped(self, *args: Any, **kwargs: Any):
            previous = trace.current_operator
            trace.current_operator = operator_name
            trace.event("bootstrap_enter", operator_name=operator_name)
            try:
                result = original(self, *args, **kwargs)
                trace.event("bootstrap_exit", operator_name=operator_name, status="completed")
                return result
            except Exception as exc:
                trace.event(
                    "bootstrap_exit",
                    operator_name=operator_name,
                    status="error",
                    error_type=type(exc).__name__,
                    error=_redact(exc),
                )
                raise
            finally:
                trace.current_operator = previous

        setattr(cls, method_name, wrapped)

    wrap_bootstrap(
        bootstrap_operator_nodes.ProgrammerBootstrapNode,
        "_forward_programmer",
        "Programmer",
    )
    wrap_bootstrap(
        bootstrap_operator_nodes.GenerateBootstrapNode,
        "_forward_generate",
        "Generate",
    )


def _instrument_controller(controller: Any, trace: TraceRecorder) -> None:
    original_forward = controller.forward

    for layer_index, layer in enumerate(getattr(controller, "layers", ())):
        original_layer_forward = layer.forward

        @wraps(original_layer_forward)
        def traced_layer_forward(*args: Any, _original=original_layer_forward, _index=layer_index, **kwargs: Any):
            result = _original(*args, **kwargs)
            probabilities = result[1]
            try:
                values = probabilities.detach().cpu().reshape(-1).tolist()
                values = [round(float(value), 8) for value in values]
            except Exception:
                values = []
            trace.event("policy_layer_scores", layer_index=_index, probabilities=values)
            return result

        layer.forward = traced_layer_forward

    @wraps(original_forward)
    def traced_forward(*args: Any, **kwargs: Any):
        result = original_forward(*args, **kwargs)
        logs, selected_layers = result
        serialized_logs = []
        for value in logs:
            try:
                serialized_logs.append(float(value.detach().cpu().item()))
            except AttributeError:
                serialized_logs.append(float(value))
        trace.event(
            "policy_route_selected",
            selected_layers=[list(layer) for layer in selected_layers],
            log_prob_layers=serialized_logs,
        )
        return result

    controller.forward = traced_forward


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


def _instrument_model(model: Any, trace: TraceRecorder) -> None:
    original_invoke = model.invoke

    @wraps(original_invoke)
    def traced_invoke(*args: Any, **kwargs: Any):
        trace.model_invocations += 1
        call_index = trace.model_invocations
        before_input, before_output = _tracker_counts(model)
        messages = args[0] if args else kwargs.get("messages", [])
        trace.event(
            "model_invoke_enter",
            call_index=call_index,
            operator_name=trace.current_operator or "unknown",
            message_count=len(messages) if isinstance(messages, list) else None,
            message_chars=sum(len(str(item.get("content", ""))) for item in messages if isinstance(item, Mapping))
            if isinstance(messages, list)
            else None,
        )
        started = time.perf_counter()
        try:
            response = original_invoke(*args, **kwargs)
        except Exception as exc:
            trace.event(
                "model_invoke_exit",
                call_index=call_index,
                operator_name=trace.current_operator or "unknown",
                status="error",
                error_type=type(exc).__name__,
                error=_redact(exc),
                duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
            )
            raise

        after_input, after_output = _tracker_counts(model)
        explicit_input, explicit_output = _response_usage(response)
        prompt_tokens = explicit_input if explicit_input is not None else max(after_input - before_input, 0)
        completion_tokens = explicit_output if explicit_output is not None else max(after_output - before_output, 0)
        cost = prompt_tokens / 1000.0 * INPUT_RATE_PER_1K + completion_tokens / 1000.0 * OUTPUT_RATE_PER_1K
        trace.event(
            "model_invoke_exit",
            call_index=call_index,
            operator_name=trace.current_operator or "unknown",
            status="completed",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=round(cost, 10),
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
        return response

    model.invoke = traced_invoke

    # LegacyOpenAIModel retries the low-level Chat Completions request inside
    # invoke().  Count that boundary too when the provider exposes it.
    completions = getattr(getattr(model, "_client", None), "chat", None)
    completions = getattr(completions, "completions", None)
    original_create = getattr(completions, "create", None)
    if not callable(original_create):
        return

    @wraps(original_create)
    def traced_create(*args: Any, **kwargs: Any):
        trace.api_attempts += 1
        attempt = trace.api_attempts
        try:
            response = original_create(*args, **kwargs)
            usage = getattr(response, "usage", None)
            prompt = getattr(usage, "prompt_tokens", None) if usage else None
            completion = getattr(usage, "completion_tokens", None) if usage else None
            trace.event(
                "api_request",
                attempt=attempt,
                operator_name=trace.current_operator or "unknown",
                model=str(kwargs.get("model", getattr(model, "model_name", "unknown"))),
                status="completed",
                prompt_tokens=int(prompt) if prompt is not None else None,
                completion_tokens=int(completion) if completion is not None else None,
            )
            return response
        except Exception as exc:
            trace.event(
                "api_request",
                attempt=attempt,
                operator_name=trace.current_operator or "unknown",
                model=str(kwargs.get("model", getattr(model, "model_name", "unknown"))),
                status="error",
                error_type=type(exc).__name__,
                error=_redact(exc),
            )
            raise

    try:
        completions.create = traced_create
    except Exception as exc:
        trace.event("instrumentation_warning", target="chat.completions.create", error_type=type(exc).__name__)


def _public_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"api_key", "prompt", "response", "expected_answer"}:
            continue
        if isinstance(item, (str, int, float, bool)) or item is None:
            result[str(key)] = _redact(item) if isinstance(item, str) else item
    return result


def _summary(report: dict[str, Any], operator_catalog: tuple[str, ...]) -> dict[str, Any]:
    events = report["events"]
    selected = Counter()
    actual = Counter()
    api_calls = Counter()
    api_total = 0
    model_total = 0
    for event in events:
        kind = event.get("kind")
        if kind == "policy_route_selected":
            for layer in event.get("selected_layers", []):
                selected.update(layer)
        elif kind == "dispatch_enter":
            actual[event.get("operator_name", "unknown")] += 1
        elif kind == "api_request":
            operator = event.get("operator_name", "unknown")
            api_calls[operator] += 1
            api_total += 1
        elif kind == "model_invoke_enter":
            model_total += 1

    operators = []
    names = list(dict.fromkeys([*operator_catalog, *selected.keys(), *actual.keys(), *api_calls.keys()]))
    for name in names:
        operators.append(
            {
                "operator_name": name,
                "selected_count": selected.get(name, 0),
                "dispatch_count": actual.get(name, 0),
                "api_request_count": api_calls.get(name, 0),
            }
        )
    return {
        "operator_counts": operators,
        "model_invoke_count": model_total,
        "api_request_count": api_total,
        "total_cost": report.get("cost", {}).get("total_cost"),
        "prompt_tokens": report.get("cost", {}).get("prompt_tokens"),
        "completion_tokens": report.get("cost", {}).get("completion_tokens"),
    }


def _write_html(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    cards = "".join(
        f'<div class="card"><span>{html.escape(label)}</span><strong>{html.escape(str(value))}</strong></div>'
        for label, value in (
            ("API requests", summary["api_request_count"]),
            ("Model invokes", summary["model_invoke_count"]),
            ("Prompt tokens", summary["prompt_tokens"]),
            ("Completion tokens", summary["completion_tokens"]),
            ("Total cost", f'${float(summary["total_cost"] or 0):.8f}'),
        )
    )
    operator_rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row[key]))}</td>"
            for key in ("operator_name", "selected_count", "dispatch_count", "api_request_count")
        )
        + "</tr>"
        for row in summary["operator_counts"]
    )
    flow_rows = "".join(
        "<li>"
        f'<code>#{event["event_index"]}</code> '
        f'<b>{html.escape(str(event.get("kind")))}</b> '
        f'{html.escape(str(event.get("operator_name", "")))} '
        f'{html.escape(_flow_details(event))}'
        "</li>"
        for event in report["events"]
    )
    route = html.escape(json.dumps(report.get("route", {}), ensure_ascii=False, indent=2))
    errors = html.escape(json.dumps(report.get("errors", []), ensure_ascii=False, indent=2))
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>MaAS MATH trace</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#f4f6f8;color:#17202a;margin:28px}}
h1{{margin-bottom:4px}} .muted{{color:#667085}} .cards{{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}}
.card{{background:white;border:1px solid #d9dee7;border-radius:10px;padding:12px 18px;min-width:130px;box-shadow:0 1px 3px #0001}}
.card span{{display:block;color:#667085;font-size:12px}} .card strong{{display:block;font-size:20px;margin-top:5px}}
section{{background:white;border:1px solid #d9dee7;border-radius:10px;padding:18px;margin:16px 0}}
table{{border-collapse:collapse;width:100%}} th,td{{border-bottom:1px solid #e7eaf0;padding:8px;text-align:left}} th{{background:#f8fafc}}
ol{{padding-left:25px}} li{{padding:5px 0;border-bottom:1px dashed #e7eaf0}} code{{color:#475467}}
pre{{background:#101828;color:#e6edf3;padding:14px;border-radius:8px;overflow:auto}}
</style></head><body>
<h1>MaAS reproduction · MATH 第 1 题调用追踪</h1>
<div class="muted">仅记录选择、dispatch、模型调用、API 请求和费用；不记录 API Key、prompt 或完整模型回答。</div>
<div class="cards">{cards}</div>
<section><h2>Operator 对比</h2><table><tr><th>Operator</th><th>策略选择次数</th><th>实际 dispatch 次数</th><th>API 请求次数</th></tr>{operator_rows}</table></section>
<section><h2>策略路线与概率</h2><pre>{route}</pre></section>
<section><h2>实际调用时间线</h2><ol>{flow_rows}</ol></section>
<section><h2>错误</h2><pre>{errors}</pre></section>
</body></html>"""
    path.write_text(document, encoding="utf-8")


def _flow_details(event: Mapping[str, Any]) -> str:
    parts = []
    for key in ("sequence_index", "layer_index", "position", "status", "prompt_tokens", "completion_tokens", "cost"):
        if key in event:
            parts.append(f"{key}={event[key]}")
    return " ".join(parts)


def _load_env() -> None:
    env_path = APP_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path, override=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace exactly one MATH test sample")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fake-model", action="store_true", help="offline preflight without API calls")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    os.chdir(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    _load_env()

    trace = TraceRecorder()
    _patch_execution_seams(trace)
    from applications.maas_reproduction.maas_reproduction.runtime.settings import load_settings
    from applications.maas_reproduction.maas_reproduction.runtime.bootstrap import build_runtime

    output_dir = args.output_dir or (
        APP_ROOT / "assets" / "output" / f"trace_MATH_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace.event(
        "test_started",
        dataset="MATH",
        split="test",
        problem_index=0,
        checkpoint=str(args.checkpoint.resolve()),
        output_dir=str(output_dir),
        fake_model=bool(args.fake_model),
    )

    errors: list[dict[str, str]] = []
    runtime = None
    sample_record: Mapping[str, Any] | None = None
    try:
        settings = load_settings(
            args.config_root,
            mode_override="test",
            dataset_override="MATH",
            split_override="test",
            sample_override=1,
        )
        runtime = build_runtime(
            settings,
            fake=args.fake_model,
            output_root_override=output_dir,
            subset=1,
        )
        payload = runtime.checkpoint_manager.load(
            args.checkpoint,
            controller=runtime.policy_controller,
            expected_operator_catalog=runtime.operator_catalog,
        )
        trace.event(
            "weights_loaded",
            checkpoint_epoch=payload.get("epoch"),
            operator_catalog=list(runtime.operator_catalog),
        )
        _instrument_controller(runtime.policy_controller, trace)
        _instrument_model(runtime.model, trace)

        for item in runtime.dataset_runner.iter_run(limit=None):
            sample_record = dict(item)
            trace.event(
                "sample_completed",
                problem_index=sample_record.get("problem_index", 0),
                status=sample_record.get("status"),
                score=sample_record.get("score"),
                cost_delta=sample_record.get("cost_delta"),
                cost_reliable=sample_record.get("cost_reliable"),
                evaluation_reliable=sample_record.get("evaluation_reliable"),
            )
    except Exception as exc:
        errors.append({"type": type(exc).__name__, "message": _redact(exc)})
        trace.event("test_error", error_type=type(exc).__name__, error=_redact(exc))
    finally:
        cost = {"total_cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "reliable": False}
        if runtime is not None:
            try:
                snapshot = runtime.cost_tracker.snapshot()
                cost = {
                    "total_cost": float(snapshot.total_cost),
                    "prompt_tokens": snapshot.prompt_tokens,
                    "completion_tokens": snapshot.completion_tokens,
                    "reliable": bool(snapshot.reliable),
                }
            except Exception as exc:
                errors.append({"type": type(exc).__name__, "message": _redact(exc)})
        trace.event("test_finished", status="error" if errors else "completed", **cost)
        report = {
            "test": {
                "dataset": "MATH",
                "split": "test",
                "problem_index": 0,
                "checkpoint": str(args.checkpoint.resolve()),
                "billing": {
                    "input_cost_per_1k_tokens": INPUT_RATE_PER_1K,
                    "output_cost_per_1k_tokens": OUTPUT_RATE_PER_1K,
                },
            },
            "route": {
                "policy_layers": [event for event in trace.events if event["kind"] == "policy_layer_scores"],
                "selected": next((event for event in trace.events if event["kind"] == "policy_route_selected"), None),
                "normalized": next((event for event in trace.events if event["kind"] == "route_plan"), None),
            },
            "cost": cost,
            "sample": {
                key: sample_record.get(key)
                for key in ("problem_index", "status", "score", "cost_delta", "cost_reliable", "evaluation_reliable")
            }
            if sample_record is not None
            else None,
            "errors": errors,
            "events": trace.events,
        }
        report["summary"] = _summary(report, tuple(runtime.operator_catalog) if runtime is not None else ())
        json_path = output_dir / "trace.json"
        html_path = output_dir / "trace.html"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        _write_html(report, html_path)
        print(f"[MaAS-TRACE] trace_json={json_path}")
        print(f"[MaAS-TRACE] trace_html={html_path}")
        print(
            f"[MaAS-COST] prompt_tokens={cost['prompt_tokens']} "
            f"completion_tokens={cost['completion_tokens']} "
            f"total_cost=${cost['total_cost']:.8f}",
            flush=True,
        )

    if errors:
        raise RuntimeError(errors[-1]["message"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
