from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any

from masfactory.compatibility.errors import CompatibilityImportError


STATE_KEY = "_dify_state"


@dataclass
class DifyRuntimeState:
    conversation: dict[str, Any] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=dict)
    sys: dict[str, Any] = field(default_factory=dict)
    node_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    workflow_result: dict[str, Any] | None = None
    iteration_item: Any = None
    loop_index: int | None = None

    @classmethod
    def from_store(cls, store: dict[str, Any]) -> DifyRuntimeState:
        raw = store.get(STATE_KEY)
        if isinstance(raw, DifyRuntimeState):
            return raw
        if isinstance(raw, dict):
            state = cls(
                conversation=dict(raw.get("conversation") or {}),
                env=dict(raw.get("env") or {}),
                sys=dict(raw.get("sys") or {}),
                node_outputs={
                    str(k): dict(v) if isinstance(v, dict) else {}
                    for k, v in (raw.get("node_outputs") or {}).items()
                },
                workflow_result=dict(raw["workflow_result"])
                if isinstance(raw.get("workflow_result"), dict)
                else None,
            )
            store[STATE_KEY] = state
            return state
        state = cls()
        store[STATE_KEY] = state
        return state

    def attach(self, store: dict[str, Any]) -> None:
        store[STATE_KEY] = self

    def set_node_output(self, node_id: str, output: dict[str, Any]) -> None:
        self.node_outputs[str(node_id)] = dict(output)

    def get_node_output(self, node_id: str) -> dict[str, Any]:
        return dict(self.node_outputs.get(str(node_id), {}))

    def resolve_selector(self, selector: list[Any] | tuple[Any, ...]) -> Any:
        if not selector:
            return None
        head = str(selector[0])
        tail = selector[1:]

        if head == "conversation":
            cur: Any = self.conversation
            for part in tail:
                if not isinstance(cur, dict):
                    return None
                cur = cur.get(str(part))
            return cur

        if head == "env":
            cur = self.env
            for part in tail:
                if not isinstance(cur, dict):
                    return None
                cur = cur.get(str(part))
            return cur

        if head == "sys":
            cur = self.sys
            for part in tail:
                if not isinstance(cur, dict):
                    return None
                cur = cur.get(str(part))
            return cur

        if head == "item" and not tail:
            return self.iteration_item

        node_out = self.get_node_output(head)
        if not tail:
            return node_out
        cur = node_out
        for part in tail:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(str(part))
        return cur

    def write_selector(self, selector: list[Any] | tuple[Any, ...], value: Any) -> None:
        if len(selector) < 2:
            raise ValueError(f"Invalid write selector: {selector!r}")
        head = str(selector[0])
        leaf = str(selector[-1])
        parents = [str(part) for part in selector[1:-1]]

        if head == "conversation":
            target = self.conversation
        elif head == "env":
            target = self.env
        elif head == "sys":
            target = self.sys
        else:
            raise ValueError(f"Unsupported write selector root: {head!r}")

        for part in parents:
            nxt = target.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                target[part] = nxt
            target = nxt
        target[leaf] = value


def collect_node_inputs(data: dict[str, Any], state: DifyRuntimeState) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for item in data.get("variables") or []:
        if not isinstance(item, dict):
            continue
        name = item.get("variable")
        selector = item.get("value_selector")
        if name is None or not selector:
            continue
        inputs[str(name)] = state.resolve_selector(selector)
    return inputs


def run_dify_code(code: str, variables: dict[str, Any]) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    try:
        exec(textwrap.dedent(code), {"__builtins__": __builtins__}, namespace)  # noqa: S102
    except ModuleNotFoundError as exc:
        name = getattr(exc, "name", None) or str(exc)
        raise CompatibilityImportError(
            f"Dify code node requires Python package {name!r}. "
            "Install extras with: uv sync --extra dify-preview"
        ) from exc
    main = namespace.get("main")
    if not callable(main):
        raise CompatibilityImportError("Dify code node must define a callable `main(...)`.")
    result = main(**variables)
    if not isinstance(result, dict):
        raise CompatibilityImportError("Dify code node `main` must return a dict.")
    return result


def init_dify_state(metadata: dict[str, Any] | None, outer: dict[str, Any]) -> DifyRuntimeState:
    state = DifyRuntimeState()
    dify = (metadata or {}).get("dify") or {}
    for item in dify.get("conversation_variables") or []:
        if isinstance(item, dict) and item.get("name") is not None:
            state.conversation[str(item["name"])] = item.get("value")
    for item in dify.get("environment_variables") or []:
        if isinstance(item, dict) and item.get("name") is not None:
            state.env[str(item["name"])] = item.get("value")

    sys_input = outer.get("sys")
    if isinstance(sys_input, dict):
        state.sys.update(sys_input)
    for key in ("query", "user_id", "files"):
        if key in outer and key not in state.sys:
            state.sys[key] = outer[key]
    return state


def _coerce_value(raw: Any, var_type: str | None) -> Any:
    if raw is None:
        return None
    if var_type == "number":
        if isinstance(raw, (int, float)):
            return raw
        text = str(raw).strip()
        if not text:
            return None
        if "." in text:
            return float(text)
        return int(text)
    if var_type == "boolean":
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}
    return raw


def _compare(left: Any, operator: str, right: Any, var_type: str | None) -> bool:
    op = (operator or "").strip()
    left_val = _coerce_value(left, var_type)
    right_val = _coerce_value(right, var_type)

    if op in {"empty", "is empty"}:
        return left_val is None or str(left_val).strip() == ""
    if op in {"not empty", "is not empty"}:
        return left_val is not None and str(left_val).strip() != ""
    if op in {"contains", "not contains"}:
        left_text = "" if left_val is None else str(left_val)
        right_text = "" if right_val is None else str(right_val)
        matched = right_text in left_text
        return not matched if op == "not contains" else matched

    if var_type == "number":
        try:
            left_num = float(left_val)
            right_num = float(right_val)
        except (TypeError, ValueError):
            return False
        if op in {"=", "=="}:
            return left_num == right_num
        if op == "!=":
            return left_num != right_num
        if op == ">":
            return left_num > right_num
        if op == "<":
            return left_num < right_num
        if op == ">=":
            return left_num >= right_num
        if op == "<=":
            return left_num <= right_num
        return False

    left_text = "" if left_val is None else str(left_val)
    right_text = "" if right_val is None else str(right_val)
    if op in {"=", "=="}:
        return left_text == right_text
    if op == "!=":
        return left_text != right_text
    return False


def evaluate_case(case: dict[str, Any], state: DifyRuntimeState) -> bool:
    conditions = case.get("conditions") or []
    if not conditions:
        return False
    logical = str(case.get("logical_operator") or "and").lower()
    results: list[bool] = []
    for cond in conditions:
        if not isinstance(cond, dict):
            continue
        selector = cond.get("variable_selector") or []
        left = state.resolve_selector(selector)
        results.append(
            _compare(
                left,
                str(cond.get("comparison_operator") or "="),
                cond.get("value"),
                str(cond.get("varType")) if cond.get("varType") is not None else None,
            )
        )
    if not results:
        return False
    if logical == "or":
        return any(results)
    return all(results)


def evaluate_handle(handle: str, cases: list[dict[str, Any]], state: DifyRuntimeState) -> bool:
    handle_id = str(handle)
    if handle_id == "false":
        return not any(evaluate_case(case, state) for case in cases if isinstance(case, dict))
    for case in cases:
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("case_id") or case.get("id") or "")
        if case_id == handle_id:
            return evaluate_case(case, state)
    return False
