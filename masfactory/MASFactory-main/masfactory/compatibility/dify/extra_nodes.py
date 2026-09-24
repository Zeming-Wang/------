"""Dify workflow node forwards beyond the core set (tool, http, knowledge, classifier, iteration)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable

from masfactory.compatibility.dify.options import DifyCompileOptions
from masfactory.compatibility.dify.runtime import DifyRuntimeState
from masfactory.compatibility.dify.template import render_dify_template
from masfactory.compatibility.errors import CompatibilityImportError


def _render_tool_param_value(spec: Any, state: DifyRuntimeState) -> Any:
    if isinstance(spec, dict):
        t = spec.get("type")
        val = spec.get("value")
        if t == "mixed" and isinstance(val, str):
            return render_dify_template(val, state)
        if t == "constant":
            return val
        return val
    return spec


def make_tool_forward(
    node_id: str,
    data: dict[str, Any],
    store: dict[str, Any],
    tool_executor: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None,
):
    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        state = DifyRuntimeState.from_store(store)
        params: dict[str, Any] = {}
        raw_params = data.get("tool_parameters") or {}
        if isinstance(raw_params, dict):
            for key, spec in raw_params.items():
                params[str(key)] = _render_tool_param_value(spec, state)
        tool_meta = {
            "provider_id": data.get("provider_id"),
            "provider_name": data.get("provider_name"),
            "tool_name": data.get("tool_name"),
            "tool_label": data.get("tool_label"),
            "parameters": params,
        }
        if tool_executor is not None:
            out = tool_executor(dict(data), params)
        else:
            out = {
                "text": json.dumps(tool_meta, ensure_ascii=False),
                "json": tool_meta,
            }
        if not isinstance(out, dict):
            raise CompatibilityImportError("tool_executor must return a dict.")
        state.set_node_output(node_id, out)
        return {**input, **out}

    return forward


def make_http_request_forward(
    node_id: str,
    data: dict[str, Any],
    store: dict[str, Any],
    http_client: Callable[[dict[str, Any]], dict[str, Any]] | None,
):
    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        state = DifyRuntimeState.from_store(store)
        if http_client is not None:
            payload = http_client(dict(data))
        else:
            url = render_dify_template(str(data.get("url") or ""), state)
            method = str(data.get("method") or "get").upper()
            if not url:
                payload = {"body": "", "status_code": 0, "error": "empty_url"}
            else:
                try:
                    req = urllib.request.Request(url, method=method if method != "GET" else "GET")
                    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                        body = resp.read().decode("utf-8", errors="replace")
                        payload = {"body": body, "status_code": getattr(resp, "status", 200)}
                except (urllib.error.URLError, OSError, ValueError) as exc:
                    payload = {"body": str(exc), "status_code": -1, "error": str(exc)}
        state.set_node_output(node_id, payload)
        return {**input, **payload}

    return forward


def make_knowledge_retrieval_forward(
    node_id: str,
    data: dict[str, Any],
    store: dict[str, Any],
    retriever: Callable[[dict[str, Any], str], str] | None,
):
    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        state = DifyRuntimeState.from_store(store)
        sel = data.get("query_variable_selector") or []
        query = str(state.resolve_selector(sel) if isinstance(sel, list) else "") or ""
        if retriever is not None:
            result = retriever(dict(data), query)
        else:
            ids = data.get("dataset_ids") or []
            n = len(ids) if isinstance(ids, list) else 0
            result = (
                f"[knowledge-retrieval placeholder] datasets={n} query={query!r}. "
                "Pass DifyCompileOptions.knowledge_retriever=... for real retrieval."
            )
        out = {"result": result, "text": result}
        state.set_node_output(node_id, out)
        return {**input, **out}

    return forward


def _first_matching_class_id(classes: list[dict[str, Any]], query: str) -> str:
    q = query.strip()
    for c in classes:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or "")
        name = str(c.get("name") or "")
        if name and name in q:
            return cid
    if classes and isinstance(classes[-1], dict):
        return str(classes[-1].get("id") or "")
    return ""


def make_question_classifier_routes(
    node_id: str,
    data: dict[str, Any],
    store: dict[str, Any],
    edges: list,
    id_to_graph_name: dict[str, str],
) -> dict[str, Callable[[dict, dict[str, object]], bool]]:
    classes = [c for c in (data.get("classes") or []) if isinstance(c, dict)]
    query_sel = data.get("query_variable_selector") or []

    routes: dict[str, Callable[[dict, dict[str, object]], bool]] = {}
    for edge in edges:
        if edge.source != node_id:
            continue
        handle = str(edge.source_handle or "source")
        receiver = id_to_graph_name.get(edge.target, edge.target)

        def _pred(
            message: dict,
            attributes: dict[str, object],
            *,
            handle_id: str = handle,
        ) -> bool:
            state = DifyRuntimeState.from_store(store)
            query = str(state.resolve_selector(query_sel) if isinstance(query_sel, list) else "")
            return _first_matching_class_id(classes, query) == handle_id

        routes[receiver] = _pred
    return routes


def _inner_node_map(inner: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for raw in inner.get("nodes") or []:
        if not isinstance(raw, dict):
            continue
        nid = raw.get("id")
        if nid is None:
            continue
        data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
        out[str(nid)] = {"kind": str(data.get("type") or ""), "raw": raw, "data": data}
    return out


def make_passthrough_forward(
    node_id: str,
    store: dict[str, Any],
    *,
    outputs: dict[str, Any] | None = None,
):
    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        state = DifyRuntimeState.from_store(store)
        payload = dict(outputs or {})
        if payload:
            state.set_node_output(node_id, payload)
        return {**input, **payload}

    return forward


def _file_value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "content", "name", "filename", "url"):
            if value.get(key):
                return str(value[key])
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "\n".join(_file_value_to_text(v) for v in value)
    return str(value)


def make_document_extractor_forward(node_id: str, data: dict[str, Any], store: dict[str, Any]):
    sel = data.get("variable_selector") or []

    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        state = DifyRuntimeState.from_store(store)
        raw = state.resolve_selector(sel) if isinstance(sel, list) else None
        text = _file_value_to_text(raw)
        out = {"text": text, "result": text}
        state.set_node_output(node_id, out)
        return {**input, **out}

    return forward


def _list_filter_string(item: str, op: str, value: str) -> bool:
    match op:
        case "contains":
            return value in item
        case "not contains":
            return value not in item
        case "start with":
            return item.startswith(value)
        case "end with":
            return item.endswith(value)
        case "is":
            return item == value
        case "is not":
            return item != value
        case "in":
            return item in value.split(",") if "," in value else item in value
        case "not in":
            return not _list_filter_string(item, "in", value)
        case "empty":
            return not item
        case "not empty":
            return bool(item)
        case _:
            return True


def _list_filter_number(item: Any, op: str, value: str) -> bool:
    try:
        n = float(item)
        v = float(value)
    except (TypeError, ValueError):
        return False
    match op:
        case "=":
            return n == v
        case "≠":
            return n != v
        case "<":
            return n < v
        case ">":
            return n > v
        case "≤":
            return n <= v
        case "≥":
            return n >= v
        case _:
            return True


def make_list_operator_forward(node_id: str, data: dict[str, Any], store: dict[str, Any]):
    sel = data.get("variable") or []

    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        state = DifyRuntimeState.from_store(store)
        raw = state.resolve_selector(sel) if isinstance(sel, list) else None
        items: list[Any] = list(raw) if isinstance(raw, (list, tuple)) else []

        filter_by = data.get("filter_by") if isinstance(data.get("filter_by"), dict) else {}
        if filter_by.get("enabled"):
            filtered: list[Any] = []
            for item in items:
                keep = True
                for cond in filter_by.get("conditions") or []:
                    if not isinstance(cond, dict):
                        continue
                    op = str(cond.get("comparison_operator") or "")
                    val = cond.get("value")
                    value = render_dify_template(str(val), state) if isinstance(val, str) else val
                    if isinstance(item, (int, float)) or (
                        isinstance(value, str) and value.replace(".", "", 1).isdigit()
                    ):
                        keep = keep and _list_filter_number(item, op, str(value))
                    else:
                        keep = keep and _list_filter_string(str(item), op, str(value))
                if keep:
                    filtered.append(item)
            items = filtered

        extract_by = data.get("extract_by") if isinstance(data.get("extract_by"), dict) else {}
        if extract_by.get("enabled"):
            serial = render_dify_template(str(extract_by.get("serial") or "1"), state)
            try:
                idx = max(int(serial) - 1, 0)
                items = [items[idx]] if idx < len(items) else []
            except ValueError:
                items = []

        order_by = data.get("order_by") if isinstance(data.get("order_by"), dict) else {}
        if order_by.get("enabled"):
            reverse = str(order_by.get("value") or "asc").lower() == "desc"
            try:
                items = sorted(items, reverse=reverse)
            except TypeError:
                items = sorted(items, key=str, reverse=reverse)

        limit = data.get("limit") if isinstance(data.get("limit"), dict) else {}
        if limit.get("enabled"):
            size = int(limit.get("size") or len(items))
            if size >= 0:
                items = items[:size]

        out = {
            "result": items,
            "first_record": items[0] if items else None,
            "last_record": items[-1] if items else None,
        }
        state.set_node_output(node_id, out)
        return {**input, **out}

    return forward


def make_knowledge_index_forward(
    node_id: str,
    data: dict[str, Any],
    store: dict[str, Any],
    retriever: Callable[[dict[str, Any], str], str] | None,
):
    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        state = DifyRuntimeState.from_store(store)
        sel = data.get("query_variable_selector") or data.get("variable_selector") or []
        query = str(state.resolve_selector(sel) if isinstance(sel, list) else "") or ""
        if retriever is not None:
            result = retriever(dict(data), query)
        else:
            result = (
                f"[knowledge-index placeholder] query={query!r}. "
                "Pass DifyCompileOptions.knowledge_retriever=... for real indexing."
            )
        out = {"result": result, "text": result}
        state.set_node_output(node_id, out)
        return {**input, **out}

    return forward


def make_agent_forward(
    node_id: str,
    data: dict[str, Any],
    store: dict[str, Any],
    options: DifyCompileOptions,
    *,
    llm_forward: Callable[..., dict[str, object]] | None = None,
):
    """Agent nodes: use LLM when `model` is present, otherwise emit a structured stub."""

    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        state = DifyRuntimeState.from_store(store)
        model_cfg = data.get("model") if isinstance(data.get("model"), dict) else {}
        if model_cfg and llm_forward is not None:
            return llm_forward(input, attributes)
        out = {
            "text": json.dumps(
                {
                    "agent_strategy": data.get("agent_strategy_name") or data.get("strategy"),
                    "tools": data.get("tools") or data.get("tool_list"),
                },
                ensure_ascii=False,
            )
        }
        state.set_node_output(node_id, out)
        return {**input, **out}

    return forward


def make_human_input_forward(node_id: str, data: dict[str, Any], store: dict[str, Any]):
    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        state = DifyRuntimeState.from_store(store)
        merged = {**attributes, **input}
        title = str(data.get("title") or "human-input")
        text = str(merged.get("human_input") or merged.get("query") or merged.get("text") or "")
        if not text:
            text = f"[human-input placeholder] Provide `{title}` via invoke payload."
        out = {"text": text, "answer": text}
        state.set_node_output(node_id, out)
        return {**input, **out}

    return forward


def _run_inner_subgraph(
    state: DifyRuntimeState,
    *,
    inner: dict[str, Any],
    options: DifyCompileOptions,
    item: Any = None,
    entry_sources: tuple[str, ...] = ("iteration_input",),
    start_node_id: str | None = None,
) -> str:
    nodes = _inner_node_map(inner)
    edges = inner.get("edges") or []
    start_targets: list[str] = []
    if start_node_id:
        start_targets = [start_node_id]
    else:
        for e in edges:
            if not isinstance(e, dict):
                continue
            if str(e.get("source")) in entry_sources:
                start_targets.append(str(e.get("target")))
    if not start_targets:
        return ""
    current = start_targets[0]
    prev_out: dict[str, Any] = {"item": item}
    visited = 0
    while current and visited < 20:
        visited += 1
        info = nodes.get(current)
        if not info:
            break
        kind = info["kind"]
        data = info["data"]
        if kind == "llm":
            prev_item = state.iteration_item
            state.iteration_item = item
            try:
                model_cfg = data.get("model") if isinstance(data.get("model"), dict) else {}
                model = options.resolve_model(model_cfg)
                messages: list[dict[str, str]] = []
                for pt in data.get("prompt_template") or []:
                    if not isinstance(pt, dict):
                        continue
                    role = str(pt.get("role") or "user")
                    text = render_dify_template(str(pt.get("text") or ""), state)
                    if text:
                        messages.append({"role": role, "content": text})
                if not messages:
                    messages = [{"role": "user", "content": str(item)}]
                resp = model.invoke(messages, tools=None, settings=model_cfg.get("completion_params"))
                prev_out = {"text": str(resp.get("content") or ""), "item": item}
                state.set_node_output(current, prev_out)
            finally:
                state.iteration_item = prev_item
        elif kind in {"template-transform"}:
            text = render_dify_template(str(data.get("template") or ""), state)
            prev_out = {**prev_out, "output": text, "text": text}
            state.set_node_output(current, prev_out)
        elif kind == "code":
            from masfactory.compatibility.dify.runtime import collect_node_inputs, run_dify_code

            variables = collect_node_inputs(data, state)
            result = run_dify_code(str(data.get("code") or ""), variables)
            prev_out = {**prev_out, **result}
            state.set_node_output(current, prev_out)
        elif kind == "answer":
            text = render_dify_template(str(data.get("answer") or ""), state)
            prev_out = {**prev_out, "text": text, "answer": text}
            state.set_node_output(current, prev_out)
            return text
        elif kind in {"loop-end", "iteration-start", "loop-start"}:
            state.set_node_output(current, prev_out)
            return str(prev_out.get("text") or "")
        else:
            break
        nxt = None
        for e in edges:
            if not isinstance(e, dict):
                continue
            if str(e.get("source")) == current:
                nxt = str(e.get("target"))
                break
        current = nxt or ""
    return str(prev_out.get("text") or "")


def _run_inner_llm_answer(
    state: DifyRuntimeState,
    item: Any,
    inner: dict[str, Any],
    options: DifyCompileOptions,
) -> str:
    return _run_inner_subgraph(state, inner=inner, options=options, item=item)


def make_loop_forward(
    node_id: str,
    data: dict[str, Any],
    store: dict[str, Any],
    loop_subgraph: dict[str, Any] | None,
    options: DifyCompileOptions,
):
    loop_count = int(data.get("loop_count") or 1)
    start_node_id = str(data.get("start_node_id") or "")

    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        state = DifyRuntimeState.from_store(store)
        inner = loop_subgraph if isinstance(loop_subgraph, dict) else {}
        lines: list[str] = []
        prev_index = state.loop_index
        for i in range(max(loop_count, 0)):
            state.loop_index = i
            if inner.get("nodes"):
                lines.append(
                    _run_inner_subgraph(
                        state,
                        inner=inner,
                        options=options,
                        entry_sources=("loop-start",),
                        start_node_id=start_node_id or None,
                    )
                )
            else:
                lines.append(str(i))
        state.loop_index = prev_index
        text = "\n".join(lines)
        outputs = data.get("outputs") if isinstance(data.get("outputs"), dict) else {}
        out = {"output": text, "text": text, **outputs}
        state.set_node_output(node_id, out)
        return {**input, **out}

    return forward


def make_iteration_forward(
    node_id: str,
    data: dict[str, Any],
    store: dict[str, Any],
    iteration_graph: dict[str, Any] | None,
    options: DifyCompileOptions,
):
    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        state = DifyRuntimeState.from_store(store)
        sel = data.get("iterator_selector") or []
        arr = state.resolve_selector(sel) if isinstance(sel, list) else None
        if not isinstance(arr, (list, tuple)):
            arr = []
        lines: list[str] = []
        inner = iteration_graph if isinstance(iteration_graph, dict) else {}
        for item in arr:
            if inner.get("nodes"):
                lines.append(_run_inner_llm_answer(state, item, inner, options))
            else:
                lines.append(str(item))
        text = "\n".join(lines)
        out = {"output": text, "text": text}
        state.set_node_output(node_id, out)
        return {**input, **out}

    return forward
