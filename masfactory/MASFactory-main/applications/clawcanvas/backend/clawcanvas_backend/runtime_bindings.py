from __future__ import annotations

import json
from collections import Counter
from typing import Any, Callable


def builtin_echo(text: str) -> dict[str, str]:
    """Return the input text unchanged."""

    return {"text": str(text)}


def builtin_json_inspect(payload: object) -> dict[str, str]:
    """Render any payload as pretty JSON for inspection."""

    try:
        rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    except Exception:
        rendered = str(payload)
    return {"json": rendered}


def builtin_list_keys(payload: object) -> dict[str, list[str]]:
    """List top-level keys from a dict payload."""

    if isinstance(payload, dict):
        keys = [str(key) for key in payload.keys()]
    else:
        keys = []
    return {"keys": keys}


def builtin_concat_text(left: str, right: str, separator: str = "\n") -> dict[str, str]:
    """Concatenate two text fragments with a configurable separator."""

    return {"text": f"{left}{separator}{right}"}


BUILTIN_TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "echo": builtin_echo,
    "json_inspect": builtin_json_inspect,
    "list_keys": builtin_list_keys,
    "concat_text": builtin_concat_text,
}
SUPPORTED_BUILTIN_TOOL_NAMES = tuple(sorted(BUILTIN_TOOL_REGISTRY.keys()))


class RuntimeWarnings:
    def __init__(self, add_warning: Callable[[str], None]):
        self._add_warning = add_warning

    def add(self, message: str) -> None:
        if message:
            self._add_warning(message)


def resolve_agent_tools(
    declarations: list[dict[str, Any]],
    *,
    owner: str,
    warnings: RuntimeWarnings,
) -> list[Callable[..., Any]]:
    tools: list[Callable[..., Any]] = []
    seen: set[str] = set()
    for declaration in declarations:
        tool = _resolve_tool_declaration(declaration, owner=owner, warnings=warnings)
        if tool is None:
            continue
        name = getattr(tool, "__name__", None) or ""
        if name in seen:
            continue
        seen.add(name)
        tools.append(tool)
    return tools


def _resolve_tool_declaration(
    declaration: dict[str, Any],
    *,
    owner: str,
    warnings: RuntimeWarnings,
) -> Callable[..., Any] | None:
    name = str(declaration.get("name") or "").strip()
    binding = str(declaration.get("binding") or "builtin").strip().lower()
    description = str(declaration.get("description") or "").strip()
    config = dict(declaration.get("config") or {})
    if not name:
        warnings.add(f"{owner}: encountered a tool declaration with empty name; skipped")
        return None
    if binding == "builtin":
        tool = BUILTIN_TOOL_REGISTRY.get(name)
        if tool is None:
            warnings.add(
                f"{owner}: builtin tool '{name}' is not supported by ClawCanvas runtime; "
                f"supported builtins are {sorted(BUILTIN_TOOL_REGISTRY.keys())}"
            )
            return None
        return tool
    if binding == "mcp":
        warnings.add(
            f"{owner}: tool '{name}' uses binding 'mcp' but ClawCanvas runtime has no MCP endpoint config for it yet"
        )
        return None
    if binding == "api":
        tool = _build_api_tool(name, description, config=config, owner=owner, warnings=warnings)
        if tool is None:
            return None
        return tool
    warnings.add(f"{owner}: tool '{name}' uses unsupported binding '{binding}' and was skipped")
    return None


def merge_tool_declarations(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: Counter[tuple[str, str]] = Counter()
    for group in groups:
        for item in group or []:
            declaration = dict(item or {})
            name = str(declaration.get("name") or "").strip()
            binding = str(declaration.get("binding") or "").strip().lower()
            if not name:
                merged.append(declaration)
                continue
            token = (name, binding)
            if seen[token]:
                continue
            seen[token] += 1
            merged.append(declaration)
    return merged


def validate_tool_declarations(declarations: list[dict[str, Any]], *, owner: str) -> list[str]:
    warnings: list[str] = []
    collector = RuntimeWarnings(lambda message: warnings.append(message))
    for declaration in declarations:
        _resolve_tool_declaration(declaration, owner=owner, warnings=collector)
    return warnings


def _extract_float(description: str, key: str, *, default: float) -> float:
    raw = _extract_setting(description, key)
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _extract_setting(description: str, key: str) -> str | None:
    for chunk in str(description or "").splitlines():
        for piece in chunk.split(";"):
            piece = piece.strip()
            if not piece or "=" not in piece:
                continue
            current_key, value = piece.split("=", 1)
            if current_key.strip().lower() == key.lower():
                return value.strip()
    return None


def _build_api_tool(
    name: str,
    description: str,
    *,
    config: dict[str, Any] | None = None,
    owner: str,
    warnings: RuntimeWarnings,
) -> Callable[..., Any] | None:
    try:
        import requests
    except ImportError:
        warnings.add(f"{owner}: tool '{name}' uses binding 'api' but Python package 'requests' is not installed")
        return None

    config = dict(config or {})
    method = str(config.get("method") or _extract_setting(description, "method") or "POST").strip().upper()
    url = str(config.get("url") or _extract_setting(description, "url") or "").strip()
    timeout_raw = config.get("timeout")
    timeout = float(timeout_raw) if timeout_raw not in (None, "") else _extract_float(description, "timeout", default=20.0)
    if not url:
        warnings.add(f"{owner}: api tool '{name}' requires url=<endpoint> in description")
        return None

    raw_headers = config.get("headers")
    raw_static_params = config.get("params")
    raw_static_body = config.get("body")
    response_mode = str(config.get("response") or _extract_setting(description, "response") or "json").strip().lower()

    static_headers = _parse_json_mapping_value(raw_headers, owner=owner, label=f"api tool '{name}' headers", warnings=warnings)
    static_params = _parse_json_mapping_value(raw_static_params, owner=owner, label=f"api tool '{name}' params", warnings=warnings)
    static_body = _parse_json_value_any(raw_static_body, owner=owner, label=f"api tool '{name}' body", warnings=warnings)

    def api_tool(**arguments):
        request_url = _render_api_template(url, arguments)
        headers = {key: _render_api_template(str(value), arguments) for key, value in static_headers.items()}
        params = {key: _render_api_template(str(value), arguments) for key, value in static_params.items()}
        payload = _render_json_payload(static_body, arguments)
        if payload is None and arguments:
            payload = dict(arguments)

        request_kwargs: dict[str, Any] = {
            "headers": headers or None,
            "params": params or None,
            "timeout": timeout,
        }
        if method in {"GET", "DELETE"}:
            if payload is not None and not params:
                request_kwargs["params"] = payload
        else:
            request_kwargs["json"] = payload

        response = requests.request(method, request_url, **request_kwargs)
        response.raise_for_status()
        if response_mode == "text":
            return {"status_code": response.status_code, "text": response.text}
        try:
            data = response.json()
        except ValueError:
            data = {"text": response.text}
        return {
            "status_code": response.status_code,
            "data": data,
        }

    api_tool.__name__ = f"api_{_safe_callable_name(name)}"
    api_tool.__doc__ = f"HTTP API tool '{name}'."
    return api_tool


def _parse_json_mapping_value(raw: Any, *, owner: str, label: str, warnings: RuntimeWarnings) -> dict[str, str]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    parsed = _parse_json_value_any(raw, owner=owner, label=label, warnings=warnings)
    if isinstance(parsed, dict):
        return {str(key): str(value) for key, value in parsed.items()}
    warnings.add(f"{owner}: {label} must be a JSON object")
    return {}


def _parse_json_value_any(raw: Any, *, owner: str, label: str, warnings: RuntimeWarnings) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (dict, list, int, float, bool)):
        return raw
    raw = str(raw).strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        warnings.add(f"{owner}: {label} must be valid JSON")
        return None


def _render_json_payload(payload: Any, arguments: dict[str, Any]) -> Any:
    if payload is None:
        return None
    if isinstance(payload, str):
        return _render_api_template(payload, arguments)
    if isinstance(payload, list):
        return [_render_json_payload(item, arguments) for item in payload]
    if isinstance(payload, dict):
        return {str(key): _render_json_payload(value, arguments) for key, value in payload.items()}
    return payload


def _render_api_template(template: str, arguments: dict[str, Any]) -> str:
    rendered = str(template)
    for key, value in arguments.items():
        rendered = rendered.replace("{" + str(key) + "}", str(value))
    return rendered


def _safe_callable_name(name: str) -> str:
    chars = []
    for char in str(name):
        if char.isalnum() or char == "_":
            chars.append(char.lower())
        else:
            chars.append("_")
    collapsed = "".join(chars).strip("_")
    return collapsed or "tool"
