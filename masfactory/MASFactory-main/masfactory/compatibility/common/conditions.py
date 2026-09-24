from __future__ import annotations

from typing import Any

from masfactory.compatibility.common.blueprint import ExternalEdge


def is_always_true_condition(cond: Any) -> bool:
    return cond is None or cond is True or cond == "true"


def edge_condition_value(edge: ExternalEdge) -> Any:
    raw = edge.raw if isinstance(edge.raw, dict) else {}
    return raw.get("condition", edge.source_handle)


def is_conditional_edge(edge: ExternalEdge) -> bool:
    return not is_always_true_condition(edge_condition_value(edge))


def needs_logic_switch(edges: list[ExternalEdge]) -> bool:
    return sum(1 for e in edges if is_conditional_edge(e)) > 1
