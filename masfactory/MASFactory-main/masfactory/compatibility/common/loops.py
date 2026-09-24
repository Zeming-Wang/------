from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from masfactory.compatibility.common.blueprint import ENTRY_TOKEN, EXIT_TOKEN, ExternalEdge, ExternalNode, GraphBlueprint


@dataclass(frozen=True)
class LoopPlan:
    """A strongly-connected region compiled into one MASFactory `Loop` node."""

    loop_id: str
    member_ids: frozenset[str]
    entry_ids: frozenset[str]
    internal_edges: tuple[ExternalEdge, ...]
    back_edges: tuple[tuple[str, str], ...]
    enter_edges: tuple[tuple[str, str], ...]
    exit_edges: tuple[tuple[str, str], ...]
    loop_counter_ids: frozenset[str] = field(default_factory=frozenset)
    max_iterations: int = 20


def _internal_edges(blueprint: GraphBlueprint) -> list[ExternalEdge]:
    node_ids = {n.id for n in blueprint.nodes}
    out: list[ExternalEdge] = []
    for edge in blueprint.edges:
        src = edge.source
        dst = edge.target
        if src in node_ids and dst in node_ids:
            out.append(edge)
    return out


def _tarjan_scc(adj: dict[str, list[str]]) -> list[list[str]]:
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    result: list[list[str]] = []

    def strongconnect(v: str) -> None:
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)
        for w in adj.get(v, []):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])
        if lowlink[v] == indices[v]:
            component: list[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                component.append(w)
                if w == v:
                    break
            result.append(component)

    for node in adj:
        if node not in indices:
            strongconnect(node)
    return result


def _detect_back_edges(
    member_ids: set[str],
    edges: list[ExternalEdge],
    entry_ids: set[str],
) -> set[tuple[str, str]]:
    """DFS back-edges on the subgraph induced by member_ids."""
    adj: dict[str, list[str]] = {n: [] for n in member_ids}
    for e in edges:
        if e.source in member_ids and e.target in member_ids:
            adj[e.source].append(e.target)

    back: set[tuple[str, str]] = set()
    visited: set[str] = set()
    stack: set[str] = set()

    def dfs(u: str) -> None:
        visited.add(u)
        stack.add(u)
        for v in adj.get(u, []):
            if v not in visited:
                dfs(v)
            elif v in stack:
                back.add((u, v))
        stack.remove(u)

    for start in entry_ids or member_ids:
        if start not in visited:
            dfs(start)
    for n in member_ids:
        if n not in visited:
            dfs(n)
    return back


def _loop_counter_max(member_ids: set[str], nodes_by_id: dict[str, ExternalNode]) -> tuple[frozenset[str], int]:
    counter_ids: set[str] = set()
    max_it = 20
    for nid in member_ids:
        ext = nodes_by_id.get(nid)
        if ext is None or ext.kind != "loop_counter":
            continue
        counter_ids.add(nid)
        cfg = ext.raw.get("config") if isinstance(ext.raw.get("config"), dict) else {}
        raw_max = cfg.get("max_iterations") or cfg.get("limit") or cfg.get("max")
        try:
            max_it = max(max_it, int(raw_max))
        except (TypeError, ValueError):
            pass
    return frozenset(counter_ids), max_it


def plan_loop_regions(blueprint: GraphBlueprint) -> list[LoopPlan]:
    """Find cyclic SCCs in a ChatDev blueprint and describe how to lower them to `Loop` nodes."""
    if not blueprint.nodes:
        return []

    nodes_by_id = {n.id: n for n in blueprint.nodes}
    node_ids = set(nodes_by_id)
    internal = _internal_edges(blueprint)

    adj: dict[str, list[str]] = {n: [] for n in node_ids}
    for e in internal:
        adj[e.source].append(e.target)

    plans: list[LoopPlan] = []
    seen_members: set[str] = set()

    for component in _tarjan_scc(adj):
        members = set(component)
        if len(members) == 1:
            only = next(iter(members))
            if not any(e.source == only and e.target == only for e in internal):
                continue
        if members & seen_members:
            continue
        seen_members |= members

        comp_edges = [e for e in internal if e.source in members and e.target in members]

        enter_edges = [
            (e.source, e.target)
            for e in blueprint.edges
            if e.target in members
            and e.source in node_ids
            and e.source not in members
            and e.source not in {ENTRY_TOKEN, EXIT_TOKEN}
        ]
        exit_edges = [
            (e.source, e.target)
            for e in blueprint.edges
            if e.source in members
            and e.target in node_ids
            and e.target not in members
            and e.target not in {ENTRY_TOKEN, EXIT_TOKEN}
        ]

        internal_targets = {e.target for e in comp_edges}
        entry_ids = {n for n in members if n not in internal_targets}
        if not entry_ids:
            entry_ids = set(members)

        back_edges = _detect_back_edges(members, comp_edges, entry_ids)
        internal_acyclic = tuple(
            e for e in comp_edges if (e.source, e.target) not in back_edges
        )

        counter_ids, max_it = _loop_counter_max(members, nodes_by_id)
        loop_id = f"loop_{min(members, key=lambda x: (len(x), x))}"

        plans.append(
            LoopPlan(
                loop_id=loop_id,
                member_ids=frozenset(members),
                entry_ids=frozenset(entry_ids),
                internal_edges=internal_acyclic,
                back_edges=tuple(sorted(back_edges)),
                enter_edges=tuple(enter_edges),
                exit_edges=tuple(exit_edges),
                loop_counter_ids=counter_ids,
                max_iterations=max_it,
            )
        )

    return plans


def nodes_in_loops(plans: list[LoopPlan]) -> frozenset[str]:
    out: set[str] = set()
    for plan in plans:
        out |= set(plan.member_ids)
    return frozenset(out)
