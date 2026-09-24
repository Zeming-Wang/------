from __future__ import annotations

from collections import defaultdict
from typing import Callable

from masfactory.compatibility.common.blueprint import ENTRY_TOKEN, EXIT_TOKEN, ExternalEdge, GraphBlueprint
from masfactory.compatibility.common.builder import _dedupe_wiring_edges, _resolve_edge_endpoint
from masfactory.compatibility.common.loops import LoopPlan


def connect_loop_controller(
    loop,
    plan: LoopPlan,
    inner_instances: dict,
) -> None:
    controller_from: set[str] = set()
    for src, dst in plan.back_edges:
        loop.edge_to_controller(inner_instances[src], keys={})
        controller_from.add(dst)
    for entry in plan.entry_ids:
        controller_from.add(entry)
    for dst in controller_from:
        loop.edge_from_controller(inner_instances[dst], keys={})
    for src, _dst in plan.exit_edges:
        loop.edge_to_terminate_node(inner_instances[src], keys={})


def build_loop_aware_wire_pairs(
    blueprint: GraphBlueprint,
    *,
    loop_plans: list[LoopPlan],
    looped_ids: frozenset[str],
    plan_for_member: dict[str, LoopPlan],
    member_id_set: set[str],
    skip_edge_sources: set[str] | None = None,
) -> tuple[list[tuple[str, str]], list[dict]]:
    skip_edge_sources = skip_edge_sources or set()
    node_id_set = {n.id for n in blueprint.nodes}
    exit_pairs = {(s, t) for p in loop_plans for s, t in p.exit_edges}
    enter_pairs = {(s, t) for p in loop_plans for s, t in p.enter_edges}

    wire_pairs: list[tuple[str, str]] = []
    for edge in blueprint.edges:
        orig_src = edge.source
        orig_dst = edge.target
        src = _resolve_edge_endpoint(orig_src, node_id_set)
        dst = _resolve_edge_endpoint(orig_dst, node_id_set)
        if orig_src in looped_ids and orig_dst in looped_ids:
            if plan_for_member.get(orig_src) is plan_for_member.get(orig_dst):
                continue
        if orig_src in looped_ids and orig_dst not in looped_ids and (orig_src, orig_dst) in exit_pairs:
            continue
        if orig_src not in looped_ids and orig_dst in looped_ids and (orig_src, orig_dst) in enter_pairs:
            continue
        if src == ENTRY_TOKEN and dst in looped_ids:
            dst = plan_for_member[dst].loop_id
        if dst == EXIT_TOKEN and src in looped_ids:
            src = plan_for_member[src].loop_id
        if orig_src in looped_ids:
            src = plan_for_member[orig_src].loop_id
        if orig_dst in looped_ids:
            dst = plan_for_member[orig_dst].loop_id
        if src == ENTRY_TOKEN and dst in member_id_set:
            wire_pairs.append((ENTRY_TOKEN, dst))
        elif dst == EXIT_TOKEN and src in member_id_set:
            wire_pairs.append((src, EXIT_TOKEN))
        elif src in member_id_set and dst in member_id_set:
            if src in skip_edge_sources:
                continue
            wire_pairs.append((src, dst))

    for plan in loop_plans:
        loop_id = plan.loop_id
        for src_out, _dst_in in plan.enter_edges:
            mapped_src = plan_for_member[src_out].loop_id if src_out in looped_ids else src_out
            wire_pairs.append((mapped_src, loop_id))
        for _src_in, dst_out in plan.exit_edges:
            mapped_dst = plan_for_member[dst_out].loop_id if dst_out in looped_ids else dst_out
            wire_pairs.append((loop_id, mapped_dst))

    incoming: defaultdict[str, int] = defaultdict(int)
    outgoing: defaultdict[str, int] = defaultdict(int)
    from_entry: set[str] = set()
    to_exit: set[str] = set()
    for src, dst in wire_pairs:
        if src == ENTRY_TOKEN:
            from_entry.add(dst)
        elif dst == EXIT_TOKEN:
            to_exit.add(src)
        else:
            incoming[dst] += 1
            outgoing[src] += 1

    for nid in member_id_set:
        if incoming[nid] == 0 and nid not in from_entry:
            wire_pairs.append((ENTRY_TOKEN, nid))
        if outgoing[nid] == 0 and nid not in to_exit:
            wire_pairs.append((nid, EXIT_TOKEN))

    return _dedupe_wiring_edges(wire_pairs)


def apply_wire_pairs(
    graph,
    wire_pairs: list[tuple[str, str]],
    resolve_instance: Callable[[str], object],
    *,
    error_label: str = "workflow",
) -> None:
    from masfactory.compatibility.errors import CompatibilityImportError

    try:
        for src, dst in wire_pairs:
            if src == ENTRY_TOKEN:
                graph.edge_from_entry(resolve_instance(dst), keys={})
            elif dst == EXIT_TOKEN:
                graph.edge_to_exit(resolve_instance(src), keys={})
            else:
                graph.create_edge(resolve_instance(src), resolve_instance(dst), keys={})
    except ValueError as exc:
        raise CompatibilityImportError(
            f"Failed to wire {error_label} graph edges (cycle, duplicate edge, or invalid topology). "
            f"Original error: {exc}"
        ) from exc
