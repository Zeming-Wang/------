from masfactory import Graph, LogicSwitch, Loop, NodeTemplate, RootGraph, CustomNode
from copy import deepcopy
from typing import Any

from ddo.graphs.g_confidence_estimation import g_confidence_estimation
from ddo.graphs.g_actions_generation_and_selection import g_actions_generation_and_selection
from ddo.graphs.g_response_simulation import g_response_simulation
from ddo.runtime.app_runtime import app_runtime
from ddo.state_model.state import ConsultationState

MAX_CONSULTATION_LOOP_ITERATIONS = 20

# Utils
def ensure_case_attrs(attrs: dict[str, Any]) -> None:
    attrs.setdefault("initial_diagnostic_confidence", None)
    attrs.setdefault("interactions", [])
    attrs.setdefault("warnings", [])

# CustomNode
def capture_initial_confidence(d: dict[str, Any], attrs: dict[str, Any]) -> dict[str, Any]:
    ensure_case_attrs(attrs)

    state: ConsultationState = attrs["state"]

    state.diagnosis_state.initial_diagnostic_confidence = deepcopy(
        state.diagnosis_state.normalized_diagnosis_confidence
    )

    return {}

def reset_round_state(d: dict[str, Any], attrs: dict[str, Any]) -> dict[str, Any]:
    ensure_case_attrs(attrs)

    state: ConsultationState = attrs["state"]
    state.reset_round_state()

    return {}

def collect_round_snapshot(d: dict[str, Any], attrs: dict[str, Any]) -> dict[str, Any]:
    ensure_case_attrs(attrs)

    state: ConsultationState = attrs["state"]

    # termination action 不算一次有效问诊轮次
    if state.control_state.should_terminate:
        return {}

    # 没有实际选择症状，也不算一次有效问诊轮次
    if state.inquiry_state.selected_symptom is None:
        return {}
    
    attrs["interactions"].append({
        "turn_id": state.control_state.turn_id,
        "retry_history": deepcopy(state.inquiry_state.retry_history),

        "candidate_symptoms": deepcopy(state.inquiry_state.candidate_symptoms),
        "selection_reasoning": state.inquiry_state.selection_reasoning,
        "selected_symptom": state.inquiry_state.selected_symptom,
        "is_recorded": state.response_state.is_recorded,
        "response_reasoning": state.response_state.response_reasoning,
        "symptom_status": state.response_state.symptom_status,
        "response_confidence": state.response_state.response_source,

        "diagnostic_confidence": deepcopy(state.diagnosis_state.normalized_diagnosis_confidence),

        "selection_raw_output": state.inquiry_state.raw_output,
        "response_raw_output": state.response_state.raw_output,
        "btp_debug": deepcopy(state.diagnosis_state.btp_debug),
    })

    return {}

# Condition
def consultation_terminate_cond(d: dict[str, Any], attrs: dict[str, Any]) -> bool:
    state: ConsultationState = attrs["state"]

    # should_terminate 由 Policy Agent 决定，优先级最高
    if state.control_state.should_terminate:
        return True
    
    if state.control_state.turn_id >= app_runtime.cfg.max_turns:
        return True
    
    return False

def route_to_skip_response_and_confidence(d: dict[str, Any], attrs: dict[str, Any]) -> bool:
    state: ConsultationState = attrs["state"]

    if state.control_state.should_terminate:
        return True
    
    if state.inquiry_state.selected_symptom is None:
        return True
    
    return False

def route_to_response_after_action(d: dict[str, Any], attrs: dict[str, Any]) -> bool:
    return not route_to_skip_response_and_confidence(d, attrs)

# LogicSwitch
after_action_router = NodeTemplate(
    LogicSwitch,
    routes={
        "collect_round_snapshot":route_to_skip_response_and_confidence,
        "g_response_simulation":route_to_response_after_action,
    }
)

# Graph
round_step_graph = NodeTemplate(
    Graph,
    nodes = [
        ("g_actions_generation_and_selection", g_actions_generation_and_selection),
        ("after_action_router", after_action_router),

        ("g_response_simulation", g_response_simulation),
        ("g_confidence_estimation", g_confidence_estimation),

        ("collect_round_snapshot", CustomNode, collect_round_snapshot),
    ],
    edges = [
        ("ENTRY", "g_actions_generation_and_selection", {}),

        ("g_actions_generation_and_selection", "after_action_router", {}),

        # branch 1: termination
        ("after_action_router", "collect_round_snapshot", {}),
        # branch 2: normal response + confidence update
        ("after_action_router", "g_response_simulation", {}),
        ("g_response_simulation", "g_confidence_estimation", {}),
        ("g_confidence_estimation", "collect_round_snapshot", {}),

        ("collect_round_snapshot", "EXIT", {}),
    ],
)

consultation_loop = NodeTemplate(
    Loop,
    max_iterations=MAX_CONSULTATION_LOOP_ITERATIONS,
    terminate_condition_function=consultation_terminate_cond,
    nodes = [
        ("reset_round_state", CustomNode,reset_round_state),
        ("round_step_graph", round_step_graph),
    ],
    edges = [
        ("CONTROLLER", "reset_round_state", {}),
        ("reset_round_state", "round_step_graph", {}),
        ("round_step_graph", "CONTROLLER", {}),
    ],
)

g_consultation = RootGraph(
    name="g_consultation",
    nodes=[
        ("g_initial_confidence_estimation", g_confidence_estimation),
        ("capture_initial_confidence", CustomNode, capture_initial_confidence),

        ("consultation_loop", consultation_loop),
    ],
    edges=[
        ("ENTRY", "g_initial_confidence_estimation", {}),
        ("g_initial_confidence_estimation", "capture_initial_confidence", {}),
        ("capture_initial_confidence", "consultation_loop", {}),
        ("consultation_loop", "EXIT", {}),
    ],
)
