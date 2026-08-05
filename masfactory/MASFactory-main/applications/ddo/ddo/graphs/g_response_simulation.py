import re
from masfactory import Graph, CustomNode, NodeTemplate, LogicSwitch
from ddo.runtime.app_runtime import app_runtime
from ddo.state_model.state import ConsultationState

# Utils 
def update_symptom_state(state: ConsultationState, selected_symptom: str, symptom_status: bool) -> None:
    diagnosis_state = state.diagnosis_state

    positive_symptoms = diagnosis_state.positive_symptoms
    negative_symptoms = diagnosis_state.negative_symptoms

    if symptom_status is True:
        if selected_symptom not in positive_symptoms:
            positive_symptoms.append(selected_symptom)
        if selected_symptom in negative_symptoms:
            negative_symptoms.remove(selected_symptom)

    else:
        if selected_symptom not in negative_symptoms:
            negative_symptoms.append(selected_symptom)
        if selected_symptom in positive_symptoms:
            positive_symptoms.remove(selected_symptom)

def append_patient_trace(
    *,
    state: ConsultationState,
    selected_symptom: str,
    symptom_status: bool,
    response_source: str,
    raw_output: str | None,
) -> None:
    control_state = state.control_state
    diagnosis_state = state.diagnosis_state
    inquiry_state = state.inquiry_state

    state.trace.append({
        "turn_id": control_state.turn_id,
        "selected_symptom": selected_symptom,
        "patient_response": symptom_status,
        "patient_response_source": response_source,
        "patient_raw_output": raw_output,
        "candidate_symptoms": inquiry_state.candidate_symptoms,
        "candidate_actions_ids": inquiry_state.candidate_actions_ids,
        "has_termination_action": inquiry_state.has_termination_action,
        "selection_reasoning": inquiry_state.selection_reasoning,
        "diagnosis_confidence": diagnosis_state.normalized_diagnosis_confidence.copy(),
        "raw_diagnosis_confidence": diagnosis_state.raw_diagnosis_confidence.copy(),
        "positive_symptoms_after_response": diagnosis_state.positive_symptoms.copy(),
        "negative_symptoms_after_response": diagnosis_state.negative_symptoms.copy(),
    })

def parse_patient_bool_output(text: str | bool | None) -> bool | None:
    if text is None:
        return None

    if isinstance(text, bool):
        return text

    text = str(text).strip()
    if not text:
        return None

    patterns = [
        r"最终判断结果\s*[为是:：]?\s*(True|False)",
        r"判断结果\s*[为是:：]?\s*(True|False)",
        r"最终结果\s*[为是:：]?\s*(True|False)",
        r"结果\s*[为是:：]?\s*(True|False)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].lower() == "true"

    # 兜底：匹配中文前后的 True / False
    matches = re.findall(
        r"(?<![A-Za-z])(True|False)(?![A-Za-z])",
        text,
        flags=re.IGNORECASE,
    )
    if matches:
        return matches[-1].lower() == "true"

    return None

# Custom Nodes 
def prepare_patient_response_input(d: dict, attrs: dict) -> dict:
    state = attrs["state"]

    inquiry_state = state.inquiry_state

    selected_symptom = inquiry_state.selected_symptom
    disease_label = state.case_state.disease_label
    disease_knowledge = app_runtime.dataset_ctx.disease_knowledge

    empirical_knowledge = disease_knowledge[disease_label]["empirical_knowledge"]

    return {
        "disease_label": disease_label,
        "empirical_knowledge": empirical_knowledge,
        "inquiried_symptom": selected_symptom,
    }

def mcr_record(d: dict, attrs: dict) -> dict:
    state = attrs["state"]

    inquiry_state = state.inquiry_state

    selected_symptom = inquiry_state.selected_symptom
    all_symptoms = state.case_state.all_symptoms

    symptom_status = bool(all_symptoms.get(selected_symptom, False))

    update_symptom_state(
        state=state,
        selected_symptom=selected_symptom,
        symptom_status=symptom_status,
    ) 

    state.response_state.symptom_status = symptom_status
    state.response_state.is_recorded = True
    state.response_state.response_reasoning = "mcr_record"
    state.response_state.response_source = "mcr_record"

    state.control_state.turn_id = int(state.control_state.turn_id) + 1

    append_patient_trace(
        state=state,
        selected_symptom=selected_symptom,
        symptom_status=symptom_status,
        response_source="mcr_record",
        raw_output=None,
    )

    return {}

def parse_patient_response_result(d: dict, attrs: dict) -> dict:
    state = attrs["state"]

    inquiry_state = state.inquiry_state

    patient_output = d.get("patient_response", "")
    state.response_state.raw_output = patient_output

    selected_symptom = inquiry_state.selected_symptom

    symptom_status = parse_patient_bool_output(patient_output)
    response_source = "patient_agent"

    if symptom_status is None:
        symptom_status = False
        response_source = "llm_output_parsing_failed, default to False"
    
    update_symptom_state(
        state=state,
        selected_symptom=selected_symptom,
        symptom_status=symptom_status,
    )

    state.response_state.symptom_status = symptom_status
    state.response_state.is_recorded = False
    state.response_state.response_reasoning = patient_output
    state.response_state.response_source = response_source

    state.control_state.turn_id = int(state.control_state.turn_id) + 1

    append_patient_trace(
        state=state,
        selected_symptom=selected_symptom,
        symptom_status=symptom_status,
        response_source=response_source,
        raw_output=patient_output,
    )

    return {}

def frequency_based_response(d: dict, attrs: dict) -> dict:
    state = attrs["state"]
    selected_symptom = state.inquiry_state.selected_symptom
    symptom_status = False

    update_symptom_state(
        state=state,
        selected_symptom=selected_symptom,
        symptom_status=symptom_status,
    )

    state.response_state.symptom_status = symptom_status
    state.response_state.is_recorded = False
    state.response_state.response_reasoning = "frequency too low"
    state.response_state.response_source = "frequency_based_response"

    state.control_state.turn_id = int(state.control_state.turn_id) + 1

    append_patient_trace(
        state=state,
        selected_symptom=selected_symptom,
        symptom_status=symptom_status,
        response_source="frequency_based_response",
        raw_output=None,
    )

    return {}

# Conditions
def route_to_mcr_record(msg: dict, attrs: dict) -> bool:
    state = attrs["state"]

    inquiry_state = state.inquiry_state

    selected_symptom = inquiry_state.selected_symptom
    all_symptoms = state.case_state.all_symptoms

    return selected_symptom in all_symptoms

def route_to_frequency_router(msg: dict, attrs: dict) -> bool:
    return not route_to_mcr_record(msg, attrs)

def route_to_frequency_based_response(msg: dict, attrs: dict) -> bool:  
    state = attrs["state"]
    selected_symptom = state.inquiry_state.selected_symptom

    frequency = app_runtime.dataset_ctx.disease_knowledge[state.case_state.disease_label]["empirical_knowledge"].get(selected_symptom, 0.0)

    return frequency < app_runtime.cfg.frequency_threshold - 0.05

def route_to_patient_agent(msg: dict, attrs: dict) -> bool:
    return not route_to_frequency_based_response(msg, attrs)

# Graphs
mcr_router = NodeTemplate(
    LogicSwitch,
    routes={
        "mcr_record": route_to_mcr_record,
        "frequency_router": route_to_frequency_router,
    },
)

frequency_router = NodeTemplate(
    LogicSwitch,
    routes={
        "frequency_based_response": route_to_frequency_based_response,
        "prepare_patient_response_input": route_to_patient_agent,
    },
)

patient_agent = app_runtime.base_agent(
    instructions="""你是一个患者模拟器，你所模拟的患者真实患有的疾病为'{disease_label}'。

疾病'{disease_label}'的症状知识如下：
- 基于已有的一些诊断结果为'{disease_label}'的病例统计的症状出现频率：{empirical_knowledge}

你需要根据疾病的症状知识，判断患者是否有较大的可能性存在症状'{inquiried_symptom}'，判断结果只有True和False两种，True表示患者有较大的可能性存在该症状，False表示患者不太可能存在该症状。判断的标准如下：
- 如果该症状在疾病'{disease_label}'中较为典型（该症状在该疾病中出现频率的排名比较靠前），则可以认为患者有较大的可能性存在该症状，判断结果为True。
- 如果该症状在疾病'{disease_label}'中不典型，则认为患者不太可能存在该症状，判断结果为False。
    
请你一步步思考，判断患者是否有较大的可能性存在症状'{inquiried_symptom}'。""".strip(),
)

g_response_simulation = NodeTemplate(
    Graph,
    nodes=[
        ("mcr_router", mcr_router),
        ("mcr_record", CustomNode, mcr_record),

        ("frequency_router", frequency_router),
        ("frequency_based_response", CustomNode, frequency_based_response),

        ("prepare_patient_response_input", CustomNode, prepare_patient_response_input),
        ("patient_agent", patient_agent),
        ("parse_patient_response_result", CustomNode, parse_patient_response_result),
    ],
    edges=[
        ("ENTRY", "mcr_router", {}),

        # mcr_branch 1
        ("mcr_router", "mcr_record", {}),
        ("mcr_record", "EXIT", {}),

        # mcr_branch 2
        ("mcr_router", "frequency_router", {}),

        # frequency_branch 1
        ("frequency_router", "frequency_based_response", {}),
        ("frequency_based_response", "EXIT", {}),

        # frequency_branch 2
        ("frequency_router", "prepare_patient_response_input", {}),
        ("prepare_patient_response_input", "patient_agent", {
            "disease_label": "患者真实患有的疾病",
            "empirical_knowledge": "疾病的临床表现知识",
            "inquiried_symptom": "被询问的症状",
        }),
        ("patient_agent", "parse_patient_response_result", {
            "patient_response": "Patient Agent 的回答",
        }),
        ("parse_patient_response_result", "EXIT", {}),
    ]
)
