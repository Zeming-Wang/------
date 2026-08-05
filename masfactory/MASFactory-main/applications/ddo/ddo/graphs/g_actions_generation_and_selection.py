from masfactory import Graph, CustomNode, LogicSwitch, Loop, NodeTemplate
from ddo.runtime.app_runtime import app_runtime
from ddo.state_model.state import ConsultationState
from collections import Counter
from typing import Any

RETRY_LOOP_MAX_ITERATIONS = 5

# == utils ==
def get_sorted_diagnosis_confidence(state: ConsultationState) -> dict:
    conf = state.diagnosis_state.normalized_diagnosis_confidence

    return dict(
        sorted(
            conf.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

def find_substr_close_to_end_local(s: str, candidates: list[str]) -> str:
    selected = ""
    selected_pos = -1

    for symptom in candidates:
        pos = s.rfind(symptom)
        if pos > selected_pos:
            selected = symptom
            selected_pos = pos

    return selected

def fallback_select_by_frequency(state: ConsultationState) -> str | None:
    inquiry_state = state.inquiry_state
    candidate_symptoms = inquiry_state.candidate_symptoms
    candidate_actions_ids = inquiry_state.candidate_actions_ids

    index_to_symptom = app_runtime.sampler.index_to_symptom
    symptom_num = app_runtime.sampler.symptom_num

    candidate_symptoms_list = []
    
    for action_id in candidate_actions_ids:
        if 0 <= action_id < symptom_num:
            symptom = index_to_symptom[action_id]
            if symptom in candidate_symptoms:
                candidate_symptoms_list.append(symptom)
    
    if not candidate_symptoms_list:
        return candidate_symptoms[0] if candidate_symptoms else None

    counter = Counter(candidate_symptoms_list)
    return max(counter, key=lambda k: counter[k])

def handle_retry_or_fallback(
    *,
    state: ConsultationState,
    reason: str
) -> dict[str, Any]:
    inquiry_state = state.inquiry_state

    retry_count = int(inquiry_state.retry_count) + 1
    max_retry = int(app_runtime.cfg.max_retry)

    candidate_actions_ids = inquiry_state.candidate_actions_ids
    candidate_symptoms = inquiry_state.candidate_symptoms

    # 无法选出的候选动作id加入mask，避免下一轮继续被选中
    old_mask = inquiry_state.extra_masking_action_ids
    new_mask = [
        int(action_id)
        for action_id in candidate_actions_ids
        if 0 <= int(action_id) < app_runtime.sampler.symptom_num
    ]

    inquiry_state.extra_masking_action_ids = sorted(set(old_mask + new_mask))
    inquiry_state.retry_count = retry_count
    inquiry_state.selection_reasoning = reason

    # 正常retry
    if retry_count <= max_retry:
        inquiry_state.selected_symptom = None
        inquiry_state.selected_action_id = None
        inquiry_state.need_retry = True
        state.control_state.should_terminate = False

        # 记录 retry 历史
        inquiry_state.retry_history.append({
            "retry_index": retry_count,
            "retry_reason": reason,
            "candidate_actions_ids": candidate_actions_ids,
            "candidate_symptoms": candidate_symptoms,
            "llm_raw_output": inquiry_state.raw_output,
        })

        return {}
    
    # retry 超出次数，fallback
    fallback_symptom = fallback_select_by_frequency(state)

    inquiry_state.selected_symptom = fallback_symptom
    inquiry_state.selected_action_id = (
        app_runtime.sampler.symptom_index_dict.get(fallback_symptom)
        if fallback_symptom is not None
        else None
    )
    inquiry_state.need_retry = False

    if fallback_symptom is None:
        inquiry_state.selection_reasoning = f"Retry 超过最大次数，进行 fallback 失败，终止"
        state.control_state.should_terminate = True
    else:
        inquiry_state.selection_reasoning = f"Retry 超过最大次数，进行 fallback"
        state.control_state.should_terminate = False

    return {}

def format_symptoms_for_prompt(symptoms: list[str]) -> str:
    return "、".join(symptoms) if symptoms else "无"

def should_accept_termination_action(state: ConsultationState) -> bool:
    inquiry_state = state.inquiry_state
    control_state = state.control_state

    has_termination_action = inquiry_state.has_termination_action
    candidate_symptoms = inquiry_state.candidate_symptoms

    if not has_termination_action:
        return False

    current_turn = int(control_state.turn_id)
    floor_turns = int(app_runtime.cfg.floor_turns)

    # 如果全是 termination action，则直接终止
    if not candidate_symptoms:
        return True
    # 如果采到 termination action，且已经达到 floor_turns，则终止
    if current_turn >= floor_turns:
        return True
    
    return False

# == Custom Nodes ==
def init_retry_loop_state(d: dict, attrs: dict) -> dict:
    state = attrs["state"]
    inquiry_state = state.inquiry_state

    inquiry_state.retry_count = 0
    inquiry_state.need_retry = False
    inquiry_state.extra_masking_action_ids = []
    inquiry_state.retry_history = []

    inquiry_state.selected_action_id = None
    inquiry_state.selected_symptom = None
    inquiry_state.selection_reasoning = None

    return {}

def prepare_inquiry_selection_input(d: dict[str, Any], attrs: dict) -> dict[str, Any]:
    state = attrs["state"]

    diagnosis_state = state.diagnosis_state
    inquiry_state = state.inquiry_state
    candidate_symptoms = inquiry_state.candidate_symptoms

    # 选取top diseases
    window_size = app_runtime.sampler.window_size
    sorted_conf = get_sorted_diagnosis_confidence(state)
    top_diseases = list(sorted_conf.keys())[:window_size]

    top_diseases_diagnostic_confidence = {
        disease: sorted_conf[disease]
        for disease in top_diseases
    }

    top_diseases_diagnostic_confidence = {
        disease: round(confidence, 2)
        for disease, confidence in top_diseases_diagnostic_confidence.items()
    }

    # 归一化
    total_confidence = sum(top_diseases_diagnostic_confidence.values())
    if total_confidence > 1e-8:
        top_diseases_diagnostic_confidence = {
            disease: round(confidence / total_confidence, 2)
            for disease, confidence in top_diseases_diagnostic_confidence.items()
        }

    # 收集top diseases的临床表现知识
    disease_knowledge = app_runtime.dataset_ctx.disease_knowledge
    top_diseases_empirical_knowledge = {
        disease: disease_knowledge[disease]["empirical_knowledge"]
        for disease in top_diseases
    }

    return {
        "positive_symptoms": format_symptoms_for_prompt(diagnosis_state.positive_symptoms),
        "negative_symptoms": format_symptoms_for_prompt(diagnosis_state.negative_symptoms),
        "top_diseases_diagnostic_confidence": top_diseases_diagnostic_confidence,
        "top_diseases_empirical_knowledge": top_diseases_empirical_knowledge,
        "candidate_symptoms": candidate_symptoms,
    }

def parse_inquiry_selection_result(d: dict, attrs: dict) -> dict:
    state = attrs["state"]
    inquiry_state = state.inquiry_state

    # 应该是没用，但是不敢删
    if state.control_state.should_terminate:
        inquiry_state.need_retry = False
        return {}

    candidate_symptoms = inquiry_state.candidate_symptoms
    output_reasoning = d.get("selection_reasoning", "")
    output = d.get("selection_output", "")
    state.inquiry_state.raw_output = output_reasoning+"\n"+output
    if output is None:
        output = ""

    output = str(output)

    # 只有一个候选症状，直接选
    if len(candidate_symptoms) == 1:
        selected_symptom = candidate_symptoms[0]
        inquiry_state.selected_symptom = selected_symptom
        inquiry_state.selected_action_id = app_runtime.sampler.symptom_index_dict.get(selected_symptom)
        inquiry_state.selection_reasoning = "only_one_candidate"
        inquiry_state.need_retry = False
        state.control_state.should_terminate = False
        return {}
    # else为多个候选症状的情况，调用llm
    else:
        output_tail = output[-30:]

        selected_symptom = find_substr_close_to_end_local(
            output_tail,
            candidate_symptoms,
        )

        # llm解析失败，retry或者fallback
        if "需要重新提供候选症状" in output_tail:
            return handle_retry_or_fallback(
                state=state,
                reason="LLM认为当前候选症状不合适，要求重新提供候选症状。",
            )
        elif selected_symptom == "":
            parse_failed = True
            return handle_retry_or_fallback(
                state=state,
                reason="LLM解析症状选择结果失败，无法从输出中提取出合法的症状名称。",
            )

    # 只有多个候选症状且调用llm成功的情况下，才会走到这里
    inquiry_state.selected_symptom = selected_symptom
    inquiry_state.selected_action_id = app_runtime.sampler.symptom_index_dict.get(selected_symptom)
    inquiry_state.selection_reasoning = output_reasoning
    inquiry_state.need_retry = False
    state.control_state.should_terminate = False

    return {}

def prepare_for_termination(d: dict, attrs: dict) -> dict:
    inquiry_state = attrs["state"].inquiry_state
    control_state = attrs["state"].control_state

    inquiry_state.selected_action_id = app_runtime.sampler.symptom_num
    inquiry_state.selected_symptom = None
    inquiry_state.selection_reasoning = "termination_action_by_policy"
    inquiry_state.need_retry = False
    control_state.should_terminate = True
    # control_state.last_action_was_termination = True

    return {}

def generate_candidate_symptoms(d: dict[str, Any], attrs: dict) -> dict[str, Any]:
    state = attrs["state"]
    inquiry_state = state.inquiry_state

    result = app_runtime.sampler.sample_candidate_symptoms(
        state,
        extra_masking_action_ids=inquiry_state.extra_masking_action_ids,
    )

    inquiry_state.candidate_actions_ids = result["candidate_actions_ids"]
    inquiry_state.candidate_symptoms = result["candidate_symptoms"]
    inquiry_state.has_termination_action = result["has_termination_action"]
    inquiry_state.top_disease_by_policy_obs = result.get("top_disease_by_policy_obs")

    return {
        "candidate_symptoms": inquiry_state.candidate_symptoms,
        "candidate_actions_ids": inquiry_state.candidate_actions_ids,
        "has_termination_action": inquiry_state.has_termination_action,
    }

# Condition
def retry_terminate_cond(d: dict, attrs: dict) -> bool:
    state = attrs["state"]
    inquiry_state = state.inquiry_state

    if state.control_state.should_terminate:
        return True
    
    if inquiry_state.selected_symptom is not None:
        return True

    retry_count = int(inquiry_state.retry_count)

    # 为了第一次通过
    if retry_count == 0 and not inquiry_state.need_retry:
        return False
    
    if not inquiry_state.need_retry:
        return True
    
    max_retry = int(app_runtime.cfg.max_retry)

    # retry_count 不是retry之后才加1的
    return retry_count > max_retry

def route_to_llm(d: dict, attrs: dict) -> bool:
    if should_accept_termination_action(attrs["state"]):
        return False
    
    candidate_symptoms = d["candidate_symptoms"]
    return len(candidate_symptoms) > 1
    
def route_to_parse(d: dict, attrs: dict) -> bool:
    if should_accept_termination_action(attrs["state"]):
        return False

    candidate_symptoms = d["candidate_symptoms"]
    return len(candidate_symptoms) <= 1

def route_to_termination(d: dict, attrs: dict) -> bool:
    return should_accept_termination_action(attrs["state"])

# Other Nodes
candidate_generation_router = NodeTemplate(
    LogicSwitch,
    routes={
        "parse_inquiry_selection_result": route_to_parse,
        "prepare_inquiry_selection_input": route_to_llm,
        "prepare_for_termination": route_to_termination,
    }
)

inquiry_agent = app_runtime.base_agent(
    instructions="""你是一位经验丰富的医学专家，现在需要你帮助从给定的候选症状中选择一个接下来向患者询问的症状，以进一步收集患者的症状表现情况。
为你提供如下信息：
# 当前已知的患者的症状表现情况（已经询问过的部分症状）：
- 存在的症状：{positive_symptoms}。
- 不存在的症状：{negative_symptoms}。

# 疾病诊断置信度（置信度范围为0~1，数值越高表示该疾病的可能性越大）：
{top_diseases_diagnostic_confidence}

# 疾病的临床表现知识（基于历史病例统计的症状出现频率）：
{top_diseases_empirical_knowledge}

# 候选症状：
{candidate_symptoms}

根据上述信息，从候选症状中选择一个适合接下来向患者询问的症状。症状的选择策略如下：
- 策略1（优先）：若诊断置信度排名第一的疾病的诊断置信度明显高于其它疾病，则应从候选症状中选择一个在排名第一的疾病中较为典型的症状，从而有助于确认该疾病的可能性。
- 策略2：若候选症状中不存在符合策略1的症状，则应从候选症状中选择一个和患者的症状表现情况比较相关的症状。

输出格式：
- 若候选症状中存在符合上述策略的症状'xx'，则输出'选择xx作为接下来向患者询问的症状'。
- 若候选症状中无合适症状，则输出'需要重新提供候选症状'。
请你一步步思考。
    """.strip(),
)

# Graphs
inquiry_retry_loop = NodeTemplate(
    Loop,
    max_iterations=RETRY_LOOP_MAX_ITERATIONS,
    terminate_condition_function= retry_terminate_cond,
    nodes=[
        ("generate_candidate_symptoms", CustomNode, generate_candidate_symptoms),

        ("candidate_generation_router", candidate_generation_router), 
        ("prepare_for_termination", CustomNode, prepare_for_termination),

        ("prepare_inquiry_selection_input", CustomNode, prepare_inquiry_selection_input),
        ("inquiry_agent", inquiry_agent),
        ("parse_inquiry_selection_result", CustomNode, parse_inquiry_selection_result),
    ],
    edges=[
        ("CONTROLLER", "generate_candidate_symptoms", {}),
        ("generate_candidate_symptoms", "candidate_generation_router", {
            "candidate_symptoms": "候选症状列表",
        }),
        
        # switch 1
        ("candidate_generation_router", "prepare_inquiry_selection_input", {
            "candidate_symptoms": "候选症状列表",
        }),
        # switch 2
        ("candidate_generation_router", "parse_inquiry_selection_result", {
            "candidate_symptoms": "候选症状列表",
        }),
        # switch 3
        ("candidate_generation_router", "prepare_for_termination", {}),
        ("prepare_for_termination", "CONTROLLER", {}),

        ("prepare_inquiry_selection_input", "inquiry_agent", {
            "positive_symptoms": "患者已确认存在的症状",
            "negative_symptoms": "患者已确认不存在的症状",
            "top_diseases_diagnostic_confidence": "当前 top 疾病及诊断置信度",
            "top_diseases_empirical_knowledge": "当前 top 疾病的临床表现知识",
            "candidate_symptoms": "候选症状列表",
        }),
        ("inquiry_agent", "parse_inquiry_selection_result", {
            "selection_output": "Inquiry Agent 的症状选择结果",
        }),
        ("parse_inquiry_selection_result", "CONTROLLER", {}),
    ]
)
    
g_actions_generation_and_selection = NodeTemplate(
    Graph,
    nodes=[
        ("init_retry_loop_state", CustomNode, init_retry_loop_state),
        ("inquiry_retry_loop", inquiry_retry_loop),
    ],
    edges=[
        ("ENTRY", "init_retry_loop_state", {}),
        ("init_retry_loop_state", "inquiry_retry_loop", {}),
        ("inquiry_retry_loop", "EXIT", {}),
    ]
)
