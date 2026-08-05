from masfactory import CustomNode, Graph, Loop, NodeTemplate
import math
from typing import Any

from ddo.runtime.app_runtime import app_runtime
from ddo.services.api_btp import estimate_btp_confidence_by_api
from ddo.state_model.state import ConsultationState

# Utils
def get_disease_to_update(state: ConsultationState) -> tuple[str,...] | list[str]:
    reserved = state.diagnosis_state.reserved_candidate_diseases
    if not reserved:
        return app_runtime.dataset_ctx.candidate_diseases
    return reserved

# CustomNode
def normalize_result(d: dict, attrs: dict) -> dict:
    state: ConsultationState = attrs["state"]

    candidate_diseases = app_runtime.dataset_ctx.candidate_diseases
    diagnosis_state = state.diagnosis_state

    raw_diagnosis_confidence = diagnosis_state.raw_diagnosis_confidence

    raw_scores = {}
    for disease in candidate_diseases:
        raw_scores[disease] = float(raw_diagnosis_confidence[disease])
    
    temp = app_runtime.cfg.disease_temperature
    max_score = max(raw_scores.values())
    exp_scores = {
        disease: math.exp((score - max_score) / temp)
        for disease, score in raw_scores.items()
    }

    total = sum(exp_scores.values())

    # 防止除零
    if total <= 1e-8:
        normalized_scores = {
            disease: 1.0 / len(candidate_diseases)
            for disease in candidate_diseases
        }
    else:
        normalized_scores = {
            disease: score / total
            for disease, score in exp_scores.items()
        }


    # 保存 保留候选疾病
    sorted_scores = dict(
        sorted(
            normalized_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    diagnosis_state.normalized_diagnosis_confidence = sorted_scores

    if diagnosis_state.reserved_candidate_diseases is None:
        reserve_k = app_runtime.cfg.reserve_k
        diagnosis_state.reserved_candidate_diseases = list(sorted_scores.keys())[:reserve_k]


    return {}

def init_loop_state(d: dict[str, Any])-> dict[str, Any]:
    return {
        "idx": 0,
    }

def prepare_one_disease(d: dict[str, Any], attrs: dict) -> dict[str, Any]:
    idx = d["idx"]
    candidate_diseases = get_disease_to_update(attrs["state"])
    candidate_disease = candidate_diseases[idx]

    disease_knowledge = app_runtime.dataset_ctx.disease_knowledge[candidate_disease]["empirical_knowledge"]
    state: ConsultationState = attrs["state"]

    return {
        "positive_symptoms": state.diagnosis_state.positive_symptoms,
        "negative_symptoms": state.diagnosis_state.negative_symptoms,
        "candidate_disease": candidate_disease,
        "current_disease_knowledge": disease_knowledge,
        "idx": idx,
    }

def diagnosis_one_disease_btp(d: dict[str, Any], attrs: dict) -> dict[str, Any]:
    state: ConsultationState = attrs["state"]
    idx = int(d["idx"])

    candidate_disease = d["candidate_disease"]
    
    btp = estimate_btp_confidence_by_api(
        client=app_runtime.client,
        api_cfg={"for_confidence_agent": app_runtime.cfg.api_cfg.for_confidence_agent},
        positive_symptoms=d["positive_symptoms"],
        negative_symptoms=d["negative_symptoms"],
        candidate_disease=candidate_disease,
        current_disease_knowledge=d["current_disease_knowledge"],
        temperature_for_model=app_runtime.cfg.confidence_model_temperature,
        temperature_for_btp=app_runtime.cfg.btp_temperature,
        fallback_margin=app_runtime.cfg.fallback_margin,
    )

    confidence = btp["confidence"]
    judgment = btp["judgment"]

    # 1. 保存 True/False 判断
    state.diagnosis_state.diagnosis_results[candidate_disease] = judgment

    # 2. 保存 BTP 原始 confidence
    state.diagnosis_state.raw_diagnosis_confidence[candidate_disease] = confidence

    # 3. 保存 debug 信息，后面写复现文档很有用
    state.diagnosis_state.btp_debug[candidate_disease] = btp["debug"]

    return {
        "idx": idx + 1,
    }    
    
# Conditions
def terminate_cond(d:dict, _attr:dict) -> bool:
    idx = int (d["idx"])

    state: ConsultationState = _attr["state"]
    
    candidate_diseases = get_disease_to_update(state)
    return idx >= len(candidate_diseases)

# Graphs
diagnosis_loop = NodeTemplate(
    Loop,
    max_iterations=100,
    terminate_condition_function=terminate_cond,
    nodes=[
        ("prepare_one_disease", CustomNode, prepare_one_disease),
        # ("diagnosis_agent", diagnosis_agent),
        ("diagnosis_agent", CustomNode, diagnosis_one_disease_btp),
        # ("collect_result", CustomNode, collect_result),
    ],
    edges=[
        ("CONTROLLER", "prepare_one_disease",{
            "idx":"当前诊断到的疾病的索引",
        }),
        ("prepare_one_disease", "diagnosis_agent",{
            "idx":"当前诊断到的疾病的索引",
            "positive_symptoms":"患者存在的症状",
            "negative_symptoms":"患者不存在的症状",
            "candidate_disease":"候选疾病",
            "current_disease_knowledge":"当前疾病的症状知识",
        }),
        ("diagnosis_agent", "CONTROLLER",{
            "idx":"当前诊断到的疾病的索引",
        })
    ]
)

g_confidence_estimation = NodeTemplate(
    Graph,
    nodes=[
        ("init_loop_state", CustomNode, init_loop_state),
        ("diagnosis_loop", diagnosis_loop),
        ("normalize_result", CustomNode, normalize_result),
    ],
    edges=[
        ("ENTRY", "init_loop_state",{}),
        ("init_loop_state", "diagnosis_loop",{
            "idx":"当前诊断到的疾病的索引",
        }),
        ("diagnosis_loop", "normalize_result",{}),
        ("normalize_result", "EXIT",{}),
    ]
)
