from typing import Any, Iterable
from openai import OpenAI
import math

from masfactory.adapters.token_usage_tracker import TokenUsageTracker

from ddo.runtime.app_runtime import app_runtime
from ddo.runtime.openai_retry import call_with_openai_retry

# 弃用
DIAGNOSIS_INSTRUCTIONS = """
你是一位经验丰富的医学专家。
请你根据患者的症状表现情况，结合疾病的症状知识，判断是否能够诊断为该疾病。

患者存在该疾病的相关症状会提升诊断信心，不存在该疾病的相关症状会降低诊断信心。
症状在患者身上的表现情况对诊断信心的影响，随该症状在疾病中的典型性的增加而增加。

判断结果有 True 和 False 两种：
- 如果你认为能够诊断为该疾病，则输出 True；
- 如果你认为不能够诊断为该疾病，则输出 False。

请直接输出判断结果，不要输出其它内容。
""".strip()

def _format_symptoms_for_prompt(symptoms: Any) -> str:
    if symptoms is None:
        return "无"
    
    if isinstance(symptoms, str):
        symptoms = symptoms.strip()
        return symptoms if symptoms else "无"
    
    if isinstance(symptoms, Iterable):
        items = [
            str(symptom).strip()
            for symptom in symptoms
            if str(symptom).strip()
        ]
        return "、".join(items) if items else "无"
    
    text = str(symptoms).strip()
    return text if text else "无"

def build_diagnosis_user_prompt(
    positive_symptoms,
    negative_symptoms,
    candidate_disease,
    current_disease_knowledge,
) -> str:
    
    positive_symptoms = _format_symptoms_for_prompt(positive_symptoms)
    negative_symptoms = _format_symptoms_for_prompt(negative_symptoms)

    return f"""你是一位经验丰富的医学专家，现在为你提供如下信息：

## 患者的症状表现情况
- 患者存在的症状：{positive_symptoms}。
- 患者不存在的症状：{negative_symptoms}。

## 疾病'{candidate_disease}'的症状知识
- 基于已有的一些诊断结果为'{candidate_disease}'的病例统计的症状出现频率：{current_disease_knowledge}

请你根据患者的症状表现情况，结合疾病'{candidate_disease}'的症状知识，判断是否能够诊断为该疾病。患者存在该疾病的相关症状会提升诊断信心，不存在该疾病的相关症状会降低诊断信心。症状在患者身上的表现情况对诊断信心的影响随该症状在疾病中的典型性的增加而增加。判断结果有True和False两种：如果你认为能够诊断为该疾病则输出True；如果你认为不能够诊断为该疾病则输出False。请直接输出判断结果，不要输出其它内容。
""".strip()

def normalize_token(token: str | None) -> str:
    if token is None:
        return ""
    return token.strip().lower()

def get_fallback_logprob(top_logprobs, fallback_margin) -> float:
    values = []

    for item in top_logprobs:
        value = _get_field(item, "logprob")
        if isinstance(value, (int, float)):
            values.append(float(value))
    
    if not values:
        return -20.0
    
    return min(values) - fallback_margin

def _get_field(obj, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)

def binary_softmax(logprob_true: float, logprob_false: float, temperature: float = 1.0) -> float:
    x = (logprob_false - logprob_true) / temperature

    if x > 60:
        return 0.0
    if x < -60:
        return 1.0

    return 1.0 / (1.0 + math.exp(x))

def update_best_logprob(current, candidate):
    if candidate is None:
        return current
    if current is None:
        return candidate
    return max(current, candidate) 

def _get_usage_value(usage: Any, *names: str) -> int:
    if usage is None:
        return 0

    if not isinstance(usage, dict):
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        elif hasattr(usage, "dict"):
            usage = usage.dict()

    for name in names:
        value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        if isinstance(value, (int, float)):
            return int(value)

    return 0

def _build_btp_usage(
    *,
    response_usage: Any,
    model_name: str,
    messages: list[dict[str, str]],
) -> dict[str, int]:
    input_tokens = _get_usage_value(response_usage, "input_tokens", "prompt_tokens")
    output_tokens = _get_usage_value(response_usage, "output_tokens", "completion_tokens")
    total_tokens = _get_usage_value(response_usage, "total_tokens")

    if input_tokens == 0:
        input_tokens = TokenUsageTracker(model_name=model_name).count_message_tokens(messages)

    if output_tokens == 0 and total_tokens > input_tokens:
        output_tokens = total_tokens - input_tokens

    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }



def estimate_btp_confidence_by_api(
    *,
    client,
    api_cfg,
    positive_symptoms,
    negative_symptoms,
    candidate_disease: str,
    current_disease_knowledge,
    temperature_for_model: float = 0.1,
    temperature_for_btp: float = 1.0,
    fallback_margin: float = 2.0,
) -> dict[str, Any]:
    
    # 发送请求给模型
    user_prompt = build_diagnosis_user_prompt(
        positive_symptoms=positive_symptoms,
        negative_symptoms=negative_symptoms,
        candidate_disease=candidate_disease,
        current_disease_knowledge=current_disease_knowledge,
    )

    request_kwargs = {
        "model": api_cfg["for_confidence_agent"].model_name,
        "messages": [
            # {"role": "system", "content": DIAGNOSIS_INSTRUCTIONS},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 1,
        "temperature": temperature_for_model,
        "logprobs": True,
        "top_logprobs": 20,
    }

    if api_cfg["for_confidence_agent"].provider == "deepseek":
        request_kwargs["extra_body"] = {
            "thinking": {"type": "disabled"},
        }

    resp = call_with_openai_retry(
        lambda: client.chat.completions.create(**request_kwargs),
        operation_name=f"confidence_agent.chat_completions[{candidate_disease}]",
        max_attempts=app_runtime.cfg.confidence_agent_max_attempts,
        base_delay=app_runtime.cfg.retry_base_delay_seconds,
        max_delay=app_runtime.cfg.retry_max_delay_seconds,
        retry_unknown_status=True,
    )
    
    # 计算token用量
    if app_runtime.token_usage is not None:
        usage = _build_btp_usage(
            response_usage=getattr(resp, "usage", None),
            model_name=api_cfg["for_confidence_agent"].model_name,
            messages=request_kwargs["messages"],
        )
        app_runtime.token_usage.add_usage(
            component="confidence_agent",
            provider=api_cfg["for_confidence_agent"].provider,
            model=api_cfg["for_confidence_agent"].model_name,
            usage=usage,
        )

    # 解析模型输出，计算诊断置信度
    choice = resp.choices[0]
    content_logprobs = choice.logprobs.content

    # 边界1：没有返回任何 logprobs
    if not content_logprobs:
        return {
            "confidence": 0.5,
            "judgment": None,
            "debug": {
                "reason": "no content logprobs",
                "raw_response": str(resp),
            },
        }
    
    first_token_logprobs = content_logprobs[0]

    chosen_token = getattr(first_token_logprobs, "token", "")
    chosen_logprob = getattr(first_token_logprobs, "logprob", None)
    top_logprobs = getattr(first_token_logprobs, "top_logprobs", [])

    # 计算 True 和 False 的 logprob

    logprob_true = None
    logprob_false = None

    for item in top_logprobs:
        token = _get_field(item, "token")
        logprob = _get_field(item, "logprob")

        normalized = normalize_token(token)

        if normalized == "true":
            logprob_true = update_best_logprob(logprob_true, logprob)
        elif normalized == "false":
            logprob_false = update_best_logprob(logprob_false, logprob)
    
    chosen_norm = normalize_token(chosen_token)
    if chosen_norm == "true" and logprob_true is None:
        logprob_true = chosen_logprob
    elif chosen_norm == "false" and logprob_false is None:
        logprob_false = chosen_logprob

    # fallback
    # 如果 top_logprobs 里没有 True/False 的 logprob，就用一个 fallback 的 logprob 来代替，避免后续计算出错。两者都没有就输出 0.5 的中性置信度。

    missing = []
    if logprob_true is None:
        missing.append("True")
    if logprob_false is None:
        missing.append("False")
    
    if logprob_true is None and logprob_false is None:
        confidence = 0.5
        judgment = None
    else:
        fall_back_logprob = get_fallback_logprob(top_logprobs, fallback_margin)

        if logprob_true is None:
            logprob_true = fall_back_logprob
        if logprob_false is None:
            logprob_false = fall_back_logprob
        
        confidence = binary_softmax(
            logprob_true,
            logprob_false,
            temperature_for_btp,
        )
        judgment = confidence >= 0.5

    return {
        "confidence": confidence,
        "judgment": judgment,
        "debug": {
            "candidate_disease": candidate_disease,
            "chosen_token": chosen_token,
            "chosen_logprob": chosen_logprob,
            "logprob_true": logprob_true,
            "logprob_false": logprob_false,
            "missing": missing,
            "top_logprobs": [
                {
                    "token": _get_field(item, "token"),
                    "logprob": _get_field(item, "logprob"),
                }
                for item in top_logprobs
            ],
        },
    }
