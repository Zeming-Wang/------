from dataclasses import dataclass, field
from typing import Any

@dataclass
class CaseState:
    case_id: str | int
    disease_label: str
    self_report: dict[str, bool]
    all_symptoms: dict[str, bool]

    @classmethod
    def from_case(cls, case: dict[str, Any]) -> "CaseState":
        return cls(
            case_id=case["id"],
            disease_label=case["disease_label"],
            self_report=case["self_report"],
            all_symptoms=case["all_symptoms"],
        )

@dataclass
class ControlState:
    turn_id: int = 0
    # 用于循环终止条件判断
    should_terminate: bool = False
    termination_reason: str | None = None

@dataclass
class DiagnosisState:
    positive_symptoms: list[str] = field(default_factory=list)
    negative_symptoms: list[str] = field(default_factory=list)

    reserved_candidate_diseases: list[str] | None = None

    # {disease: True / False / None}
    diagnosis_results: dict[str, bool | None] = field(default_factory=dict)

    # {disease: raw_btp_confidence}
    raw_diagnosis_confidence: dict[str, float] = field(default_factory=dict)

    # {disease: normalized_confidence} 排序的
    normalized_diagnosis_confidence: dict[str, float] = field(default_factory=dict)

    initial_diagnostic_confidence: dict[str, float] = field(default_factory=dict)

    # {disease: raw btp debug info}
    btp_debug: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_self_report(cls, self_report: dict[str, bool]) -> "DiagnosisState":
        positive_symptoms : list[str] = []
        negative_symptoms : list[str] = []

        for symptom, status in self_report.items():
            if status is True:
                positive_symptoms.append(symptom)
            elif status is False:
                negative_symptoms.append(symptom)
        
        return cls(positive_symptoms=positive_symptoms, negative_symptoms=negative_symptoms)

@dataclass
class InquiryState:
    
    # Policy Agent 输出
    candidate_actions_ids: list[int] = field(default_factory=list) # 不去重，有终止动作
    candidate_symptoms: list[str] = field(default_factory=list) #去重，没有终止动作
    has_termination_action: bool = False
    top_disease_by_policy_obs: list[str] = field(default_factory=list)

    # Inquiry Agent / fallback / termination 选择结果
    selected_action_id: int | None = None
    selected_symptom: str | None = None
    selection_reasoning: str | None = None
    raw_output: str | None = None

    # retry / fallback
    need_retry: bool = False
    retry_count: int = 0
    retry_history: list[dict[str, Any]] = field(default_factory=list)
    extra_masking_action_ids: list[int] = field(default_factory=list)

@dataclass
class ResponseState:
    symptom_status: bool | None = None
    is_recorded: bool | None = None
    response_reasoning: str | None = None
    response_source: str | None = None
    raw_output: str | None = None

@dataclass
class ConsultationState:
    case_state: CaseState
    control_state: ControlState
    diagnosis_state: DiagnosisState
    inquiry_state: InquiryState = field(default_factory=InquiryState)
    response_state: ResponseState = field(default_factory=ResponseState)
    trace: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_case(cls, case: dict[str, Any]) -> "ConsultationState":
        case_state = CaseState.from_case(case)

        diagnosis_state = DiagnosisState.from_self_report(case_state.self_report)

        return cls(
            case_state=case_state,
            control_state=ControlState(),
            diagnosis_state=diagnosis_state,
            inquiry_state=InquiryState(),
            response_state=ResponseState(),
            trace=[],
        )
    
    def reset_round_state(self):
        self.inquiry_state = InquiryState()
        self.response_state = ResponseState()
