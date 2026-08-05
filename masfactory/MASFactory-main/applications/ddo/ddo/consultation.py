from typing import Any

from ddo.state_model.state import ConsultationState

def run_consultation(g_consultation, case: dict[str, Any]) -> dict[str, Any]:
    state = ConsultationState.from_case(case)

    attrs = {
        "state": state,
        "interactions": [],
    }

    out, attrs = g_consultation.invoke(
        {},
        attributes=attrs,
    )
    state = attrs["state"]

    return {
        "case_id": state.case_state.case_id,
        "disease_label": state.case_state.disease_label,
        "initial_symptoms_status": state.case_state.self_report,
        "initial_diagnostic_confidence": state.diagnosis_state.initial_diagnostic_confidence,
        "final_diagnostic_confidence": dict(state.diagnosis_state.normalized_diagnosis_confidence),
        "interactions": attrs["interactions"],
    }
