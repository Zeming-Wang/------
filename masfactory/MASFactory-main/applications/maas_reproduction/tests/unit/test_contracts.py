import pytest

from maas_reproduction.contracts import (
    BOOTSTRAP_OPERATOR_ORDER,
    FIRST_LAYER_EARLY_STOP_LOG_PROB_ADJUSTMENT,
    FIRST_LAYER_EARLY_STOP_REPLACEMENT,
    OPERATOR_SAMPLING_WITH_REPLACEMENT,
    OPERATOR_SELECTION_THRESHOLD,
    POLICY_LAYER_COUNT,
    POLICY_LOG_PROB_REDUCTION,
    POLICY_LOSS_REDUCTION,
    POLICY_LOSS_SIGN,
    ROUTE_EXPANSION_ORDER,
    UTILITY_COST_COEFFICIENT,
    bootstrap_operator_order_for,
    operator_catalog_for,
)


def test_operator_catalog_order_matches_maas_source() -> None:
    assert operator_catalog_for("GSM8K") == (
        "Generate",
        "GenerateCoT",
        "MultiGenerateCoT",
        "ScEnsemble",
        "Programmer",
        "SelfRefine",
        "EarlyStop",
    )
    assert operator_catalog_for("MATH") == operator_catalog_for("GSM8K")
    assert operator_catalog_for("HumanEval") == (
        "Generate",
        "GenerateCoT",
        "MultiGenerateCoT",
        "ScEnsemble",
        "Test",
        "SelfRefine",
        "EarlyStop",
    )


def test_only_math_has_fixed_bootstrap_operators() -> None:
    assert bootstrap_operator_order_for("MATH") == ("Programmer", "Generate")
    assert bootstrap_operator_order_for("GSM8K") == ()
    assert bootstrap_operator_order_for("HumanEval") == ()
    assert dict(BOOTSTRAP_OPERATOR_ORDER)["MATH"] == ("Programmer", "Generate")


def test_sampling_route_and_training_formulas_are_frozen() -> None:
    assert ROUTE_EXPANSION_ORDER == "layer_major_then_position"
    assert POLICY_LAYER_COUNT == 4
    assert OPERATOR_SELECTION_THRESHOLD == 0.3
    assert OPERATOR_SAMPLING_WITH_REPLACEMENT is False
    assert FIRST_LAYER_EARLY_STOP_REPLACEMENT == "Generate"
    assert FIRST_LAYER_EARLY_STOP_LOG_PROB_ADJUSTMENT == -1.5
    assert POLICY_LOG_PROB_REDUCTION == "sum_selected_then_sum_layers"
    assert UTILITY_COST_COEFFICIENT == 3.0
    assert POLICY_LOSS_SIGN == -1.0
    assert POLICY_LOSS_REDUCTION == "mean_over_eligible_batch"


def test_unknown_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported dataset"):
        operator_catalog_for("unknown")
