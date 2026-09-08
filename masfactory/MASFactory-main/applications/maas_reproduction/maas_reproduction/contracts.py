"""Frozen MaAS reproduction contracts.

This module is the machine-readable part of Task 0.  It intentionally records
the reproduction target instead of trying to make the original MaAS runtime
more featureful.

Source-compatible facts
-----------------------
* Operator catalogs retain the order from ``experiment_configs.py``.
* The policy controller has four layers and samples without replacement until
  selected probability mass reaches 0.3.  Sampling probabilities are detached,
  while selected log-probabilities remain live for policy gradient.
* A route is expanded in layer-major, position-major order.
* Policy log-probability is the sum of selected operator log-probabilities in
  each sampled layer, followed by a sum across sampled layers.
* A first-layer EarlyStop is replaced by Generate, receives a -1.5 log-prob
  adjustment, and ends further layer sampling.
* Otherwise, the first layer must contain an operator whose name contains
  ``generate`` (case-insensitive).  If none was sampled, the layer is replaced
  by Generate; if one was sampled later in the layer, the first such operator
  is moved to the front while preserving the remaining order.
* Utility is ``score - 3.0 * cost_delta``.
* Policy loss is the negative mean of ``policy_log_prob * utility`` over the
  eligible samples in the batch.

Spec-governed compatibility decisions
-------------------------------------
* MATH bootstrap is Programmer followed by Generate refinement and is outside
  the policy route/log-probability.
* EarlyStop is a deterministic control marker.  It contributes its sampled
  log-probability, but dispatch terminates when the marker is reached.
* Cost is a per-sample snapshot/delta.  It is reliable only when both snapshots
  and the delta succeed, the finite delta is non-negative, the tracker was not
  reset, and all retries were accounted for.  The original cross-sample
  ``previous_cost`` subtraction is deliberately not reproduced because it can
  produce negative or unrelated deltas and conflicts with the reproduction
  specification.
* Evaluation is reliable only when the architecture result is valid, the
  dataset scorer completes successfully, and it returns a finite score.
* Failures never add a reward, advantage, or penalty.  Training eligibility is
  based only on result validity, cost reliability, evaluation reliability, and
  the presence of a policy log-probability.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping


GENERATE_OPERATOR: Final = "Generate"
EARLY_STOP_OPERATOR: Final = "EarlyStop"
GENERATOR_NAME_FRAGMENT: Final = "generate"

OPERATOR_CATALOGS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "GSM8K": (
            "Generate",
            "GenerateCoT",
            "MultiGenerateCoT",
            "ScEnsemble",
            "Programmer",
            "SelfRefine",
            "EarlyStop",
        ),
        "MATH": (
            "Generate",
            "GenerateCoT",
            "MultiGenerateCoT",
            "ScEnsemble",
            "Programmer",
            "SelfRefine",
            "EarlyStop",
        ),
        "HumanEval": (
            "Generate",
            "GenerateCoT",
            "MultiGenerateCoT",
            "ScEnsemble",
            "Test",
            "SelfRefine",
            "EarlyStop",
        ),
    }
)

BOOTSTRAP_OPERATOR_ORDER: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "GSM8K": (),
        "MATH": ("Programmer", "Generate"),
        "HumanEval": (),
    }
)

ROUTE_EXPANSION_ORDER: Final = "layer_major_then_position"
POLICY_LAYER_COUNT: Final = 4
OPERATOR_SELECTION_THRESHOLD: Final = 0.3
OPERATOR_SAMPLING_WITH_REPLACEMENT: Final = False
POLICY_LOG_PROB_REDUCTION: Final = "sum_selected_then_sum_layers"
FIRST_LAYER_EARLY_STOP_LOG_PROB_ADJUSTMENT: Final = -1.5
UTILITY_COST_COEFFICIENT: Final = 3.0
POLICY_LOSS_SIGN: Final = -1.0
POLICY_LOSS_REDUCTION: Final = "mean_over_eligible_batch"

TRAINING_SKIP_REASONS: Final = frozenset(
    {
        "invalid_architecture_result",
        "unreliable_cost",
        "unreliable_evaluation",
        "missing_policy_log_prob",
    }
)


def operator_catalog_for(dataset: str) -> tuple[str, ...]:
    """Return the source-ordered policy operator catalog for ``dataset``."""

    try:
        return OPERATOR_CATALOGS[dataset]
    except KeyError as exc:
        raise ValueError(f"unsupported dataset: {dataset!r}") from exc


def bootstrap_operator_order_for(dataset: str) -> tuple[str, ...]:
    """Return fixed, non-policy bootstrap operators for ``dataset``."""

    try:
        return BOOTSTRAP_OPERATOR_ORDER[dataset]
    except KeyError as exc:
        raise ValueError(f"unsupported dataset: {dataset!r}") from exc


__all__ = [
    "BOOTSTRAP_OPERATOR_ORDER",
    "EARLY_STOP_OPERATOR",
    "FIRST_LAYER_EARLY_STOP_LOG_PROB_ADJUSTMENT",
    "GENERATE_OPERATOR",
    "GENERATOR_NAME_FRAGMENT",
    "OPERATOR_CATALOGS",
    "OPERATOR_SAMPLING_WITH_REPLACEMENT",
    "OPERATOR_SELECTION_THRESHOLD",
    "POLICY_LAYER_COUNT",
    "POLICY_LOG_PROB_REDUCTION",
    "POLICY_LOSS_REDUCTION",
    "POLICY_LOSS_SIGN",
    "ROUTE_EXPANSION_ORDER",
    "TRAINING_SKIP_REASONS",
    "UTILITY_COST_COEFFICIENT",
    "bootstrap_operator_order_for",
    "operator_catalog_for",
]
