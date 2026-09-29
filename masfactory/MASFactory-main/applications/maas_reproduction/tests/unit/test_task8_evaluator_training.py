from pathlib import Path

import pytest
from applications.maas_reproduction.maas_reproduction.benchmarks import GSM8KBenchmark, MATHBenchmark
from applications.maas_reproduction.maas_reproduction.runtime.prompt_loader import PromptLoader
from applications.maas_reproduction.maas_reproduction.schemas import ArchitectureResult, EvaluationContext, FailureSource
from applications.maas_reproduction.maas_reproduction.training.training_signal import build_training_signal
from applications.maas_reproduction.components.evaluator_node import EvaluatorNode


def test_scorers_follow_dataset_rules():
    assert GSM8KBenchmark().score("work; answer 1,000", "#### 1000") == 1.0
    assert MATHBenchmark().score("therefore \\boxed{2}", "\\boxed{2.0}") == 1.0


def test_math_scorer_supports_source_latex_and_numeric_symbolic_fallbacks():
    scorer = MATHBenchmark()
    assert scorer.score(r"\boxed{\frac{1}{2}}", r"\boxed{\frac{2}{4}}") == 1.0
    assert scorer.score(r"\boxed{\sqrt{2}}", r"\boxed{1.4142135623730951}") == 1.0


@pytest.mark.parametrize(
    ("prediction", "expected", "problem"),
    [
        ("The proposed answer is correct.\n\n**Final answer: 10**", r"The answer is \boxed{10}.", "Count the integers."),
        (r"\[\boxed{x=\frac{11}{13}}\]", r"Therefore x=\boxed{\frac{11}{13}}.", "Solve for x."),
        (r"\[\boxed{8\sqrt3}\]", r"The area is \boxed{8\sqrt{3}}.", "Find the area."),
        (r"\[\boxed{16+12\sqrt2}\]", r"The perimeter is \boxed{16+12\sqrt{2}}.", "Find the perimeter."),
        (r"\[\boxed{80^\circ}\]", r"The angle is \boxed{80} degrees.", "Find the angle in degrees."),
        ("Their sum is 298.\n\nFinal answer: **298**", r"The sum is \boxed{298}.", "Find the sum."),
        ("The calculation gives 3.\n\n**Final answer: 3**", r"Thus x=\boxed{3}.", "How many cars?"),
        (r"Rounded to a whole number, \boxed{162}.", r"It takes \boxed{162\text{ minutes}}.", "How many minutes?"),
        ("20%", r"The answer is \boxed{20} percent.", "What percent are children?"),
        ("Calculation: 45*40/10=180\n\n**Answer: 180**", r"The estimate is \boxed{180}.", "Estimate the frogs."),
    ],
)
def test_math_scorer_accepts_equivalent_answers_seen_in_real_math_run(prediction, expected, problem):
    assert MATHBenchmark().score(prediction, expected, context={"problem": problem}) == 1.0


@pytest.mark.parametrize(
    ("prediction", "expected", "problem"),
    [
        ("2", r"\boxed{4}", "How many integer divisors does 7 have?"),
        (r"\boxed{\sqrt3}", r"\boxed{\sqrt{6}}", "Find the hypotenuse."),
        ("60 cartons per school day.", r"\boxed{400}", "How many cartons each day?"),
    ],
)
def test_math_scorer_does_not_turn_real_math_errors_into_matches(prediction, expected, problem):
    assert MATHBenchmark().score(prediction, expected, context={"problem": problem}) == 0.0


@pytest.mark.parametrize(
    "operator",
    ["Generate", "GenerateCoT", "MultiGenerateCoT", "SelfRefine", "BootstrapGenerate"],
)
def test_math_prompts_preserve_the_source_boxed_answer_contract(operator):
    prompt_root = Path(__file__).parents[2] / "assets" / "prompts"
    prompt = PromptLoader(prompt_root).load(operator, "MATH")
    assert r"\boxed" in prompt


def test_evaluator_failure_is_unreliable_and_training_skips():
    node = EvaluatorNode(scorer=GSM8KBenchmark())
    result = node._forward({
        "architecture_result": ArchitectureResult("answer", 0.1, object(), "success", None, True, True),
        "evaluation_context": EvaluationContext("question", 0, "not-a-number"),
    })["evaluation_result"]
    assert not result.evaluation_reliable
    assert result.failure_source is FailureSource.EVALUATION
    signal = build_training_signal(result)
    assert not signal.should_update
    assert signal.skip_reason == "unreliable_evaluation"


def test_failure_source_does_not_change_utility():
    common = {"result_valid": True, "cost_reliable": True, "evaluation_reliable": True, "score": 1.0, "cost_delta": 0.2, "policy_log_prob": object()}
    a = build_training_signal({**common, "failure_source": FailureSource.ROUTE_EXECUTION})
    b = build_training_signal({**common, "failure_source": FailureSource.BOOTSTRAP})
    assert a.utility == b.utility
    assert a.utility == pytest.approx(0.4)
