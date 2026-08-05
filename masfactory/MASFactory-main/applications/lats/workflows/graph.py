"""
LATS workflow: build RootGraph + Loop (LLM -> Executor -> Reflection -> controller), run_one_problem.
Uses only system Agent and CustomNode via NodeTemplate (no custom agent classes).
"""
import os
from typing import List, Tuple

from masfactory import RootGraph, NodeTemplate, Loop, Agent, CustomNode, Shared
from masfactory.core.message import ParagraphMessageFormatter, TwinsFieldTextFormatter
from masfactory import OpenAIModel

from ..components.agents import run_humaneval_forward
from ..components.tree import LATSNode, TreeManager
from .controller import lats_controller_logic, set_lats_tree
from ..humaneval.executor import full_evaluate

# Paper/source run_lats_gpt4.sh: max_iters=8, number_of_tests=2
LATS_MAX_ITERS = int(os.environ.get("LATS_MAX_ITERS", "8"))
NUMBER_OF_TESTS = int(os.environ.get("NUMBER_OF_TESTS", "2"))

EXECUTOR_PUSH_KEYS = {
    "observation": "observation",
    "reward": "reward",
    "action": "action",
    "full_passed": "full_passed",
}

model_instance = OpenAIModel(
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    base_url=os.environ.get("OPENAI_API_BASE", ""),
    model_name=os.environ.get("LATS_MODEL", "gpt-4"),
)

LLMAgentTemplate = NodeTemplate(
    Agent,
    model=model_instance,
    instructions=(
        "You output ONLY Python code in a ```python ... ``` block. "
        "No explanations. Restate the function signature in your implementation."
    ),
    prompt_template="{reflexion_prompt}",
    formatters=[
        ParagraphMessageFormatter(),
        TwinsFieldTextFormatter(),
    ],
)

ExecutorTemplate = NodeTemplate(
    CustomNode,
    forward=run_humaneval_forward,
    pull_keys={
        "problem": "problem",
        "internal_tests": "internal_tests",
        "content": "content",
    },
    push_keys=dict(EXECUTOR_PUSH_KEYS),
)

ReflectionTemplate = NodeTemplate(
    Agent,
    model=model_instance,
    instructions=(
        "You are a Python programming assistant. Given a function implementation "
        "and unit test results, write a few sentences explaining why the "
        "implementation is wrong. Do NOT output code, only the explanation."
    ),
    prompt_template=(
        "[function impl]:\n```python\n{action}\n```\n\n"
        "[unit test results]:\n{observation}\n\n[self-reflection]:"
    ),
    pull_keys={
        "action": "Candidate implementation under review.",
        "observation": "Unit-test feedback for the candidate implementation.",
    },
    formatters=[
        ParagraphMessageFormatter(),
        TwinsFieldTextFormatter(),
    ],
)

# Shared() prevents NodeTemplate from being deepcopied when the Loop config is cloned (avoids RLock/pickle errors).
loop_nodes = [
    ("LLM_Agent", Shared(LLMAgentTemplate)),
    ("Executor", Shared(ExecutorTemplate)),
    ("Reflection", Shared(ReflectionTemplate)),
]

LATSTemplate = NodeTemplate(
    Loop,
    max_iterations=LATS_MAX_ITERS,
    terminate_condition_function=lats_controller_logic,
    nodes=loop_nodes,
    edges=[
        # Controller provides the next prompt; problem/internal_tests in loop attributes.
        (
            "controller",
            "LLM_Agent",
            {"reflexion_prompt": "reflexion_prompt"},
        ),
        # Controller also sends problem/internal_tests to Executor (so Executor gets them without LLM pass-through).
        (
            "controller",
            "Executor",
            {"problem": "problem", "internal_tests": "internal_tests"},
        ),
        # LLM_Agent outputs candidate code only.
        (
            "LLM_Agent",
            "Executor",
            {"content": "content"},
        ),
        # Executor sends implementation and test output to Reflection.
        (
            "Executor",
            "Reflection",
            {"action": "action", "observation": "observation"},
        ),
        # Executor sends evaluation state to controller.
        (
            "Executor",
            "controller",
            {
                "action": "action",
                "observation": "observation",
                "reward": "reward",
                "full_passed": "full_passed",
            },
        ),
        # Reflection sends only the explanation to controller.
        (
            "Reflection",
            "controller",
            {"reflection": "reflection"},
        ),
    ],
    pull_keys={"problem": "problem", "internal_tests": "internal_tests"},
    push_keys={"final_code": "final_code", "final_passed": "final_passed"},
)


def build_graph() -> RootGraph:
    """Build LATS RootGraph with single LATS node (Loop)."""
    g = RootGraph(
        name="LATS_Runner",
        nodes=[("LATS", LATSTemplate)],
        edges=[
            ("entry", "LATS", {"problem": "problem", "internal_tests": "internal_tests"}),
            ("LATS", "exit", {"final_code": "final_code", "final_passed": "final_passed"}),
        ],
    )
    g.build()
    try:
        lats_loop = getattr(g, "_nodes", {}).get("LATS")
        if lats_loop is not None and hasattr(lats_loop, "_nodes"):
            env_node = lats_loop._nodes.get("Executor")
            if env_node is not None and hasattr(env_node, "set_push_keys"):
                env_node.set_push_keys(dict(EXECUTOR_PUSH_KEYS))
    except Exception:
        pass
    return g


def run_one_problem(
    problem: dict,
    graph: RootGraph,
    internal_tests: List[str],
    max_iters: int,
    number_of_tests: int,
) -> Tuple[str, bool]:
    """Run LATS for one problem; return (best_solution, passed)."""
    set_lats_tree(None)
    prompt = problem.get("prompt", "")
    simple_prompt = (
        "You output ONLY Python code in a ```python ... ``` block. No explanations. "
        "Write your full implementation (restate the function signature).\n\n" + prompt
    )
    root = LATSNode(solution="", context=prompt)
    tm = TreeManager(problem, root)
    tm._max_iters = max_iters
    set_lats_tree(tm)

    initial_input = {
        "problem": problem,
        "internal_tests": internal_tests,
        "reflexion_prompt": simple_prompt,
    }
    result, _ = graph.invoke(initial_input)
    final_code = result.get("final_code", "") or ""
    final_passed = result.get("final_passed", False)
    if isinstance(final_passed, str) and "(not set yet)" in str(final_passed):
        final_passed = False
    final_passed = bool(final_passed)
    if not final_code and tm and tm.root:
        best_node = tm.root.best_child_value() if tm.root.children else tm.root
        if best_node and getattr(best_node, "solution", None):
            final_code = best_node.solution
        elif getattr(tm.root, "solution", None):
            final_code = tm.root.solution
        if final_code and not final_passed:
            final_passed = full_evaluate(
                tm.problem.get("entry_point", ""),
                final_code,
                tm.problem.get("test", ""),
                timeout=10,
            )
    set_lats_tree(None)
    return final_code, final_passed
