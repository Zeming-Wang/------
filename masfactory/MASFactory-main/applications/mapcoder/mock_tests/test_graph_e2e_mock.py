"""
End-to-end smoke for the MapCoder graph using a stub Model (no API calls).

Strategy:
- Subclass `masfactory.Model` to return canned responses keyed by which
  Agent's prompt is being processed.
- Build the real `build_graph(model, k, t)` graph and call `g.invoke(...)` on
  one hand-rolled HumanEval-style problem.
- Assert: graph terminates, final_code passes the in-graph Tester, and the
  final_code also passes the public hidden test (verify_humaneval_solution).

Run:
    python -m applications.mapcoder.tests.test_graph_e2e_mock
"""

from __future__ import annotations

import sys

from masfactory import Model
from masfactory.adapters.model.base import ModelResponseType, ModelCapabilities

from applications.mapcoder.workflows.graph import build_graph
from applications.mapcoder.humaneval.dataset import (
    extract_sample_io_from_prompt,
    verify_humaneval_solution,
)


PROBLEM_PROMPT = '''\
def sum_squares(lst):
    """\
    You are given a list of numbers.
    Return the sum of squares of the numbers in the list that are odd.
    Ignore numbers that are negative or not integers.
    If the input list is empty, return 0.
    >>> sum_squares([1, 3, 2, 0])
    10
    >>> sum_squares([-1, -2, 0])
    0
    >>> sum_squares([9, -2])
    81
    >>> sum_squares([0])
    0
    """
'''

PROBLEM_TEST = '''\
def check(candidate):
    assert candidate([1, 3, 2, 0]) == 10
    assert candidate([-1, -2, 0]) == 0
    assert candidate([9, -2]) == 81
    assert candidate([0]) == 0
    assert candidate([]) == 0
'''


CORRECT_CANDIDATE = '''\
def sum_squares(lst):
    return sum(x * x for x in lst if isinstance(x, int) and x > 0 and x % 2 == 1)
'''


# Coding agent's FIRST emission: deliberately wrong (no filtering) so that
# Tester rejects it and the DebugLoop has to kick in.
BROKEN_CANDIDATE = '''\
def sum_squares(lst):
    # Bug: sums ALL squares (no odd/positive filtering).
    return sum(x * x for x in lst)
'''


# ---------------------------------------------------------------------------
# Stub model: returns canned responses by detecting which Agent's prompt it
# sees.
# ---------------------------------------------------------------------------

RETRIEVAL_REPLY = """\
<root>
<problem>
<description> Sum the squares of all positive odd integers in a list. </description>
<code>
```python
def example(lst):
    return sum(x * x for x in lst if x > 0 and x % 2 == 1)
```
</code>
<planning> Iterate through the list, skip negatives and even numbers, square the rest, sum. </planning>
</problem>
<problem>
<description> Sum cubes of even numbers in a list. </description>
<code>
```python
def cubes_even(lst):
    return sum(x ** 3 for x in lst if x % 2 == 0)
```
</code>
<planning> Iterate, filter to even, cube each, sum. </planning>
</problem>
<algorithm>
List comprehension + filter + reduction (sum). Tutorial: walk the input once, apply a predicate to each element, transform, then aggregate.
</algorithm>
</root>
"""


HIGH_PLAN = (
    "1. Iterate over the input list.\n"
    "2. Keep only positive odd integers.\n"
    "3. Square each kept value.\n"
    "4. Return the sum (0 if list is empty)."
)
LOW_PLAN = (
    "Just sum the squares of all elements without filtering. (incorrect)"
)


HIGH_CONFIDENCE = "<root><explanation> ok </explanation><confidence>92</confidence></root>"
LOW_CONFIDENCE = "<root><explanation> bad </explanation><confidence>10</confidence></root>"


CODING_REPLIES = [BROKEN_CANDIDATE, CORRECT_CANDIDATE]
DEBUG_REPLIES = [CORRECT_CANDIDATE]


class StubModel(Model):
    """Routes responses by detecting Agent intent in the prompt."""

    def __init__(self):
        super().__init__(
            model_name="stub-mapcoder",
            capabilities=ModelCapabilities(image_input=False, pdf_input=False),
        )
        self._coding_idx = 0
        self._debug_idx = 0
        self._plan_idx = 0
        self._conf_idx = 0
        self.calls: list[str] = []

    def invoke(self, messages, tools=None, settings=None, **kwargs):  # type: ignore[override]
        sys_msg = ""
        usr_msg = ""
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    getattr(b, "text", str(b)) if not isinstance(b, str) else b
                    for b in content
                )
            if role == "system":
                sys_msg += "\n" + str(content)
            elif role == "user":
                usr_msg += "\n" + str(content)

        full = sys_msg + "\n" + usr_msg

        if "Recall" in full and "<root>" in full and "Algorithm" in full:
            tag = "RETRIEVAL"
            text = RETRIEVAL_REPLY
        elif "tell whether the plan is correct" in full or "<confidence>" in full:
            tag = "CONFIDENCE"
            text = HIGH_CONFIDENCE if self._conf_idx == 0 else LOW_CONFIDENCE
            self._conf_idx += 1
        elif "generate a concrete planning" in full or "Planning to solve" in full or "Relevant Algorithm" in full and "Coding" not in full and "Debugging" not in full and "code can not pass" not in full:
            # PlanGen comes before CODING in the loop sequence.
            if "code can not pass" in full:
                tag = "DEBUG"
                text = DEBUG_REPLIES[min(self._debug_idx, len(DEBUG_REPLIES) - 1)]
                self._debug_idx += 1
            elif "## Sample Input/Outputs" in full and "## Planning" in full and "## Code" not in full:
                tag = "CODING"
                text = "```python\n" + CODING_REPLIES[min(self._coding_idx, len(CODING_REPLIES) - 1)] + "```"
                self._coding_idx += 1
            else:
                tag = "PLAN_GEN"
                text = HIGH_PLAN if self._plan_idx == 0 else LOW_PLAN
                self._plan_idx += 1
        elif "code can not pass" in full or "## Test Report" in full:
            tag = "DEBUG"
            text = DEBUG_REPLIES[min(self._debug_idx, len(DEBUG_REPLIES) - 1)]
            self._debug_idx += 1
        elif "## Code" in full and "## Planning" in full:
            tag = "CODING"
            text = "```python\n" + CODING_REPLIES[min(self._coding_idx, len(CODING_REPLIES) - 1)] + "```"
            self._coding_idx += 1
        else:
            tag = "FALLBACK"
            text = "(unrecognized prompt)"
        self.calls.append(tag)
        return {"type": ModelResponseType.CONTENT, "content": text}


def main() -> None:
    model = StubModel()
    g = build_graph(model, k=2, t=2)
    sample_io = extract_sample_io_from_prompt(PROBLEM_PROMPT, "sum_squares")
    assert sample_io, "expected sample I/O extracted from docstring"
    print(f"Sample I/O: {len(sample_io)} pair(s)")

    out, _attrs = g.invoke(
        {"problem": PROBLEM_PROMPT, "language": "Python3", "k": 2},
        attributes={"sample_io": sample_io},
    )
    print("graph output keys:", sorted(out.keys()) if isinstance(out, dict) else type(out))
    print("model call sequence:", model.calls)

    final_code = out.get("final_code", "") if isinstance(out, dict) else ""
    final_passed = bool(out.get("final_passed", False))
    print(f"final_passed={final_passed}, final_code chars={len(final_code)}")
    assert final_code, "graph produced no final_code"

    hidden_passed, hidden_msg = verify_humaneval_solution(final_code, PROBLEM_TEST, "sum_squares")
    print(f"hidden_passed={hidden_passed}, hidden_msg={hidden_msg[:120]}")
    assert hidden_passed, f"final_code failed hidden tests: {hidden_msg}"

    print("\nMapCoder graph end-to-end mock smoke PASSED.")


if __name__ == "__main__":
    main()
