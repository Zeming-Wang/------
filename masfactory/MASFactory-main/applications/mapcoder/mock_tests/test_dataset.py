"""Unit tests for HumanEval dataset utilities (no API key required)."""

from applications.mapcoder.humaneval.dataset import (
    extract_sample_io_from_prompt,
    verify_humaneval_solution,
)


SAMPLE_PROMPT = '''\
def has_close_elements(numbers, threshold):
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
'''

SAMPLE_TEST = '''\
def check(candidate):
    assert candidate([1.0, 2.0, 3.0], 0.5) == False
    assert candidate([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True
'''


def main() -> None:
    sample_io = extract_sample_io_from_prompt(SAMPLE_PROMPT, entry_point="has_close_elements")
    assert len(sample_io) == 2, f"expected 2 sample I/O pairs, got {sample_io!r}"
    assert "has_close_elements" in sample_io[0]["call"]
    assert sample_io[0]["expected"] == "False"
    assert sample_io[1]["expected"] == "True"
    print("[OK  ] extract_sample_io_from_prompt: ", sample_io)

    candidate = '''\
def has_close_elements(numbers, threshold):
    for i, a in enumerate(numbers):
        for j, b in enumerate(numbers):
            if i != j and abs(a - b) < threshold:
                return True
    return False
'''
    passed, msg = verify_humaneval_solution(candidate, SAMPLE_TEST, "has_close_elements")
    assert passed, f"verify should pass: {msg}"
    print("[OK  ] verify_humaneval_solution(passing candidate)")

    bad = "def has_close_elements(*a, **k):\n    return None\n"
    passed, msg = verify_humaneval_solution(bad, SAMPLE_TEST, "has_close_elements")
    assert not passed, f"verify should fail: {msg}"
    print("[OK  ] verify_humaneval_solution(failing candidate):", msg.split('\n')[0][:80])

    print("\nAll dataset tests passed.")


if __name__ == "__main__":
    main()
