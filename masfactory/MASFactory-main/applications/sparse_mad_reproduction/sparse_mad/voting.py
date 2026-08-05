from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class VoteResult:
    final_answer: str
    vote_summary: dict[str, int]


def majority_vote(agent_answers: dict[str, str]) -> VoteResult:
    if not agent_answers:
        raise ValueError("agent_answers cannot be empty")

    counts = Counter(agent_answers.values())
    final_answer, _count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]

    return VoteResult(
        final_answer=final_answer,
        vote_summary=dict(sorted(counts.items())),
    )
