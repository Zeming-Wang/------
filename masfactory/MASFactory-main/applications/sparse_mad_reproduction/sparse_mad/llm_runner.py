from __future__ import annotations

import os
import re
from typing import Protocol

from openai import OpenAI

from sparse_mad.metrics import estimate_messages_tokens
from sparse_mad.topology import build_topology
from sparse_mad.voting import majority_vote


class LLMClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str:
        ...


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.25,
    ) -> None:
        self.model_name = model_name or os.getenv("OPENAI_MODEL_NAME") or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.temperature = temperature
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""


def run_sparse_mad_llm(
    *,
    question: str,
    client: LLMClient,
    num_agents: int = 6,
    topology_type: str = "ring",
    max_rounds: int = 3,
) -> dict:
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")

    topology = build_topology(num_agents=num_agents, topology_type=topology_type)
    responses_by_round: dict[int, dict[str, dict[str, str]]] = {}
    visibility_by_round: dict[int, dict[str, list[str]]] = {}

    responses_by_round[1] = {
        agent: _call_initial_agent(client=client, agent=agent, question=question)
        for agent in topology.agents
    }

    for round_id in range(2, max_rounds + 1):
        previous = responses_by_round[round_id - 1]
        visibility_by_round[round_id] = topology.neighbors
        responses_by_round[round_id] = {
            agent: _call_debate_agent(
                client=client,
                agent=agent,
                question=question,
                round_id=round_id,
                own_previous=previous[agent],
                neighbor_responses={neighbor: previous[neighbor] for neighbor in topology.neighbors[agent]},
            )
            for agent in topology.agents
        }

    final_round = responses_by_round[max_rounds]
    vote = majority_vote({agent: response["answer"] for agent, response in final_round.items()})
    token_cost = sum(
        response["input_tokens"]
        for round_responses in responses_by_round.values()
        for response in round_responses.values()
    )

    return {
        "question": question,
        "topology": topology.as_dict(),
        "responses_by_round": responses_by_round,
        "visibility_by_round": visibility_by_round,
        "final_agent_answers": {agent: response["answer"] for agent, response in final_round.items()},
        "final_answer": vote.final_answer,
        "vote_summary": vote.vote_summary,
        "token_cost": token_cost,
    }


def extract_answer(text: str) -> str:
    final_match = re.search(r"FINAL\s*:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if final_match:
        return _clean_answer(final_match.group(1))

    boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed_match:
        return _clean_answer(boxed_match.group(1))

    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]

    return _clean_answer(text)


def _call_initial_agent(*, client: LLMClient, agent: str, question: str) -> dict[str, str | int]:
    messages = [
        {"role": "system", "content": _system_prompt()},
        {
            "role": "user",
            "content": (
                f"You are {agent}. Solve the following problem independently.\n\n"
                f"Problem: {question}\n\n"
                "Give concise reasoning and end with `FINAL: <answer>`."
            ),
        },
    ]
    input_tokens = estimate_messages_tokens(messages)
    content = client.complete(messages)
    return {"reasoning": content, "answer": extract_answer(content), "input_tokens": input_tokens}


def _call_debate_agent(
    *,
    client: LLMClient,
    agent: str,
    question: str,
    round_id: int,
    own_previous: dict[str, str],
    neighbor_responses: dict[str, dict[str, str]],
) -> dict[str, str | int]:
    references = "\n\n".join(
        f"{neighbor} previous response:\n{response['reasoning']}"
        for neighbor, response in neighbor_responses.items()
    )
    messages = [
        {"role": "system", "content": _system_prompt()},
        {
            "role": "user",
            "content": (
                f"You are {agent} in debate round {round_id}.\n\n"
                f"Problem: {question}\n\n"
                f"Your previous response:\n{own_previous['reasoning']}\n\n"
                f"Neighbor responses visible to you:\n{references}\n\n"
                "Use only your own previous response and the visible neighbor responses. "
                "Update your answer if needed. End with `FINAL: <answer>`."
            ),
        },
    ]
    input_tokens = estimate_messages_tokens(messages)
    content = client.complete(messages)
    return {"reasoning": content, "answer": extract_answer(content), "input_tokens": input_tokens}


def _system_prompt() -> str:
    return (
        "You are a careful reasoning agent in a multi-agent debate system. "
        "Keep reasoning concise. Always finish with exactly one line: FINAL: <answer>."
    )


def _clean_answer(answer: str) -> str:
    return answer.strip().strip("`").strip().rstrip(".")

