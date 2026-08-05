from __future__ import annotations

import argparse
import os
from pathlib import Path

from masfactory import Agent, JsonMessageFormatter, Loop, Node, OpenAIModel, RootGraph


APP_DIR = Path(__file__).resolve().parent
TIKTOKEN_CACHE_DIR = APP_DIR / "number_off_assets" / "tiktoken_cache"

COUNTING_INSTRUCTIONS = """
You are {role_name}, a counting Agent in a MASFactory workflow.

Your sole task:
- Read the incoming field named "number".
- If "number" is a single number, add 1 to it.
- If "number" is a list of numbers, use the largest number in the list, then add 1.
- If "number" is missing or invalid, treat it as 0, then add 1.

Output rules:
- Return strictly valid JSON only.
- The JSON object must contain exactly one field: "number".
- The "number" value must be an integer.
"""

COUNTING_PROMPT = """
Incoming number:
{number}

Return the next count as JSON.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MASFactory number-off demo with real Agent nodes.")
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", ""),
        help="OpenAI-compatible API key. Defaults to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL") or None,
        help="OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL or BASE_URL.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL_NAME") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
        help="Model name. Defaults to OPENAI_MODEL_NAME, OPENAI_MODEL, or gpt-4o-mini.",
    )
    return parser.parse_args()


def build_graph(model: OpenAIModel) -> RootGraph:
    graph = RootGraph(name="number_off_demo")
    formatter = JsonMessageFormatter()

    def print_count(agent: Agent, output: dict[str, object], _input: dict[str, object]) -> None:
        print(f"{agent.name} : {output['number']}")

    agent_a = graph.create_node(
        Agent,
        name="AgentA",
        instructions=COUNTING_INSTRUCTIONS,
        prompt_template=COUNTING_PROMPT,
        model=model,
        formatters=formatter,
        model_settings={"temperature": 0},
        role_name="AgentA",
    )
    agent_b = graph.create_node(
        Agent,
        name="AgentB",
        instructions=COUNTING_INSTRUCTIONS,
        prompt_template=COUNTING_PROMPT,
        model=model,
        formatters=formatter,
        model_settings={"temperature": 0},
        role_name="AgentB",
    )
    agent_c = graph.create_node(
        Agent,
        name="AgentC",
        instructions=COUNTING_INSTRUCTIONS,
        prompt_template=COUNTING_PROMPT,
        model=model,
        formatters=formatter,
        model_settings={"temperature": 0},
        role_name="AgentC",
    )
    agent_d = graph.create_node(
        Agent,
        name="AgentD",
        instructions=COUNTING_INSTRUCTIONS,
        prompt_template=COUNTING_PROMPT,
        model=model,
        formatters=formatter,
        model_settings={"temperature": 0},
        role_name="AgentD",
    )

    loop = graph.create_node(Loop, name="counting_loop", max_iterations=3)
    agent_e = loop.create_node(
        Agent,
        name="AgentE",
        instructions=COUNTING_INSTRUCTIONS,
        prompt_template=COUNTING_PROMPT,
        model=model,
        formatters=formatter,
        model_settings={"temperature": 0},
        role_name="AgentE",
    )
    agent_f = loop.create_node(
        Agent,
        name="AgentF",
        instructions=COUNTING_INSTRUCTIONS,
        prompt_template=COUNTING_PROMPT,
        model=model,
        formatters=formatter,
        model_settings={"temperature": 0},
        role_name="AgentF",
    )
    agent_g = loop.create_node(
        Agent,
        name="AgentG",
        instructions=COUNTING_INSTRUCTIONS,
        prompt_template=COUNTING_PROMPT,
        model=model,
        formatters=formatter,
        model_settings={"temperature": 0},
        role_name="AgentG",
    )

    agent_h = graph.create_node(
        Agent,
        name="AgentH",
        instructions=COUNTING_INSTRUCTIONS,
        prompt_template=COUNTING_PROMPT,
        model=model,
        formatters=formatter,
        model_settings={"temperature": 0},
        role_name="AgentH",
    )

    for agent in (agent_a, agent_b, agent_c, agent_d, agent_e, agent_f, agent_g, agent_h):
        agent.hooks.register(Node.Hook.FORWARD.AFTER, print_count)

    graph.edge_from_entry(agent_a, {"number": "current number"})
    graph.create_edge(agent_a, agent_b, {"number": "current number"})
    graph.create_edge(agent_a, agent_c, {"number": "current number"})
    graph.create_edge(agent_b, agent_d, {"number": "current number from AgentB"})
    graph.create_edge(agent_c, agent_d, {"number": "current number from AgentC"})

    loop.edge_from_controller(agent_e, {"number": "current number"})
    loop.create_edge(agent_e, agent_f, {"number": "current number"})
    loop.create_edge(agent_f, agent_g, {"number": "current number"})
    loop.edge_to_controller(agent_g, {"number": "current number"})

    graph.create_edge(agent_d, loop, {"number": "current number"})
    graph.create_edge(loop, agent_h, {"number": "current number"})
    graph.edge_to_exit(agent_h, {"number": "final number"})

    graph.build()
    return graph


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Missing API key: set OPENAI_API_KEY or pass --api-key.")

    os.environ["TIKTOKEN_CACHE_DIR"] = str(TIKTOKEN_CACHE_DIR)
    model = OpenAIModel(
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model,
    )
    graph = build_graph(model)
    output, _attrs = graph.invoke({"number": 0})

    print("Final output:", output)
    print("Expected final number: 13")


if __name__ == "__main__":
    main()
