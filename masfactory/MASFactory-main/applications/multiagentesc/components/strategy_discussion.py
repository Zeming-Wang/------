"""
Strategy discussion: 3-agent round-robin via MeshGraph.
"""

import re
from collections import Counter
from masfactory import Agent, OpenAIModel, NodeTemplate, MeshGraph
from masfactory.adapters.memory import HistoryMemory
from ..patch_masfactory import PlainTextFormatter
from ..prompts import get_prompt


def create_strategy_discussion_graph(model: OpenAIModel):
    shared_memory = HistoryMemory(top_k=100, memory_size=100)
    agents = [
        NodeTemplate(
            Agent,
            role_name="agent_0",
            instructions="You are a psychological counseling expert.",
            prompt_template="{message}",
            model_settings={"max_tokens": 400},
            formatters=PlainTextFormatter(),
        ),
        NodeTemplate(
            Agent,
            role_name="agent_1",
            instructions="You are a psychological counseling expert.",
            prompt_template="{message}",
            model_settings={"max_tokens": 400},
            formatters=PlainTextFormatter(),
        ),
        NodeTemplate(
            Agent,
            role_name="agent_2",
            instructions="You are a psychological counseling expert.",
            prompt_template="{message}",
            model_settings={"max_tokens": 400},
            formatters=PlainTextFormatter(),
        ),
    ]
    graph = MeshGraph(
        name="strategy_discussion",
        agents=agents,
        model=model,
        shared_memory=shared_memory,
        max_iterations=3,
        terminate_condition_function=lambda _m, _a: False,
    )
    graph.build()
    return graph


def run_strategy_discussion(graph, context, emo_and_reason, cau_and_reason, int_and_reason, examples):
    admin_prompt = get_prompt("select_strategy").format(
        context=context,
        emo_and_reason=emo_and_reason,
        cau_and_reason=cau_and_reason,
        int_and_reason=int_and_reason,
        examples=examples,
    )
    graph._shared_memory.reset()
    graph._forward({"message": admin_prompt})
    messages = graph._shared_memory.get_messages(top_k=0)
    discussion_history = [
        msg["content"] for msg in messages if msg["role"] == "assistant"
    ]
    strategies = re.findall(
        r"Strategy:\s*\[?([a-zA-Z ]+)\]?", "\n".join(discussion_history)
    )
    if not strategies:
        strategies = re.findall(r'\[([a-zA-Z ]+)\]', examples)
        counter = Counter(strategies)
        most_common = counter.most_common(3)
        strategies = [item[0] for item in most_common]
    cleaned = []
    for s in strategies:
        s = s.strip()
        if s and s not in cleaned:
            cleaned.append(s)
    return cleaned
