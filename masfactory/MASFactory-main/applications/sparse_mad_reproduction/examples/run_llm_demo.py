from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparse_mad.llm_runner import OpenAICompatibleClient
from sparse_mad.workflow import build_sparse_mad_llm_graph


def main() -> None:
    client = OpenAICompatibleClient()
    graph = build_sparse_mad_llm_graph(client=client)
    output, _attrs = graph.invoke(
        {
            "question": "If a train travels 60 km in 1.5 hours, what is its average speed in km/h?",
            "num_agents": 4,
            "topology_type": "ring",
            "max_rounds": 2,
        }
    )

    print(output["summary"])
    print("\nFinal agent answers:")
    for agent, answer in output["final_agent_answers"].items():
        print(f"{agent}: {answer}")

    print("\nFinal-round reasoning:")
    final_round = max(output["responses_by_round"])
    for agent, response in output["responses_by_round"][final_round].items():
        print(f"\n[{agent}]")
        print(response["reasoning"])


if __name__ == "__main__":
    main()
