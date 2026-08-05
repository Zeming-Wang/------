from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparse_mad.workflow import build_sparse_mad_dry_graph


def main() -> None:
    graph = build_sparse_mad_dry_graph()
    output, _attrs = graph.invoke(
        {
            "question": "What is 20 + 22?",
            "num_agents": 6,
            "topology_type": "ring",
            "max_rounds": 3,
            "correct_answer": "42",
        }
    )

    print(output["summary"])
    print("\nTopology neighbors:")
    for agent, neighbors in output["topology"]["neighbors"].items():
        print(f"{agent}: {', '.join(neighbors)}")

    print("\nFinal agent answers:")
    for agent, answer in output["final_agent_answers"].items():
        print(f"{agent}: {answer}")


if __name__ == "__main__":
    main()
