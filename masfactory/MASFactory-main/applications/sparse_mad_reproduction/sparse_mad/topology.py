from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Topology:
    agents: list[str]
    edges: list[tuple[str, str]]
    neighbors: dict[str, list[str]]
    density: float

    def as_dict(self) -> dict:
        return {
            "agents": self.agents,
            "edges": self.edges,
            "neighbors": self.neighbors,
            "density": self.density,
        }


def graph_density(num_agents: int, edge_count: int) -> float:
    if num_agents < 2:
        raise ValueError("num_agents must be at least 2")
    return 2 * edge_count / (num_agents * (num_agents - 1))


def build_topology(num_agents: int, topology_type: str) -> Topology:
    if num_agents < 2:
        raise ValueError("num_agents must be at least 2")

    agents = [f"Agent_{index}" for index in range(1, num_agents + 1)]
    topology_type = topology_type.lower()

    if topology_type in {"ring", "neighbor_connected", "neighbor"}:
        edges = [(agents[index], agents[(index + 1) % num_agents]) for index in range(num_agents)]
    elif topology_type in {"fully_connected", "full"}:
        edges = [
            (agents[left], agents[right])
            for left in range(num_agents)
            for right in range(left + 1, num_agents)
        ]
    else:
        raise ValueError(f"Unsupported topology_type: {topology_type}")

    neighbors = {agent: [] for agent in agents}
    for left, right in edges:
        neighbors[left].append(right)
        neighbors[right].append(left)

    return Topology(
        agents=agents,
        edges=edges,
        neighbors=neighbors,
        density=graph_density(num_agents, len(edges)),
    )
