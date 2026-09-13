"""Run a small live GSM8K Graph smoke test with controller update evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from maas_reproduction.config import MaASRuntimeSettings
from maas_reproduction.nodes.config_node import config_forward
from maas_reproduction.runtime import build_runtime_attributes
from maas_reproduction.workflow import build_maas_reproduction_graph


class RecordingWorkflow:
    def __init__(self, workflow: Any, calls: list[dict[str, Any]]) -> None:
        self._workflow = workflow
        self._calls = calls

    async def __call__(self, *args: Any) -> Any:
        prediction, cost, logprob = await self._workflow(*args)
        self._calls.append(
            {
                "args": [str(arg)[:160] for arg in args],
                "prediction": str(prediction),
                "cost": float(cost),
                "logprob": _describe_logprob(logprob),
            }
        )
        return prediction, cost, logprob


def run_smoke(
    indices: list[int],
    batch_size: int,
    opt_model_name: str,
    exec_model_name: str,
) -> dict[str, Any]:
    application_root = Path(__file__).resolve().parents[1]
    input_data = {
        "application_root": application_root,
        "dataset": "GSM8K",
        "mode": "Graph",
        "sample": 1,
        "round_number": 1,
        "batch_size": batch_size,
        "learning_rate": 0.01,
        "is_textgrad": False,
        "opt_model_name": opt_model_name,
        "exec_model_name": exec_model_name,
    }
    config = config_forward(input_data, {})
    settings: MaASRuntimeSettings = config["settings"]
    attributes = build_runtime_attributes(settings, specific_indices=indices)
    workflow_calls: list[dict[str, Any]] = []
    attributes["architecture_workflow"] = RecordingWorkflow(
        attributes["architecture_workflow"],
        workflow_calls,
    )

    before = _clone_controller_state(attributes["controller"])
    graph = build_maas_reproduction_graph()
    graph.build()
    output, returned_attributes = graph.invoke(input_data, attributes=attributes)
    after = attributes["controller"].state_dict()
    changed = _changed_tensors(before, after)

    return {
        "output": output,
        "dataset_file": str(settings.dataset_file),
        "loaded_problem_count": len(attributes["problems"]),
        "loaded_problem_preview": [
            {
                "problem_index": index,
                "question": problem["question"],
                "answer": problem["answer"].split("####")[-1].strip(),
            }
            for index, problem in zip(indices, attributes["problems"])
        ],
        "workflow_call_count": len(workflow_calls),
        "workflow_calls": workflow_calls,
        "all_scores": attributes["all_scores"],
        "optimizer_state_entries": len(attributes["optimizer"].state),
        "changed_controller_tensors": changed,
        "changed_controller_tensor_count": len(changed),
        "max_controller_param_delta": _max_delta(before, after),
        "checkpoint_exists": Path(str(output["checkpoint_path"])).exists(),
        "batch_buffers_after": {
            "logprobs": len(attributes["batch_logprobs"]),
            "scores": len(attributes["batch_scores"]),
            "costs": len(attributes["batch_costs"]),
        },
        "returned_attribute_keys": sorted(returned_attributes.keys()),
    }


def _clone_controller_state(controller: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in controller.state_dict().items()
    }


def _changed_tensors(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> list[str]:
    return [
        name
        for name, before_tensor in before.items()
        if not torch.equal(before_tensor, after[name].detach().cpu())
    ]


def _max_delta(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> float:
    max_delta = 0.0
    for name, before_tensor in before.items():
        delta = (after[name].detach().cpu() - before_tensor).abs().max().item()
        max_delta = max(max_delta, float(delta))
    return max_delta


def _describe_logprob(logprob: Any) -> dict[str, Any]:
    if isinstance(logprob, torch.Tensor):
        return {
            "type": "tensor",
            "value": float(logprob.detach().cpu().item()),
            "requires_grad": bool(logprob.requires_grad),
        }
    return {"type": type(logprob).__name__, "value": float(logprob)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--opt-model-name", default="gpt-4o-mini")
    parser.add_argument("--exec-model-name", default="gpt-4o-mini")
    args = parser.parse_args()
    print(
        json.dumps(
            run_smoke(
                args.indices,
                args.batch_size,
                args.opt_model_name,
                args.exec_model_name,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
