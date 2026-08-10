"""Loss accumulation and controller update node for MaAS training."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def loss_update_forward(
    input_data: dict[str, object],
    attributes: dict[str, object],
) -> dict[str, object]:
    score = float(input_data["score"])
    cost = float(input_data["cost"])
    logprob = input_data["logprob"]
    previous_cost = float(attributes.get("previous_cost", 0.0))
    cost_delta = cost - previous_cost
    attributes["previous_cost"] = cost

    attributes.setdefault("batch_logprobs", []).append(_as_logprob_tensor(logprob, attributes))
    attributes.setdefault("batch_scores", []).append(score)
    attributes.setdefault("batch_costs", []).append(cost_delta)
    attributes.setdefault("all_scores", []).append(score)

    loss_value = None
    update_performed = False
    batch_size = int(attributes["batch_size"])
    settings = attributes["settings"]

    if len(attributes["batch_logprobs"]) >= batch_size:
        if settings.mode == "Graph":
            loss = _compute_loss(attributes)
            loss_value = float(loss.detach().cpu().item())
            if loss.requires_grad:
                loss.backward()
                attributes["optimizer"].step()
                attributes["optimizer"].zero_grad()
                update_performed = True
                logger.info("MaAS controller updated at problem %s with loss %.6f", input_data["problem_index"], loss_value)
            else:
                logger.info("MaAS batch loss at problem %s has no gradient and update was skipped", input_data["problem_index"])

        attributes["batch_logprobs"] = []
        attributes["batch_scores"] = []
        attributes["batch_costs"] = []

    return {
        "result_score": score,
        "result_cost": cost,
        "result_logprob": logprob,
        "result_loss": loss_value,
        "result_update_performed": update_performed,
        "result_problem_index": input_data["problem_index"],
    }


def _compute_loss(attributes: dict[str, object]) -> torch.Tensor:
    device = _resolve_device(attributes)
    logprobs = torch.stack(attributes["batch_logprobs"]).to(device)
    scores = torch.tensor(attributes["batch_scores"], dtype=torch.float32, device=device)
    costs = torch.tensor(attributes["batch_costs"], dtype=torch.float32, device=device)
    utilities = scores - 3 * costs
    return -(logprobs * utilities).mean()


def _as_logprob_tensor(logprob: object, attributes: dict[str, object]) -> torch.Tensor:
    if isinstance(logprob, torch.Tensor):
        return logprob
    device = _resolve_device(attributes)
    return torch.tensor(float(logprob), dtype=torch.float32, device=device)


def _resolve_device(attributes: dict[str, object]) -> torch.device:
    if "device" in attributes:
        return attributes["device"]
    if "controller" in attributes:
        return next(attributes["controller"].parameters()).device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
