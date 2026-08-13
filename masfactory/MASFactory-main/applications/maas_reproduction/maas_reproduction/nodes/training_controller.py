"""Loop controller function for MaAS training and evaluation."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from maas_reproduction.nodes.artifact_writer import append_round_summary, author_round_directory, write_results_csv

logger = logging.getLogger(__name__)


def training_controller(input_data: dict[str, object], attributes: dict[str, object]) -> bool:
    for key in list(input_data.keys()):
        if key.startswith("result_"):
            del input_data[key]

    settings = attributes["settings"]
    problems = attributes["problems"]
    problem_index = int(attributes.get("problem_index", 0))
    repetition = int(attributes.get("repetition", 1))
    max_repetitions = int(settings.optimizer.sample)

    if problem_index >= len(problems):
        from maas_reproduction.nodes.loss_update_node import flush_remaining_batch

        flush_remaining_batch(attributes)
        current_repetition_score = _finish_repetition(attributes)
        if repetition >= max_repetitions:
            _write_final_result(input_data, attributes)
            logger.info("MaAS training loop finished with average score %.5f", input_data["average_score"])
            return True
        _maybe_run_textgrad(settings, attributes, current_repetition_score)
        repetition += 1
        problem_index = 0
        attributes["repetition"] = repetition
        logger.info("MaAS training loop starting repetition %s/%s", repetition, max_repetitions)

    problem = problems[problem_index]
    _write_problem_message(input_data, settings.dataset, problem, problem_index)
    attributes["problem_index"] = problem_index + 1
    attributes["repetition"] = repetition
    return False


def _write_problem_message(
    input_data: dict[str, object],
    dataset: str,
    problem: dict[str, object],
    problem_index: int,
) -> None:
    if dataset == "GSM8K":
        input_data["problem"] = problem["question"]
        input_data["entry_point"] = ""
        input_data["expected_answer"] = problem["answer"]
    elif dataset == "MATH":
        input_data["problem"] = problem["problem"]
        input_data["entry_point"] = ""
        input_data["expected_answer"] = problem["solution"]
    elif dataset == "HumanEval":
        input_data["problem"] = problem["prompt"]
        input_data["entry_point"] = problem["entry_point"]
        input_data["expected_answer"] = problem["test"]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    input_data["problem_index"] = problem_index


def _write_final_result(input_data: dict[str, object], attributes: dict[str, object]) -> None:
    settings = attributes["settings"]
    all_scores = attributes.get("all_scores", [])
    sample_results = list(attributes.get("sample_results", []))
    average_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    fallback_avg_cost, fallback_total_cost = _summarize_costs(sample_results)
    usage = _workflow_usage(attributes.get("architecture_workflow"))
    if usage is None:
        avg_cost, total_cost = fallback_avg_cost, fallback_total_cost
        prompt_tokens = completion_tokens = total_tokens = 0
    else:
        total_cost = usage["total_cost"]
        avg_cost = total_cost / len(sample_results) if sample_results else 0.0
        prompt_tokens = usage["prompt_tokens"]
        completion_tokens = usage["completion_tokens"]
        total_tokens = prompt_tokens + completion_tokens
    textgrad_events = list(attributes.get("textgrad_events", []))
    checkpoint_path = settings.paths.controller_checkpoint(
        settings.dataset,
        settings.optimizer.round_number,
        settings.optimizer.sample,
    )
    if getattr(settings, "mode", "Graph") == "Graph":
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(attributes["controller"].state_dict(), checkpoint_path)
        logger.info("Saved MaAS controller parameters to %s", checkpoint_path)

    artifact_directory = author_round_directory(settings)
    csv_path = write_results_csv(
        artifact_directory,
        list(attributes.get("result_columns", _default_result_columns(settings.dataset))),
        sample_results,
        average_score,
    )
    results_json_path = append_round_summary(
        _results_json_path(settings),
        settings.optimizer.round_number,
        average_score,
        avg_cost=avg_cost,
        total_cost=total_cost,
        token=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        textgrad_events=textgrad_events,
    )

    input_data["average_score"] = average_score
    input_data["round"] = settings.optimizer.round_number
    input_data["checkpoint_path"] = str(checkpoint_path)
    input_data["result_path"] = str(artifact_directory)
    input_data["runtime_metadata"] = {
        "dataset": settings.dataset,
        "mode": getattr(settings, "mode", "Graph"),
        "processed_problems": len(all_scores),
        "avg_cost": avg_cost,
        "total_cost": total_cost,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "textgrad_applied": any(event.get("applied", False) for event in textgrad_events),
        "textgrad_events": textgrad_events,
        "csv_path": str(csv_path),
        "results_json_path": str(results_json_path),
        "run_directory": str(settings.run_directory),
    }


def _finish_repetition(attributes: dict[str, object]) -> float:
    scores = list(attributes.get("current_repetition_scores", []))
    current_repetition_score = sum(scores) / len(scores) if scores else 0.0
    attributes["last_repetition_score"] = current_repetition_score
    attributes["current_repetition_scores"] = []
    return current_repetition_score


def _maybe_run_textgrad(settings, attributes: dict[str, object], current_repetition_score: float) -> None:
    if not getattr(settings.optimizer, "is_textgrad", False):
        attributes["previous_repetition_score"] = current_repetition_score
        return
    if attributes.get("textgrad_consumed"):
        attributes["previous_repetition_score"] = current_repetition_score
        return

    previous_repetition_score = attributes.get("previous_repetition_score")
    if previous_repetition_score is not None and current_repetition_score < float(previous_repetition_score):
        update_event = _run_textgrad_update(settings)
        attributes.setdefault("textgrad_events", []).append(update_event)
        attributes["textgrad_applied"] = bool(update_event["applied"])
        attributes["textgrad_consumed"] = True

    attributes["previous_repetition_score"] = current_repetition_score


def _run_textgrad_update(settings) -> dict[str, object]:
    from pydantic import BaseModel, Field

    from maas.actions.action_node import ActionNode
    from maas.provider.llm_provider_registry import create_llm_instance
    from maas_reproduction.runtime.async_runner import run_async_once

    prompt_directory = author_round_directory(settings)
    prompt_path = _prompt_file_path(prompt_directory)
    prompt_name, prompt_content = _extract_random_prompt(prompt_directory)
    if prompt_name is None:
        logger.info("MaAS textgrad skipped because no *_PROMPT assignment was found")
        return {"applied": False, "prompt_name": None, "prompt_path": str(prompt_path)}

    class TextGrad(BaseModel):
        prompt: str = Field(default="", description="prompt")

    context = TEXT_GRAD_PROMPT.format(
        dataset=settings.dataset,
        prompt_name=prompt_name,
        prompt_content=prompt_content,
    )
    llm = create_llm_instance(settings.opt_llm_config)
    node = run_async_once(ActionNode.from_pydantic(TextGrad).fill(context=context, mode="xml_fill", llm=llm))
    response = node.instruct_content.model_dump()
    prompt = str(response["prompt"]).strip()
    if not prompt:
        logger.info("MaAS textgrad skipped prompt %s because the generated prompt was empty", prompt_name)
        return {"applied": False, "prompt_name": prompt_name, "prompt_path": str(prompt_path)}
    applied = _update_prompt_in_file(prompt_directory, prompt_name, prompt)
    if applied:
        logger.info("MaAS textgrad updated prompt %s at repetition boundary", prompt_name)
    return {"applied": applied, "prompt_name": prompt_name, "prompt_path": str(prompt_path)}


def _extract_random_prompt(log_directory: Path):
    import ast
    import random

    prompt_file = _prompt_file_path(log_directory)
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt does not exist: {prompt_file}")

    tree = ast.parse(prompt_file.read_text(encoding="utf-8"))
    prompts = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.endswith("_PROMPT"):
                if isinstance(node.value, ast.Constant):
                    prompts[target.id] = node.value.value
                elif isinstance(node.value, ast.Str):
                    prompts[target.id] = node.value.s

    if not prompts:
        return None, None
    return random.choice(list(prompts.items()))


def _update_prompt_in_file(log_directory: Path, prompt_name: str, prompt_content: str) -> bool:
    import re
    from collections import Counter

    prompt_file = _prompt_file_path(log_directory)
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt does not exist: {prompt_file}")

    content = prompt_file.read_text(encoding="utf-8")
    pattern = rf'{re.escape(prompt_name)}\s*=\s*"""(.*?)"""'
    match = re.search(pattern, content, flags=re.DOTALL)
    if match:
        old_placeholders = re.findall(r"{([^{}]+)}", match.group(1))
        new_placeholders = re.findall(r"{([^{}]+)}", prompt_content)
        if Counter(old_placeholders) != Counter(new_placeholders):
            logger.info("MaAS textgrad skipped prompt %s because placeholders changed", prompt_name)
            return False

    new_assignment = f'{prompt_name} = """\n{prompt_content}\n"""'
    if match:
        content = re.sub(pattern, new_assignment, content, flags=re.DOTALL)
    else:
        content = content.rstrip() + "\n\n" + new_assignment + "\n"
    prompt_file.write_text(content, encoding="utf-8")
    return True


def _prompt_file_path(log_directory: Path) -> Path:
    return log_directory.parent / "template" / "op_prompt.py"


def _results_json_path(settings) -> Path:
    root = getattr(settings, "architecture_root", None)
    if root is None:
        root = Path(settings.run_directory).parent
    return Path(root) / "results.json"


def _default_result_columns(dataset: str) -> list[str]:
    if dataset == "HumanEval":
        return ["inputs", "prediction", "expected_output", "score", "cost", "logprob"]
    return ["question", "prediction", "expected_output", "score", "cost", "logprob"]


def _summarize_costs(sample_results: list[list[object]]) -> tuple[float, float]:
    if not sample_results:
        return 0.0, 0.0
    costs = []
    previous_cost = 0.0
    for row in sample_results:
        if len(row) > 4:
            try:
                cumulative_cost = float(row[4])
            except (TypeError, ValueError):
                continue
            costs.append(max(cumulative_cost - previous_cost, 0.0))
            previous_cost = max(previous_cost, cumulative_cost)
    if not costs:
        return 0.0, 0.0
    total_cost = sum(costs)
    return total_cost / len(costs), total_cost


def _workflow_usage(workflow: object) -> dict[str, float | int] | None:
    """Read provider-reported usage from the active MaAS workflow's cost manager."""
    while hasattr(workflow, "_workflow"):
        workflow = getattr(workflow, "_workflow")
    cost_manager = getattr(getattr(workflow, "llm", None), "cost_manager", None)
    if cost_manager is None:
        return None
    return {
        "prompt_tokens": int(getattr(cost_manager, "total_prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(cost_manager, "total_completion_tokens", 0) or 0),
        "total_cost": float(getattr(cost_manager, "total_cost", 0.0) or 0.0),
    }


TEXT_GRAD_PROMPT = """
Please act as an expert with extensive experience in Natural Language Processing and model tuning. Your task is to generate a detailed and specific prompt tailored to the {dataset} dataset.

The original prompt content is as follows: {prompt_name} = {prompt_content}

Please adhere to the following requirements:
1. Clearly describe the core characteristics and challenges of the dataset, analyze the type of tasks involved, and make targeted prompt adjustments accordingly.
2. Strictly preserve all placeholders in the original prompt content without any modifications, ensuring that the number and names of the placeholders remain unchanged for subsequent replacements.
3. Return the newly generated prompt content in the 'prompt' field. This field should include only the new prompt content (excluding the name) and should not contain any triple quotation marks.
4.  Include your single prompt in XML tags in your reply.
"""
