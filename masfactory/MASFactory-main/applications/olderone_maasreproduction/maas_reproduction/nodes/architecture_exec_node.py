"""Architecture execution node for MaAS workflows.

The node keeps MaAS architecture execution intact: it calls the migrated
``Workflow.__call__`` once for the current problem and returns the original fields needed by
the evaluator plus MaAS' ``prediction``, ``cost``, and ``logprob`` result tuple.
"""

from __future__ import annotations

import asyncio
import logging

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed

from maas_reproduction.runtime.async_runner import AsyncRunnerContextError, run_async_once

logger = logging.getLogger(__name__)
_MAAS_WORKFLOW_TIMEOUT_SECONDS = 200
_MAAS_WORKFLOW_MAX_RETRIES = 2


async def _call_workflow(workflow, *args):
    """Call a MaAS workflow with the same timeout used by the original benchmarks."""
    return await asyncio.wait_for(workflow(*args), timeout=_MAAS_WORKFLOW_TIMEOUT_SECONDS)


@retry(
    stop=stop_after_attempt(_MAAS_WORKFLOW_MAX_RETRIES),
    wait=wait_fixed(1),
    retry=retry_if_exception(lambda exc: not isinstance(exc, AsyncRunnerContextError)),
    reraise=True,
)
def _run_workflow_with_retry(workflow, *args):
    """Run one MaAS workflow call with the original benchmark retry policy."""
    return run_async_once(_call_workflow(workflow, *args))


def architecture_exec_forward(
    input_data: dict[str, object],
    attributes: dict[str, object],
) -> dict[str, object]:
    """Run the current problem through the active MaAS architecture workflow."""
    settings = attributes["settings"]
    workflow = attributes["architecture_workflow"]
    problem = str(input_data["problem"])
    entry_point = str(input_data["entry_point"])

    try:
        if settings.dataset == "HumanEval":
            result = _run_workflow_with_retry(
                workflow,
                problem,
                entry_point,
                str(settings.run_directory),
            )
        else:
            result = _run_workflow_with_retry(workflow, problem)
        prediction, cost, logprob = result
    except AsyncRunnerContextError:
        raise
    except Exception as exc:
        logger.exception(
            "Architecture execution failed for problem index %s",
            input_data["problem_index"],
        )
        prediction, cost, logprob = str(exc), 0.0, 0.0

    return {
        "problem": problem,
        "entry_point": entry_point,
        "expected_answer": input_data["expected_answer"],
        "prediction": prediction,
        "cost": cost,
        "logprob": logprob,
        "problem_index": input_data["problem_index"],
    }
