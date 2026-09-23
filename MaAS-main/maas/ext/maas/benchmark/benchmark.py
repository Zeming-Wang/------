import asyncio
import csv
import json
import os
import torch
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
from pydantic import BaseModel, Field
from maas.actions.action_node import ActionNode
import aiofiles
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from maas.configs.models_config import ModelsConfig
from maas.provider.llm_provider_registry import create_llm_instance
from maas.logs import logger
from maas.utils.common import write_json_file
from maas.ext.maas.scripts.utils import extract_random_prompt, update_prompt_in_file
from maas.ext.maas.scripts.textgrad.textual_gradient import TEXT_GRAD_PROMPT

class TextGrad(BaseModel):
    prompt: str = Field(default="", description="prompt")


class _TrainingInterrupted(Exception):
    """Carry Ctrl+C through asyncio.gather without losing the batch boundary."""


def _normalize_resume_position(
    repetition: int,
    next_batch_idx: int,
    total_batches: int,
) -> Tuple[int, int]:
    """Move a completed repetition to the first batch of the next one."""
    if total_batches > 0 and next_batch_idx >= total_batches:
        return repetition + 1, 0
    return repetition, next_batch_idx


class BaseBenchmark(ABC):
    def __init__(
        self,
        name: str,
        file_path: str,
        log_path: str,
        batch_size: int,
        controller: torch.nn.Module,
        operator_embeddings,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.name = name
        self.file_path = file_path
        self.log_path = log_path
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.controller = controller.to(self.device)
        self.operator_embeddings = operator_embeddings.to(self.device)
        self.optimizer = optimizer

    PASS = "PASS"
    FAIL = "FAIL"

    async def load_data(self, specific_indices: List[int] = None) -> List[dict]:
        data = []
        async with aiofiles.open(self.file_path, mode="r", encoding="utf-8") as file:
            async for line in file:
                data.append(json.loads(line))
        if specific_indices is not None:
            filtered_data = [data[i] for i in specific_indices if i < len(data)]
            return filtered_data
        return data

    def save_results_to_csv(self, results: List[Tuple[Any, ...]], columns: List[str]):
        df = pd.DataFrame(results, columns=columns)
        avg_score = df["score"].mean()
        if "cost" in df.columns:
            df = df.drop(columns=["cost"])
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{avg_score:.5f}_{current_time}.csv"
        output_file = os.path.join(self.log_path, filename)
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        return avg_score

    def save_cost_summary(
        self,
        graph,
        mode: str,
        sample: int,
        problem_count: int,
        average_score: Optional[float],
        status: str,
    ) -> None:
        """Append one main-graph usage row for a complete train/test invocation.

        TextGrad intentionally uses a separate LLM and is outside this summary's
        accounting boundary. Reading only ``graph.llm`` keeps the totals aligned
        with the operator execution costs used by the existing training logic.
        """
        costs = graph.llm.get_costs()
        summary_file = Path(self.log_path).parent.parent / "cost_token_summary.csv"
        summary_file.parent.mkdir(parents=True, exist_ok=True)

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": mode,
            "dataset": self.name,
            "sample_count": sample,
            "problem_count": problem_count,
            "average_score": "" if average_score is None else f"{average_score:.5f}",
            "prompt_tokens": costs.total_prompt_tokens,
            "completion_tokens": costs.total_completion_tokens,
            "total_tokens": costs.total_prompt_tokens + costs.total_completion_tokens,
            "total_cost": f"{float(costs.total_cost):.8f}",
            "status": status,
        }
        write_header = not summary_file.exists() or summary_file.stat().st_size == 0

        with summary_file.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        logger.info(f"Cost summary saved to {summary_file}")

    def save_training_checkpoint(
        self,
        graph,
        repetition: int,
        next_batch_idx: int,
        filename: str = "latest.pt",
    ) -> Path:
        checkpoint_dir = Path(self.log_path) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / filename
        temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        payload = {
            "controller": self.controller.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "repetition": repetition,
            "next_batch_idx": next_batch_idx,
            "total_cost": float(graph.llm.get_costs().total_cost),
        }
        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
        logger.info(
            f"Training checkpoint saved to {checkpoint_path}: "
            f"repetition={repetition}, next_batch_idx={next_batch_idx}"
        )
        return checkpoint_path

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {key: BaseBenchmark._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [BaseBenchmark._json_safe(item) for item in value]
        if hasattr(value, "detach"):
            value = value.detach().cpu()
            return value.item() if value.numel() == 1 else value.tolist()
        return value

    def _test_results_path(self) -> Path:
        path = Path(self.log_path) / "test_progress.jsonl"
        legacy_path = Path(self.log_path) / "results.jsonl"
        if not path.exists() and legacy_path.exists():
            os.replace(legacy_path, path)
            logger.info(f"Migrated test progress from {legacy_path} to {path}")
        return path

    def _load_test_results(self) -> dict:
        path = self._test_results_path()
        if not path.exists():
            return {}

        completed = {}
        valid_rows = []
        repair_required = False
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    repair_required = True
                    logger.warning(
                        f"Ignoring incomplete test result at {path}:{line_number}"
                    )
                    continue
                problem_id = row.get("problem_id")
                if problem_id is None:
                    repair_required = True
                    continue
                problem_id = str(problem_id)
                row["problem_id"] = problem_id
                completed[problem_id] = row
                valid_rows.append(row)

        if repair_required:
            temporary_path = path.with_suffix(path.suffix + ".tmp")
            with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
                for row in valid_rows:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
            os.replace(temporary_path, path)
        return completed

    def _append_test_result(self, problem_id: str, result: Tuple[Any, ...]) -> dict:
        columns = self.get_result_columns()
        row = {"problem_id": str(problem_id)}
        row.update(
            {
                column: self._json_safe(value)
                for column, value in zip(columns, result)
            }
        )
        path = self._test_results_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
        return row

    def log_mismatch(
        self,
        problem: str,
        expected_output: Any,
        prediction: str,
        extracted_output: Any,
        extract_answer_code: str = "None",
    ):
        log_data = {
            "question": problem,
            "right_answer": expected_output,
            "model_output": prediction,
            "extracted_output": extracted_output,
            "extract_answer_code": extract_answer_code,
        }
        log_file = Path(self.log_path) / "log.json"
        if log_file.exists():
            with log_file.open("r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        data.append(log_data)
        write_json_file(log_file, data, encoding="utf-8", indent=4)

    @abstractmethod
    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[Any, ...]:
        pass

    @abstractmethod
    def calculate_score(self, expected_output: Any, prediction: Any) -> Tuple[float, Any]:
        pass

    @abstractmethod
    def get_result_columns(self) -> List[str]:
        pass

    async def evaluate_all_problems(
        self,
        data: List[dict],
        graph: Callable,
        max_concurrent_tasks: int = 30,
        repetitions: int = 4,
        is_textgrad: bool = False,
        start_repetition: int = 1,
        start_batch_idx: int = 0,
        resume_total_cost: float = 0.0,
    ):
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        results = []
        previous_cost = float(resume_total_cost)
        textgrad = False           
        prev_rep_score = None   
        total_batches = (len(data) + self.batch_size - 1) // self.batch_size
        start_repetition, start_batch_idx = _normalize_resume_position(
            start_repetition,
            start_batch_idx,
            total_batches,
        )
        interrupted_repetition = start_repetition
        interrupted_batch_idx = start_batch_idx

        async def sem_evaluate(problem):
            async with semaphore:
                try:
                    return await self.evaluate_problem(problem, graph)
                except KeyboardInterrupt as error:
                    raise _TrainingInterrupted(str(error)) from error
                except Exception as e:
                    logger.error(f"Error evaluating problem: {e}")
                    return ("", "", "", 0.0, 0.0, 0.0)
        
        try:
            for rep in range(start_repetition, repetitions + 1):
                first_batch_idx = start_batch_idx if rep == start_repetition else 0
                interrupted_repetition = rep
                interrupted_batch_idx = first_batch_idx
                logger.info(
                    f"Starting training repetition {rep}/{repetitions} "
                    f"from batch {first_batch_idx}/{total_batches}"
                )
                rep_scores = []

                if textgrad and is_textgrad:
                    prompt_name, prompt_content = extract_random_prompt(self.log_path)
                    textgrad_prompt = TEXT_GRAD_PROMPT.format(dataset = self.name, prompt_name = prompt_name, prompt_content = prompt_content)
                    textgrad_llm_config = ModelsConfig.default().get("gpt-4o-mini")
                    textgrad_llm = create_llm_instance(textgrad_llm_config)
                    textgrad_node = await ActionNode.from_pydantic(TextGrad).fill(context=textgrad_prompt, mode="xml_fill", llm=textgrad_llm)
                    response = textgrad_node.instruct_content.model_dump()
                    update_prompt_in_file(prompt_name, response["prompt"])
                    is_textgrad = False

                for batch_idx in range(first_batch_idx, total_batches):
                    interrupted_batch_idx = batch_idx
                    batch_start = batch_idx * self.batch_size
                    batch = data[batch_start:batch_start + self.batch_size]
                    tasks = [sem_evaluate(problem) for problem in batch]
                    batch_results = await tqdm_asyncio.gather(
                        *tasks,
                        desc=f"Repetition {rep}: Executing batch {batch_idx + 1}",
                        total=len(batch)
                    )
                    results.extend(batch_results)

                    logprobs = []
                    scores = []
                    costs = []
                    for r in batch_results:
                        logprob = r[5]
                        cost = r[4]
                        score = r[3]
                        logprobs.append(logprob)
                        scores.append(score)
                        costs.append(cost - previous_cost)
                        previous_cost = cost
                        rep_scores.append(score)

                    if len(logprobs) > 0:
                        logprobs = torch.stack(logprobs).to(self.device)
                        scores_tensor = torch.tensor(scores, dtype=torch.float32, device=self.device)
                        costs_tensor = torch.tensor(costs, dtype=torch.float32, device=self.device)
                        utilities = scores_tensor - 3 * costs_tensor
                        loss = -(logprobs * utilities).mean()
                        if loss.requires_grad:
                            loss.backward()
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            logger.info(f"Repetition {rep}: Batch {batch_idx + 1} Loss: {loss.item()}")
                        else:
                            logger.info(f"Repetition {rep}: Batch {batch_idx + 1} Loss does not require grad and was skipped.")
                    else:
                        logger.info(f"Repetition {rep}: Batch {batch_idx + 1} skipped due to invalid logprobs.")

                    interrupted_batch_idx = batch_idx + 1
                    self.save_training_checkpoint(
                        graph,
                        repetition=rep,
                        next_batch_idx=batch_idx + 1,
                    )

                if rep_scores:
                    current_rep_score = sum(rep_scores) / len(rep_scores)
                else:
                    current_rep_score = 0.0

                if not textgrad:
                    if prev_rep_score is not None and current_rep_score < prev_rep_score:
                        textgrad = True
                    prev_rep_score = current_rep_score
        except (_TrainingInterrupted, KeyboardInterrupt) as error:
            try:
                self.save_training_checkpoint(
                    graph,
                    repetition=interrupted_repetition,
                    next_batch_idx=interrupted_batch_idx,
                    filename="interrupted.pt",
                )
            except Exception as checkpoint_error:
                logger.error(
                    f"Failed to save interrupted checkpoint: {checkpoint_error}"
                )
            if isinstance(error, _TrainingInterrupted):
                raise KeyboardInterrupt(str(error)) from error
            raise

        return results
    
    async def evaluate_all_problems_test(self, data: List[dict], graph: Callable, max_concurrent_tasks: int = 10):
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        columns = self.get_result_columns()
        persisted = self._load_test_results()
        results_by_id = {}

        async def sem_evaluate(problem_id, problem):
            async with semaphore:
                result = await self.evaluate_problem(problem, graph)
                self._append_test_result(problem_id, result)
                return problem_id, result

        tasks = []
        ordered_problem_ids = []
        for index, problem in enumerate(data):
            problem_id = str(problem.get("__maas_problem_id", f"{self.name}:{index}"))
            ordered_problem_ids.append(problem_id)
            stored = persisted.get(problem_id)
            if stored is not None and all(column in stored for column in columns):
                results_by_id[problem_id] = tuple(stored[column] for column in columns)
                continue
            tasks.append(asyncio.create_task(sem_evaluate(problem_id, problem)))

        for task in asyncio.as_completed(tasks):
            problem_id, result = await task
            results_by_id[problem_id] = result

        return [results_by_id[problem_id] for problem_id in ordered_problem_ids]
    
    async def run_evaluation(
        self,
        graph: Callable,
        va_list: List[int],
        is_test: bool,
        sample: int,
        is_textgrad: bool = False,
        max_concurrent_tasks: int = 30,
        start_repetition: int = 1,
        start_batch_idx: int = 0,
        resume_total_cost: float = 0.0,
    ):
        mode = "test" if is_test else "train"
        data = []
        average_score = None
        status = "failed"

        try:
            cost_manager = getattr(getattr(graph, "llm", None), "cost_manager", None)
            if cost_manager is not None and resume_total_cost:
                cost_manager.total_cost = float(resume_total_cost)
            data = await self.load_data(va_list)

            if is_test == True:
                results = await self.evaluate_all_problems_test(data, graph, max_concurrent_tasks)
                columns = self.get_result_columns()
                average_score = self.save_results_to_csv(results, columns)
                logger.info(f"Average score on {self.name} dataset: {average_score:.5f}")
                status = "success"
                return average_score

            results = await self.evaluate_all_problems(
                data,
                graph,
                max_concurrent_tasks,
                sample,
                is_textgrad,
                start_repetition,
                start_batch_idx,
                resume_total_cost,
            )

            columns = self.get_result_columns()
            average_score = self.save_results_to_csv(results, columns)
            logger.info(f"Average score on {self.name} dataset: {average_score:.5f}")

            try:
                os.makedirs(self.log_path, exist_ok=True)
                controller_path = os.path.join(self.log_path, f"{self.name}_controller_sample{sample}.pth")
                torch.save(self.controller.state_dict(), controller_path)
                logger.info(f"Saved controller parameters to {controller_path}")
                logger.info("Successfully Finish Training")
            except Exception as e:
                logger.error(f"Failed to save controller parameters: {e}")

            status = "success"
            return average_score
        finally:
            try:
                self.save_cost_summary(
                    graph=graph,
                    mode=mode,
                    sample=sample,
                    problem_count=len(data),
                    average_score=average_score,
                    status=status,
                )
            except Exception as summary_error:
                # Accounting must not replace the run's original return value or
                # exception. The error remains visible in the normal application log.
                logger.error(f"Failed to save cost summary: {summary_error}")
