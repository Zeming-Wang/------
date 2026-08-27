import asyncio
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import torch

from maas.configs.models_config import ModelsConfig
from maas.ext.maas.benchmark.experiment_configs import EXPERIMENT_CONFIGS
from maas.ext.maas.benchmark.gsm8k import GSM8KBenchmark
from maas.ext.maas.scripts.optimizer import Optimizer
from maas.ext.maas.scripts.optimizer_utils.graph_utils import GraphUtils
from maas.ext.maas.models.utils import get_sentence_embedding
from maas.logs import logger


DATA_LIMIT = 10
REPETITIONS = 2
BATCH_SIZE = 4
OPTIMIZED_PATH = "maas/ext/maas/scripts/optimized"
SMOKE_PATH = Path(OPTIMIZED_PATH) / "GSM8K" / "train" / "smoke_10x2"


async def run_smoke_test():
    SMOKE_PATH.mkdir(parents=True, exist_ok=True)
    log_sink = logger.add(SMOKE_PATH / "smoke.log", level="DEBUG")

    try:
        config = EXPERIMENT_CONFIGS["GSM8K"]
        models_config = ModelsConfig.default()
        exec_llm_config = models_config.get("gpt-4o-mini")
        if exec_llm_config is None:
            raise ValueError("Model 'gpt-4o-mini' was not found in config/config2.yaml")

        optimizer = Optimizer(
            dataset=config.dataset,
            question_type=config.question_type,
            opt_llm_config=exec_llm_config,
            exec_llm_config=exec_llm_config,
            operators=config.operators,
            optimized_path=OPTIMIZED_PATH,
            sample=REPETITIONS,
            round=1,
            batch_size=BATCH_SIZE,
            lr=0.01,
            is_textgrad=False,
        )

        graph_path = f"{optimizer.root_path}/train"
        graph_utils = GraphUtils(optimizer.root_path)
        operator_descriptions = graph_utils.load_operators_description_maas(config.operators)
        operator_embeddings = torch.stack(
            [get_sentence_embedding(description) for description in operator_descriptions]
        ).to(optimizer.device)

        graph_class = graph_utils.load_graph_maas(graph_path)
        graph = graph_class(
            name=config.dataset,
            llm_config=exec_llm_config,
            dataset=config.dataset,
            controller=optimizer.controller,
            operator_embeddings=operator_embeddings,
        )

        benchmark = GSM8KBenchmark(
            name="GSM8K",
            file_path="maas/ext/maas/data/gsm8k_train.jsonl",
            log_path=str(SMOKE_PATH),
            batch_size=BATCH_SIZE,
            controller=optimizer.controller,
            operator_embeddings=operator_embeddings,
            optimizer=optimizer.optimizer,
        )

        all_data = await benchmark.load_data()
        data = all_data[:DATA_LIMIT]
        if len(data) < DATA_LIMIT:
            raise ValueError(f"Expected at least {DATA_LIMIT} GSM8K samples, found {len(data)}")

        logger.info(
            f"Starting GSM8K smoke test: samples={len(data)}, "
            f"repetitions={REPETITIONS}, batch_size={BATCH_SIZE}"
        )

        results = await benchmark.evaluate_all_problems(
            data=data,
            graph=graph,
            max_concurrent_tasks=BATCH_SIZE,
            repetitions=REPETITIONS,
            is_textgrad=False,
        )

        average_score = benchmark.save_results_to_csv(
            results,
            benchmark.get_result_columns(),
        )

        costs = [float(result[4]) for result in results]
        total_cost = sum(costs)
        average_cost = total_cost / len(costs) if costs else 0.0
        result_data = optimizer.data_utils.create_result_data(
            REPETITIONS,
            float(average_score),
            average_cost,
            total_cost,
            0,
        )
        optimizer.data_utils.save_results(
            str(SMOKE_PATH / "results.json"),
            [result_data],
        )

        checkpoint_path = SMOKE_PATH / "GSM8K_controller_sample2.pth"
        torch.save(optimizer.controller.state_dict(), checkpoint_path)
        logger.info(f"Smoke test completed: score={average_score:.5f}")
        logger.info(f"Saved smoke checkpoint to {checkpoint_path}")
        logger.info(f"Smoke output directory: {SMOKE_PATH}")
    finally:
        logger.remove(log_sink)


if __name__ == "__main__":
    asyncio.run(run_smoke_test())
