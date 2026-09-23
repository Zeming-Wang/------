import asyncio
import time
import torch
import os
import numpy as np
from pathlib import Path
from typing import List, Literal

from pydantic import BaseModel, Field
from maas.ext.maas.scripts.evaluator import DatasetType
from maas.ext.maas.scripts.optimizer_utils.data_utils import DataUtils               
from maas.ext.maas.scripts.optimizer_utils.experience_utils import ExperienceUtils
from maas.ext.maas.scripts.optimizer_utils.evaluation_utils import EvaluationUtils
from maas.ext.maas.scripts.optimizer_utils.graph_utils import GraphUtils           
from maas.logs import logger
from maas.ext.maas.models.utils import get_sentence_embedding
from maas.ext.maas.models.controller import MultiLayerController

QuestionType = Literal["math", "code", "qa"]
OptimizerType = Literal["Graph", "Test"]

class GraphOptimize(BaseModel):
    modification: str = Field(default="", description="modification")
    graph: str = Field(default="", description="graph")
    prompt: str = Field(default="", description="prompt")


class Optimizer:
    def __init__(   
        self,
        dataset: DatasetType,
        question_type: QuestionType,
        opt_llm_config,
        exec_llm_config,
        operators: List,
        sample: int,
        optimized_path: str = None,
        round: int = 1,
        batch_size: int = 4,
        lr: float = 0.01,
        is_textgrad: bool = False,
        resume: str = None,
    ) -> None:
        self.optimize_llm_config = opt_llm_config
        self.execute_llm_config = exec_llm_config
        self.dataset = dataset
        self.type = question_type
        self.graph = None
        self.operators = operators
        self.root_path = f"{optimized_path}/{self.dataset}"
        self.sample = sample
        self.top_scores = []
        self.round = round
        self.batch_size = batch_size
        self.lr = lr
        self.is_textgrad = is_textgrad
        self.graph_utils = GraphUtils(self.root_path)
        self.data_utils = DataUtils(self.root_path)
        self.experience_utils = ExperienceUtils(self.root_path)
        self.evaluation_utils = EvaluationUtils(self.root_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.controller = MultiLayerController(device=self.device).to(self.device)
        #将控制器实例化 
        self.optimizer = torch.optim.Adam(self.controller.parameters(), lr=self.lr)
        self.resume_repetition = 1
        self.resume_next_batch_idx = 0
        self.resume_total_cost = 0.0
        if resume is not None:
            self._load_resume_checkpoint(resume)

    def _load_resume_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)

        required = {
            "controller",
            "optimizer",
            "repetition",
            "next_batch_idx",
            "total_cost",
        }
        if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
            missing = sorted(required.difference(checkpoint if isinstance(checkpoint, dict) else {}))
            raise ValueError(f"Invalid resume checkpoint; missing fields: {missing}")

        repetition = checkpoint["repetition"]
        next_batch_idx = checkpoint["next_batch_idx"]
        total_cost = checkpoint["total_cost"]
        if isinstance(repetition, bool) or not isinstance(repetition, int) or repetition < 1:
            raise ValueError("Resume checkpoint repetition must be a positive integer")
        if isinstance(next_batch_idx, bool) or not isinstance(next_batch_idx, int) or next_batch_idx < 0:
            raise ValueError("Resume checkpoint next_batch_idx must be a non-negative integer")
        if isinstance(total_cost, bool) or not isinstance(total_cost, (int, float)) or total_cost < 0:
            raise ValueError("Resume checkpoint total_cost must be a non-negative number")

        self.controller.load_state_dict(checkpoint["controller"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.resume_repetition = repetition
        self.resume_next_batch_idx = next_batch_idx
        self.resume_total_cost = float(total_cost)
        logger.info(
            f"Resumed training from {path}: repetition={repetition}, "
            f"next_batch_idx={next_batch_idx}, total_cost={self.resume_total_cost}"
        )

    def optimize(self, mode: OptimizerType = "Graph"):
        if mode == "Test":
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            score = loop.run_until_complete(self.test())
            return None

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        retry_count = 0
        max_retries = 1
        round = 1

        while retry_count < max_retries:
            try:
                score = loop.run_until_complete(self._optimize_graph_maas()) 
                break
            except KeyboardInterrupt:
                try:
                    self._save_interrupted_checkpoint()
                except Exception as checkpoint_error:
                    logger.error(
                        f"Failed to save interrupted checkpoint: {checkpoint_error}"
                    )
                raise
            except Exception as e:
                retry_count += 1
                logger.info(f"Error occurred: {e}. Retrying... (Attempt {retry_count}/{max_retries})")
                if retry_count == max_retries:
                    logger.info("Max retries reached. Moving to next round.")
                    score = None

                wait_time = 5 * retry_count
                time.sleep(wait_time)

            if retry_count < max_retries: 
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

        logger.info(f"Score for round {round}: {score}")
        round += 1
        
        time.sleep(5)

    def _save_interrupted_checkpoint(self) -> Path:
        checkpoint_dir = (
            Path(self.root_path)
            / "train"
            / f"round_{self.round}"
            / "checkpoints"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        latest_path = checkpoint_dir / "latest.pt"
        interrupted_path = checkpoint_dir / "interrupted.pt"
        temporary_path = interrupted_path.with_suffix(interrupted_path.suffix + ".tmp")

        payload = None
        if latest_path.is_file():
            try:
                payload = torch.load(latest_path, map_location=self.device, weights_only=False)
            except TypeError:
                payload = torch.load(latest_path, map_location=self.device)
            except Exception as error:
                logger.info(f"Could not reuse latest checkpoint during interrupt: {error}")

        if not isinstance(payload, dict):
            total_cost = self.resume_total_cost
            if self.graph is not None:
                try:
                    total_cost = float(self.graph.llm.get_costs().total_cost)
                except Exception:
                    pass
            payload = {
                "controller": self.controller.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "repetition": self.resume_repetition,
                "next_batch_idx": self.resume_next_batch_idx,
                "total_cost": total_cost,
            }

        torch.save(payload, temporary_path)
        os.replace(temporary_path, interrupted_path)
        logger.info(f"Interrupted training checkpoint saved to {interrupted_path}")
        return interrupted_path

    #是整个训练的主循环
    async def _optimize_graph_maas(self):
        graph_path = f"{self.root_path}/train"
        data = self.data_utils.load_results(graph_path)

        operator_descriptions = self.graph_utils.load_operators_description_maas(self.operators) 
        #将自然语言转为Tensor向量，并存入GPU 算子向量化 对应公式9
        precomputed_operator_embeddings = torch.stack([get_sentence_embedding(op_desc) for op_desc in operator_descriptions]).to(self.device)
        directory = self.graph_utils.create_round_directory(graph_path, self.round)
        logger.info(directory)

        self.graph = self.graph_utils.load_graph_maas(graph_path)

        params = {
            "operator_embeddings": precomputed_operator_embeddings,
            "controller": self.controller,
            "execute_llm_config": self.execute_llm_config, 
            "dataset": self.dataset, 
            "optimizer": self.optimizer,
            "sample": self.sample,
            "is_textgrad": self.is_textgrad,
            "resume_repetition": self.resume_repetition,
            "resume_next_batch_idx": self.resume_next_batch_idx,
            "resume_total_cost": self.resume_total_cost,
        }

        avg_score = await self.evaluation_utils.evaluate_graph_maas(self, directory, data, initial=False, params=params)

        return avg_score
    
        #异步执行一轮完整的多智能体采样执行与梯度更新

    async def test(self):
        data = []
        graph_path = f"{self.root_path}/test"
        
        json_file_path = self.data_utils.get_results_file_path(graph_path)
        data = self.data_utils.load_results(graph_path)

        operator_descriptions = self.graph_utils.load_operators_description_maas(self.operators) 
        precomputed_operator_embeddings = torch.stack([get_sentence_embedding(op_desc) for op_desc in operator_descriptions]).to(self.device)

        self.graph = self.graph_utils.load_graph_maas(graph_path)
        directory = self.graph_utils.create_round_directory(graph_path, self.round)

        #加载已经训练好的权重即整个训练成果 并设为评估模式
        pth_path = f"{self.root_path}/train"
        pth_directory = self.graph_utils.create_round_directory(pth_path, self.round)
        controller_path = os.path.join(pth_directory,  f"{self.dataset}_controller_sample{self.sample}.pth")
        logger.info(controller_path)

        if os.path.exists(controller_path):
            checkpoint = torch.load(controller_path, map_location=self.device)
            self.controller.load_state_dict(checkpoint)
            self.controller.eval()
        else:
            raise FileNotFoundError(f"Controller model file not found at {controller_path}")         

        params = {
            "operator_embeddings": precomputed_operator_embeddings,
            "controller": self.controller,
            "execute_llm_config": self.execute_llm_config,  
            "dataset": self.dataset,                        
            "optimizer": self.optimizer,
            "sample": self.sample,
            "is_textgrad": False
        }

        score = await self.evaluation_utils.evaluate_graph_test_maas(self, directory, is_test=True, params=params)

        new_data = self.data_utils.create_result_data(self.round, score)
        data.append(new_data)

        self.data_utils.save_results(json_file_path, data)

        return score
