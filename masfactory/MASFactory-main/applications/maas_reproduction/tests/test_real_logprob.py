import asyncio
import os
import sys
from types import SimpleNamespace

import torch

maas_project_root = os.environ["METAGPT_PROJECT_ROOT"]
if maas_project_root in sys.path:
    sys.path.remove(maas_project_root)
sys.path.insert(0, maas_project_root)

from assets.optimized.GSM8K.train.graph import Workflow


class GradientController(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, problem, operator_embeddings, selection_operator_names):
        return [torch.log(torch.sigmoid(self.weight))], [["Generate"]]


class GenerateOperator:
    async def __call__(self, input, instruction):
        return {"response": "2"}


class ProgrammerOperator:
    async def __call__(self, problem, analysis):
        return {"code": "def solve(): return 2", "output": "2"}


def test_real_workflow_returns_gradient_logprob() -> None:
    controller = GradientController()
    workflow = Workflow.__new__(Workflow)
    workflow.device = torch.device("cpu")
    workflow.controller = controller
    workflow.operator_embeddings = torch.zeros(1, 384)
    workflow.selection_operator_names = ["Generate"]
    workflow.selection_operator_instances = {"Generate": GenerateOperator()}
    workflow.programmer = ProgrammerOperator()
    workflow.llm = SimpleNamespace(cost_manager=SimpleNamespace(total_cost=0.0))

    prediction, cost, logprob = asyncio.run(workflow("What is 1 + 1?"))

    print("prediction:", prediction)
    print("cost:", cost)
    print("logprob type:", type(logprob))
    print("requires_grad:", logprob.requires_grad)
    print("grad_fn:", logprob.grad_fn)

    assert prediction == "2"
    assert cost == 0.0
    assert isinstance(logprob, torch.Tensor)
    assert logprob.requires_grad
    assert logprob.grad_fn is not None
