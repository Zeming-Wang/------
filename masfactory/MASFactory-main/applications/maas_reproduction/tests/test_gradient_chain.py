import torch

from maas_reproduction.nodes.loss_update_node import loss_update_forward


class DummySettings:
    mode = "Graph"


def test_gradient_chain() -> None:
    parameter = torch.nn.Parameter(torch.tensor(0.5))
    controller = torch.nn.Module()
    controller.register_parameter("weight", parameter)
    optimizer = torch.optim.Adam(controller.parameters(), lr=0.1)
    logprob = torch.log(torch.sigmoid(parameter))

    attributes = {
        "previous_cost": [0.0],
        "batch_logprobs": [],
        "batch_scores": [],
        "batch_costs": [],
        "all_scores": [],
        "batch_size": 1,
        "settings": DummySettings(),
        "optimizer": optimizer,
        "controller": controller,
        "device": parameter.device,
    }

    before = parameter.detach().clone()
    result = loss_update_forward(
        {
            "score": 1.0,
            "cost": 0.1,
            "logprob": logprob,
            "problem_index": 0,
        },
        attributes,
    )
    after = parameter.detach().clone()

    print("logprob type:", type(logprob))
    print("requires_grad:", logprob.requires_grad)
    print("grad_fn:", logprob.grad_fn)
    print("loss:", result["result_loss"])
    print("update_performed:", result["result_update_performed"])
    print("parameter before:", before.item())
    print("parameter after:", after.item())
    print("parameter changed:", not torch.equal(before, after))

    assert logprob.requires_grad
    assert logprob.grad_fn is not None
    assert result["result_update_performed"]
    assert not torch.equal(before, after)
