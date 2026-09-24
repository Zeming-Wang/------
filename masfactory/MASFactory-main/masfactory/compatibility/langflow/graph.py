from masfactory.compatibility.common.output_root import WorkflowOutputRootGraph


class LangflowRootGraph(WorkflowOutputRootGraph):
    _outputs_key = "langflow_outputs"
    _node_id_hints = ("ChatOutput",)
